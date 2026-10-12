from dataclasses import replace
from pathlib import Path
import math
import tempfile
import unittest
import torch
from race_rl.architecture import (DesignEnvironment,DesignOracle,CountAction,RetileAction,
    resized_mapping,retiled_mapping,nondominated)
from race_rl.examples import toy_workload,initial_mapping
from race_rl.mesh import mesh_hardware,mesh_shape
from race_rl.network import evaluate_network
from race_rl.serialization import digest
from race_rl.oracle import Oracle,BudgetExhausted
from race_rl.training import RACETrainer


class UnifiedDesignTests(unittest.TestCase):
    def env(self,**kw):
        return DesignEnvironment(toy_workload(),compute_counts=(2,4,8),budget=12,**kw)

    def test_complete_mesh_and_constant_per_die_specs(self):
        for count in (1,2,4,6,8):
            h = mesh_hardware(count)
            rows,cols = mesh_shape(count)
            self.assertEqual(len(h.nodes),rows*cols)
            self.assertEqual(len(h.links),rows*(cols-1)+cols*(rows-1))
            for link in h.links:
                a,b = h.node_map[link.a],h.node_map[link.b]
                self.assertEqual(abs(a.x_mm-b.x_mm)+abs(a.y_mm-b.y_mm),2)
            self.assertEqual({n.operations_per_s for n in h.nodes if n.kind == 'compute'},{1e9})
            self.assertEqual(evaluate_network(h).area_mm2,count)
            self.assertEqual(len(h.memory_ids),2)
        for bad in (0,-1,True,1.5):
            with self.assertRaises(ValueError): mesh_hardware(bad)

    def test_global_budget_cache_distinguishes_hardware(self):
        w = toy_workload()
        a,b = mesh_hardware(2),mesh_hardware(4)
        mapping = initial_mapping(w,a)
        oracle = DesignOracle(w,2)
        x,y = oracle.evaluate(a,mapping),oracle.evaluate(b,mapping)
        self.assertEqual(x.mapping_hash,y.mapping_hash)
        self.assertNotEqual(x.hardware_hash,y.hardware_hash)
        self.assertEqual(oracle.evaluations,2)
        self.assertIs(oracle.evaluate(a,mapping),x)
        self.assertEqual(oracle.cache_hits,1)
        with self.assertRaises(BudgetExhausted): oracle.evaluate(mesh_hardware(8),mapping)
        self.assertEqual(oracle.requests,3)

    def test_resize_updates_state_and_revisits_paid_cache(self):
        env = self.env()
        old = env.reset()
        initial = digest(env.current_mapping)
        env.step(CountAction(8))
        self.assertEqual(env.observation.compute_count,8)
        self.assertNotEqual(len(old.graph.hardware_ids),len(env.observation.graph.hardware_ids))
        env.step(CountAction(2))
        self.assertEqual(digest(env.current_mapping),initial)
        self.assertEqual(env.oracle.evaluations,2)
        self.assertEqual(env.oracle.cache_hits,1)
        self.assertEqual(env.oracle.requests,3)
        self.assertIn(CountAction(8),env.actions)

    def test_poor_mapping_does_not_prune_count_or_reject_worse_state(self):
        env = self.env(); env.reset()
        h = env.hardware_candidates[4]
        backend = Oracle(env.workload,h,budget=12,scheduling_policy='serial_node_tiles')
        original = backend._run
        def costly(mapping,started):
            r = original(mapping,started)
            return replace(r,energy_j=r.energy_j*100,edp_js=r.edp_js*100,average_power_w=r.average_power_w*100)
        backend._run = costly
        env.oracle.backends[digest(h)] = backend
        env.step(CountAction(4))
        self.assertGreater(env.current_objective,1)
        self.assertEqual(env.best_objective,1)
        self.assertEqual(env.compute_count,4)
        self.assertTrue(any(isinstance(a,RetileAction) for a in env.actions))
        env.step(CountAction(2))
        self.assertIn(CountAction(4),env.actions)
        env.step(CountAction(4))
        self.assertEqual(env.compute_count,4)
        self.assertEqual(env.best_objective,1)

    def test_whole_operator_retile_and_shrink_preserve_exact_coverage(self):
        env = self.env(); env.reset(); env.step(CountAction(8))
        action = RetileAction('conv_a',4,'c7',True)
        self.assertIn(action,env.actions)
        env.step(action)
        self.assertEqual(len([t for t in env.current_mapping.tiles if t.operator == 'conv_a']),4)
        env.current_mapping.validate(env.workload,env.hardware)
        env.step(CountAction(2))
        self.assertTrue(set(env.current_mapping.placements.values()) <= {'c0','c1'})
        env.current_mapping.validate(env.workload,env.hardware)
        self.assertTrue(any(isinstance(a,RetileAction) and a.parts == 1 and a.operator == 'conv_a' for a in env.actions))
        self.assertAlmostEqual(sum(r['reward'] for r in env.trace),math.log(1/env.best_objective),places=12)

    def test_hard_area_mask_and_failed_candidate_counts(self):
        env = self.env(max_area_mm2=4); env.reset()
        self.assertIn(CountAction(4),env.actions)
        self.assertNotIn(CountAction(8),env.actions)
        with self.assertRaises(ValueError): env.step(CountAction(8))
        h = env.hardware_candidates[4]
        backend = Oracle(env.workload,h,budget=12,scheduling_policy='serial_node_tiles')
        original = backend._run
        backend._run = lambda m,s: replace(original(m,s),feasible=False,violations=('forced capacity regression',))
        env.oracle.backends[digest(h)] = backend
        before = digest(env.current_mapping)
        env.step(CountAction(4))
        self.assertEqual(env.oracle.evaluations,2)
        self.assertEqual(env.compute_count,2)
        self.assertEqual(digest(env.current_mapping),before)
        self.assertIn(CountAction(4),env.actions)

    def test_one_float64_policy_updates_count_retile_mapping_and_critic(self):
        env = self.env(); o = env.reset()
        trainer = RACETrainer(seed=3,width=8,epochs=2,precision='float64')
        before = {k:v.clone() for k,v in trainer.policy.state_dict().items()}
        samples = []
        for kind in ('resize_mesh','retile_operator','cut_migrate'):
            d,v = trainer.policy.distribution(o)
            self.assertEqual(d.probs.dtype,torch.float64)
            self.assertAlmostEqual(d.probs.sum().item(),1,places=12)
            a = next(a for a in o.legal_actions if a.kind == kind)
            i = o.legal_actions.index(a)
            next_o,r,_ = env.step(a)
            samples.append((o,i,d.log_prob(torch.tensor(i)).item(),v.item(),r))
            o = next_o
        stats = {'parameter_updates':0}; trainer.update(samples,stats)
        after = trainer.policy.state_dict()
        for prefix in ('count_score','retile_score','root_score','encoder','critic'):
            self.assertTrue(any(not torch.equal(before[k],after[k]) for k in before if k.startswith(prefix)),prefix)
        self.assertEqual(stats['parameter_updates'],2)
        self.assertTrue(all(torch.isfinite(v).all() for v in after.values()))

    def test_exploration_floor_masks_and_double_checkpoint(self):
        env = self.env(); o = env.reset()
        trainer = RACETrainer(seed=4,width=8,epochs=1,precision='float64')
        d,_ = trainer.policy.distribution(o)
        ci = [i for i,a in enumerate(o.legal_actions) if isinstance(a,CountAction)]
        self.assertTrue(all(d.probs[i].item() >= 0.05/3*0.05/len(ci)-1e-14 for i in ci))
        masks = tuple(i != ci[0] for i in range(len(o.legal_actions)))
        masked,_ = trainer.policy.distribution(replace(o,legal_action_masks=masks))
        self.assertEqual(masked.probs[ci[0]].item(),0)
        trainer.run_episode(env)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.pt'; trainer.save(path,experiment_identity='joint')
            resumed = RACETrainer.load(path,experiment_identity='joint')
            self.assertEqual(resumed.config['precision'],'float64')
            a,b = self.env(),self.env()
            sa = trainer.run_episode(a)
            resumed = RACETrainer.load(path,experiment_identity='joint')
            sb = resumed.run_episode(b)
            self.assertEqual(sa,sb); self.assertEqual(a.trace,b.trace)

    def test_actual_capacity_failure_does_not_mask_repartition(self):
        from race_rl.workload import Workload,Tensor,Operator
        w = Workload((Tensor('x',(1,1024)),Tensor('y',(1,8192))),
            (Operator('fc','Linear',('x',),'y'),),('y',))
        env = DesignEnvironment(w,compute_counts=(2,4),budget=4)
        env.reset()
        bad = RetileAction('fc',1,'c0',False)
        self.assertIn(bad,env.actions)
        before = digest(env.current_mapping)
        env.step(bad)
        self.assertFalse(env.trace[-1]['feasible'])
        self.assertEqual(env.oracle.evaluations,2)
        self.assertEqual(digest(env.current_mapping),before)
        self.assertIn(CountAction(4),env.actions)
        self.assertTrue(any(isinstance(a,RetileAction) and a.parts >= 2 for a in env.actions))
        self.assertIn(bad,env.actions)

    def test_whole_operator_retiling_keeps_independent_numerical_semantics(self):
        import numpy as np
        from race_rl.reference import execute
        w = toy_workload(); h = mesh_hardware(4); mapping = initial_mapping(w,h)
        rng = np.random.default_rng(7)
        x = rng.normal(size=w.tensor_map['x'].shape)
        weights = {o.id:rng.normal(size=w.weight_shape(o)) for o in w.operators if w.weight_shape(o)}
        expected = execute(w,mapping.tiles,{'x':x},weights)['y']
        for operator in ('conv_a','conv_b','residual'):
            for count in (1,2,3,4):
                mapping = retiled_mapping(w,h,mapping,RetileAction(operator,count,'c3',True))
                mapping.validate(w,h)
                actual = execute(w,mapping.tiles,{'x':x},weights)['y']
                np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)

    def test_frontier_uses_absolute_ppa(self):
        rows = [dict(feasible=True,average_power_w=p,energy_j=e,latency_s=d,area_mm2=a,compute_count=n)
            for p,e,d,a,n in ((1,1,1,4,2),(2,2,2,6,4),(2,0.5,0.5,10,8))]
        self.assertEqual({r['compute_count'] for r in nondominated(rows)},{2,8})
