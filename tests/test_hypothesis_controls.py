from dataclasses import replace
import math
import unittest
import torch
from race_rl.architecture import DesignEnvironment,DEFAULT_COUNTS,CountAction,IndependentAction,RetileAction
from race_rl.examples import toy_workload
from race_rl.mesh import mesh_hardware,mesh_shape
from race_rl.training import RACETrainer
from race_rl.serialization import digest


class HypothesisControlTests(unittest.TestCase):
    def env(self,**kw):
        return DesignEnvironment(toy_workload(),compute_counts=(2,4,16),budget=4,**kw)

    def test_sixteen_compute_mesh_is_real_reachable_state(self):
        self.assertEqual(DEFAULT_COUNTS,tuple(range(2,17,2)))
        self.assertEqual(mesh_shape(16),(4,4))
        env=self.env();env.reset();env.step(CountAction(16))
        self.assertEqual(len(env.hardware.compute_ids),16)
        self.assertEqual(len(env.hardware.nodes),16)
        self.assertEqual(env.current_record.area_mm2,16)
        self.assertEqual(env.oracle.evaluations,2)
        self.assertEqual(env.observation.compute_count,16)
        self.assertIn(CountAction(2),env.actions)

    def test_single_edits_change_one_design_variable(self):
        env=self.env(action_mode='independent');env.reset()
        self.assertTrue(all(isinstance(a,(CountAction,IndependentAction)) for a in env.actions))
        old=env.current_mapping
        action=next(a for a in env.actions if a.kind=='place_tile')
        env.step(action)
        self.assertEqual(sum(old.placements[t]!=env.current_mapping.placements[t] for t in old.placements),1)
        self.assertEqual(old.storage,env.current_mapping.storage)
        previous=env.current_mapping
        action=next(a for a in env.actions if a.kind=='store_object')
        env.step(action)
        self.assertEqual(previous.placement,env.current_mapping.placement)
        self.assertEqual(sum(previous.storages[n]!=env.current_mapping.storages[n] for n in previous.storages),1)
        storage_after_edit=env.current_mapping.storage
        action=next(a for a in env.actions if a.kind=='partition_operator')
        env.step(action)
        env.current_mapping.validate(env.workload,env.hardware)
        self.assertEqual(storage_after_edit,env.current_mapping.storage)

    def test_reward_variants_telescope_over_same_cross_hardware_trajectory(self):
        objectives=[]
        for variant,c in (('dense',0),('terminal',0),('potential',0),('potential',0.1)):
            env=self.env(reward_variant=variant,lambda_c=c);env.reset()
            initial_phi=env.phi()
            for count in (4,16,2):env.step(CountAction(count))
            env.step(RetileAction('conv_a',2,'c0',False))
            self.assertTrue(env.done)
            actual=sum(r['reward'] for r in env.trace)
            expected=math.log(1/env.best_objective)-(initial_phi if variant=='potential' else 0)
            self.assertAlmostEqual(actual,expected,places=12)
            self.assertEqual(env.phi(),0)
            objectives.append(env.best_objective)
        self.assertEqual(len(set(objectives)),1)

    def test_state_ablation_switches_change_actual_action_probabilities(self):
        env=self.env();o=env.reset()
        baseline=None
        for state in ('full','compact','dnngraph','hardwaregraph','no_attribution','no_criticality'):
            trainer=RACETrainer(seed=0,width=16,precision='float64',state_variant=state)
            distribution,_=trainer.policy.distribution(o)
            if baseline is None:baseline=distribution.probs.detach()
            else:self.assertGreater(torch.max(torch.abs(distribution.probs.detach()-baseline)).item(),1e-12,state)

    def test_parameter_counts_match_state_variants_and_frozen_policy_does_not_update(self):
        counts=[]
        for state in ('full','compact','dnngraph','hardwaregraph','no_attribution','no_criticality'):
            trainer=RACETrainer(seed=0,width=8,epochs=1,precision='float64',state_variant=state)
            counts.append(sum(p.numel() for p in trainer.policy.parameters()))
            for mode in ('coupled','independent'):
                env=self.env(action_mode=mode)
                before={k:v.clone() for k,v in trainer.policy.state_dict().items()}
                trainer.run_episode(env,learn=False)
                self.assertTrue(all(torch.equal(v,trainer.policy.state_dict()[k]) for k,v in before.items()))
                self.assertEqual(trainer.training_evaluations,0)
        self.assertEqual(len(set(counts)),1)
