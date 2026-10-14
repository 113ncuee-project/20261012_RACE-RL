"""Action-triggered Gemmini feedback, real REINFORCE wiring and missing-PPA gates."""
from dataclasses import replace, asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from race_rl.accelerators.gemmini_environment import GemminiDesignEnvironment, GemminiDesign
from race_rl.accelerators.gemmini_model import GemmShape, GemminiModel
from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
from race_rl.accelerators.gemmini_ppa import GemminiCostProfile, load_cost_profile
from race_rl.training import RACETrainer
from race_rl.requirements import PPARequirements
from race_rl.recording import policy_hash


def fixture():
    keys = [key for key in GemminiCostProfile.__dataclass_fields__ if key not in ('name','evidence_status','sources','memory_hierarchy_identity')]
    return GemminiCostProfile('not_hardware_data','TEST_FIXTURE',{key:'synthetic integration test' for key in keys},
        100e6,1,MemoryHierarchy().identity,1e-12,1e-12,1e-12,1e-12,1e-12,1e-11,1e-12,1e-12,.001,.001)


class GemminiRLTests(unittest.TestCase):
    def env(self, **kwargs):
        options = dict(compute_counts=(1,2),budget=4,block_profiles=((16,16,16),(32,16,32)))
        options.update(kwargs)
        return GemminiDesignEnvironment(GemmShape(64,16,32),**options)

    def scored_env(self, **kwargs):
        return self.env(cost_profile=fixture(),allow_test_fixture=True,**kwargs)

    def test_reset_catalogue_encoder_actor_have_zero_model_and_ppa_calls(self):
        original = GemminiModel.evaluate
        with patch.object(GemminiModel,'evaluate',autospec=True,side_effect=original) as evaluate:
            env = self.scored_env()
            observation = env.reset()
            trainer = RACETrainer(width=8,precision='float64',policy_backend='gemmini')
            distribution,value = trainer.policy.distribution(observation)
            self.assertIsNone(value)
            self.assertTrue(torch.isfinite(distribution.logits).all())
            self.assertEqual(evaluate.call_count,0)
            self.assertEqual(env.oracle.requests,0)
            self.assertEqual(env.oracle.evaluations,0)
            action,_,_ = trainer.policy.select(observation)
            self.assertEqual(evaluate.call_count,0)
            new_observation,r,done = env.step(action)
            self.assertEqual(evaluate.call_count,1)
            self.assertEqual(env.oracle.requests,1)
            self.assertTrue(new_observation.ppa_valid)
            self.assertTrue(new_observation.last_reward_valid)
            self.assertEqual(new_observation.graph.global_features[9],r)

    def test_no_ppa_still_returns_model_state_but_does_not_invent_reward(self):
        env = self.env()
        observation = env.reset()
        self.assertFalse(observation.model_feedback_valid)
        next_o,r,done = env.step(env.actions[0])
        self.assertIsNone(r)
        self.assertTrue(done)
        self.assertTrue(next_o.model_feedback_valid)
        self.assertFalse(next_o.ppa_valid)
        self.assertFalse(next_o.last_reward_valid)
        self.assertIsNone(env.trace[0]['ppa_assessment'])
        self.assertEqual(env.termination,'ppa_unavailable')

    def test_trainer_refuses_missing_reward_without_changing_parameters(self):
        trainer = RACETrainer(width=8,precision='float64',policy_backend='gemmini')
        before = policy_hash(trainer.policy)
        env = self.env()
        with self.assertRaisesRegex(ValueError,'unscorable feedback'):
            trainer.run_episode(env)
        self.assertEqual(env.oracle.requests,1)
        self.assertEqual(before,policy_hash(trainer.policy))
        self.assertEqual(trainer.episodes,0)

    def test_same_reinforce_trainer_updates_once_and_frozen_mode_does_not_update(self):
        trainer = RACETrainer(width=8,precision='float64',policy_backend='gemmini')
        before = policy_hash(trainer.policy)
        env = self.scored_env()
        stats = trainer.run_episode(env)
        self.assertEqual(stats['steps'],4)
        self.assertEqual(stats['parameter_updates'],1)
        self.assertEqual(stats['algorithm'],'reinforce')
        self.assertFalse(stats['critic_used'])
        self.assertNotEqual(before,policy_hash(trainer.policy))
        trained = policy_hash(trainer.policy)
        frozen = trainer.run_episode(self.scored_env(),learn=False)
        self.assertEqual(frozen['parameter_updates'],0)
        self.assertEqual(trained,policy_hash(trainer.policy))
        self.assertEqual(trainer.episodes,1)

    def test_schema7_exposes_banks_coordinates_execution_flags_and_actual_feedback(self):
        env = self.scored_env()
        o = env.reset()
        graph = o.graph
        self.assertEqual(graph.schema_version,7)
        self.assertEqual(len(graph.tile_features[0]),23)
        self.assertEqual(len(graph.global_features),22)
        self.assertIn('c0.gb0',graph.hardware_ids)
        self.assertIn('c0.acc0',graph.hardware_ids)
        self.assertIn('c0.dma',graph.hardware_ids)
        sp = graph.hardware_features[graph.hardware_ids.index('c0.gb0')]
        self.assertAlmostEqual(float(torch.expm1(torch.tensor(sp[4],dtype=torch.float64))),65536)
        next_o,r,_ = env.step(env.actions[0])
        self.assertEqual(next_o.compute_count,env.current_design.compute_count)
        self.assertTrue(next_o.graph.global_features[8])
        self.assertEqual(next_o.graph.global_features[9],r)
        self.assertEqual(next_o.graph.global_features[4],env.current_record.average_power_w)
        self.assertEqual(next_o.graph.global_features[5],env.current_record.latency_s)

    def test_all_counts_are_present_even_when_area_limit_is_exceeded(self):
        env = GemminiDesignEnvironment(GemmShape(64,16,32),budget=4,
            requirements=PPARequirements(1,1,2),cost_profile=fixture(),allow_test_fixture=True,
            block_profiles=((16,16,16),))
        env.reset()
        self.assertEqual({action.compute_count for action in env.actions},set(range(1,17)))
        first = next(a for a in env.actions if a.compute_count==1 and a.retain_weights)
        env.step(first)
        best = env.best_record
        worse = next(a for a in env.actions if a.compute_count==16 and a.target==0 and a.retain_weights)
        _,r,_ = env.step(worse)
        self.assertEqual(env.compute_count,16)
        self.assertEqual(env.best_record,best)
        self.assertFalse(env.feasible(env.current_record))
        self.assertEqual(r,0)

    def test_revisiting_selected_design_is_one_feedback_but_zero_new_model_calls(self):
        env = self.scored_env(budget=5)
        env.reset()
        a = next(action for action in env.actions if action.compute_count==1)
        original = GemminiModel.evaluate
        with patch.object(GemminiModel,'evaluate',autospec=True,side_effect=original) as evaluate:
            env.step(a)
            b = next(action for action in env.actions if action.compute_count==2)
            env.step(b)
            env.step(a)
            self.assertEqual(evaluate.call_count,2)
        self.assertEqual((env.oracle.requests,env.oracle.evaluations,env.oracle.cache_hits),(3,2,1))

    def test_capacity_failure_keeps_current_and_has_no_synthetic_fallback(self):
        env = GemminiDesignEnvironment(GemmShape(16,64,4096),compute_counts=(1,2),budget=4,
            cost_profile=fixture(),allow_test_fixture=True,block_profiles=((16,16,16),(64,64,64)))
        env.reset()
        bad = next(action for action in env.actions if action.n_tile==64 and action.retain_weights)
        current = env.current_design
        with patch('race_rl.oracle.Oracle.evaluate',side_effect=AssertionError('legacy oracle fallback')):
            _,r,_ = env.step(bad)
        self.assertEqual(env.current_design,current)
        self.assertEqual(r,0)
        self.assertFalse(env.trace[0]['physical_feasible'])
        self.assertEqual(env.trace[0]['status'],'PHYSICAL_INVALID')

    def test_cost_sources_and_test_fixture_permission_are_required(self):
        profile = fixture()
        with self.assertRaisesRegex(ValueError,'allow_test_fixture'): self.env(cost_profile=profile)
        with self.assertRaises(ValueError): load_cost_profile(asdict(replace(profile,clock_hz=None)),allow_test_fixture=True)
        with self.assertRaises(ValueError): replace(profile,sources={}).validate(allow_test_fixture=True)
        with self.assertRaises(ValueError): self.env(cost_profile=replace(profile,evidence_status='UNAVAILABLE'))
        record = self.scored_env(); record.reset(); record.step(record.actions[0])
        detail = record.current_record.ppa_detail
        self.assertEqual(detail['confidence'],'TEST_ONLY')
        self.assertEqual(detail['timing_calibration_status'],'ANALYTICAL_UNVALIDATED')
        self.assertAlmostEqual(detail['energy_j'],sum(detail['energy_components'].values()),places=18)

    def test_gemmini_checkpoint_reloads_its_backend_and_rejects_other_model_identity(self):
        trainer = RACETrainer(width=8,precision='float64',policy_backend='gemmini')
        env = self.scored_env()
        trainer.run_episode(env)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'checkpoint.pt'
            trainer.save(path,experiment_identity=env.experiment_identity)
            restored = RACETrainer.load(path,experiment_identity=env.experiment_identity)
            self.assertEqual(restored.config['policy_backend'],'gemmini')
            self.assertEqual(policy_hash(restored.policy),policy_hash(trainer.policy))
            with self.assertRaises(ValueError): RACETrainer.load(path,experiment_identity='legacy-design-oracle')

    def test_legacy_policy_still_rejects_gemmini_schema_and_graph_tasks_are_explicit(self):
        from race_rl.examples import toy_workload
        with self.assertRaises(ValueError): GemminiDesignEnvironment(toy_workload())
        env = self.scored_env(); o = env.reset()
        legacy = RACETrainer(width=8)
        with self.assertRaises(ValueError): legacy.policy.encoder(o.graph)
        with self.assertRaises(ValueError): GemminiDesign(2,16,16,16,True,True,2,True)


if __name__=='__main__': unittest.main()
