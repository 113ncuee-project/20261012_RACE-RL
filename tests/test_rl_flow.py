import unittest
from unittest.mock import patch
from race_rl.architecture import DesignEnvironment,DesignOracle,CountAction,search_mesh
from race_rl.environment import RACEEnvironment
from race_rl.examples import toy_workload,toy_hardware,initial_mapping
from race_rl.training import RACETrainer
from race_rl.oracle import Oracle


class RLFlowTests(unittest.TestCase):
    def test_reset_has_no_oracle_or_native_setup_and_unknown_features_are_marked(self):
        w,h=toy_workload(),toy_hardware()
        for env in (DesignEnvironment(w,budget=1),RACEEnvironment(w,h,initial_mapping(w,h),budget=1)):
            with patch.object(Oracle,'__init__',side_effect=AssertionError('native setup before action')):
                with patch.object(DesignOracle,'evaluate',side_effect=AssertionError('PPA before action')):
                    o=env.reset()
            graph=getattr(o,'graph',o)
            self.assertEqual(env.oracle.evaluations,0)
            self.assertEqual(env.oracle.requests,0)
            self.assertIsNone(env.current_record)
            self.assertIsNone(env.best_record)
            self.assertFalse(env.done)
            self.assertEqual(graph.global_features[8],0)
            self.assertFalse(graph.feature_valid_masks['global'][4])
            self.assertTrue(all(not mask[-1] for mask in graph.feature_valid_masks['tile']))

    def test_one_action_budget_observes_reward_and_terminates(self):
        env=DesignEnvironment(toy_workload(),compute_counts=(2,4),budget=1)
        env.reset()
        o,r,done=env.step(CountAction(4))
        self.assertTrue(done)
        self.assertEqual(env.oracle.evaluations,1)
        self.assertEqual(env.oracle.requests,1)
        self.assertEqual(o.graph.global_features[8],1)
        self.assertEqual(o.graph.global_features[9],r)
        self.assertTrue(o.graph.feature_valid_masks['global'][4])

    def test_all_failed_actions_do_not_invent_a_best_design(self):
        from dataclasses import replace
        from pathlib import Path
        import tempfile
        real=DesignOracle.evaluate
        def fail(oracle,h,m):
            return replace(real(oracle,h,m),feasible=False,violations=('forced physical failure',))
        with tempfile.TemporaryDirectory() as folder,patch.object(DesignOracle,'evaluate',fail):
            summary=search_mesh(toy_workload(),Path(folder)/'run',compute_counts=(2,4),episodes=1,budget=1,width=8)
            self.assertIsNone(summary['selected_compute_count'])
            self.assertIsNone(summary['selected_assessment']['metrics']['latency_s'])
            self.assertFalse(list(Path(folder).rglob('*_evaluation.json')))

    def test_action_selection_does_not_call_ppa_and_transition_calls_environment(self):
        env=DesignEnvironment(toy_workload(),compute_counts=(2,1,4,16),budget=3)
        observation=env.reset()
        trainer=RACETrainer(seed=7,width=8,precision='float64')
        before=env.oracle.evaluations
        with patch.object(env.oracle,'evaluate',side_effect=AssertionError('PPA called during actor selection')):
            with patch.object(Oracle,'evaluate',side_effect=AssertionError('native PPA called during actor selection')):
                action,log_probability,value=trainer.policy.select(observation)
        self.assertEqual(env.oracle.evaluations,before)
        original=env.oracle.evaluate
        with patch.object(env.oracle,'evaluate',wraps=original) as oracle_call:
            next_o,reward,done=env.step(action)
        oracle_call.assert_called_once()
        self.assertEqual(env.oracle.requests,1)
        self.assertAlmostEqual(next_o.graph.global_features[4],env.current_record.average_power_w)
        self.assertAlmostEqual(next_o.graph.global_features[5],env.current_record.latency_s)
        self.assertEqual(env.trace[-1]['reward'],reward)
        self.assertEqual(next_o.compute_count,env.compute_count)
        with patch.object(env.oracle,'evaluate',side_effect=AssertionError('PPA called during next actor selection')):
            trainer.policy.select(next_o)

    def test_candidate_generation_does_not_query_oracle(self):
        env=DesignEnvironment(toy_workload(),compute_counts=(2,4),budget=3)
        env.reset()
        with patch.object(env.oracle,'evaluate',side_effect=AssertionError('PPA called by candidate generator')):
            with patch.object(Oracle,'evaluate',side_effect=AssertionError('PPA called by candidate generator')):
                env._refresh()
        self.assertEqual(env.oracle.evaluations,0)

    def test_training_updates_shared_parameters_but_frozen_policy_does_not(self):
        import torch
        env=DesignEnvironment(toy_workload(),compute_counts=(2,4),budget=4)
        trainer=RACETrainer(seed=3,width=8,precision='float64')
        before={k:v.clone() for k,v in trainer.policy.state_dict().items()}
        stats=trainer.run_episode(env)
        self.assertEqual(stats['parameter_updates'],1)
        self.assertTrue(any(not torch.equal(v,trainer.policy.state_dict()[k]) for k,v in before.items() if k.startswith('encoder')))
        self.assertFalse(any(k.startswith('critic') for k in before))
        self.assertTrue(any(not torch.equal(v,trainer.policy.state_dict()[k]) for k,v in before.items() if not k.startswith(('encoder','critic'))))
        frozen={k:v.clone() for k,v in trainer.policy.state_dict().items()}
        trainer.run_episode(env,learn=False)
        self.assertTrue(all(torch.equal(v,trainer.policy.state_dict()[k]) for k,v in frozen.items()))
