"""Budget and reproducibility checks for CNN comparison controllers."""
import unittest
import importlib.util
from pathlib import Path

from test_cnn_hierarchy import bundle
from race_rl.accelerators.cnn_baselines import run_cnn_baseline
from race_rl.accelerators.cnn_environment import GemminiCNNEnvironment
from race_rl.accelerators.cnn_search import fixture
from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
from race_rl.requirements import PPARequirements
from race_rl.comparisons.cnn_ppo import CNNPPOTrainer, ChipletGymAdaptedTrainer, FlatCNNEncoder
from race_rl.recording import policy_hash


class CNNBaselineTests(unittest.TestCase):
    def environment(self):
        workload, _ = bundle()
        memory = MemoryHierarchy()
        return GemminiCNNEnvironment(workload.workload, memory=memory,
            cost_profile=fixture(memory), allow_test_fixture=True, budget=5,
            requirements=PPARequirements(10, 100, 100))

    def test_random_and_sa_use_exact_request_budget_and_are_repeatable(self):
        for method in ('random', 'sa'):
            first = run_cnn_baseline(self.environment(), method, seed=19)
            second = run_cnn_baseline(self.environment(), method, seed=19)
            self.assertEqual(first['oracle_requests'], 5)
            self.assertEqual(first['oracle_requests'], first['oracle_evaluations'] + first['cache_hits'])
            self.assertEqual([row['choices'] for row in first['trace']],
                             [row['choices'] for row in second['trace']])
            self.assertEqual(first['assessment'], second['assessment'])

    def test_unscored_runs_are_rejected(self):
        env = self.environment()
        env.cost_profile = None
        with self.assertRaisesRegex(ValueError, 'cost profile'):
            run_cnn_baseline(env, 'random')

    def test_cnn_ppo_consumes_budget_and_updates_policy(self):
        env = self.environment()
        trainer = CNNPPOTrainer(seed=19, epochs=2)
        before = policy_hash(trainer.policy)
        stats = trainer.run_episode(env)
        self.assertEqual(stats['oracle_requests'], 5)
        self.assertEqual(stats['parameter_updates'], 2)
        self.assertTrue(stats['critic_used'])
        self.assertNotEqual(policy_hash(trainer.policy), before)

    def test_chiplet_gym_adaptation_uses_flat_encoder_and_runs(self):
        env = self.environment()
        trainer = ChipletGymAdaptedTrainer(seed=19, epochs=2)
        self.assertIsInstance(trainer.policy.encoder, FlatCNNEncoder)
        self.assertEqual(trainer.config['gamma'], 0.99)
        self.assertEqual(trainer.config['gae_lambda'], 0.95)
        stats = trainer.run_episode(env)
        self.assertEqual(stats['oracle_requests'], 5)
        self.assertEqual(stats['parameter_updates'], 2)
        self.assertTrue(stats['critic_used'])

    def test_exploration_limits_have_a_real_witness(self):
        path = Path(__file__).resolve().parents[1] / 'tools/explore_cnn_requirements.py'
        spec = importlib.util.spec_from_file_location('requirements_exploration', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaisesRegex(ValueError, 'no physically valid'):
            module.choose_limits([{'physical_feasible': False, 'ppa': None}])
        row = {'physical_feasible': True, 'ppa': {'average_power_w': 2.0,
               'latency_s': 3.0, 'area_mm2': 5.0}, 'design_hash': 'sample'}
        limits, witness, count = module.choose_limits([row])
        self.assertEqual(count, 1)
        self.assertEqual(witness, row)
        self.assertGreater(limits.max_power_w, row['ppa']['average_power_w'])
        self.assertGreater(limits.max_latency_s, row['ppa']['latency_s'])
        self.assertGreater(limits.max_area_mm2, row['ppa']['area_mm2'])


if __name__ == '__main__':
    unittest.main()
