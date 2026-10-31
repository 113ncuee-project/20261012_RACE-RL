"""Check Chiplet-Gym's adapted PPO and SA use the common request budget."""
import importlib.util
from pathlib import Path
import unittest

from test_cnn_hierarchy import bundle
from race_rl.accelerators.cnn_environment import GemminiCNNEnvironment
from race_rl.accelerators.cnn_search import fixture
from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
from race_rl.requirements import PPARequirements

path = Path(__file__).resolve().parents[1]/'tools/chiplet_gym_common.py'
spec = importlib.util.spec_from_file_location('chiplet_gym_common',path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ChipletGymCommonTests(unittest.TestCase):
    @unittest.skipUnless((module.DEPS/'stable_baselines3').exists(),'optional Chiplet-Gym dependencies missing')
    def test_ppo_and_sa_together_use_exact_budget(self):
        workload, _ = bundle()
        memory = MemoryHierarchy()
        def make_env(budget):
            return GemminiCNNEnvironment(workload.workload, memory=memory,
                cost_profile=fixture(memory), allow_test_fixture=True, budget=budget,
                requirements=PPARequirements(10,100,100))
        result = module.run_chiplet_gym_common(make_env, budget=4, seed=31)
        self.assertEqual(result['oracle_requests'],4)
        self.assertEqual(result['oracle_evaluations']+result['cache_hits'],4)
        self.assertEqual([row['phase'] for row in result['trace']], ['ppo','ppo','sa','sa'])
        self.assertFalse(result['protocol']['original_pretrained_used'])


if __name__ == '__main__':
    unittest.main()
