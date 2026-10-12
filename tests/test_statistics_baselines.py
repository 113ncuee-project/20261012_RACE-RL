import unittest
from race_rl.statistics import paired_summary,holm_adjust
from race_rl.examples import toy_workload,toy_hardware,initial_mapping
from race_rl.environment import RACEEnvironment
from race_rl.baselines import run_baseline


class StatisticsBaselineTests(unittest.TestCase):
    def test_constant_paired_difference_ci(self):
        result = paired_summary([1,2,3],[2,3,4],bootstrap_samples=100)
        self.assertEqual(result["paired_bootstrap_ci95"],[-1,-1])
        self.assertEqual(result["paired_std"],0)
        self.assertEqual(holm_adjust([0.01,0.04,0.03]),[0.03,0.06,0.06])

    def test_rejects_unpaired_or_missing_statistics(self):
        for a,b in (([1],[2]),([1,2],[2]),([1,float("nan")],[1,2])):
            with self.assertRaises(ValueError): paired_summary(a,b)

    def test_baselines_enforce_actual_oracle_budget(self):
        w,h = toy_workload(),toy_hardware()
        for method in ("random","greedy","sa"):
            env = RACEEnvironment(w,h,initial_mapping(w,h),budget=4,max_scope=2)
            result = run_baseline(env,method)
            self.assertLessEqual(result["oracle_evaluations"],4)
            self.assertEqual(result["oracle_requests"],result["oracle_evaluations"]+result["cache_hits"])
            self.assertLessEqual(result["best_objective"],1)
