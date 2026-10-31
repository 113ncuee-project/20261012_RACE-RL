import math
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import common_dse_bayes
import common_dse_timeloop_restricted
from race_rl.serialization import digest

class FakeEvaluator:
    def __init__(self,budget):
        self.environment=SimpleNamespace(workload=SimpleNamespace(operators=[SimpleNamespace(kind='Conv') for _ in range(4)]))
        self.budget=budget;self.requests=0;self.trace=[]
    def evaluate(self,design):
        self.requests+=1
        assert all(p.target<design.compute_count for p in design.plans)
        result={'assessment':{'search_merit':design.compute_count+sum(p.h for p in design.plans)/100},'design_hash':digest(design)}
        self.trace.append(result);return result
    def snapshot(self):return {'requests':self.requests,'trace':self.trace}

class BaselineTests(unittest.TestCase):
    def test_gp_expected_improvement_is_finite_with_duplicate_observations(self):
        score=common_dse_bayes.acquisition([np.array([0.,1.])]*2,[1.,1.],
                                            [np.array([0.,1.]),np.array([1.,0.])])
        self.assertTrue(all(math.isfinite(x) and x>=0 for x in score))
    def test_reproducible_and_every_proposal_charged(self):
        for module in (common_dse_bayes,common_dse_timeloop_restricted):
            a=FakeEvaluator(12);b=FakeEvaluator(12)
            module.run(a,seed=311);module.run(b,seed=311)
            self.assertEqual(a.requests,12)
            self.assertEqual([v['design_hash'] for v in a.trace],[v['design_hash'] for v in b.trace])
