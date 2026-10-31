import unittest
from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from common_dse_genetic import run
from race_rl.serialization import digest

class FakeEvaluator:
    def __init__(self, budget):
        self.environment=SimpleNamespace(workload=SimpleNamespace(operators=[SimpleNamespace(kind='Conv') for _ in range(5)]))
        self.budget=budget
        self.requests=0
        self.trace=[]
    def evaluate(self,design):
        self.requests+=1
        assert len(design.plans)==5
        assert all(0<=p.target<design.compute_count for p in design.plans)
        score=design.compute_count+sum(p.h for p in design.plans)/100
        result={'assessment':{'search_merit':score},'design_hash':digest(design)}
        self.trace.append(result)
        return result
    def snapshot(self):
        return {'requests':self.requests,'trace':self.trace}

class GeneticTest(unittest.TestCase):
    def test_genetic_charges_every_proposal_and_reproduces(self):
        first=FakeEvaluator(16);second=FakeEvaluator(16)
        a=run(first,seed=42);b=run(second,seed=42)
        self.assertEqual(a['result']['requests'],16)
        self.assertEqual([x['design_hash'] for x in first.trace],[x['design_hash'] for x in second.trace])
        self.assertEqual(first.trace[0]['phase'],'initial')
        self.assertTrue(any(x['phase']=='tournament_crossover_mutation' for x in first.trace))
