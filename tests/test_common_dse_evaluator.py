"""Common DSE evaluation is independent of the RACE action generator."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import unittest

from test_cnn_hierarchy import bundle
from race_rl.accelerators.cnn_design import CNNDesign, LayerPlan, coupled_scope, transaction
from race_rl.accelerators.cnn_environment import GemminiCNNEnvironment
from race_rl.accelerators.cnn_search import fixture
from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
from race_rl.oracle import BudgetExhausted
from race_rl.requirements import PPARequirements

path = Path(__file__).resolve().parents[1]/'tools/common_dse_evaluator.py'
sys.path.insert(0,str(path.parent))
spec = importlib.util.spec_from_file_location('common_dse_evaluator',path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def load_tool(name):
    tool_spec = importlib.util.spec_from_file_location(name,path.parent/f'{name}.py')
    tool = importlib.util.module_from_spec(tool_spec)
    tool_spec.loader.exec_module(tool)
    return tool


class CommonDSEEvaluatorTests(unittest.TestCase):
    def test_complete_designs_cache_and_invalid_proposals_share_budget(self):
        bundle_, _ = bundle()
        memory = MemoryHierarchy()
        env = GemminiCNNEnvironment(bundle_.workload, memory=memory, cost_profile=fixture(memory),
            allow_test_fixture=True, budget=4, requirements=PPARequirements(10,100,100))
        plans = [LayerPlan() for _ in bundle_.workload.operators]
        plans[0] = replace(plans[0],target=1)
        plans[-1] = replace(plans[-1],target=1,keep_weights=True)
        design = CNNDesign(2,tuple(plans))
        evaluator = module.CommonDSEEvaluator(env,3)
        first = evaluator.evaluate(design)
        second = evaluator.evaluate(design)
        third = evaluator.evaluate(None)
        self.assertTrue(first['physical_feasible'])
        self.assertEqual(module.decode_design(first['candidate_encoding']),design)
        self.assertTrue(second['cached'])
        self.assertEqual(third['status'],'STRUCTURALLY_INVALID')
        self.assertEqual(evaluator.requests,3)
        self.assertEqual(evaluator.oracle.evaluations,1)
        self.assertEqual(evaluator.oracle.cache_hits,1)
        self.assertEqual(evaluator.snapshot()['best_design']['compute_count'],2)
        with self.assertRaises(BudgetExhausted): evaluator.evaluate(design)

    def test_invalid_candidate_encoding_is_rejected(self):
        with self.assertRaises(ValueError):
            module.decode_design({'version':1,'compute_count':1,'plans':[[8,0,1,0]]})

    def test_same_design_has_same_feedback_as_race_environment(self):
        bundle_, _ = bundle()
        memory = MemoryHierarchy()
        def environment():
            return GemminiCNNEnvironment(bundle_.workload, memory=memory, cost_profile=fixture(memory),
                allow_test_fixture=True, budget=2, requirements=PPARequirements(10,100,100))
        reference = environment()
        observation = reference.reset()
        anchor = observation.anchor_indices[0]
        size = len(coupled_scope(observation.workload,anchor,observation.max_scope))
        action = transaction(observation.workload,observation.current_design,
                             tuple([1,anchor] + [0,0,1,0]*size),observation.max_scope)
        reference.step(action)
        evaluator = module.CommonDSEEvaluator(environment(),2)
        result = evaluator.evaluate(action.design)
        self.assertEqual(result['assessment'], reference.scorer.assess(reference.current_record))
        self.assertEqual(result['design_hash'], reference.current_record.design_hash)

    def test_gemini_search_uses_common_request_budget(self):
        bundle_, _ = bundle()
        memory = MemoryHierarchy()
        env = GemminiCNNEnvironment(bundle_.workload, memory=memory, cost_profile=fixture(memory),
            allow_test_fixture=True, budget=4, requirements=PPARequirements(10,100,100))
        result = load_tool('gemini_common_dse').run(module.CommonDSEEvaluator(env,4),seed=9)
        self.assertEqual(result['result']['requests'],4)
        self.assertEqual(len(result['result']['trace']),4)

    def test_local_baselines_charge_initial_and_next_candidates(self):
        bundle_, _ = bundle()
        memory = MemoryHierarchy()
        def environment():
            return GemminiCNNEnvironment(bundle_.workload, memory=memory, cost_profile=fixture(memory),
                allow_test_fixture=True, budget=4, requirements=PPARequirements(10,100,100))
        baseline=load_tool('common_dse_baselines')
        for name in ('local_random','sa'):
            result=baseline.run(module.CommonDSEEvaluator(environment(),4),method=name,seed=9)
            self.assertEqual(result['result']['requests'],4)
            self.assertTrue(result['result']['trace'][0]['physical_feasible'])
            trace=result['result']['trace']
            self.assertTrue(all(trace[i]['proposal_parent_hash']==trace[i-1]['accepted_design_hash']
                                for i in range(1,len(trace))))

    @unittest.skipUnless((path.parents[1]/'.cache/chiplet-gym-deps/stable_baselines3').exists(),
                         'optional Chiplet-Gym dependencies missing')
    def test_chiplet_gym_combines_ppo_and_sa_on_common_evaluator(self):
        bundle_, _ = bundle()
        memory = MemoryHierarchy()
        env = GemminiCNNEnvironment(bundle_.workload, memory=memory, cost_profile=fixture(memory),
            allow_test_fixture=True, budget=8, requirements=PPARequirements(10,100,100))
        result = load_tool('chiplet_gym_dse').run(module.CommonDSEEvaluator(env,8),seed=9)
        self.assertEqual(result['result']['requests'],8)
        self.assertEqual([row['phase'] for row in result['result']['trace']],['ppo']*4+['sa']*4)
        self.assertEqual(result['protocol']['n_steps'],2)
        self.assertEqual(result['protocol']['ppo_parameter_updates'],20)


if __name__ == '__main__':
    unittest.main()
