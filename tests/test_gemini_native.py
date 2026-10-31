"""Guard artifact identity and native output parsing."""
import importlib.util
from pathlib import Path
import unittest


path = Path(__file__).resolve().parents[1]/'tools/gemini_native.py'
spec = importlib.util.spec_from_file_location('gemini_native', path)
gemini = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gemini)
audit_spec = importlib.util.spec_from_file_location('gemini_bridge_audit', path.parent/'gemini_bridge_audit.py')
bridge_audit = importlib.util.module_from_spec(audit_spec)
audit_spec.loader.exec_module(bridge_audit)
owner_spec = importlib.util.spec_from_file_location('gemini_ownership_bridge', path.parent/'gemini_ownership_bridge.py')
owner_bridge = importlib.util.module_from_spec(owner_spec)
owner_spec.loader.exec_module(owner_bridge)
adapt_spec = importlib.util.spec_from_file_location('gemini_constrained_baseline', path.parent/'gemini_constrained_baseline.py')
adapt = importlib.util.module_from_spec(adapt_spec)
adapt_spec.loader.exec_module(adapt)


class GeminiNativeTests(unittest.TestCase):
    def test_archive_has_pinned_identity(self):
        self.assertEqual(gemini.digest_file(gemini.ARCHIVE), gemini.EXPECTED)

    def test_results_parse_native_fields_without_claiming_ppa(self):
        fields = [str(i) for i in range(36)]
        fields[21] = '3'
        row = gemini.parse_results(' '.join(fields))[0]
        self.assertEqual(row['mapping_index'], 3)
        self.assertEqual(row['cycles'], 18)
        self.assertEqual(row['energy_raw'], 17)
        self.assertNotIn('average_power_w', row)
        with self.assertRaises(ValueError):
            gemini.parse_results('1 2 3')

    def test_incompatible_workloads_are_rejected_without_estimated_ppa(self):
        mapping = {'schema':1, 'origin':'gemini_hpca2024', 'network_id':1,
            'mesh_x':2, 'mesh_y':2, 'chiplet_x':1, 'chiplet_y':1, 'batch':1,
            'layers':[{'name':'conv1', 'kind':'Conv', 'shape_chw':[4,8,8], 'output_tiles':[]}],
            'segments':[{'partitions':[], 'dram':[]}]}
        workload = {'name':'test', 'operators':[{'kind':'ReLU', 'output':'out'}],
                    'tensors':[{'id':'out', 'shape':[1,4,8,8]}]}
        result = bridge_audit.audit(mapping, workload)
        self.assertFalse(result['comparison_ready'])
        self.assertTrue(any('ReLU' in reason for reason in result['reasons']))
        self.assertTrue(any('4 cores across 1 chiplets' in reason for reason in result['reasons']))
        self.assertNotIn('ppa', result)

    def test_tile_coverage_detects_overlap(self):
        layer = {'shape_chw':[1,2,2], 'output_tiles':[
            {'bounds':[[0,1],[0,1],[0,2],[0,2]]},
            {'bounds':[[0,1],[0,1],[0,2],[0,2]]}]}
        self.assertEqual(bridge_audit.validate_tiles(layer, 1), ['output tile bounds overlap'])
        layer['output_tiles'].pop()
        self.assertEqual(bridge_audit.validate_tiles(layer, 1), [])

    def test_exact_tile_ownership_requires_explicit_core_mapping(self):
        workload = owner_bridge.Workload(
            (owner_bridge.Tensor('input',(1,16,8,16)), owner_bridge.Tensor('out',(1,16,8,16))),
            (owner_bridge.Operator('op','ReLU',('input',),'out'),), ('out',))
        layer = {'shape_chw':[16,8,16], 'output_tiles':[
            {'bounds':[[0,1],[0,16],[0,8],[0,8]], 'core_xy':[0,0]},
            {'bounds':[[0,1],[0,16],[0,8],[8,16]], 'core_xy':[1,0]}]}
        result = owner_bridge.matching_plans(workload,'op',layer,{'0,0':0,'1,0':1},2)
        self.assertTrue(any(p['h']==8 and p['w']==8 and p['target']==0 and p['spread'] for p in result))
        with self.assertRaisesRegex(ValueError, 'no explicit chiplet translation'):
            owner_bridge.matching_plans(workload,'op',layer,{'0,0':0},2)
        self.assertEqual(owner_bridge.matching_plans(workload,'op',layer,{'0,0':0,'1,0':0},2),
                         [p for p in owner_bridge.matching_plans(workload,'op',layer,
                          {'0,0':0,'1,0':0},2) if not p['spread']])

    def test_adapted_gemini_uses_shared_budget_and_is_repeatable(self):
        from test_cnn_hierarchy import bundle
        from race_rl.accelerators.cnn_environment import GemminiCNNEnvironment
        from race_rl.accelerators.cnn_search import fixture
        from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
        from race_rl.requirements import PPARequirements
        workload, _ = bundle()
        memory = MemoryHierarchy()
        def environment():
            return GemminiCNNEnvironment(workload.workload, memory=memory,
                cost_profile=fixture(memory), allow_test_fixture=True, budget=6,
                requirements=PPARequirements(10,100,100))
        first = adapt.run_gemini_constrained(environment(), seed=21)
        second = adapt.run_gemini_constrained(environment(), seed=21)
        self.assertEqual(first['oracle_requests'], 6)
        self.assertEqual(first['oracle_requests'], first['oracle_evaluations']+first['cache_hits'])
        self.assertEqual([row['choices'] for row in first['trace']],
                         [row['choices'] for row in second['trace']])
        self.assertEqual(first['assessment'], second['assessment'])


if __name__ == '__main__':
    unittest.main()
