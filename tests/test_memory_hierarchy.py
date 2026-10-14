"""Independent hierarchy counts, capacity/port sensitivity and RL visibility."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from race_rl.accelerators.gemmini_model import GemminiModel, GemminiAction, GemmShape
from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
from race_rl.accelerators.gemmini_environment import GemminiDesignEnvironment
from test_gemmini_foundation import assert_resource_schedule
from test_gemmini_rl import fixture


class MemoryHierarchyTests(unittest.TestCase):
    def test_repository_profile_matches_defaults_and_missing_parameters_are_rejected(self):
        root = Path(__file__).resolve().parents[1]
        record = json.loads((root/'configs/memory_hierarchy.json').read_text())
        self.assertEqual(record,MemoryHierarchy().record())
        self.assertEqual(MemoryHierarchy.from_record(record),MemoryHierarchy())
        del record['parameters']['rf_a_bytes']
        with self.assertRaisesRegex(ValueError,'every capacity'):
            MemoryHierarchy.from_record(record)

    def record(self, **kwargs):
        return GemminiModel(**kwargs).evaluate(GemmShape(16,16,16),GemminiAction(placement=(0,)))

    def test_single_microtile_has_all_boundaries_and_exact_endpoint_counts(self):
        record = self.record()
        self.assertEqual(record['traffic']['hierarchy_movement_bytes'],{
            'dram->gb':512,'gb->lb':512,'lb->rf':512,
            'rf->lb':1024,'lb->accumulator':1024,
            'accumulator->lb':1024,'lb->gb':1024,'gb->dram':1024})
        # RF: operand fills 512, save 1024, two reads/MAC (8192),
        # 15 psum reads per element (15360), 16 psum writes (16384).
        self.assertEqual(record['traffic']['storage_access_bytes'],{
            'gb':3072,'lb':5120,'rf':41472,'accumulator':2048})
        self.assertNotIn('dram->rf',record['traffic']['hierarchy_movement_bytes'])
        assert_resource_schedule(self,record['events'])

    def test_k_reduction_traverses_accumulator_lb_rf_both_directions(self):
        record = GemminiModel().evaluate(GemmShape(16,16,32),GemminiAction(placement=(0,)))
        moves = record['traffic']['hierarchy_movement_bytes']
        self.assertEqual(moves['accumulator->lb'],2048) # one reload plus final drain
        self.assertEqual(moves['lb->rf'],2048) # operands 1024 + psum reload 1024
        self.assertEqual(moves['rf->lb'],2048)
        self.assertEqual(record['operations']['acc_read_bytes'],1024)

    def test_panel_blocking_reuses_a_in_lb_and_weights_in_rf(self):
        shape = GemmShape(32,32,16)
        combined = GemminiModel().evaluate(shape,GemminiAction(32,32,16,placement=(0,)))
        separated = GemminiModel().evaluate(shape,GemminiAction(16,16,16,placement=(0,0,0,0)))
        self.assertEqual(combined['operations']['padded_macs'],separated['operations']['padded_macs'])
        self.assertLess(combined['traffic']['by_operand']['A'],separated['traffic']['by_operand']['A'])
        self.assertLess(combined['traffic']['hierarchy_movement_bytes']['gb->lb'],separated['traffic']['hierarchy_movement_bytes']['gb->lb'])
        self.assertLess(combined['operations']['lb_b_read_bytes'],separated['operations']['lb_b_read_bytes'])

    def test_lb_and_rf_capacity_are_not_infinite_or_implicitly_retiled(self):
        shape,action = GemmShape(64,64,64),GemminiAction(64,64,64,placement=(0,))
        GemminiModel().evaluate(shape,action)
        with self.assertRaisesRegex(ValueError,'local buffer'):
            GemminiModel(memory=replace(MemoryHierarchy(),lb_operand_slot_bytes=2048)).evaluate(shape,action)
        with self.assertRaisesRegex(ValueError,'RF'):
            self.record(memory=replace(MemoryHierarchy(),rf_psum_bytes=512))
        with self.assertRaisesRegex(ValueError,'output LB'):
            self.record(memory=replace(MemoryHierarchy(),lb_output_bytes=512))
        with self.assertRaises(ValueError): MemoryHierarchy(lb_bank_bytes_per_cycle=0)

    def test_lb_and_rf_bandwidth_affect_service_without_changing_work_or_bytes(self):
        normal = self.record()
        for memory in (replace(MemoryHierarchy(),lb_bank_bytes_per_cycle=4),
                       replace(MemoryHierarchy(),rf_operand_bytes_per_cycle=4)):
            slow = self.record(memory=memory)
            self.assertGreater(slow['modeled_cycles'],normal['modeled_cycles'])
            self.assertEqual(slow['operations'],normal['operations'])
            self.assertEqual(slow['traffic'],normal['traffic'])
            self.assertNotEqual(slow['identity'],normal['identity'])

    def test_output_streams_microtiles_without_whole_tile_lb_allocation(self):
        record = GemminiModel().evaluate(GemmShape(128,128,16),GemminiAction(128,128,16,placement=(0,)))
        block = record['storage']['blocks'][0]
        self.assertEqual(block['accumulator_bytes'],65536)
        self.assertEqual(block['lb_output_peak_bytes'],1024)
        self.assertLess(block['lb_output_capacity_bytes'],block['accumulator_bytes'])
        self.assertEqual(record['traffic']['hierarchy_movement_bytes']['lb->gb'],65536)

    def test_ppa_has_separate_endpoint_costs_and_binds_exact_hardware_profile(self):
        record,profile = self.record(),fixture().validate(allow_test_fixture=True)
        detail = profile.assess(record)
        for name,level in (('global_buffer','gb'),('local_buffer','lb'),('register_file','rf'),('accumulator','accumulator')):
            self.assertAlmostEqual(detail['energy_components'][name],record['traffic']['storage_access_bytes'][level]*1e-12,places=18)
        self.assertAlmostEqual(detail['energy_components']['mac'],4096e-12,places=18)
        with self.assertRaisesRegex(ValueError,'different'):
            profile.assess(self.record(memory=replace(MemoryHierarchy(),lb_output_bytes=2048)))
        with self.assertRaisesRegex(ValueError,'does not match'):
            GemminiDesignEnvironment(GemmShape(16,16,16),cost_profile=profile,allow_test_fixture=True,
                                    memory=replace(MemoryHierarchy(),lb_output_bytes=2048))

    def test_rl_graph_has_real_hierarchy_edges_and_post_action_internal_traffic(self):
        env = GemminiDesignEnvironment(GemmShape(16,16,16),budget=2,compute_counts=(1,),initial_count=1)
        with patch.object(GemminiModel,'evaluate',autospec=True,side_effect=GemminiModel.evaluate) as evaluate:
            first = env.reset()
            self.assertEqual(evaluate.call_count,0)
            self.assertEqual(first.graph.schema_version,7)
            next_o,_,_ = env.step(env.actions[0])
            self.assertEqual(evaluate.call_count,1)
        graph = next_o.graph
        names = graph.hardware_ids
        edges = {(names[a],names[b]):features for (a,b),features in zip(graph.hardware_edge_index,graph.hardware_edge_features)}
        for edge in (('c0.gb0','c0.lb0'),('c0.lb0','c0.rfA'),('c0.rfA','c0.array'),
                     ('c0.rfpsum','c0.lbC'),('c0.lbC','c0.acc0'),('c0.lbC','c0.gb0')):
            self.assertIn(edge,edges)
            self.assertGreater(edges[edge][1],0)
            self.assertGreater(edges[edge][2],0)
        self.assertNotIn(('c0.gb0','c0.array'),edges)
        self.assertNotIn(('c0.dma','c0.acc0'),edges)
        changed = GemminiDesignEnvironment(GemmShape(16,16,16),budget=2,
                   memory=replace(MemoryHierarchy(),lb_output_bytes=2048))
        self.assertNotEqual(env.experiment_identity,changed.experiment_identity)


if __name__=='__main__': unittest.main()
