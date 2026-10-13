from dataclasses import replace
import unittest
import torch
from race_rl.mesh import mesh_hardware
from race_rl.hardware import Link
from race_rl.network import evaluate_network
from race_rl.oracle import Oracle
from race_rl.workload import Workload,Tensor,Operator
from race_rl.examples import initial_mapping,toy_workload
from race_rl.architecture import DesignEnvironment,CountAction
from race_rl.training import RACETrainer


class ExternalMemoryTests(unittest.TestCase):
    def test_native_mesh_contains_compute_dies_only(self):
        h=mesh_hardware(16);net=evaluate_network(h)
        self.assertEqual(len(h.nodes),16)
        self.assertTrue(all(n.kind=='compute' for n in h.nodes))
        self.assertEqual(net.area_mm2,16)
        self.assertEqual(set(net.inputs['chiplets']),set(h.compute_ids))
        self.assertEqual(len(h.memory_ports),2)
        self.assertTrue(all(n.id not in net.inputs['chiplets'] for n in h.memory_ports))
        self.assertEqual(len(h.links),24)

    def test_ports_attach_to_gateways_and_never_relay(self):
        h=mesh_hardware(16);net=evaluate_network(h)
        self.assertEqual(h.route('dram0','c0'),())
        self.assertEqual(len(h.route('dram0','c15')),6)
        self.assertEqual(h.route('dram0','c15'),h.route('c0','c15'))
        self.assertEqual(net.latency_s['dram0','c15'],net.latency_s['c0','c15'])
        self.assertTrue(all(a.startswith('c') and b.startswith('c') for _,a,b in h.route('dram0','dram1')))
        with self.assertRaises(ValueError):replace(h,links=(*h.links,Link('bad','dram0','c0',1e9))).validate()
        with self.assertRaises(ValueError):replace(h,memory_ports=(replace(h.memory_ports[0],attached_to='missing'),)).validate()

    def test_external_access_costs_remain_even_without_mesh_hops(self):
        w=Workload((Tensor('x',(1,8)),Tensor('y',(1,8))),(Operator('fc','Linear',('x',),'y'),),('y',))
        h=mesh_hardware(1);oracle=Oracle(w,h,scheduling_policy='serial_node_tiles')
        r=oracle.evaluate(initial_mapping(w,h))
        self.assertTrue(r.feasible,r.violations)
        self.assertTrue(all(not f['route'] for f in r.traffic_ledger))
        self.assertTrue(all(f['memory_service_s']>0 for f in r.traffic_ledger))
        self.assertGreater(r.diagnostics['energy_components']['external_memory'],0)
        self.assertGreater(r.diagnostics['energy_components']['local_memory'],0)
        expected=(oracle.network.idle_power_w+sum(p.static_power_w for p in h.memory_ports))*r.latency_s
        self.assertAlmostEqual(r.diagnostics['energy_components']['static'],expected,places=18)

    def test_external_capacity_checked_separately_from_die_area(self):
        w=Workload((Tensor('x',(1,8)),Tensor('y',(1,8))),(Operator('fc','Linear',('x',),'y'),),('y',))
        h=mesh_hardware(2,external_memory_bytes=100)
        r=Oracle(w,h,scheduling_policy='serial_node_tiles').evaluate(initial_mapping(w,h))
        self.assertFalse(r.feasible)
        self.assertTrue(any('dram0' in v for v in r.violations))
        self.assertGreater(r.buffer_ledger['peak_bytes']['dram0'],100)
        self.assertEqual(r.area_mm2,2)

    def test_logical_memory_relations_are_distinct_from_physical_links(self):
        env=DesignEnvironment(toy_workload(),compute_counts=(2,16),budget=4)
        o=env.reset()
        self.assertEqual(o.graph.schema_version,5)
        self.assertEqual(len(o.graph.memory_port_edge_index),4)
        self.assertEqual(len(o.graph.hardware_edge_index),2)
        env.step(CountAction(16))
        self.assertEqual(len(env.hardware.nodes),16)
        self.assertEqual(len(env.observation.graph.hardware_edge_index),48)
        self.assertEqual(len(env.observation.graph.memory_port_edge_index),4)
        env.current_mapping.validate(env.workload,env.hardware)

    def test_single_compute_mesh_without_links_has_finite_neural_policy(self):
        env=DesignEnvironment(toy_workload(),compute_counts=(1,),budget=3)
        o=env.reset()
        trainer=RACETrainer(width=8,state_variant='no_criticality',precision='float64')
        d,v=trainer.policy.distribution(o)
        self.assertTrue(torch.isfinite(d.logits).all())
        self.assertIsNone(v)
        self.assertAlmostEqual(d.probs.sum().item(),1,places=12)
