from dataclasses import replace
import unittest
from race_rl.workload import Tensor,Operator,Workload
from race_rl.examples import toy_hardware,toy_workload,initial_mapping
from race_rl.initialization import conservative_initial_mapping
from race_rl.oracle import Oracle
from race_rl.actions import legal_actions,dependency_edges
from race_rl.observation import observe
from race_rl.environment import RACEEnvironment


class GuardedMemoryTests(unittest.TestCase):
    def fixture(self,capacity=175):
        w = Workload((Tensor('x',(1,8)),Tensor('y',(1,8))),(Operator('fc','Linear',('x',),'y'),),('y',))
        h = toy_hardware()
        h = replace(h,nodes=tuple(replace(n,memory_bytes=capacity if n.kind == 'compute' else 256) for n in h.nodes))
        return w,h

    def test_compiler_fits_local_tiles_and_distributes_external_constants(self):
        w,h = self.fixture()
        mapping,meta = conservative_initial_mapping(w,h)
        self.assertEqual(meta['oracle_calls_for_compilation'],0)
        self.assertEqual(meta['tile_counts']['fc'],3)
        self.assertLessEqual(meta['max_tile_footprint_bytes'],175)
        self.assertEqual(set(mapping.storages.values()),{'m0','m1'})
        r = Oracle(w,h,scheduling_policy='serial_node_tiles').evaluate(mapping)
        self.assertTrue(r.feasible,r.violations)
        self.assertTrue(all(amount <= h.node_map[node].memory_bytes for node,amount in r.buffer_ledger['peak_bytes'].items()))
        self.assertEqual(r.diagnostics['scheduling_policy'],'serial_node_tiles')

    def test_no_free_shared_read_under_tile_admission(self):
        w,h = self.fixture()
        mapping,_ = conservative_initial_mapping(w,h)
        r = Oracle(w,h,scheduling_policy='serial_node_tiles').evaluate(mapping)
        x_flows = [f for f in r.traffic_ledger if f['tensor'] == 'x']
        self.assertEqual(len(x_flows),3)
        self.assertEqual(sum(f['bytes'] for f in x_flows),3*w.tensor_map['x'].bytes)
        self.assertTrue(all(len(f['consumer_tiles']) == 1 for f in x_flows))
        trace = {e['job']['id']:e for e in r.schedule_trace}
        for t,previous_write in r.diagnostics['tile_control_predecessors'].items():
            reads = [e for e in r.schedule_trace if e['job']['kind'] == 'memory_read' and t in e['job']['metadata']['consumer_tiles']]
            self.assertTrue(all(e['start_s'] >= trace[previous_write]['finish_s'] for e in reads))

    def test_control_dependencies_do_not_become_tensor_link_attribution(self):
        w,h = toy_workload(),toy_hardware()
        mapping = initial_mapping(w,h)
        oracle = Oracle(w,h,scheduling_policy='serial_node_tiles')
        r = oracle.evaluate(mapping)
        actions = legal_actions(w,h,mapping,1)
        o = observe(w,h,mapping,r,actions,current=1,best=1,remaining=63,budget=64)
        deps = dependency_edges(w,mapping)
        flows = {f['id']:f for f in r.traffic_ledger}
        relations = [row for row in o.dependency_link_relations if row[3] == 'read' and flows[row[2]]['tensor'] == 'a']
        self.assertTrue(relations)
        self.assertTrue(all(deps[row[0]][0] == 'conv_a:0' for row in relations))
        # Residual a-read carries conv_b's write for admission, but that is not
        # the source of tensor a. This guards against false physical attribution.
        residual_a = next(f for f in r.traffic_ledger if f['tensor'] == 'a' and f['consumer_tiles'] == ['residual:0'])
        self.assertIn('write:conv_b:0',residual_a['dependencies'])
        self.assertNotIn('write:conv_b:0',residual_a['data_dependencies'])

    def test_unfittable_working_set_and_memory_object_fail_explicitly(self):
        w,h = self.fixture(64)
        with self.assertRaisesRegex(ValueError,'no supported output partition'):
            conservative_initial_mapping(w,h)
        w,h = self.fixture()
        h = replace(h,nodes=tuple(replace(n,memory_bytes=128) if n.kind == 'memory' else n for n in h.nodes))
        with self.assertRaisesRegex(ValueError,'external residency'):
            conservative_initial_mapping(w,h)

    def test_compiler_determinism_and_schedule_identity(self):
        w,h = self.fixture()
        a,ma = conservative_initial_mapping(w,h)
        b,mb = conservative_initial_mapping(w,h)
        self.assertEqual((a,ma),(b,mb))
        x,y = Oracle(w,h),Oracle(w,h,scheduling_policy='serial_node_tiles')
        self.assertNotEqual(x.identity,y.identity)
        with self.assertRaises(ValueError): Oracle(w,h,scheduling_policy='unknown')

    def test_guarded_environment_only_counts_actions(self):
        w,h = self.fixture()
        mapping,_ = conservative_initial_mapping(w,h)
        env = RACEEnvironment(w,h,mapping,budget=1,max_scope=1,scheduling_policy='serial_node_tiles')
        env.reset()
        self.assertIsNone(env.best_record)
        self.assertEqual(env.oracle.evaluations,0)
        # Retiling a complete multi-tile operator needs scope closure; migrate
        # one tile to another compute resource while keeping the paid contract.
        action = next(a for a in env.actions if a.kind == 'cut_migrate')
        env.step(action)
        self.assertEqual(env.oracle.evaluations,1)
        self.assertTrue(env.best_record.feasible)
        self.assertTrue(env.done)
        self.assertLessEqual(env.best_objective,1)
