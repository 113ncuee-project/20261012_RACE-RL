"""Cached lookup acceleration must preserve immutable input and hash contracts."""
import unittest
from race_rl.workload import Tensor,Operator,Workload
from race_rl.examples import toy_hardware
from race_rl.hardware import Hardware
from race_rl.serialization import digest


class IdentityInvariantTests(unittest.TestCase):
    def test_caller_list_mutation_cannot_change_cached_workload(self):
        shape = [1,8]
        inputs = ['x']
        tensors = [Tensor('x',shape),Tensor('y',(1,8))]
        ops = [Operator('fc','Linear',inputs,'y')]
        metadata = [['origin','fixture']]
        w = Workload(tensors,ops,['y'],source_metadata=metadata)
        before = digest(w)
        w.tensor_map; w.operator_map; w.parameter_map; w.producers
        shape[1] = 999
        inputs.append('unknown')
        tensors.clear(); ops.clear(); metadata[0][1] = 'changed'
        self.assertEqual(digest(w),before)
        self.assertTrue(w.validate())
        with self.assertRaises(TypeError): w.tensor_map['x'] = Tensor('x',(1,9))
        with self.assertRaises(TypeError): w.parameter_map['weights:fc'] = (1,1)

    def test_hardware_cache_does_not_change_identity_or_expose_mutation(self):
        original = toy_hardware()
        nodes,links = list(original.nodes),list(original.links)
        h = Hardware(nodes,links)
        before = digest(h)
        h.node_map
        nodes.clear(); links.clear()
        self.assertEqual(digest(h),before)
        self.assertTrue(h.validate())
        with self.assertRaises(TypeError): h.node_map['unknown'] = original.nodes[0]
