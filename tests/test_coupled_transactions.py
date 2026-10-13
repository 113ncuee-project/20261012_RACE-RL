"""Coupled edits are atomic, with zero PPA calls between parameter decisions."""
from dataclasses import replace
import unittest
from unittest.mock import patch
from race_rl.architecture import DesignEnvironment,DesignOracle
from race_rl.actions import legal_actions,apply
from race_rl.examples import toy_workload
from race_rl.initialization import conservative_initial_mapping
from race_rl.mesh import mesh_hardware
from race_rl.oracle import Oracle
from action_reference import exhaustive_actions


class CoupledTransactionTests(unittest.TestCase):
    def test_fast_masks_equal_independent_full_compilation_in_order(self):
        w=toy_workload()
        for count in (1,2,4,16):
            h=mesh_hardware(count);m,_=conservative_initial_mapping(w,h)
            for canonical in (False,True):
                if canonical:m=replace(m,tiles=tuple(sorted(m.tiles,key=lambda t:t.id)),
                    placement=tuple(sorted(m.placement)),storage=tuple(sorted(m.storage)))
                for scope in (1,2,3):
                    with self.subTest(count=count,canonical=canonical,scope=scope):
                        self.assertEqual(legal_actions(w,h,m,scope),exhaustive_actions(w,h,m,scope))

    def test_repartitioned_partial_ownership_masks_match_reference(self):
        w=toy_workload();h=mesh_hardware(4);m,_=conservative_initial_mapping(w,h)
        a=next(a for a in legal_actions(w,h,m,2) if a.kind=='work_share' and a.count==2)
        m=apply(w,h,m,a)
        for scope in (1,2,3,4):
            self.assertEqual(legal_actions(w,h,m,scope),exhaustive_actions(w,h,m,scope))

    def test_two_layer_edit_is_complete_before_one_environment_query(self):
        env=DesignEnvironment(toy_workload(),compute_counts=(2,4),budget=2,max_scope=2)
        with patch.object(Oracle,'evaluate',side_effect=AssertionError('pre-action PPA')):
            env.reset()
            action=next(a for a in env.actions if a.kind=='work_share' and len(getattr(a,'scope',()))==2 and a.count==2)
        initial=env.current_mapping
        evaluate=env.oracle.evaluate
        def check_complete(h,m):
            # Both whole operators must have changed before reaching the oracle.
            selected={next(t.operator for t in initial.tiles if t.id==name) for name in action.scope}
            self.assertEqual(len(selected),2)
            for op in selected:self.assertEqual(sum(t.operator==op for t in m.tiles),2)
            self.assertEqual(env.current_mapping,initial)  # no partial commit
            return evaluate(h,m)
        with patch.object(env.oracle,'evaluate',side_effect=check_complete) as called:
            o,r,done=env.step(action)
        called.assert_called_once()
        self.assertEqual(env.oracle.evaluations,1)
        self.assertEqual(env.oracle.requests,1)
        self.assertEqual(o.graph.global_features[9],r)
        self.assertEqual(env.trace[-1]['action'],action)
        self.assertNotEqual(env.current_mapping,initial)

    def test_failed_multi_tile_edit_rolls_back_entire_transaction(self):
        env=DesignEnvironment(toy_workload(),compute_counts=(2,4),budget=1,max_scope=2)
        env.reset();initial=env.current_mapping
        action=next(a for a in env.actions if a.kind=='memory_affinity' and len(getattr(a,'scope',()))==2)
        evaluate=env.oracle.evaluate
        with patch.object(env.oracle,'evaluate',side_effect=lambda h,m:replace(evaluate(h,m),feasible=False,violations=('capacity failure',))) as called:
            _,_,done=env.step(action)
        called.assert_called_once()
        self.assertTrue(done)
        self.assertEqual(env.current_mapping,initial)
        self.assertIsNone(env.best_record)
