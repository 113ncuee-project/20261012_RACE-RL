import unittest
from race_rl.examples import toy_workload,toy_hardware,initial_mapping
from race_rl.actions import Action,apply,legal_actions
from race_rl.tiles import validate_coverage
from race_rl.rewards import potential,reward
from race_rl.environment import RACEEnvironment


class ActionRewardTests(unittest.TestCase):
    def setUp(self):
        self.w,self.h = toy_workload(),toy_hardware()
        self.m = initial_mapping(self.w,self.h)

    def test_atomic_transaction_original_unchanged(self):
        original = self.m
        a = Action("conv_a:0",("conv_a:0","conv_b:0"),"cut_migrate","c1")
        candidate = apply(self.w,self.h,self.m,a)
        self.assertEqual(self.m,original)
        self.assertEqual(candidate.placements["conv_a:0"],"c1")
        self.assertEqual(candidate.placements["conv_b:0"],"c1")
        with self.assertRaises(ValueError): apply(self.w,self.h,self.m,Action(a.root,a.scope,a.kind,"absent"))
        self.assertEqual(self.m,original)

    def test_retile_coverage_and_routes_rebuilt(self):
        a = Action("conv_a:0",("conv_a:0","conv_b:0"),"work_share","c1",2)
        candidate = apply(self.w,self.h,self.m,a)
        self.assertTrue(validate_coverage(self.w,candidate.tiles))
        self.assertEqual(len(candidate.tiles),5)
        self.assertEqual(set(candidate.placements.values()),{"c0","c1"})

    def test_all_candidates_structurally_valid(self):
        actions = legal_actions(self.w,self.h,self.m)
        self.assertTrue(actions)
        for a in actions: self.assertTrue(apply(self.w,self.h,self.m,a).validate(self.w,self.h))
        self.assertTrue(all(len(a.scope) == 1 for a in legal_actions(self.w,self.h,self.m,1)))

    def test_dense_and_potential_telescope(self):
        for sequence in ((1,1,0.8,0.8,0.6),(1,0.9),(1,1,1)):
            dense,shaped = [],[]
            initial_phi = potential(1,sequence[0],0.2,1,0.3)
            for i,(a,b) in enumerate(zip(sequence,sequence[1:])):
                terminal = i == len(sequence)-2
                dense.append(reward(1,a,b,terminal=terminal))
                shaped.append(reward(1,a,b,variant="potential",terminal=terminal,
                    old_phi=potential(1,a,0.2,1,0.3),new_phi=potential(1,b,0.2,1,0.3)))
            import math
            self.assertAlmostEqual(sum(dense),math.log(1/sequence[-1]))
            self.assertAlmostEqual(sum(shaped),math.log(1/sequence[-1])-initial_phi)

    def test_budget_one_and_strict_initial_limits(self):
        env = RACEEnvironment(self.w,self.h,self.m,budget=1)
        env.reset()
        self.assertTrue(env.done)
        self.assertEqual(env.oracle.evaluations,1)
        with self.assertRaises(ValueError):
            RACEEnvironment(self.w,self.h,self.m,limits={"area_mm2":1}).reset()

    def test_worse_feasible_current_does_not_replace_archive(self):
        env = RACEEnvironment(self.w,self.h,self.m,budget=4)
        env.reset()
        action = next(a for a in env.actions if a.kind == "memory_affinity" and a.target == "m1"
                      and a.compute_target == "c0" and len(a.scope) == 3)
        env.step(action)
        self.assertGreater(env.current_objective,1)
        self.assertEqual(env.best_objective,1)
        self.assertEqual(env.best_mapping,self.m)

    def test_physical_failure_reward_telescope(self):
        from dataclasses import replace
        h = replace(self.h,nodes=tuple(replace(n,memory_bytes=1) if n.id == "c1" else n for n in self.h.nodes))
        env = RACEEnvironment(self.w,h,self.m,budget=2,reward_variant="potential",lambda_c=0.3)
        env.reset()
        phi0 = env.phi()
        action = next(a for a in env.actions if a.kind == "cut_migrate" and a.target == "c1")
        _,r,done = env.step(action)
        self.assertTrue(done)
        self.assertEqual(env.oracle.evaluations,2)
        self.assertEqual(env.best_objective,1)
        self.assertAlmostEqual(r,-phi0)

    def test_cache_request_limit_terminal_reward(self):
        env = RACEEnvironment(self.w,self.h,self.m,budget=100,max_requests=4,reward_variant="potential",lambda_c=0.2)
        env.reset()
        phi0 = env.phi()
        total = 0
        while not env.done:
            action = next(a for a in env.actions if a.kind == "cut_migrate" and len(a.scope) == 1)
            _,r,_ = env.step(action)
            total += r
        import math
        self.assertEqual(env.termination,"request_limit")
        self.assertAlmostEqual(total,math.log(1/env.best_objective)-phi0)
        self.assertTrue(all(t["best_objective"] <= 1 for t in env.trace))
