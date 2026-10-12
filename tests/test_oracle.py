from dataclasses import replace
import unittest
from race_rl.examples import toy_workload,toy_hardware,initial_mapping
from race_rl.hardware import Node,Link,Hardware
from race_rl.workload import Workload,Tensor,Operator
from race_rl.oracle import Oracle,BudgetExhausted
from race_rl.parameters import exploratory_profile
from race_rl.network import evaluate_network
from race_rl.serialization import plain


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.w,self.h = toy_workload(),toy_hardware()
        self.m = initial_mapping(self.w,self.h)

    def test_hand_calculated_linear_job_energy_and_time(self):
        w = Workload((Tensor("x",(1,1)),Tensor("y",(1,1))),
                     (Operator("fc","Linear",("x",),"y"),),("y",))
        h = Hardware((Node("c","compute",0,0,1,1,1000,1e9,1e9,static_power_w=0.1),
                      Node("m","memory",2,0,1,1,1000,1e9,static_power_w=0.1)),
                     (Link("link","c","m",1e9),))
        r = Oracle(w,h).evaluate(initial_mapping(w,h))
        # Three 4B transfers: 4ns DRAM + 9ns native path + 4ns wire + 4ns local each.
        # One compute: 1ns op + 12ns local. Total=3*21+13=76ns.
        self.assertTrue(r.feasible)
        self.assertAlmostEqual(r.latency_s,76e-9,places=18)
        expected = 1e-12+24e-12+120e-12+48e-12+0.2*76e-9
        self.assertAlmostEqual(r.energy_j,expected,places=20)
        self.assertAlmostEqual(r.average_power_w,r.energy_j/r.latency_s)
        self.assertEqual(r.area_mm2,2)

    def test_ledger_and_native_backend(self):
        r = Oracle(self.w,self.h).evaluate(self.m)
        self.assertTrue(r.feasible)
        self.assertAlmostEqual(r.energy_j,sum(v["energy_j"] for v in r.energy_ledger),places=18)
        self.assertEqual(r.diagnostics["native_backend"],"official_rapidchiplet")
        self.assertEqual(set(r.diagnostics["energy_components"]),{"compute","local_memory","external_memory","interconnect","static"})
        self.assertEqual(len(r.source_hashes["native"]),4)

    def test_cache_initial_and_failed_budget_accounting(self):
        oracle = Oracle(self.w,self.h,budget=2)
        oracle.evaluate(self.m)
        oracle.evaluate(self.m)
        bad = replace(self.m,placement=((self.m.tiles[0].id,"missing"),))
        self.assertFalse(oracle.evaluate(bad).feasible)
        self.assertEqual((oracle.evaluations,oracle.requests,oracle.cache_hits),(2,3,1))
        oracle.evaluate(self.m)  # cached requests remain accessible at exhausted budget
        with self.assertRaises(BudgetExhausted): oracle.evaluate(replace(self.m,placement=()))
        self.assertEqual(oracle.requests,4)

    def test_buffer_capacity_violation(self):
        h = replace(self.h,nodes=tuple(replace(n,memory_bytes=1) if n.kind == "compute" else n for n in self.h.nodes))
        r = Oracle(self.w,h).evaluate(self.m)
        self.assertFalse(r.feasible)
        self.assertTrue(any("buffer capacity" in v for v in r.violations))
        self.assertIsNotNone(r.latency_s)  # expensive failed physical checks count and preserve ledgers

    def test_connectivity_failure(self):
        with self.assertRaisesRegex(ValueError,"disconnected"):
            Oracle(self.w,replace(self.h,links=()))

    def test_deterministic_trace(self):
        a = plain(Oracle(self.w,self.h).evaluate(self.m))
        b = plain(Oracle(self.w,self.h).evaluate(self.m))
        a.pop("wall_time_s"); b.pop("wall_time_s")
        self.assertEqual(a,b)

    def test_fanout_read_shared_same_destination(self):
        r = Oracle(self.w,self.h).evaluate(self.m)
        shared = [f for f in r.traffic_ledger if f["tensor"] == "a" and len(f["consumer_tiles"]) == 2]
        self.assertEqual(len(shared),1)
        self.assertEqual(shared[0]["bytes"],self.w.tensor_map["a"].bytes)

    def test_generic_topologies_and_phy_ceiling(self):
        for topology in ("chain","star","ring"):
            h = toy_hardware(4,topology)
            net = evaluate_network(h)
            self.assertEqual(net.area_mm2,6)
            self.assertGreater(net.latency_s[("c0","m1")],0)
        h = replace(self.h,bump_pitch_mm=1)
        with self.assertRaisesRegex(ValueError,"bump budget"): evaluate_network(h)

    def test_unit_checks(self):
        p = exploratory_profile()
        with self.assertRaisesRegex(ValueError,"unit mismatch"):
            replace(p,link_bit_j=replace(p.link_bit_j,unit="pJ/bit")).validate()
