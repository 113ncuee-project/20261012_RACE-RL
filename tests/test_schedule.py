import unittest
from race_rl.schedule import Job,schedule,criticality,peak_bytes


class GoldenScheduleTests(unittest.TestCase):
    def test_single_compute(self):
        self.assertEqual(schedule([Job("a",(),("core",),2)]).makespan_s,2)

    def test_dependent_jobs(self):
        s = schedule([Job("b",("a",),("other",),3),Job("a",(),("core",),2)])
        self.assertEqual(s.makespan_s,5)
        self.assertEqual(s.event_map["b"].start_s,2)

    def test_parallel_jobs(self):
        s = schedule([Job("a",(),("c0",),2),Job("b",(),("c1",),3)])
        self.assertEqual(s.makespan_s,3)

    def test_shared_resource(self):
        self.assertEqual(schedule([Job(str(i),(),("core",),2) for i in range(3)]).makespan_s,6)

    def test_multi_resource_contention(self):
        s = schedule([Job("a",(),("c0","link"),2),Job("b",(),("c1","link"),3)])
        self.assertEqual(s.makespan_s,5)
        self.assertTrue(s.validate())

    def test_calendar_gap(self):
        s = schedule([Job("a",(),("x",),1),Job("b",("a",),("r",),5),Job("c",(),("r",),0.5)])
        self.assertEqual(s.event_map["c"].start_s,0)
        self.assertEqual(s.event_map["b"].start_s,1)

    def test_all_critical_chains(self):
        s = schedule([Job("a",(),("c0",),2),Job("b",(),("c1",),2),Job("c",("a","b"),(),1)])
        diagnostics = criticality(s)
        self.assertTrue(all(v["critical"] for v in diagnostics.values()))

    def test_slack_and_resource_precedence(self):
        s = schedule([Job("a",(),("x",),2),Job("b",(),("x",),1),Job("c",(),("y",),1)])
        d = criticality(s)
        self.assertIn("a",d["b"]["predecessors"])
        self.assertEqual(d["c"]["slack_s"],2)

    def test_cycles_and_unknown_dependencies(self):
        for jobs in ([Job("a",("b",),(),1)], [Job("a",("b",),(),1),Job("b",("a",),(),1)]):
            with self.assertRaises(ValueError): schedule(jobs)

    def test_invalid_duration(self):
        for value in (-1,float("nan"),float("inf")):
            with self.assertRaises(ValueError): Job("a",(),(),value)

    def test_buffer_release_at_equal_time(self):
        self.assertEqual(peak_bytes([(0,1,5),(1,2,7),(0,2,1)]),8)
