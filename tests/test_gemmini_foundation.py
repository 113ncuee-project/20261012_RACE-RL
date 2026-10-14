"""Source facts, independent arithmetic, storage limits and resource accounting."""
from dataclasses import replace
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np

from race_rl.accelerators.gemmini_config import default_config, verify_reference
from race_rl.accelerators.gemmini_model import (
    GemmShape, GemminiAction, GemminiModel, PlatformScenario, CycleJob,
    execute_gemm, lower_conv2d, schedule_cycles, xy_route,
)

ROOT = Path(__file__).resolve().parents[1]


def assert_resource_schedule(test, events):
    by_id = {event["job"]["id"]: event for event in events}
    calendars = {}
    for event in events:
        for dependency in event["job"]["dependencies"]:
            test.assertLessEqual(by_id[dependency]["finish_cycle"], event["start_cycle"])
        for resource in event["job"]["resources"]:
            calendars.setdefault(resource, []).append(event)
    for resource, spans in calendars.items():
        spans.sort(key=lambda event: event["start_cycle"])
        for a, b in zip(spans, spans[1:]):
            test.assertLessEqual(a["finish_cycle"], b["start_cycle"], resource)


class GemminiFoundationTests(unittest.TestCase):
    def test_pinned_default_and_derived_bank_geometry(self):
        c = default_config()
        self.assertEqual((c.rows, c.columns, c.input_bits, c.accumulator_bits), (16, 16, 8, 32))
        self.assertEqual((c.scratchpad_bytes, c.accumulator_bytes), (256*1024, 64*1024))
        self.assertEqual((c.scratchpad_row_bytes, c.accumulator_row_bytes), (16, 64))
        self.assertEqual(c.scratchpad_bank_bytes // c.scratchpad_row_bytes, 4096)
        self.assertEqual(c.accumulator_bank_bytes // c.accumulator_row_bytes, 512)
        audit = verify_reference(ROOT)
        self.assertEqual(audit["mismatches"], [])
        self.assertEqual(audit["files_checked"], 7)
        self.assertEqual(audit["parameters_checked"], 21)
        self.assertTrue(c.provenance()["parameters_match_public_default"])

    def test_source_audit_rejects_changed_upstream_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copy(ROOT/"gemmini.lock.json", root/"gemmini.lock.json")
            shutil.copytree(ROOT/"third_party/gemmini-reference", root/"third_party/gemmini-reference")
            source = root/"third_party/gemmini-reference/src/main/scala/gemmini/Configs.scala"
            source.write_text(source.read_text(encoding="utf-8").replace("meshRows = 16", "meshRows = 32", 1), encoding="utf-8")
            audit = verify_reference(root)
            self.assertEqual(audit["status"], "FAILED")
            self.assertIn("defaultConfig parameter: mesh_rows", audit["mismatches"])
            self.assertIn("src/main/scala/gemmini/Configs.scala", audit["mismatches"])

    def test_selected_action_requires_complete_placement(self):
        model = GemminiModel(chiplets=2)
        shape = GemmShape(64, 16, 32)
        for placement in ((), (0,), (0, 0, 0, 2)):
            with self.assertRaises(ValueError): model.evaluate(shape, GemminiAction(placement=placement))
        with self.assertRaises(ValueError): GemminiAction(dataflow="output_stationary")
        with self.assertRaises(ValueError): GemminiAction(m_tile=15)
        with self.assertRaises(ValueError): GemminiModel(chiplets=17)

    def test_weight_residency_changes_dram_traffic_without_changing_work(self):
        shape, model = GemmShape(64, 16, 32), GemminiModel()
        kept = model.evaluate(shape, GemminiAction(placement=(0, 0, 0, 0)))
        reloaded = model.evaluate(shape, GemminiAction(retain_weights=False, placement=(0, 0, 0, 0)))
        self.assertEqual(kept["traffic"]["by_operand"], {"A": 2048, "B": 512, "C": 4096})
        self.assertEqual(reloaded["traffic"]["by_operand"]["B"], 2048)
        self.assertEqual(kept["operations"], reloaded["operations"])
        self.assertEqual(kept["operations"]["logical_macs"], 32768)
        self.assertLess(kept["modeled_cycles"], reloaded["modeled_cycles"])

    def test_weight_reuse_is_local_to_each_chiplet_and_n_block(self):
        record = GemminiModel(chiplets=2).evaluate(GemmShape(64, 16, 32),
            GemminiAction(placement=(0, 1, 0, 1)))
        self.assertEqual(record["traffic"]["by_operand"]["B"], 1024)
        self.assertEqual(record["traffic"]["directed_link_byte_hops"], 3584)
        two_n_blocks = GemminiModel().evaluate(GemmShape(32, 32, 32),
            GemminiAction(placement=(0, 0, 0, 0)))
        self.assertEqual(two_n_blocks["traffic"]["by_operand"]["B"], 1024)
        assert_resource_schedule(self, record["events"])

    def test_separate_scratchpad_and_accumulator_capacity_limits(self):
        with self.assertRaisesRegex(ValueError, "scratchpad"):
            GemminiModel().evaluate(GemmShape(1024, 16, 1024),
                GemminiAction(m_tile=1024, k_tile=1024, placement=(0,)))
        with self.assertRaisesRegex(ValueError, "resident B"):
            GemminiModel().evaluate(GemmShape(16, 128, 2048),
                GemminiAction(n_tile=128, placement=(0,)))
        with self.assertRaisesRegex(ValueError, "accumulator"):
            GemminiModel().evaluate(GemmShape(256, 128, 16),
                GemminiAction(m_tile=256, n_tile=128, placement=(0,)))
        record = GemminiModel().evaluate(GemmShape(128, 128, 16),
            GemminiAction(m_tile=128, n_tile=128, placement=(0,)))
        self.assertEqual(record["storage"]["blocks"][0]["accumulator_bytes"], 65536)
        with self.assertRaisesRegex(ValueError, "external capacity"):
            GemminiModel(platform=PlatformScenario(dram_capacity_bytes=100)).evaluate(
                GemmShape(16, 16, 16), GemminiAction(placement=(0,)))

    def test_dma_transaction_segmentation_and_full_width_output(self):
        record = GemminiModel().evaluate(GemmShape(16, 16, 128),
            GemminiAction(k_tile=128, placement=(0,)))
        a = next(flow for flow in record["transfers"] if flow["operand"] == "A")
        self.assertEqual((a["bytes"], a["transactions"], a["dma_beats"]), (2048, 32, 128))
        output = next(flow for flow in record["transfers"] if flow["operand"] == "C")
        self.assertEqual(output["bytes"], 16*16*4)

    def test_partial_sum_reads_writes_cover_k_reduction(self):
        record = GemminiModel().evaluate(GemmShape(16, 16, 48), GemminiAction(placement=(0,)))
        self.assertEqual(record["operations"]["acc_write_bytes"], 3*16*16*4)
        self.assertEqual(record["operations"]["acc_read_bytes"], 2*16*16*4)

    def test_double_buffer_overlaps_dma_without_bank_conflicts(self):
        shape = GemmShape(16, 16, 64)
        model = GemminiModel(platform=PlatformScenario(dram_first_response_cycles=0))
        double = model.evaluate(shape, GemminiAction(placement=(0,)))
        single = model.evaluate(shape, GemminiAction(double_buffer=False, placement=(0,)))
        self.assertLess(double["modeled_cycles"], single["modeled_cycles"])
        assert_resource_schedule(self, double["events"])
        by_id = {event["job"]["id"]: event for event in double["events"]}
        compute = by_id["g0.c0.k0.execute"]
        next_fill = by_id["g0.c0.k1.A.lb"]
        self.assertLess(next_fill["start_cycle"], compute["finish_cycle"])
        self.assertGreater(next_fill["finish_cycle"], compute["start_cycle"])

    def test_tails_preserve_logical_macs_and_report_padding(self):
        record = GemminiModel().evaluate(GemmShape(29, 17, 33),
            GemminiAction(placement=(0, 0, 0, 0)))
        self.assertEqual(record["operations"]["logical_macs"], 29*17*33)
        self.assertEqual(record["operations"]["padded_macs"], 32*32*48)
        self.assertLess(record["array_padding_utilization"], 1)
        assert_resource_schedule(self, record["events"])

    def test_arithmetic_matches_independent_numpy_matmul_with_tails(self):
        rng = np.random.default_rng(19)
        for m, n, k in ((16, 16, 16), (29, 17, 33), (7, 3, 65)):
            a = rng.integers(-8, 9, (m, k), dtype=np.int8)
            b = rng.integers(-8, 9, (k, n), dtype=np.int8)
            expected = a.astype(np.int64) @ b.astype(np.int64)
            np.testing.assert_array_equal(execute_gemm(a, b, GemminiAction()), expected)

    def test_dense_convolution_lowering_matches_independent_direct_convolution(self):
        rng = np.random.default_rng(7)
        for stride, padding in ((1, 0), (1, 1), (2, 1)):
            x = rng.integers(-3, 4, (2, 4, 8, 8), dtype=np.int8)
            w = rng.integers(-3, 4, (8, 4, 3, 3), dtype=np.int8)
            a, b, info = lower_conv2d(x, w, stride=stride, padding=padding)
            actual = execute_gemm(a, b, GemminiAction()).reshape(
                info["output_shape"][0], info["output_shape"][2], info["output_shape"][3], 8).transpose(0, 3, 1, 2)
            source = np.pad(x.astype(np.int64), ((0, 0), (0, 0), (padding, padding), (padding, padding)))
            expected = np.zeros(info["output_shape"], dtype=np.int64)
            for batch in range(2):
                for channel in range(8):
                    for y in range(expected.shape[2]):
                        for z in range(expected.shape[3]):
                            expected[batch, channel, y, z] = np.sum(
                                source[batch, :, y*stride:y*stride+3, z*stride:z*stride+3]*w[channel])
            np.testing.assert_array_equal(actual, expected)
            self.assertGreater(info["im2col_bytes"], info["original_input_bytes"])
            self.assertIn("NOT_MODELED", info["preprocessing_latency_energy"])

    def test_no_silent_quantization_or_accumulator_overflow(self):
        for a in (np.ones((1, 1), dtype=np.float64), np.array([[128]]), np.array([[-129]])):
            with self.assertRaises(ValueError): execute_gemm(a, np.ones((1, 1), dtype=np.int8), GemminiAction())
        with self.assertRaisesRegex(ValueError, "overflow"):
            execute_gemm(np.full((1, 140000), 127, dtype=np.int8),
                         np.full((140000, 1), 127, dtype=np.int8), GemminiAction())

    def test_mesh_routes_are_rectangular_directed_and_gateway_based(self):
        for count in range(1, 17):
            for destination in range(count):
                route = xy_route(count, 0, destination)
                node = 0
                for a, b in route:
                    self.assertEqual(a, node)
                    self.assertTrue(0 <= b < count)
                    node = b
                self.assertEqual(node, destination)
        self.assertEqual(len(xy_route(16, 0, 15)), 6)
        self.assertEqual(xy_route(1, 0, 0), ())
        with self.assertRaises(ValueError): xy_route(2, 0, 2)

    def test_cycle_scheduler_rejects_invalid_dependency_and_enforces_resources(self):
        jobs = [CycleJob("a", (), ("bank",), 5, "fill"),
                CycleJob("b", (), ("bank",), 3, "fill"),
                CycleJob("c", ("a",), ("array",), 7, "execute")]
        events = schedule_cycles(jobs)
        assert_resource_schedule(self, events)
        by_id = {event["job"]["id"]: event for event in events}
        self.assertEqual(by_id["c"]["start_cycle"], 5)
        with self.assertRaises(ValueError): schedule_cycles([CycleJob("a", ("missing",), (), 1, "x")])
        with self.assertRaises(ValueError): schedule_cycles([
            CycleJob("a", ("b",), (), 1, "x"), CycleJob("b", ("a",), (), 1, "x")])

    def test_reports_do_not_invent_ppa_or_clock_and_identity_is_reproducible(self):
        model, shape, action = GemminiModel(), GemmShape(16, 16, 16), GemminiAction(placement=(0,))
        a, b = model.evaluate(shape, action), model.evaluate(shape, action)
        self.assertEqual(a, b)
        self.assertIsNone(a["scenario_latency_s"])
        for key in ("energy_j", "average_power_w", "area_mm2"): self.assertIsNone(a["ppa"][key])
        with self.assertRaisesRegex(ValueError, "scoring is blocked"): model.require_ppa(a)
        clocked = GemminiModel(platform=PlatformScenario(clock_hz=100e6)).evaluate(shape, action)
        self.assertEqual(clocked["scenario_latency_s"], clocked["modeled_cycles"]/100e6)
        self.assertFalse(clocked["ppa"]["eligible_for_formal_rl_scoring"])
        self.assertNotEqual(clocked["identity"], a["identity"])

    def test_bandwidth_scenario_changes_timing_but_not_work_or_traffic(self):
        shape, action = GemmShape(16, 16, 32), GemminiAction(placement=(0,))
        fast = GemminiModel().evaluate(shape, action)
        slow = GemminiModel(platform=PlatformScenario(dram_bytes_per_cycle=4)).evaluate(shape, action)
        self.assertGreater(slow["modeled_cycles"], fast["modeled_cycles"])
        self.assertEqual(slow["operations"], fast["operations"])
        self.assertEqual(slow["traffic"], fast["traffic"])


if __name__ == "__main__": unittest.main()
