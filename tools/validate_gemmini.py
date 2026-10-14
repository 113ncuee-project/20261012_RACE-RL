"""Reproduce small Gemmini foundation cases; no RL training or candidate search."""
import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import html
import json
import platform
from pathlib import Path
import sys
import time
import numpy as np
from race_rl.accelerators.gemmini_config import verify_reference, default_config
from race_rl.accelerators.memory_hierarchy import MemoryHierarchy
from race_rl.accelerators.gemmini_model import (
    GemminiAction, GemminiModel, GemmShape, PlatformScenario, execute_gemm, lower_conv2d,
)
from race_rl.serialization import save_json

ROOT = Path(__file__).resolve().parents[1]


def selected_placement(shape, action, chiplets):
    count = ((shape.m+action.m_tile-1)//action.m_tile)*((shape.n+action.n_tile-1)//action.n_tile)
    # Declared fixture placement; no optimizer, mapper or PPA-based selection.
    return tuple(index % chiplets for index in range(count))


def digest_array(value): return hashlib.sha256(value.tobytes()).hexdigest()


def direct_conv(x, weights, stride, padding):
    batches, _, height, width = x.shape
    channels, _, kh, kw = weights.shape
    oh, ow = (height+2*padding-kh)//stride+1, (width+2*padding-kw)//stride+1
    source = np.pad(x.astype(np.int64), ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    result = np.zeros((batches, channels, oh, ow), dtype=np.int64)
    for batch in range(batches):
        for channel in range(channels):
            for y in range(oh):
                for z in range(ow):
                    result[batch, channel, y, z] = np.sum(source[batch, :, y*stride:y*stride+kh,
                        z*stride:z*stride+kw]*weights[channel])
    return result


def run_validation(platform_scenario, *, include_traces=False, memory=None):
    audit = verify_reference(ROOT)
    if audit["status"] != "PASSED": raise ValueError("Gemmini source verification failed: "+str(audit["mismatches"]))
    started = time.perf_counter()
    rng = np.random.default_rng(20261014)
    matrices = []
    for name, dimensions in (("gemm_reuse", (64, 16, 32)), ("gemm_tail", (29, 17, 33))):
        m, n, k = dimensions
        a = rng.integers(-8, 9, (m, k), dtype=np.int8)
        b = rng.integers(-8, 9, (k, n), dtype=np.int8)
        reference = a.astype(np.int64) @ b.astype(np.int64)
        matrices.append((name, a, b, reference, {"kind": "GEMM"}))
    x = rng.integers(-3, 4, (1, 4, 8, 8), dtype=np.int8)
    weights = rng.integers(-3, 4, (8, 4, 3, 3), dtype=np.int8)
    a, b, lowering = lower_conv2d(x, weights)
    expected = direct_conv(x, weights, 1, 0).transpose(0, 2, 3, 1).reshape(-1, 8)
    matrices.append(("conv_8x8_dense", a, b, expected, {"kind": "Conv GEMM stage", **lowering}))
    cases = []
    for name, a, b, reference, workload_metadata in matrices:
        shape = GemmShape(a.shape[0], b.shape[1], a.shape[1])
        numerical = execute_gemm(a, b, GemminiAction()).astype(np.int64)
        equal = np.array_equal(numerical, reference)
        if not equal: raise ValueError("independent numerical mismatch: "+name)
        variants = []
        for chiplets, retain in ((1, False), (1, True), (2, True)):
            action = GemminiAction(retain_weights=retain)
            action = replace(action, placement=selected_placement(shape, action, chiplets))
            record = GemminiModel(chiplets=chiplets, platform=platform_scenario, memory=memory).evaluate(shape, action)
            if not include_traces:
                record.pop("events"); record.pop("transfers")
            variants.append(record)
        cases.append({"case": name, "workload": workload_metadata, "numerical_status": "PASSED",
            "reference": "independent NumPy INT64 matmul" if name != "conv_8x8_dense" else "independent direct convolution loops",
            "input_hashes": {"A": digest_array(a), "B": digest_array(b)},
            "result_sha256_int64": digest_array(numerical), "max_abs_error": 0, "variants": variants})
    reuse = cases[0]["variants"]
    return {"status": "PASSED", "scope": "fixed-action foundation verification; not training or accelerator performance validation",
        "recorded_at": datetime.now().astimezone().isoformat(), "wall_time_s": time.perf_counter()-started,
        "source_audit": audit, "seed": 20261014,
        "execution_platform": {"system": platform.system(), "machine": platform.machine(),
            "python": sys.version, "numpy": np.__version__},
        "configuration": default_config().provenance(), "cases": cases,
        "findings": {"gemm_weight_bytes_reload": reuse[0]["traffic"]["by_operand"]["B"],
            "gemm_weight_bytes_retained": reuse[1]["traffic"]["by_operand"]["B"],
            "gemm_weight_bytes_two_chiplets": reuse[2]["traffic"]["by_operand"]["B"],
            "interpretation": "same logical GEMM work; residency reduces model traffic, replication across chiplets costs bytes; no silicon speedup claim"},
        "trust": {"configuration": "PINNED_PRIMARY_SOURCE_VERIFIED", "arithmetic": "INDEPENDENT_INTEGER_REFERENCE_VERIFIED",
            "capacity_and_resources": "MODEL_INVARIANTS_TESTED", "timing": "ANALYTICAL_UNCALIBRATED",
            "platform_and_chiplet_mesh": "DECLARED_RESEARCH_ASSUMPTIONS", "power_area": "NOT_AVAILABLE"}}


def render_report(report):
    rows = []
    for case in report["cases"]:
        for variant in case["variants"]:
            amount = variant["traffic"]["by_operand"]
            rows.append("<tr>"+"".join(f"<td>{html.escape(str(value))}</td>" for value in (
                case["case"], variant["chiplets"], "保留" if variant["action"]["retain_weights"] else "重載",
                variant["modeled_cycles"], amount["A"], amount["B"], amount["C"],
                variant["traffic"]["directed_link_byte_hops"], case["numerical_status"]))+"</tr>")
    trust_rows = "".join(f"<tr><td>{html.escape(key)}</td><td>{html.escape(value)}</td></tr>" for key, value in report["trust"].items())
    commit = report["source_audit"]["commit"]
    return f"""<!doctype html><html lang="zh-Hant"><meta charset="utf-8">
<title>Gemmini 單晶粒基礎驗證</title><style>
body{{font-family:'Times New Roman','DFKai-SB',serif;max-width:1200px;margin:48px auto;padding:0 24px;color:#183848;line-height:1.6}}
h1{{font-size:32px}}h2{{font-size:24px}}table{{border-collapse:collapse;width:100%;margin:20px 0}}th,td{{border:1px solid #ccd8de;padding:9px;text-align:left}}th{{background:#edf5f3}}code{{overflow-wrap:anywhere}}.note{{color:#9a5424}}
</style><h1>Gemmini 單晶粒基礎驗證</h1>
<p>RACE-RL 記憶體擴充：DRAM ↔ GB ↔ LB ↔ RF ↔ PE；部分和另由 LB 連到 accumulator。16×16 陣列、INT8、GB 256 KiB 與 accumulator 64 KiB 參考固定 Gemmini 配置；LB／RF 組織及服務頻寬為研究假設，並非官方 defaultConfig RTL。</p>
<p>來源：<a href="https://github.com/ucb-bar/gemmini/tree/{commit}">ucb-bar/gemmini</a>，commit <code>{commit}</code>。7 個原始檔與 21 個配置欄位核對通過。</p>
<p class="note">下表 cycles 為未經 RTL 校準的分析式估計。DRAM 與 mesh 為宣告的研究情境。功耗、能量、面積皆留空；新版尚不能用於正式 PPA 訓練。</p>
<h2>固定動作的運算與流量核對</h2>
<p>三種方案都由驗證腳本事先指定，沒有搜尋或更新 RL。尾端補零的資料傳輸計入；輸出採 INT32。卷積案例只涵蓋 explicit-im2col 後的 GEMM 階段，未計 im2col 建立成本。</p>
<table><thead><tr><th>案例</th><th>顆數</th><th>權重</th><th>模型 cycles</th><th>A bytes</th><th>B bytes</th><th>C bytes</th><th>link byte-hops</th><th>數值核對</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>來源與可信度</h2><table><tr><th>項目</th><th>驗證狀態</th></tr>{trust_rows}</table>
<p>可用於核對容量、資料搬移與資源排程的模型行為；尚不能作為真實晶片 PPA 或全 CNN 加速的證據。完整假設、覆蓋範圍、來源 hash 及硬體／執行平台記錄於 report.json。</p>
</html>"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", type=Path, default=ROOT/"configs/gemmini_platform.json")
    parser.add_argument("--memory", type=Path, default=ROOT/"configs/memory_hierarchy.json")
    parser.add_argument("--out", type=Path, default=ROOT/"experiments/runs/gemmini_foundation")
    parser.add_argument("--full-traces", action="store_true", help="include per-job calendars; default report is compact")
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output directory is nonempty; choose a fresh --out")
    try:
        scenario_record = json.loads(args.platform.read_text(encoding="utf-8"))
        memory = MemoryHierarchy.from_record(json.loads(args.memory.read_text(encoding='utf-8')))
        report = run_validation(PlatformScenario.from_record(scenario_record), include_traces=args.full_traces, memory=memory)
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))
    report["platform_source_record"] = scenario_record
    report["platform_source_sha256"] = hashlib.sha256(args.platform.read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out/"report.json", report)
    (args.out/"report.html").write_text(render_report(report), encoding="utf-8")
    print("Gemmini 基礎驗證 PASSED；三個數值案例、九個固定配置／執行方案。")
    print("時序為分析式情境估計；PPA 未校準，未更新 RL 或替換正式評估器。")
    print("報告："+str(args.out/"report.html"))
    return 0


if __name__ == "__main__": raise SystemExit(main())
