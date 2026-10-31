#!/bin/bash
# Three predeclared PPA scenarios, RACE-RL only. Safe to rerun to resume.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/race-sensitivity-mpl"
ROOT_OUT="experiments/runs/race_ppa_sensitivity_20261031"
mkdir -p "$ROOT_OUT"
status() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$1" > "$ROOT_OUT/status.txt"; }
trap 'code=$?; status "stopped_or_failed exit=$code; rerun the same launcher to resume"' ERR INT TERM
status 'running'
for scenario in reference_v2 exploratory_latency_x2 exploratory_latency_x2_area_x05; do
  out="$ROOT_OUT/$scenario"
  cfg="configs/common_dse_limits/$scenario"
  resume_flag=""
  if [[ -f "$out/suite_protocol.json" ]]; then resume_flag="--resume"; fi
  echo "===== 情境 $scenario ====="
  .venv/bin/python tools/run_common_dse_suite.py \
    --limits-dir "$cfg" --out "$out" \
    --models resnet18 mobilenet_v2 vgg16 \
    --preferences balanced power performance area \
    --seeds 331 332 333 334 335 --model-seed 7 --input-size 64 \
    --budget 64 --rl-episode-size 16 --methods reinforce ${resume_flag:+$resume_flag}
  if [[ ! -f "$out/analysis.json" ]]; then
    PYTHONPATH=src .venv/bin/python tools/analyze_common_dse.py "$out" --out "$out/analysis.json"
  else
    .venv/bin/python -c 'import hashlib,json,sys;from pathlib import Path;p=Path(sys.argv[1]);a=json.loads((p/"analysis.json").read_text());assert a["results_sha256"]==hashlib.sha256((p/"results.json").read_bytes()).hexdigest(),"analysis is stale"' "$out"
  fi
  if [[ ! -f "$out/figures/report.md" ]]; then
    .venv/bin/python tools/summarize_race_ppa_sensitivity.py --batches "$out" --out "$out/figures"
  fi
done
combined="$ROOT_OUT/combined"
if [[ ! -f "$combined/report.md" ]]; then
  .venv/bin/python tools/summarize_race_ppa_sensitivity.py --batches \
    "$ROOT_OUT/reference_v2" "$ROOT_OUT/exploratory_latency_x2" \
    "$ROOT_OUT/exploratory_latency_x2_area_x05" --out "$combined"
fi
status 'completed'
echo "全部情境完成：$combined/report.md"
