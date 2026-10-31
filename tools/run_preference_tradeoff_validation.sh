#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/race-sensitivity-mpl"
BASE='experiments/runs/preference_tradeoff_validation_20261031'
mkdir -p "$BASE"
status() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$1" > "$BASE/status.txt"; }
trap 'code=$?; status "stopped_or_failed exit=$code; rerun launcher to resume"' ERR INT TERM
status running
for scenario in reference_v2 exploratory_latency_x08; do
  out="$BASE/$scenario"; cfg="configs/common_dse_limits/$scenario";resume_flag=''
  if [[ -f "$out/suite_protocol.json" ]]; then resume_flag='--resume'; fi
  echo "===== $scenario ====="
  .venv/bin/python tools/run_preference_tradeoff_suite.py \
    --limits-dir "$cfg" --out "$out" --models resnet18 mobilenet_v2 vgg16 \
    --preferences balanced power performance area --seeds 341 342 343 344 345 \
    --model-seed 7 --input-size 64 --budget 64 --rl-episode-size 16 \
    --methods reinforce --dominant-weight .95 ${resume_flag:+$resume_flag}
  if [[ ! -f "$out/analysis.json" ]]; then
    .venv/bin/python tools/analyze_common_dse.py "$out" --out "$out/analysis.json"
  else
    .venv/bin/python -c 'import hashlib,json,sys;from pathlib import Path;p=Path(sys.argv[1]);a=json.loads((p/"analysis.json").read_text());assert a["results_sha256"]==hashlib.sha256((p/"results.json").read_bytes()).hexdigest(),"analysis is stale"' "$out"
  fi
  if [[ ! -f "$out/figures/report.md" ]]; then
    .venv/bin/python tools/summarize_race_ppa_sensitivity.py --batches "$out" --out "$out/figures"
  fi
done
if [[ ! -f "$BASE/combined/report.md" ]]; then
  .venv/bin/python tools/summarize_race_ppa_sensitivity.py --batches "$BASE/reference_v2" "$BASE/exploratory_latency_x08" --out "$BASE/combined"
fi
status completed
echo "完成：$BASE/combined/report.md"
