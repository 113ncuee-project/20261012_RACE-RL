# 可重現流程

Python 3.12；新輸出目錄需為空：

```sh
.venv/bin/python tools/verify_sources.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/validate_cnn_suite.py --precision float64 --analytical-check --guarded-memory --out experiments/runs/new_cnn
.venv/bin/python -m race_rl.cli search-mesh --model resnet18 --compute-counts 16 2 4 8 --episodes 1 --budget 4 --max-scope 1 --out experiments/runs/new_full_search
.venv/bin/python tools/run_hypothesis_pilot.py --seeds 0 1 2 --budget 8 --train-episodes 2 --out experiments/runs/new_hypotheses
.venv/bin/python tools/audit_hypothesis_pilot.py --input experiments/runs/new_hypotheses --out experiments/runs/new_hypotheses/audit.json
```

候選列表第一個顆數為固定起始 count，完整 DNN gate 先通過才搜尋。研究主流程使用 float64。各紀錄包含實作、W/H/M 身份與來源、profile、實際 query logs、完整 raw ledgers、best hardware／mapping、policy checkpoint 與訓練／凍結 costs。

本次現行輸出在 `experiments/runs/`，大型 raw schedules／checkpoints 不納入 Git；當前小型證據與完整輸出 checksum manifest 位於 `docs/research/evidence/`。此 repository 不保留過時版本封存或 Legacy 專案。

`requirements-tested.txt` 與 environment manifest 是本機環境快照，不宣稱其他平台浮點完全相同。Native dependency 依 `rapidchiplet.lock.json` 的四個核心 hashes 核對。回合邊界 checkpoint 需相同 experiment identity，僅載入本機產生的可信檔案。

圖的重現：安裝 `.[reports]`，用支援繁體中文的字型執行 `tools/draw_rl_architecture.py`；PNG／SVG 皆存入 `docs/research/figures/`。PPO 算法、budget、外部記憶體與 state/action 定義見 [RL 架構](rl_architecture.md)。
