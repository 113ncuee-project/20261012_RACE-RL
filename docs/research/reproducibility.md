# 可重現流程

Python 3.12；新輸出目錄需為空：

```sh
.venv/bin/python tools/verify_sources.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python tools/validate_cnn_suite.py --precision float64 --analytical-check --guarded-memory --out experiments/runs/new_cnn
.venv/bin/python -m race_rl.cli search-mesh --requirements configs/vendor_requirements.json --model resnet18 --compute-counts 16 2 4 8 --episodes 1 --budget 4 --max-scope 1 --out experiments/runs/new_full_search
.venv/bin/python tools/run_hypothesis_pilot.py --seeds 0 1 2 --budget 8 --train-episodes 2 --out experiments/runs/new_hypotheses
.venv/bin/python tools/audit_hypothesis_pilot.py --input experiments/runs/new_hypotheses --out experiments/runs/new_hypotheses/audit.json
```

候選列表第一個顆數為固定起始 count，完整 DNN gate 先通過才搜尋。研究主流程使用 float64。一般 `search-mesh` 預設精簡 SQLite 互動紀錄，保留來源／身份、actual PPA／reward 與最終完整選中設計。需要每候選 raw ledgers 的研究 audit，明確加入 `--recording full`；假說 pilot 的直接 `export` 保持完整研究輸出。

新訓練輸出集中在 `experiments/runs/model_suite/`，大型資料不納入 Git。2026-10-14 已依使用者要求清除舊大型訓練 raw；小型歷史證據與舊 checksum manifest 保留於 `docs/research/evidence/`，其中的 raw 路徑已不再於本機存在，不能當成仍可直接讀取的檔案。清理範圍見 `training_output_cleanup.json`。PPO 歷史摘要只列為比較資料，不當成正式版本成果。

`requirements-tested.txt` 與 environment manifest 是本機環境快照，不宣稱其他平台浮點完全相同。Native dependency 依 `rapidchiplet.lock.json` 的四個核心 hashes 核對。回合邊界 checkpoint 需相同 experiment identity，僅載入本機產生的可信檔案。

圖的重現：安裝 `.[reports]`，用支援繁體中文的字型執行 `tools/draw_rl_architecture.py`；PNG／SVG 皆存入 `docs/research/figures/`。正式 REINFORCE 算法、budget、外部記憶體與 state/action 定義見 [RL 架構](rl_architecture.md)。

歷史正式 pilot 的 compact evidence 在 `docs/research/evidence/formal_core_hypotheses/`；逐筆 raw 評估與 checkpoint 的 SHA256 manifest 僅保留歷史身份，原大型輸出已清除。GUI 的小型真實執行證據在 `evidence/gui_validation/`。動作產生效能可用 `python tools/benchmark_action_generation.py --out experiments/runs/action_generation_benchmark.json` 重跑。PPO 必須明確用 `--comparison-model ppo`，不會被 GUI 選中。

新正式終端批次：`python tools/train_model_suite.py`，依序七模型各 60 回合、64×64、每回合 64 互動，停用提前停止。前後各同預算凍結測試與訓練分開記錄。`--resume <原目錄>` 核對來源／protocol／workload 後，從最後完整交易的 policy／optimizer／RNG 接續；精簡和完整記錄的 action、PPA、reward、policy／RNG 已做同 seed 回歸核對。完整操作見 `experiments/README.md`。

Git 簽出位元組由 `.gitattributes` 固定：文字來源使用 LF，Windows `.bat` 使用 CRLF。RapidChiplet 仍依原始位元組 SHA256 嚴格驗證，不可改 lock 值來繞過換行造成的不符。Windows 簽出回歸與 CI 狀態見[CI 紀錄](ci_verification.md)。
