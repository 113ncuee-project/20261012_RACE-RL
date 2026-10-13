# 核心假說配對 pilot：實驗設計

當前實驗版本：0.7。目的為開始檢驗原始需求的 H1／H2／H3，而非只驗證軟體可執行。此階段是開發用 Toy pilot，不是正式完整 CNN 泛化實驗。

在每筆 trial 開始前已寫入不可覆寫的 `protocol.json`，記錄 workload／來源雜湊、所有 variants、seeds、PPO／環境／budget 與 bootstrap 設定；每個 trial 帶同一個 protocol hash。現行需求定義於 `research_requirements.md`。

## 共同條件

- 一個固定 Toy workload：兩層 3×3 Conv 與 residual Add，batch=1、2 channels、8×8、float64。NumPy 獨立 Tile reference 與 Torch double 數值比對先通過，才啟動試驗。
- 運算 chiplet 候選為 2／4／6／8／10／12／14／16，只有 compute die 在實體 mesh；外部記憶體是無 die 幾何的服務介面，容量／bandwidth／能量保留。
- 每筆 trial 起始硬體／mapping 相同，固定起始 2 顆運算 chiplet；所有顆數共用一個固定初始 PPA merit reference 及同回合 query budget。
- 廠商需求固定為 `configs/research_requirements.json`：平均功耗上限 0.1 W、延遲上限 0.001 s、運算 die area 上限 16 mm²、PPA 平均。這是事前宣告的合成研究任務，不是實際廠商規格。
- 三個獨立 policy seeds：0／1／2；不挑最好 seed。每個 variant 各有相同配對 seeds。
- 每 seed／variant 訓練 2 回合，每回合最多 8 次 uncached oracle evaluations；之後做 1 回合凍結 policy 的搜尋，最多 8 次。
- 每回合初始設計與失敗皆計入，快取仍算 requests，上限為 10B。訓練最多 16 次，凍結搜尋最多 8 次，合計最多 24 次；10 variants×3 seeds 的上限為 720 次。
- 同一種含 16 維需求特徵的 encoder／actor／critic 架構，width=16、PPO epochs=4、gamma=1，features／weights／returns／optimizer arithmetic 全部 float64。
- 凍結階段獨立重設 action RNG 為 10000+seed，所有 variants 共用同樣規則；比較前後 policy 雜湊必須相同，不允許邊搜邊更新。
- 配對 outcome 為凍結搜尋的 best verified normalized PPA merit；另記 parameter count、訓練與凍結的實際 calls／requests、action support、結構生成與 wall time。

## 比較矩陣

| 假說 | 比較 | 固定項目 |
| --- | --- | --- |
| H1 | full vs compact／DNN-only／hardware-only／without attribution／without criticality | coupled actions、dense reward、oracle／budget／seeds |
| H2 | full coupled vs full independent edits | full state、dense reward、oracle／budget／seeds、顆數集合 |
| H3 | best potential vs terminal；bottleneck potential vs best potential／dense | full state、coupled actions、oracle／budget／seeds |

共 10 個 variants、30 筆獨立 seed／variant trials、60 個訓練回合與 30 個凍結回合。H1 不增加／刪除網路參數，只切換 message 路徑；每組仍報實際參數量與候選成本。

H2 的 independent edit 定義為每次選一個 Tile placement、一個完整 storage object，或一個 operator 的 partition count。partition-only 沿舊 Tile 最大輸出交集繼承 placement，不順便最佳化 compute／storage；需要 agent 後續逐筆搬移才能改配置。兩組都可使用相同 mesh resize 與其必要 deterministic relocation。因 action support／條件解碼結構不同，候選數與時間必須一起報，不能把行動設計與控制器計算差異隱藏。

H3 的 potential 為 `lambda_b*log(initial/best) - lambda_c*bottleneck`，terminal potential=0。bottleneck 使用可追溯 criticality 診斷中最長的單一非 compute critical service／makespan，是明示的 bounded proxy。lambda_b=1；best potential 的 lambda_c=0，bottleneck potential 的 lambda_c=0.1。0.1 是本次事前固定的探索值，尚未經正式 validation 選擇。

## 判讀規則

報全部 seed、平均／中位數／sample standard deviation，以及配對 seed difference。Bootstrap 固定 seed=1234、10,000 resamples，以 seed/run 作單位；只有三個 seeds，區間只作探索性描述。

此 pilot 不做正式 p-value 宣稱，不把多項 bootstrap CI 當作多重比較已校正的 hypothesis test。沒有觀察到改善不代表普遍假說被否證；有改善也不能當作跨 CNN 成立。不得因結果不利就修改本次 baseline、reward、seed 或刪掉 trial；後續改設定需另開實驗版本。

下一步必須擴充獨立 workload instances、充分訓練／搜尋預算、多初始 mapping 與 seeds，才評估完整 CNN。正式 H1/H2/H3 驗收另需 held-out split、validation-only 係數選擇、正式多重比較計畫、校準／不確定性分析，以及 state×action×reward interaction。

## 重現

```sh
.venv/bin/python tools/run_hypothesis_pilot.py \
  --seeds 0 1 2 --budget 8 --train-episodes 2 \
  --compute-counts 2 4 6 8 10 12 14 16 \
  --out experiments/runs/new_core_hypotheses
```

來源、數值 gate、trial records、配對分析與 checkpoint 皆保留；輸出目錄必須全新。大型 raw ledger／checkpoint 在 runs/ 中，本機完整保留；小型證據另存入 `docs/research/evidence/`。
