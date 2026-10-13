# 正式 REINFORCE 的核心假說結果

新正式版本已重跑 10 variants × 3 seeds，全部使用無 Critic 的 REINFORCE、float64、同一 Tiny Toy、同一需求與 1–16 顆候選集合、相同起始結構及 Actor 初始化、最多 2 個相依 Tile、關閉提前停止。每 trial 訓練 2 回合與凍結 1 回合，每回合最多 8 個完整 action。reset 零 PPA 評估。

實際訓練 469 次新評估、凍結 236 次；差額來自環境快取。共 720 次完整動作／環境回饋，其中 152 次修改多個 Tile。每個模型 5,257 個參數；全部訓練 policy 確有更新，凍結 policy 不更新。來源／protocol／計費與原始 PPA 重算 audit 通過。

| Variant | Seed 0 J | Seed 1 J | Seed 2 J |
| --- | --- | --- | --- |
| full_dense | 0.04897453 | 0.07013566 | 0.12882516 |
| compact_dense | 0.04897453 | 0.07013566 | 0.05091452 |
| dnn_dense | 0.04897453 | 0.05043657 | 0.05091452 |
| hardware_dense | 0.04897453 | 0.07013566 | 0.16489599 |
| no_attribution_dense | 0.04897453 | 0.07013566 | 0.16489599 |
| no_criticality_dense | 0.04897453 | 0.07013566 | 0.12882516 |
| independent_dense | 0.06594407 | 0.06742179 | 0.06801293 |
| full_terminal | 0.04897453 | 0.07013566 | 0.12882516 |
| full_best_potential | 0.04897453 | 0.07013566 | 0.12882516 |
| full_bottleneck_potential | 0.04897453 | 0.07013566 | 0.12882516 |

J=M/(1+M)，越低越好。以下為配對描述性差異：left−right，負值才偏向 full／left；3 seeds 不構成正式顯著性驗收。

| 假說 | Left / Right | 平均差 | Bootstrap 95% CI | 勝 / 平 / 負 |
| --- | --- | --- | --- | --- |
| H1 | full_dense / compact_dense | 0.02597021 | [0.00000000, 0.07791064] | 0 / 2 / 1 |
| H1 | full_dense / dnn_dense | 0.03253657 | [0.00000000, 0.07791064] | 0 / 1 / 2 |
| H1 | full_dense / hardware_dense | -0.01202361 | [-0.03607083, 0.00000000] | 1 / 2 / 0 |
| H1 | full_dense / no_attribution_dense | -0.01202361 | [-0.03607083, 0.00000000] | 1 / 2 / 0 |
| H1 | full_dense / no_criticality_dense | 0.00000000 | [0.00000000, 0.00000000] | 0 / 3 / 0 |
| H2 | full_dense / independent_dense | 0.01551885 | [-0.01696954, 0.06081223] | 1 / 0 / 2 |
| H3 | full_best_potential / full_terminal | 0.00000000 | [0.00000000, 0.00000000] | 0 / 3 / 0 |
| H3 | full_bottleneck_potential / full_best_potential | 0.00000000 | [0.00000000, 0.00000000] | 0 / 3 / 0 |
| H3 | full_bottleneck_potential / full_dense | 0.00000000 | [0.00000000, 0.00000000] | 0 / 3 / 0 |

**H1／H2／H3 均未成立。** H2 full 对 independent 的結果混合，不能宣稱聯合修改已優於單項修改；H3 目前完全同分。需更有辨識度的預先宣告 barrier workload、多工作負載與更多 seeds，不得為了讓結論好看而調整已執行紀錄。

PPO 只做了新環境契約下的兩種子可執行 smoke，未形成 PPO 對 REINFORCE 的優越性證據。舊 0.7.0 PPO pilot 位於[比較模型紀錄](comparison_models.md)，未重標為正式成果。

現行證據：[protocol](evidence/formal_core_hypotheses/protocol.json)、[query／source audit](evidence/formal_core_hypotheses/audit.json)、[PPA audit](evidence/formal_core_hypotheses/scoring_audit.json)、[trial results](evidence/formal_core_hypotheses/trial_results.json)。完整 raw／checkpoints 由可重現工具產生，另有 SHA256 manifest。

2026-10-14 清理大型訓練資料後，以上小型歷史證據仍保留；舊 raw／checkpoint 已刪除，原 SHA256 manifest 只保留歷史身份。新 60 回合模型訓練由終端批次重跑，尚未執行，不能取代上述 pilot 的結果或當成新增假說證據。
