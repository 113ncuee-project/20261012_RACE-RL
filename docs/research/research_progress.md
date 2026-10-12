# 研究進度 — 2026 年 10 月 12 日

目前版本 0.6.0。已建立 **單一 RL Agent 的 compute-only mesh DSE**，預設最高 16 顆運算 chiplet，16 顆形成 4×4；外部記憶體使用服務介面，不存在實體 memory chiplet。研究主流程及 PPO arithmetic 使用 float64。

本 repository 只保留現行實作及當前實驗證據。依使用者要求，Legacy、過時文件、舊輸出與舊版封存已移除；目前原生 RapidChiplet 仍是必要依賴，四個核心來源雜湊核對通過。

## 已完成工作

- 一個共享 relational encoder、條件式 Actor、Critic、PPO optimizer，任意交錯顆數／整層 retile／coupled mapping 動作。
- state 區分完整設計 S_t 與神經觀察 O_t（schema 3）；物理 mesh、memory-interface relation、dependency-link attribution 分開表示。
- external capacity／effective bandwidth／dynamic/static energy 保留；不占 physical die area、不作路由中繼。Compute gateway 到 consumer 的 mesh hops 另算。
- global hardware+mapping cache 與全域 query budget；initial／失敗照算。較差可行 current 可接受，一次差的 mapping 不淘汰顆數。
- CNN FX 解析、獨立 NumPy Tile semantics、排程／容量／traffic／energy ledger、合法 masks、reward 數學與 checkpoint。
- 架構圖已輸出為 [PNG](figures/rl_architecture.png)／[SVG](figures/rl_architecture.svg)，完整定義在 [RL 架構](rl_architecture.md)。

## 本次實際結果

| 驗證 | 當前結果 |
| --- | --- |
| 獨立測試 | **96 項通過**，含 external access 零 mesh hops 仍付費、外部容量、不得 relay、area excludes ports、16 顆狀態轉換與神經 policy |
| Native dependency | 四個核心檔案符合 source lock |
| 完整 CNN | ResNet-18／34／50、VGG-16／19、MobileNetV2、AlexNet；batch=1、224×224、float64，七個數值及容量 checks 全通過 |
| 完整 CNN PPO | ResNet-18，起始 16 顆，1 回合／4 次全域評估／4 次 PPO updates；有整層 retile，該短回合沒有 resize，不宣稱已搜尋所有顆數 |
| 核心假說 pilot | 10 variants×3 paired seeds，60 個訓練回合、30 個 frozen rollouts；480 training＋240 frozen calls；全 2～16 顆候選實際訪問 |
| 公平性核對 | source／protocol／W、同 initial H/M/EDP、budget、global cache、parameter count 4,986、training update／frozen unchanged、outcomes 全核對通過 |

模型權重為 seeded random，沒有 pretrained／dataset accuracy 主張。PPO 短 CNN pilot 只驗證可執行迴圈，完整七模型 checks 不等於七模型正式搜尋實驗。

## 核心假說的判讀

**H1 尚未成立。** Full 對 compact／DNN／hardware／without criticality 的最終 EDP 相同；對 without attribution 只有一個 seed 改善，兩個相同。

**H2 尚未成立。** Coupled 對 independent 一勝一負一平，平均 normalized EDP difference -0.008181，探索性 paired bootstrap interval [-0.044482, 0.019940] 跨零。

**H3 尚未成立。** 本次 terminal／dense／best potential／bottleneck potential 最終 EDP 相同。這些是單一 Toy、少量訓練與三 seeds 的開發結果，不足以肯定或普遍否定假說；不挑最好 seed，也不調整既有試驗規則來迎合結果。

詳細數據、全部 variants／seeds 與配對分析見 [假說結果](hypothesis_results.md) 與 [事前 protocol](core_hypothesis_protocol.md)。

## 修改與新增檔案

主要實作為 `architecture.py`、`mesh.py`、`hardware.py`、`network.py`、`oracle.py`、`observation.py`、`policy.py`、`training.py`。外部記憶體驗證在 `tests/test_external_memory.py`，核心 controls 在 `test_hypothesis_controls.py`。當前測量、audit、繪圖及來源驗證工具在 `tools/`。

當前小型證據與 full-output checksum manifests 在 `evidence/`；完整 raw schedules／checkpoint 位於忽略的 `experiments/runs/`。另從本機 Git 暫存內容重建獨立副本，四個 native hashes 與 96 項測試再次通過，紀錄在 `evidence/git_delivery_verification.log`。

## 假設、限制與下一步

Compute area／SRAM density、float64 compute／energy 及外部記憶體仍未校準。Physical area 僅 compute dies；保守 spill-all／serial admission 不代表商用封裝或精確 packet simulator。Compressed observation 未證明完整 Markov sufficiency；learned incremental scope expansion、channel reduction、跨層常駐與新的 heuristic baselines 尚未驗收。

下一步是增加有辨識度的 dependency／memory-bottleneck workload instances、充分 training/search budget、多初始 mapping／seeds，加入物理與外部服務敏感度；再固定 CNN held-out split、validation coefficients 與正式多重比較／interaction 計畫。

本次沒有提交、設定 remote 或推送 GitHub；CI 設定不代表已在 GitHub 跑過。
