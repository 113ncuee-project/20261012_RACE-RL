# 研究進度 — 2026 年 10 月 12 日

目前版本 0.7.0。已實作單一 PPO Actor–Critic、廠商 PPA 條件與五種偏好、Python 本機網頁 GUI。GUI 固定讓 Agent 自行選擇 1–16 顆 compute chiplet，mesh 固定；外部記憶體只有服務介面，不是實體 chiplet。

## 已核對的 RL 流程

Actor 根據目前 observation 選一個完整 action；環境套用設計、評估該設計、產生 reward 與下一個 observation，交回同一 Agent。候選生成與 Actor 不呼叫 PPA Oracle。每回合完成後使用 trajectory 更新同一套 encoder／Actor／Critic；PPO 不負責計算 PPA。

最新測試 123 項通過，包括禁止選 action 時偷跑 PPA、step 單次 Oracle 呼叫、實際參數更新、frozen unchanged、單一 Agent 各動作、容量／排程／能量、PPA 權重／限制、GUI／取消與 route 彙總。詳細流程見 [訓練說明](training_process.md)。

## 廠商與 GUI

三個上限必須完整輸入，偏好為平均／功耗／效能／面積／自訂。先達標，再依正規化加權成本排序；未達標結果明確標為 closest。預設最多 60 回合／每回合 64 次新設計評估，含起始；20 回合後若持續 15 回合改善不足 0.1% 可提前停止。這是建議預算，不是已驗證跨模型收斂的設定。

網頁直接顯示實際晶粒排列、Tile／operator 配置、外部記憶體讀寫與雙向 mesh bytes；可點選細節。逐步狀態可區分 Agent 選動作、環境計算 PPA、收到回饋、更新 policy。macOS 已實測；Windows matrix 已加入 CI，尚未在 Windows 實機或 GitHub 執行。

## 當前研究證據

七個完整 CNN 在 224×224、float64 的數值與分析式容量檢查通過。完整 ResNet-18 PPO 為 1 回合、4 次評估的短開發測試，不構成收斂證据。PPA pilot 為 10 variants × 3 seeds：480 次訓練、240 次凍結搜尋、相同 5,242 個參數。實際比較見 [假說結果](hypothesis_results.md)。

H1／H2／H3 尚未成立。候選集合、預算、權重、尺度與起始狀態受控，但單一 Toy／三 seeds／短訓練不足以支持正式假說。任務全可行且面積占成本約 50–80%，研究辨識力不足，詳見 [評分核對](scoring_review.md)。

## 限制與下一步

物理係數未校準；平均功耗不等於 peak power，compute area 不含封裝／DRAM。壓縮 observation 的完整 Markov sufficiency 未證明，沒有 learned incremental scope expansion 或全域最優保證。多需求／多 workload／充分訓練、正式 baselines、validation-only 係數選擇及 held-out paired comparisons 尚待完成。

過時規格與舊目標成績已移除，當前證據使用現行實作重新執行，沒有將舊數字重標。完整 raw ledgers／checkpoints 在忽略的 `experiments/runs/`，當前小型證據與 checksum manifest 在 `evidence/`。本次未提交或推送 GitHub；CI 設定不代表雲端已執行。
