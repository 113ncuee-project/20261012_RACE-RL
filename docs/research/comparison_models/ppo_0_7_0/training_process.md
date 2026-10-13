# 目前的詳細訓練流程

實作為單一 PPO Actor–Critic。嚴格遵循 `Agent → action → environment → reward + next observation → 同一 Agent`。PPA 只由環境計算。Actor 的 logits 是選擇動作的機率參數，Critic 的值是預估未來累積 reward，兩者都不是候選 PPA。

## 1. 建立任務

使用者選 DNN，輸入功耗、延遲、面積上限與五種偏好之一。模型以 seeded random weights 建立，做 FX 解析、shape／參數檢查與獨立 NumPy Tile 數值驗證。這裡驗證 DNN 計算正確，不是在訓練 DNN 的分類能力。

GUI 固定讓 Agent 選 1–16 任一顆數，系統起始 2 顆；mesh 規則、单顆硬體能力、外部記憶體係數及 workload 固定。偏好權重與限制在整個任務內固定。

## 2. 開始一回合

環境建立保守的起始 mapping，評估起始設計的 PPA，計入一次額度，再建立 observation O₀。每回合使用相同起始任務，RL 網路權重則承接上一回合的更新。環境評估快取在每回合重設。

O 包含目前顆數、Tile／operator、hardware／memory-service、依賴／placement／mesh／流量歸因關係、剩餘額度，以及前次有效設計的 PPA 與搜尋 merit。16 維任務特徵提供限制與權重。PPA 不從候選動作推測。

## 3. Agent 選一個 action

共享 encoder 將 O 轉成圖 embedding 與 context。Actor 決定一個 action family 與完整參數：

- Resize：目標顆數。
- Whole-operator retile：operator、切分數、目標 compute、集中／分散。
- Coupled mapping：root、connected scope、動作類型與目標。

合法候選由環境依 shape／coverage／placement 規則生成；不事先計算這些候選的 PPA。神經網路的條件分布展開後，抽樣**一個**完整動作。Design family／conditional probabilities 含 5% uniform exploration，masked actions 機率為零。批次計算神經機率只改變計算效率，不改變 action 支援或 PPA 評估順序。

## 4. 環境執行與回饋

環境套用該 action，重建設計／依賴／路由／排程／buffer lifetime。Oracle 查 hardware+mapping 快取，未命中才完整評估該設計。原生 RapidChiplet 處理物理 compute mesh；本專案計帳 compute／local memory／external memory／network service、排程與容量，再算平均功耗、推論延遲、die 面積。

環境判斷是否符合三項限制，依固定尺度及權重得到 C，再計算 feasibility-first merit M。物理有效 current 可接受較差或暫時超標的設計；最佳進度另保留。物理失敗保留原有效 mapping，但剩餘預算／request counters 已變，仍回傳一個 reward 與下一個 observation。失败的候選資訊留在 trace；神經觀察目前主要使用有效 current 的 PPA，沒有把原始前一步 reward 另加為 Actor 輸入欄位。

預設 dense reward 為 `rₜ = log(M_best_before/M_best_after)`。已驗證最佳進度改善時為正，沒有改善通常為零；目前不是「每個差動作立即給負 reward」的設計。浪費評估額度會減少未來改善機會。

## 5. 下一步與回合結束

把 `(Oₜ, action_index, old_log_probability, old_value, rₜ)` 記入 trajectory；下一次 Actor 使用環境回傳的 Oₜ₊₁。直到新設計評估額度用完、request ceiling 到達或沒有合法動作。

例如每回合額度 2：起始評估占 1，通常僅能再採取一次動作、評估新設計，便結束回合。這適合流程檢查，難以學習同回合內「先 resize，再 retile，再 remap」的長期效果。64 次則提供最多 63 次新動作評估機會；cache 重訪另算 request。

## 6. PPO 更新同一個 Agent

PPO（Proximal Policy Optimization，近端策略優化）是 policy-gradient 更新方法：使用剛才與環境互動的資料，調整 Actor 的行動機率，並讓 Critic 更準確地預估累積 reward。此處不是另一個設計代理或 PPA 引擎。參考 [原始 PPO 論文](https://arxiv.org/abs/1707.06347)。

目前以 gamma=1 計算完整 episodic Monte Carlo returns：`Gₜ = rₜ + rₜ₊₁ + …`；advantage 為 `Gₜ − old V(Oₜ)`，多樣本回合再標準化。正 advantage 提高相關行動的更新誘因，負 advantage 降低；value loss 讓 Critic 擬合 G。

對儲存的同一 action 重算機率，得到新／舊 probability ratio。Actor 使用 clipped surrogate（clip=0.2），另加入 0.5×value loss 與 −0.01×entropy。Clipping 是目標函數的約束形式，不是保證每個動作的最終機率只能變動 20%。

Adam lr=3e-4、gradient norm clip=0.5，每回合 4 個更新 epochs；Actor、Critic、encoder 共用同一 optimizer。現在不使用 GAE、minibatch replay 或每一步立即更新。PPO 在回合結束更新，Agent–environment 互動則逐 action 發生。

## 7. 下一回合與最終結果

使用更新後的同一個 Agent 開始下一回合。搜尋保留跨回合已評估的最佳設計；達標時輸出 best，未達標時輸出 closest 並明示違規。GUI 直接顯示選中設計的 mesh、Tile 配置與 route-ledger 彙總圖，不需下載才能查看。

一般 GUI 建議最多 60 回合、每回合 64 次評估。至少 20 回合後，連續 15 回合未達 0.1% 的顯著改善，可提前以 objective plateau 停止；正式配對研究停用提前停止以維持相同預算。這些預設是起始建議，尚未證明跨模型收斂。沒有時間上限保證。

## 8. 已核對與尚未證明

測試禁止 Actor 選動作／候選生成階段呼叫 Oracle，並驗證環境 step 恰呼叫一次評估入口；Actor／Critic／encoder 訓練確有更新、凍結 policy 不更新。批次與逐筆神經計算的 probabilities、value 及各參數梯度已比較。

這確認機制為可執行的 PPO RL。它不證明目前壓縮 O 具有完整 Markov sufficiency，不證明候選生成是 learned graph expansion，也不證明 RL 已收斂、優於基線或找到全域最優。PPA 係數尚未校準。
