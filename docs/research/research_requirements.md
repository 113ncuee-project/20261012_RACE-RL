# 現行研究需求

本文件只描述目前採用的需求；過時的原始版本與已淘汰實驗不保留。

1. 廠商先輸入平均功耗、推論延遲及運算 die 面積限制，並從 PPA 平均、偏重功耗、偏重效能、偏重面積、使用者自訂五項選一。
2. 只有一個 RL Agent、共享 Actor／encoder 與 optimizer。GUI 固定可選 1–16 顆，使用者不指定顆數；mesh 拓樸固定。
3. 嚴格流程：Agent 根據目前 observation 選一個完整 action；環境執行 action、計算該設計的 PPA、產生一個 reward 與下一個 observation；回傳同一個 Agent。候選生成只檢查結構合法性。
4. 同一 Agent 可交錯選顆數、operator 切分、工作 placement 與資料 storage。不得以某一個差的 mapping 或 partition 永久排除整個顆數。
5. 一個 episode 的所有硬體使用同一評估預算；起始不評估；action 後的失敗新設計計費，重訪快取算 request。Current 可探索較差或暫時 PPA 超標的物理有效狀態，另保留已評估的最佳進度；對外只有達標才叫 best。
6. 沒有實體 memory chiplet。外部 memory service 的容量、頻寬與能量仍計入，不占 mesh 格點、不作 relay；面積目前只含 compute dies。
7. Python local server 加網頁 GUI，面向 macOS／Windows；搜尋結果的晶粒排列、工作配置與真實 route／傳輸量直接顯示於網頁。
8. 預設提供固定搜尋預算與 objective plateau 停止條件，專家可從進階選項調整；不得將 plateau、測試通過或有限預算的 best observed 當成收斂／全域最優證明。
9. 中文研究文件、實際執行證據、來源與環境身份及可重現輸出。不得捏造結果、重標舊指標、挑最好 seed 或把未達標設計當成達標。

核心假說仍需正式驗證：H1 關聯 state 的有限預算搜尋價值；H2 dependency-coupled 原子動作相對單變量動作的價值；H3 verified-best 與 schedule-informed potential 對 credit assignment 的價值。現有開發 pilot 不構成正式驗收。

正式版本指定為不使用 PPO 與 Critic 的 REINFORCE。PPO 保留於比較模型紀錄，尚未證明較好。reset 不評估初始 PPA，所有評估由 Agent 所選 action 觸發；下一動作使用環境回傳的新 state 與實際 reward。

研究名稱為 Relational Attribution and Coupled-Edit Reinforcement Learning。一次 action 由同一 policy 選相依子圖／類型／完整參數，聯合修改原子提交；中間不評估 PPA，完整 action 立即送環境評估一次。正式預設最多 2 個相依 Tile，單項修改為比較設定；64 僅表示最多環境互動次數，不是動作內部的生成次數。
