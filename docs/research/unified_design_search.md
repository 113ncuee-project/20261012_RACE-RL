# 單一 Agent 的探索空間

廠商 GUI 固定讓 RL 選擇 1–16 任一運算顆數，起始 2 顆；當前配對研究 pilot 使用子集合 `{2,4,6,8,10,12,14,16}`，只有 compute die 在 mesh 內。外部記憶體介面為固定服務參數，沒有 memory chiplet 數量或實體座標搜尋。

| 可探索項目 | Agent 的動作 |
| --- | --- |
| 顆數 | Resize 到候選集合任一不同數量 |
| 整層 partition | Conv 等 NCHW output height；Linear output features |
| Partition count | 1／2／4／8／16／32 跳躍；目前數量±1、减半／加倍，限定合法 output 維度 |
| Work distribution | 整層同節點或 round-robin；後续每 Tile 可搬到其他 compute |
| Coupled scope | 由 connected legal scopes 中選 root／scope／type／targets；K=1～4 |
| Storage | 完整 tensor／parameter 選 dram0 或 dram1，含 compute+storage 原子 affinity 動作 |

單一 policy 可任意交錯這些動作。Resize 將刪除節點的工作按 ID modulo 新數量重定位，不使用 PPA 選址，之後可再次 retile／搬移。物理可行候選仍更新 current，即使加權成本較差或 PPA 暫時超標；已評估的最佳搜尋進度另外保留，對外只將達標方案標為 best。沒有顆數 performance blacklist，容量由候選 oracle 判定，5% exploration mixture 保持合法候選非零機率。

DNN 結構、精度、單顆規格、mesh 實體位置規則、路由與排程固定。尚不支援 channel reduction、任意不規則切分、memory sharding、data replication、precision／model architecture search 或實體 floorplan 搜尋。

合法候選生成不用 PPA 預先排名。這個有限空間與有限預算沒有全域最優保證；一次差的 mapping 不能被解讀成整個架構差。廠商 PPA 限制與五種偏好固定於 task，見 [需求輸入](vendor_requirements.md)。完整 RL state/action/reward 契約見 [架構定義](rl_architecture.md)。

研究名稱的 Coupled-Edit 指選中相依範圍的一次原子聯合修改，不要求每次重建整個全域方案；範圍、類型、參數由同一 policy 選出，中間零 PPA 評估，完整 action 立即送環境評一次。正式預設 K≤2，K=1 為比較設定；CLI／研究工具仍可宣告 1–4 的 ablation。PPA 權重與 graph attribution 是 Relational Attribution 的輸入與回饋基礎。

合法 mask 已改為直接結構計算，避免為全部候選反覆建立、驗證完整設計。所選 action 仍完整驗證。正式與 PPO 比較模型共享此環境。
