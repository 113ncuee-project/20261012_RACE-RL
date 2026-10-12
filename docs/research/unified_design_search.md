# 單一 Agent 的探索空間

預設運算顆數 `{2,4,6,8,10,12,14,16}`，只有 compute die 在 mesh 內。外部記憶體介面為固定服務參數，沒有 memory chiplet 數量或實體座標搜尋。

| 可探索項目 | Agent 的動作 |
| --- | --- |
| 顆數 | Resize 到候選集合任一不同數量 |
| 整層 partition | Conv 等 NCHW output height；Linear output features |
| Partition count | 1／2／4／8／16／32 跳躍；目前數量±1、减半／加倍，限定合法 output 維度 |
| Work distribution | 整層同節點或 round-robin；後续每 Tile 可搬到其他 compute |
| Coupled scope | 由 connected legal scopes 中選 root／scope／type／targets；K=1～4 |
| Storage | 完整 tensor／parameter 選 dram0 或 dram1，含 compute+storage 原子 affinity 動作 |

單一 policy 可任意交錯這些動作。Resize 將刪除節點的工作按 ID modulo 新數量重定位，不使用 PPA 選址，之後可再次 retile／搬移。較差可行候選仍更新 current；global best 另外保留。沒有顆數 performance blacklist，容量由候選 oracle 判定，5% exploration mixture 保持合法候選非零機率。

DNN 結構、精度、單顆規格、mesh 實體位置規則、路由與排程固定。尚不支援 channel reduction、任意不規則切分、memory sharding、data replication、precision／model architecture search 或實體 floorplan 搜尋。

合法候選生成不用 PPA 預先排名。這個有限空間與有限預算沒有全域最優保證；一次差的 mapping 不能被解讀成整個架構差。完整 RL state/action/reward 契約見 [架構定義](rl_architecture.md)。
