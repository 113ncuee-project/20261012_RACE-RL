# 正式版本的詳細訓練流程

目前正式版本為 **REINFORCE policy gradient，沒有 PPO，也沒有 Critic**。PPO 只保留在[比較模型](comparison_models.md)，不能宣稱它比正式版本好。正式 GUI 與 CLI 預設只使用 `RACETrainer`。

## 1. 任務與起始狀態

使用者輸入 DNN、三項 PPA 限制與五種偏好之一。DNN 做結構與獨立數值驗證；這是計算語意檢查，並不會預先測試搜尋候選的 PPA，也不訓練 DNN 分類權重。

GUI 固定 mesh，Agent 可選 1–16 任一運算 chiplet 數量，起始結構為 2 顆。環境建立合法起始 mapping，但 **reset 不呼叫 PPA Oracle，也不建立原生網路評估後端**。初始 PPA／排程／busy／criticality／實測流量未知，以零佔位與有效性 mask 明確標示；零不是測得的零功耗。最佳設計為空，不能把未測試起始設計當成結果。

Observation schema 5 提供 DNN／Tile、硬體、mesh、placement、外部記憶體介面、PPA 條件與權重。Global features 共 10 維，新增「目前是否已有有效 PPA」與「上一動作的實際 reward」。起始 reward 以 0 佔位。每回合重設設計與環境快取；Agent 的策略參數沿用上一回合訓練結果。

## 2. 同一個 Agent 提出一個動作

共享圖 encoder 加條件式 Actor 產生行動機率，一次抽樣一個完整 action。可改顆數、整層切分與配置，或進行 root／connected scope／type／target 的聯合 mapping。合法候選只依結構規則生成，不呼叫 PPA，不提前測試、不排序候選 PPA。顆數不因 PPA 或一次差 mapping 被預先排除。

所有輸出 head 都屬於同一個 Agent。神經 logits 只表示動作機率，不是預估功耗、延遲、面積或回報。正式 policy 沒有 value predictor。Design family 與 conditional 分布混入 5% uniform exploration；masked actions 機率為零。

## 3. 動作交給完整設計環境

環境接收所選 action，套用顆數／切分／工作與資料配置，重建依賴、路由、排程與 buffer lifetime，再查唯一的 Oracle 入口。第一次 action 才按需建立原生 RapidChiplet 後端。

環境計帳完整推論的運算、local／external memory、mesh 流量與能量、容量和排程，回傳 PPA、reward 與 next observation。PPA 不是只評一個切分或單一層的局部分數。物理有效設計可成為 current，即使 PPA 暫時超標或分數較差，讓後續動作有機會修正；物理失敗保留原有效 current，回傳失敗紀錄與剩餘步數。首次失敗時仍沒有有效 PPA，不填入假數值。

同一回合重複的 H+M 使用環境已測得的快取資料；仍是一個 action → 一次環境回饋，並計入動作步數。沒有 Agent 自己的 PPA 模型。

## 4. PPA 分數與回報

尺度固定為使用者的三項限制。研究 API 若省略某项，使用事先宣告的 SI 尺度 1 W／1 s／1 mm²，不能依未選的起始設計擬合尺度。

`C = Σ w_i × metric_i / scale_i`。達標時 `M=C/(1+C)`；超標時 `M=2+Σexcess+C/(1+C)`。這保留「任何達標方案優先於未達標方案」的排序。

已測得設計的訓練目標為 `J=M/(1+M)`，範圍 (0,1)。未知最佳檔案使用 sentinel J=1；它是狀態旗標與數學邊界，不是預測起始 PPA。第一個物理有效 action 才建立最佳檔案。

預設 dense reward 為 `r_t=log(J_best_before/J_best_after)`。第一次有效結果及最佳進度改善產生正 reward；未改善或物理失敗通常為零。總回報 telescopes 為 `log(1/J_best_final)`。這是已測得最佳進度的獎勵，不是每一個較差動作都給負分。

## 5. 回饋後才提出下一個動作

下一次 Actor 使用環境回傳的新 observation，包含已測 PPA、實際上一個 reward 與更新後的結構。互動严格為：

`O_t → Agent 抽樣 a_t → environment.step(a_t) → r_t + O_(t+1) → 同一 Agent`

GUI 已移除每回合評估／互動次數的輸入欄位。完整聯合 action 產生後立即送進環境，不等待其他 action。內部有固定的有限回合互動額度以結束 trajectory；這不是一個 action 的生成步數。研究 CLI 可用 budget 設定公平比較成本，重複設計也算 action，起始不花一次。

## 6. 用實際 trajectory 更新同一策略

回合結束後，對每一步計算實際 reward-to-go：`G_t=Σ_(k=t)^(T-1) r_k`，gamma=1。這些回報全來自已完成的環境互動，不是採取動作前預測。

REINFORCE loss：`L=-Σ_t G_t log πθ(a_t|O_t)`。每回合對 on-policy trajectory **更新一次**。沒有 Critic、value loss、advantage estimator、PPO probability ratio、PPO clip 或重複 PPO epochs。預設 entropy coefficient=0，Adam lr=3e-4，gradient norm clip=0.5；梯度裁切只處理數值幅度，與 PPO 的策略裁切不同。正式搜尋使用 float64。

更新 encoder 與 Actor 後開始下一回合；回合中每一步更新的是環境 state，策略參數在回合結束更新。凍結評估保留同一 Agent，但不更新參數。

算法出處：[Williams, 1992](https://doi.org/10.1007/BF00992696)。採用它是依使用者指定的無 PPO、無 Critic 正式流程，不表示本專案已證明其效能較好。

## 7. 輸出與目前證據

跨回合保留實際測得的最佳設計。達標才輸出 best；未達標輸出 closest 與違規原因。若所有 action 都物理失敗，不會輸出虛構 mapping／PPA／設計圖。GUI 直接畫出實測選中設計的 chiplet、Tile 配置與傳輸路徑。

最多 60 回合、每回合 64 步，以及 20 回合 warmup、15 回合無 0.1% 明顯改善提前停止，是起始預設，尚未驗證跨模型收斂；沒有時間上限保證。正式配對研究關閉提前停止。

流程測試驗證 reset／Actor／候選生成均無 Oracle 呼叫、一 action 一環境評估入口、last reward 與新 PPA 回到 observation、REINFORCE 梯度、參數更新、凍結不更新、checkpoint 算法隔離及全部失敗的輸出。這些是機制正確性，不能代替算法優越性、核心假說或全域最優的實驗。

## 動作產生效率

合法 mask 直接由相依範圍、完整 operator ownership、placement／storage 與 output 維度計算。逐候選建立完整 mapping／全域 validate 已移除；只有真正抽樣的 action 透過 apply 做完整結構驗證，再由環境算 PPA。獨立逐候選編譯 reference 核對 action 集合及順序相同，包含多顆數、scope、canonical／非 canonical 與重新切分狀態。這是加速合法性檢查，並沒有使用預測器或刪減探索空間。
