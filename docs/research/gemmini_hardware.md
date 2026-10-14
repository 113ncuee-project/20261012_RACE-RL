# Gemmini 單晶粒硬體模型與來源可信度

2026 年 10 月 14 日建立第一階段模型，以 Berkeley Gemmini 的 `GemminiConfigs.defaultConfig` 作來源參照，新增 DRAM → GB → LB → RF → PE 的記憶體研究配置，另以 LB 連到獨立 accumulator。已實作逐層容量、搬移與存取計帳、權重複用、bank／port 排程、DMA 與 1–16 顆矩形 mesh 的分析式服務模型。[Gemmini 與 RACE-RL 的串接](gemmini_rl_integration.md)已接同一 REINFORCE trainer 的獨立 GEMM 後端；目前 GUI 與七模型主入口已[接完整 CNN](cnn_hierarchy.md)；本頁以下為 GEMM foundation 的規則與歷史證據。

基礎模型輸出流量、存取次數、容量與估計 cycles；其功耗、能量與面積維持 `null`，`require_ppa()` 拒絕直接評分。串接環境另有明列來源的成本介面；未提供完整成本資料時，reward 缺值並禁止策略更新。輸入／權重的 INT8 與累加的 INT32 是這套硬體配置；原 CNN 浮點數值驗證與 RL 的 float64 不因此改變，也沒有隱式量化。

## 固定的公開配置參照（與研究擴充分開）

來源：[ucb-bar/gemmini](https://github.com/ucb-bar/gemmini)，commit `8c3f9923a44a2fe2c7930587be297d6d4f8c09ca`。

| 項目 | 固定值 | 公開原始碼依據 |
| --- | --- | --- |
| PE 陣列 | 16×16，tileRows／tileColumns 都是 1 | `Configs.scala` 第 30–33 行 |
| 輸入／權重 | signed INT8 | 第 21–22 行 |
| 部分和 | signed INT32 | 第 23 行 |
| 陣列內部輸出 | signed 20-bit，與 accumulator 不同 | 第 27 行 |
| Scratchpad | 256 KiB、4 banks、single-port | 第 38、41、44 行 |
| Accumulator | 64 KiB、2 banks、非 single-port | 第 39、42、45 行 |
| DMA | 128-bit bus，單次最大 64 bytes | 第 64–65 行 |
| 最多 in-flight requests | 16 | 第 62 行；第一階段不模擬 queue／TLB 行為 |
| Load／Store／Execute queue | 8／2／8 | 第 57–59 行；第一階段不模擬 queue 深度 |
| 硬體 dataflow | WS 與 OS | 第 35 行；第一階段 executor 只實作 WS |

固定版本連結：[Configs.scala](https://github.com/ucb-bar/gemmini/blob/8c3f9923a44a2fe2c7930587be297d6d4f8c09ca/src/main/scala/gemmini/Configs.scala)。執行用配置位於 `src/race_rl/accelerators/gemmini_default.json`，隨 Python 套件安裝。

由公開配置可推導 scratchpad 每列 16 bytes、每 bank 4096 列；accumulator 每列 64 bytes、每 bank 512 列。推導公式位於固定版本 [GemminiConfigs.scala](https://github.com/ucb-bar/gemmini/blob/8c3f9923a44a2fe2c7930587be297d6d4f8c09ca/src/main/scala/gemmini/GemminiConfigs.scala#L106)。固定配置沒有提供一個可直接當成實作結果的時脈、晶粒面積或存取能量。

`third_party/gemmini-reference/` 保存 7 個核對用途的原始檔，約 161 KiB，包含原始 `LICENSE`。這是只讀來源快照，不是可編譯的完整 Gemmini／Chipyard checkout。`gemmini.lock.json` 保存 commit、每個檔案的下載 URL、大小及 SHA-256。

## 現行研究硬體配置

![記憶體階層與資料路徑](figures/memory_hierarchy.png)

[PNG 圖檔](figures/memory_hierarchy.png)／[可編輯 SVG](figures/memory_hierarchy.svg)。中文使用標楷體、英文使用 Times New Roman。RF 與 accumulator 分開畫；部分和走支線，輸入與權重不經 accumulator。

| 每顆 chiplet 的元件 | 容量／分工 | 模型服務率／來源 |
| --- | --- | --- |
| GB SRAM | 256 KiB、4 × 64 KiB banks；共享輸入、權重與輸出暫存 | 每 bank 16 bytes/cycle、single-port；容量參照 Gemmini，GB 用途與服務率為研究配置 |
| LB A／B SRAM | A、B 各兩個 4 KiB slots，共 16 KiB；每個 slot 放一個完整 panel | 每 bank 16 bytes/cycle、single-port；新增研究配置 |
| LB output SRAM | 4 KiB；串流部分和與完成輸出的 16×16 INT32 microtiles | 16 bytes/cycle、single-port；新增研究配置 |
| RF 操作數 | A 256 bytes、B 256 bytes；各容納 16×16 INT8 microtile | 各自讀／寫 port、256 bytes/cycle；新增研究配置 |
| RF partial sums | 1024 bytes；一個 16×16 INT32 microtile | 讀／寫各 1024 bytes/cycle；新增研究配置 |
| Accumulator SRAM | 64 KiB、2 × 32 KiB banks；保留完整 output block 的部分和 | 每 bank 讀／寫各 64 bytes/cycle；容量參照 Gemmini，port 服務率為研究假設 |
| PE array | 16×16、256 MAC、一個 PE cluster | INT8 乘法／INT32 累加的分析式微程式；尚未重現 Gemmini 20-bit array 中間訊號 |

每顆晶粒固定配置同一套 GB／LB／RF／ACC。LB 實體總容量為 **20 KiB**、RF 為 **1536 bytes**；關閉雙緩衝只停用第二個 slot，不會減少實體容量或晶粒面積。GB 的 global 是「晶粒內共用」，不是整個封裝只有一個 GB；目前只有一個 16×16 cluster，尚未宣稱多 cluster 的仲裁與共享行為。

這組層級符合 CNN 加速器中分層儲存、資料複用的概念，但具體架構並不是所有產品的共同標準。原 Gemmini 使用 banked scratchpad 與 SRAM accumulator，不含這裡新增的獨立 LB 組織；因此模型標為 `race_gemmini_hierarchy_ws_v2`，不能稱為「官方 defaultConfig 的逐週期模型」。官方來源快照與配置 JSON 保留原貌，不改成我們的 GB／LB 參數。[官方架構說明](https://github.com/ucb-bar/gemmini#architecture)支持 scratchpad、DMA、array 與 accumulator 的角色；[Timeloop 階層描述](https://timeloop.csail.mit.edu/v4/input-formats/design/architecture)可參考記憶體層級的建模方式，本專案未執行 Timeloop／Accelergy。

記憶體配置見 `configs/memory_hierarchy.json`，由 `MemoryHierarchy` 嚴格載入所有容量與服務率；缺欄位不自動補齊。GB／ACC 與 PE 的來源參照參數仍放在 `gemmini_default.json`。變更記憶體配置會改變環境、模型與 checkpoint 身份，成本 profile 必須綁定同一 `memory_hierarchy_identity`。

`configs/gemmini_platform.json` 固定 DRAM 1 GiB、16 bytes/cycle、每 transaction 20-cycle response，以及 mesh 16 bytes/cycle、2-cycle hop。這些全部是研究情境，沒有官方時脈、DRAM／PHY 量測依據。DRAM 通道接 c0、讀寫共享；其他晶粒經 mesh 的 XY route 存取。尚未模擬 packet、credit、multicast、queue／TLB 或完整記憶體控制器協定。

## 已實作的資料與執行規則

一個 `GemminiAction` 完整指定 M／N／K blocking、權重駐留、雙緩衝與所有 output blocks 的 chiplet 配置。環境只執行這個 action；沒有候選試算、額外 mapper 或隱藏重新切 tile。blocking 是 16 的倍數，尾端補零；A／B 為 INT8 packed matrices、C 為 INT32。

輸入／權重路徑：**DRAM → GB → LB → RF → PE**。輸出路徑：**ACC → LB → GB → DRAM**。部分和在 **RF ↔ LB ↔ ACC** 之間保存與讀回；它不是需要每次 MAC 都去 DRAM 的資料。每個 copy 在來源端算一次讀、目的端算一次寫；MAC 的算術成本與 RF 存取成本分開。

- GB bank 0／1 放 A ping-pong panels；bank 2／3 放 B。保留 B 時，當前 N-block 的完整 K×N 權重駐留於這兩個 banks，跨同顆晶粒的 M groups 複用；不同晶粒需要各自副本。
- 每個 K panel 放進對應 LB A／B slot；所選 panel 必須完整放得下 4 KiB，不會因容量不足而偷偷替 Agent 換成較小 action。保留權重只表示 GB 駐留，LB 仍需重新填入使用的 panel。
- 固定 WS 微程式採 N16 → K16 → M16：B microtile preload 一次到 RF，跨 M16 使用；A panel 駐留 LB，跨 N16 使用。macro tile 大小因此會影響 DRAM、GB、LB 與 RF 的流量，而非只有 MAC 數。
- 一次 K16 reduction 的部分和保存至 ACC；後續 K16 由 ACC 經 LB 讀回 RF。ACC 必須放得下完整 output block。RF 只需容納一個 microtile，與 ACC 的 SRAM 容量分開限制。
- 完成輸出以 1024-byte microtiles 串流經 output LB，到已釋出的 GB A banks，再 DMA 寫回。output LB 不需要放得下整個 64 KiB output block。
- DMA 直接串流到 GB，沒有額外免費的 fill staging。GB／LB single-port banks 在 copy 與執行期間有明確占用；不同 slots 可重疊預取，未完成的 slot 不能覆寫。
- 每個 DMA row 按 64-byte transaction／16-byte beat 分段。讀寫共享 DRAM channel；同顆晶粒的 output groups 串行。

模型將每個 K panel 的 N16／K16／M16 微程式聚合成一個分析式服務工作，記錄內部各方向 bytes、RF 讀寫及服務 cycles，避免為每個 MAC 建立事件造成搜尋時間與檔案膨脹。聚合服務保守保留相關 ports；它沒有模擬實際 pipeline、broadcast 或逐週期 interleaving。估計時間尚未經 RTL 校準，也不是已證明的延遲上界。

## 卷積支援範圍

第一階段提供 dense NCHW 卷積的 explicit-im2col 數值 lowering，支援 stride 與 padding；與獨立直接卷積計算核對。輸入展開會增加儲存與傳輸，報告明列 `im2col_bytes`。

目前評估的是 **展開完成後的 GEMM 階段**。im2col 建立的延遲與能量沒有計入，因此不能把結果當成完整卷積或 CNN 的 PPA。Grouped／depthwise、dilation、bias、requantization、非卷積算子、跨層 activation 駐留、halo 直接交換及跨晶粒 reduction 尚未接入；不會偷偷使用原 spill-all 模型補算未支援的部分。

## 來源可信度與驗證

可信度依「可以支持哪一項主張」區分，不使用沒有統計依據的百分比。

| 項目 | 狀態 | 能支持的主張 |
| --- | --- | --- |
| 配置與原始碼身份 | 固定官方第一手來源，SHA-256 與欄位核對通過 | 參照容量、位寬與陣列參數對應該 defaultConfig；新組織另列研究配置 |
| 整數運算 | 獨立 NumPy INT64 matmul／直接卷積核對 | 已測案例的 INT8／INT32 算術結果一致 |
| 容量與排程 | 自動測試核對容量拒絕、相依、bank 衝突與 slot 重用 | 本模型內的帳本與資源限制一致 |
| 估計 cycles | 分析式公式，尚未對照 RTL | 只能比較所宣告情境下的模型結果 |
| DRAM 與 chiplet mesh | 專案研究假設 | 只代表 JSON 中固定的服務情境 |
| 能量、功耗與面積 | 無校準資料，維持缺值 | 不能產生正式 PPA reward 或 silicon PPA 結論 |

執行來源核對與小型驗證：

```sh
python tools/verify_sources.py
python tools/validate_gemmini.py --out experiments/runs/memory_hierarchy_validation_new
```

macOS 使用 `.venv/bin/python`；Windows 使用 `.venv\Scripts\python.exe`。腳本只執行三個預先指定的數值案例、九個固定方案，不訓練 RL。每次輸出一份 compact `report.json` 與 `report.html`；需要事件帳本才指定 `--full-traces`。非空輸出目錄會拒絕覆寫。

初版未加入 LB／RF 的歷史證據位於 [v1 驗證資料](evidence/gemmini_foundation/report.json)，不可視為目前架構的結果。新版證據位於 [記憶體階層驗證](evidence/memory_hierarchy/report.json)。64×16×32 GEMM 的 B 外部讀取，在單顆／不駐留時為 2048 bytes，單顆／駐留時為 512 bytes；兩顆各自駐留時為 1024 bytes。三者 logical MAC 數相同。這證明模型已呈現容量允許下的複用與複製成本，沒有證明晶片加速比。

新層級完整回歸 **197 個測試通過**，包含 9 個新增容量、流量、頻寬敏感度與 RL graph 測試。單個 16×16×16 microtile 的 GB／LB／RF／ACC endpoint 存取分別為 3072／5120／41472／2048 bytes，包含端點讀寫與 RF 的 MAC 操作數／部分和存取。這些是模型帳本，不是外部 DRAM 流量，也不是真實晶片量測。

另跑三個固定數值案例、九個硬體執行方案；一個凍結 Actor 動作確認回饋與缺值 gate；2×4 步 TEST_FIXTURE smoke 確認兩次 REINFORCE 更新。執行平台、來源、SHA、時間與證據位置見 [新版測試紀錄](evidence/memory_hierarchy/verification.json)。macOS 上驗證通過並包含換行轉換測試，尚未實際執行新版 Windows 雲端 CI。

## 接入正式 RL 前的工作

先以固定公開配置建立可執行的 Gemmini RTL 對照案例，核對 DMA、bank、preload、尾端利用率及 K reduction 的時序；功能模擬器只能驗證結果，不能校準週期。面積與能量要取得固定製程、SRAM 與算術元件的可追溯成本。再補完整 CNN 的資料生命週期與未支援算子，定義同一 Agent 的動作及 observation，最後接入 PPA scorer。工具只評估 Agent 已選出的方案，不使用另一個 mapper 代替 RL 作選擇。

舊合成環境的 PPA 限制、可行性憑證與訓練結果不能直接移植成新硬體的成效；新環境必須建立獨立版本與實驗身份。
