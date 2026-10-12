# 研究進度 — 2026 年 10 月 12 日

目前版本：`0.3.1`。目前進度：**已從 Toy 原型推進到完整 CNN 語意驗證、保守容量排程，以及完整 ResNet-18 的實際 PPO pilot。完整學術研究驗收仍未完成。**

本文件與後續研究進度使用繁體中文。Legacy 原始碼保持不變；本機分支為 `research/race-rl`，尚未設定遠端、提交或上傳 GitHub。

## 已完成工作

### 來源與既有測試

已審核兩份來源 ZIP，核對 RapidChiplet 四個核心檔案與 313 個原始檔案，沒有雜湊差異。Legacy 的 166 項測試已重現通過；本次沒有修改 Legacy。最後從 Git 暫存內容重建獨立副本，313 個來源檔案核對與新版 74 項測試再次通過；紀錄為 `evidence/git_delivery_v0.3.1_verification.log`。第一版 42-test 原型、CNN 擴充 66-test 版本、保守容量 72-test 版本，以及容量修正前的完整來源與結果均有封存。

### CNN 算子與獨立解析

已實作 groups／Depthwise／dilation／bias 卷積、Linear bias、ReLU／ReLU6、推論 BatchNorm、Flatten、Adaptive Average Pooling、pooling 邊界條件與 eval Dropout。bias、BN mean／variance／scale／shift 都納入儲存、流量、能量與 buffer 生命週期。

解析器直接讀取 PyTorch FX，不依賴 Legacy parser，也不默默忽略算子。NumPy reference 獨立計算每個 Tile／Operator，逐層與 PyTorch 比對。新觀察為 schema 2，Tile features 為 17 維；舊 checkpoint 必須使用對應封存版本。

七種完整模型在 batch=1、224×224、相同輸入與初始化參數的 float64 驗證下全部通過：ResNet-18／34／50、VGG-16／19、MobileNetV2、AlexNet。最大逐層絕對誤差約在 4.4e-16～3.2e-12。float32 的 ResNet-34／50 仍未達原定逐元素門檻，失敗資料完整保留。

### 保守記憶體排程與初始 Mapping

先前 VGG／AlexNet 的容量失敗已定位為工作集、prefetch／共用 buffer 生命週期與儲存分配問題。新增獨立 `serial_node_tiles` 排程版本：同一 compute node 的下一個 Tile 等上一個 Tile 完成寫回後才載入，並明示關閉跨 Tile 的 shared reads。

新初始 compiler 依結構工作集選 output partition，按完整 tensor／constant 的保守容量上界分配外部儲存。compiler 不查詢或排名 PPA；評估器仍計入初始與失敗候選。以原有 32 MiB 本地、每個 memory 1 GiB 的合成硬體，七個 float64 完整模型皆通過排程與容量檢查。

這是新的保守模型版本，會改變流量、延遲與能量，不能與原先 shared-read 版本混用比較。data dependency 與排程控制 dependency 已分開，避免把記憶體 admission 邊誤當成 tensor 的物理鏈路來源。

### 真正執行的學習與驗證

- 獨立測試：**74 項全數通過**；不是只檢查形狀或 MAC。涵蓋數值語意、容量、排程、能量帳、動作原子性、Reward、Mask、神經參數更新與 checkpoint。
- 新版完整 ResNet-18 coupled PPO pilot：**3 回合、每回合 4 次評估、合計 12 次訓練評估**，最大 scope=2，policy 確實更新。
- pilot 三回合 best normalized EDP：1.000000、0.999406、1.000000。這是一個 policy seed 的開發驗證，不是三個獨立 seed，更不是優於 Baselines 的統計證據。
- 原有 Toy 實驗仍保留：3 回合／24 次評估、8 種固定 placement 枚舉、19 組參數敏感度檢查。

## 已完成的等價效能修正

完整模型 pilot 的重複查表與 scope 列舉已修正；cached 查表設為唯讀，並凍結外部傳入的容器，避免輸入或雜湊身份被意外改變。沒有刪減動作或引入 PPA 篩選。

以同一個 ResNet-18／硬體／初始 Mapping／scope=2 測量，候選仍是完全相同順序的 2,200 個動作；單次 cold generation 由約 5.99 秒降至 1.97 秒。這是一組開發計時，不是正式多次效能統計。

同一 policy seed 的 3 回合 pilot 在修正前後，完整 search trace、動作、評估次數、Reward 與 best J 完全相同。每回合 structural generation 合計由約 24～26 秒降至約 7.8～7.9 秒；這確認加速沒有改變此次研究搜尋軌跡。證據見 `evidence/pilot_optimization_equivalence.json`。

## 階段與驗收狀態

| 階段 | 已完成的證據 | 尚待完成的 gate |
| --- | --- | --- |
| P0 來源審核 | 現有來源與 166 項 Legacy 測試已驗證 | 三研究提案原檔缺失；上游 Git 物件未獨立核實 |
| P1 IR、原生介面、CNN | 七個完整 float64 模型的解析／數值語意通過 | float32 深層模型誤差、更多算子／精度與拓樸 |
| P2 E2E 評估 | 新算子 ledger 與保守排程／容量通過 | multicast／residency realism、校準或文獻 profile、製程面積耦合 |
| P3 原子動作 | 完整 CNN pilot 使用真正 coupled actions | 開銷已降低；可擴展的逐步學習式子圖選擇仍待完成 |
| P4 關聯狀態、Criticality | trace 歸因、data／control 分離、schema 2 已測試 | 因果 counterfactual 與狀態消融 |
| P5 Actor／Critic | Toy 與完整 ResNet-18 都有實際 PPO 更新 | 足夠訓練預算、跨模型效益與裝置驗證 |
| P6 Reward | 數學邊界與新排程回歸通過 | Reward 消融與 validation 係數選擇 |
| P7 Baselines | 共用框架與原型比較已有 | 六種方法的正式配對公平比較尚未執行 |
| P8 完整 CNN | 語意、容量與單模型 coupled pilot 已執行 | 固定訓練／驗證／測試切分、profile 與正式多種子搜尋 |
| P9 消融、統計 | 分析工具已有數學測試 | 正式狀態／動作／Reward／交互作用試驗尚未執行 |
| P10 論文、重現 | 方法、來源、版本 snapshot 與重現流程已有 | 完整文獻差異與可支持的正式結果章 |

## 修改與新增檔案

- `src/race_rl/workload.py`、`tiles.py`、`reference.py`：新算子、參數、Flatten 座標與 NumPy 語意。
- `frontend.py`、`model_validation.py`：獨立 FX 解析、內容雜湊與逐層數值比對。
- `oracle.py`、`initialization.py`、`cnn_fixture.py`：常數流量、保守排程、初始工作集／外部儲存與合成硬體。
- `observation.py`、`policy.py`、`environment.py`：新算子特徵、data／control 歸因與排程版本設定。
- `tests/test_operator_extensions.py`、`test_frontend.py`、`test_guarded_memory.py`、`test_identity_invariants.py`：新語意、拒絕案例、容量與整合測試。
- `tools/validate_cnn_suite.py`、`run_cnn_pilot.py`：真正執行完整模型驗證與 PPO；新輸出資料夾不可覆寫舊證據。
- `docs/research/cnn_operator_validation.md`、相關 evidence／風險／重現文件：保留成功、失敗、來源與版本邊界。
- Legacy 與第三方原始檔案沒有修改。

## 科學假設與剩餘風險

所有新物理係數仍是未校準的探索假設。合成幾何、SRAM 容量與製程密度尚未耦合；通過 ledger／容量不是可製造 ASIC 的證明。保守版本限制 prefetch、關閉跨 Tile shared reads，會改變 timing／energy，必須與其他模型版本分開比較。

float32 ResNet-34／50 未達門檻；float64 通過支持捨入／累積誤差解釋，但不能把 float32 失敗改成通過。七個模型皆屬開發語意驗證資料，不能改稱未接觸的正式測試集。

H1／H2／H3 效益、瓶頸因果與泛化仍未證實。尚無正式獨立多種子 Baseline 比較、信賴區間或統計顯著性。GitHub CI 只有設定，尚未執行；跨裝置決定性與回合中途 resume 尚未驗證／實作。

## 下一個驗收點

下一步建立可擴展的逐步 scope 選擇，以及同一排程版本下 Random／Greedy／SA／RL 的 Baseline pilot；持續分開量測 action／policy／oracle 開銷。固定資料切分、真實參數來源與訓練／測試預算後，再執行正式配對多種子與消融。
