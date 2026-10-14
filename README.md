# RACE-RL

讓廠商先輸入 PPA 限制值與偏好，再由**單一 RL Agent** 聯合搜尋 chiplet 數量、DNN 切分、工作配置及資料配置。正式策略採 REINFORCE，沒有 PPO 或 Critic；PPO 僅列入[比較模型](docs/research/comparison_models.md)。主流程採 float64；實體硬體為 compute-only mesh，外部記憶體以服務介面建模。

![RL 架構](docs/research/figures/rl_architecture.png)

[現行硬體與資料流](docs/research/hardware_model.md)｜[七模型完整訓練基準與研究修正](docs/research/model_suite_20261014.md)

[本機 GUI 使用方法](docs/research/gui.md)｜[評分公平性核對](docs/research/scoring_review.md)｜[廠商 PPA 輸入與評分](docs/research/vendor_requirements.md)｜[RL 架構定義](docs/research/rl_architecture.md)｜[中文研究進度](docs/research/research_progress.md)｜[核心假說驗證](docs/research/hypothesis_results.md)

研究名稱：**Relational Attribution and Coupled-Edit Reinforcement Learning**。一次 action 聯合修改相依範圍，完整後立即交給環境評估一次；動作產生中間零 PPA 評估。正式預設最多 2 個相依 Tile，單項修改作比較設定。

## 本機網頁 GUI

安裝研究依賴後啟動 Python server，瀏覽器即可操作：

```sh
python -m race_gui.server
```

啟動時重用此專案已在執行的 GUI；否則優先使用 `http://127.0.0.1:8765`，占用時自動選空閒連接埠。請以終端顯示的實際網址開啟。介面提供建議搜尋設定、進階選項與實際配置／傳輸設計圖。macOS／Windows 的環境設定與雙擊啟動入口見 [GUI 使用方法](docs/research/gui.md)。

## 廠商起始輸入

三項限制：平均功耗上限（W）、單次推論延遲上限（互動用 ms）、運算晶粒面積上限（mm²）。三項必須完整輸入，再選偏好：

| 選項 | 功耗／效能／面積權重 |
| --- | --- |
| PPA 平均 | 各 1/3 |
| 偏重功耗（第一個 P） | 0.6／0.2／0.2 |
| 偏重效能（第二個 P） | 0.2／0.6／0.2 |
| 偏重面積（A） | 0.2／0.2／0.6 |
| 使用者自訂 | 三個非負權重，自動正規化 |

先判定 PPA 是否全部達標，再以正規化加權成本排序。需求直接輸入共享 encoder，影響 Actor 與環境 reward。搜尋可走過暫時超標的物理合法設計；結束仍無達標方案時，明確輸出 `NO_FEASIBLE_DESIGN_FOUND` 與 `closest_*`。

## 執行

Python 3.12 建議：

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[rl,cnn,reports]'
.venv/bin/python tools/verify_sources.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m race_rl.cli search-mesh --interactive --out experiments/runs/vendor_search
```

自動化可先修改範例需求檔，再使用：

```sh
.venv/bin/python -m race_rl.cli search-mesh --requirements configs/vendor_requirements.json --model resnet18 --episodes 3 --budget 16 --out experiments/runs/vendor_search
.venv/bin/python tools/run_hypothesis_pilot.py --requirements configs/research_requirements.json --out experiments/runs/new_hypotheses
```

JSON 延遲單位為秒。範例限制只是合成數字，需替換成廠商需求。新輸出目錄須為空；`vendor_report.json` 記錄需求、權重、原始 PPA、達標判定與違規原因。

## 七模型終端訓練

在終端機執行一次，依序訓練 ResNet18、ResNet34、ResNet50、VGG16、VGG19、MobileNetV2、AlexNet；每個模型從新策略開始，固定 60 回合、64×64 輸入、每回合最多 64 次完整互動，停用提前停止。模型不並行。

macOS：

```sh
cd /Users/hsuyuchia/Documents/RACE-RL
.venv/bin/python tools/train_model_suite.py
```

Windows（在專案資料夾）：

```powershell
.venv\Scripts\python.exe tools\train_model_suite.py
```

預設使用 `configs/model_suite_tasks.json` 的各模型合成研究限制，具有先導實測的可行設計憑證；數值與出處見[研究修正紀錄](docs/research/model_suite_20261014.md)。不再對全部 CNN 套用已證明不合理的 100 ms。指定 `--requirements <檔案>` 則完全採廠商限制，不自動放寬；GUI 也保留使用者自行輸入。

終端同一行對齊顯示模型、回合、互動、進度、已用與暫估剩餘時間。結果集中在 `experiments/runs/model_suite/training/<時間>/`，完成後開啟 `report.html`。前後各做 10 個配對 seed、各 64 次互動的凍結策略測試，還原所有 RNG、零參數更新；每模型總 5,120 次互動。比較方法及中途測試預設停用；選用 `--comparisons random sa --evaluation-interval 10` 才增加相關預算。時間與查詢均另計，不能把額外實驗成本當成訓練器變慢。

中斷後使用終端印出的目錄接續：`python tools/train_model_suite.py --resume <輸出目錄>`。只接續最後完整提交的回合，來源或條件不一致會拒絕。詳細紀錄與精簡保存規則見 [實驗說明](experiments/README.md)。正式訓練由使用者啟動；`--smoke` 是短流程驗證，不能當成 60 回合成果。

## 搜尋設定與驗證範圍

GUI 固定由 RL 選擇 1–16 任一運算顆數，使用者不指定顆數；研究 pilot 使用 2／4／6／8／10／12／14／16 子集合。16 顆為 4×4 mesh。一個 policy、encoder、Actor 與 optimizer，每步交錯改顆數、整層 retile 或 coupled mapping。外部記憶體不占 mesh 格點，不作路由转送，容量／頻寬／能量仍計入。所有硬體共用回合 oracle budget，起始不評估，只計 action 後的評估；失敗也計入；一次差的 mapping 不排除整個顆數。

當前驗證紀錄見 [研究進度](docs/research/research_progress.md)。小型證據位於 `docs/research/evidence/`；新搜尋預設保存精簡互動紀錄與最終選中設計完整帳本，大型輸出不納入 Git。舊的大型訓練輸出已依使用者要求清除，歷史摘要與 hashes 保留，見[清理紀錄](docs/research/evidence/training_output_cleanup.json)。歷史 PPO 資料只放在比較模型紀錄，不視為正式版本成果。

物理係數未校準；功耗是平均值，面積只含 compute dies。沒有 silicon PPA、全域最優或跨需求／CNN 泛化保證。授權與第三方依賴說明見 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
