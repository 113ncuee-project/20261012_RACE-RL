# 當前實驗

新輸出集中在忽略的 `runs/model_suite/`。現行 compute-only mesh／外部記憶體版本的小型可核對證據位於 `docs/research/evidence/`。每次實驗需使用全新輸出目錄，並在開始前凍結 protocol／來源身份；不得把開發用 workload 改稱 held-out test。

## 正式終端訓練

`python tools/train_model_suite.py` 一次啟動七模型，順序為 resnet18、resnet34、resnet50、vgg16、vgg19、mobilenet_v2、alexnet。前一個模型的子程序完全結束後，才啟動下一個，釋放該模型的 CNN／環境記憶體。每模型使用全新 REINFORCE 策略，固定 60 回合、64×64、float64、scope=2、起始 2 顆、可選 1–16 顆、每回合 64 次互動；不提前停止，不使用 PPO。

PPA 原樣讀取 `configs/vendor_requirements.json`，可在開始前修改或以 `--requirements <檔案>` 指定。訓練 seed 預設 7。前後測預設各一個 seed=1007 的凍結策略 rollout，互動預算相同；`--evaluation-seeds 1007 1008 1009` 可增加配對測試。這些測試會增加執行時間，不計入 60 次策略更新，也不改變訓練 RNG。

互動式終端以同一行刷新模型／階段、目前回合、當回合完整互動數、總進度、已用時間與 ETA，中文欄位依顯示寬度對齊。視窗夠寬時同列顯示目前 PPA；較窄時縮短欄位避免折行。每模型完成／失敗才新增摘要行，重新導向檔案的標準輸出也只寫摘要，不刷進度控制碼。完整逐步紀錄與時間日誌仍照常保存。ETA 依已完成互動速度更新，包含前後測的互動進度；模型匯入、策略更新、報告寫出與跨模型差異使它不是完成時間保證。進度達 100% 後可能仍在寫出最終報告。

預設輸出為 `runs/model_suite/training/<UTC時間>/`，終端印出絕對位置。主要檔案：

| 位置 | 內容 |
| --- | --- |
| `protocol.json` | 全模型順序、需求／偏好、seed、來源身份、固定訓練條件 |
| `terminal.log`、`status.json`、`model_times.json` | 整次執行的進度、累計時間、完成／中斷／失敗狀態 |
| `report.html`、`results.json` | 全模型報告、原始 PPA、達標判定與前後測對照 |
| `<模型>/execution.log` | 模型驗證、各回合時間／成本與失敗原因 |
| `<模型>/history.sqlite` | 壓縮的逐 action 回報／設計狀態、回合統計、交易式最新 checkpoint |
| `<模型>/training_curve.json`、`.svg`、`.png` | 逐回合數值與可用於報告的實測曲線 |
| `<模型>/policy_comparison.json` | 前後凍結策略相同預算／抽樣 seed 的結果 |
| `<模型>/selected_design.json.gz` | 訓練中選出方案的完整硬體、mapping、PPA、排程與流量帳本 |
| `<模型>/checkpoint.pt` | 訓練結束時的策略／optimizer／RNG；只使用可信本機檔案 |

若未安裝 matplotlib，數值與 HTML 報告仍生成；安裝 `.[reports]` 才產生 PNG／SVG。圖不把「搜尋最佳值改善」標成「策略已收斂」；需看配對的凍結策略結果。單一訓練 seed 只能作學習診斷，不足以證明核心假說或泛化能力。

Ctrl+C 停止後執行 `python tools/train_model_suite.py --resume <原輸出目錄>`；已完成模型跳過，未完成模型從最後原子提交的回合 checkpoint／RNG 繼續，未完成回合重跑。來源身份不同時拒絕接續。`--model resnet50` 僅供需要單獨重跑某模型的進階用途，預設仍是全部七模型。`--smoke` 為七模型各 1 回合、2 次互動，只有流程驗證用途。

## 紀錄精簡，不改訓練

不降低回合數、不跳過環境、不改動 action supports、PPA、reward、loss、dtype 或更新頻率。回合內 cache 仍保留，避免改變重複設計的計費與狀態。記錄器在互動／更新完成後保存資料，不參與選 action 或評分。

逐步保存 action 參數、物理成功／失敗、PPA／超標原因、reward、current／best objective、前後硬體／mapping hash、顆數、剩餘預算。結合 workload、初始規則、protocol 與順序 action 可重播設計；不重複寫出每個候選的完整排程、能量與流量帳本。SQLite 的 `episodes` 表可直接查詢 stats JSON；`interactions` BLOB 為 zlib 壓縮的 UTF-8 JSON 陣列。

GUI／一般 `search-mesh` 預設也使用精簡 `history.sqlite`，最終 `best_*`／`closest_*` 完整資料與設計圖入口保留。需要每候選原始帳本的逐項研究 audit，可明確使用 `--recording full`；此選項只改輸出量。

2026-10-14 在紀錄一致性與全部模型流程驗證後，依使用者要求刪除舊 GUI／正式 pilot／PPO 大型訓練 raw 輸出，清除約 4.55 GiB。保留來源、設定、小型歷史證據、CNN gate 與最新 smoke。舊 raw manifest 只保留歷史 hashes，已刪除資料不能宣稱仍可直接在本機開啟；詳見 `docs/research/evidence/training_output_cleanup.json`。
