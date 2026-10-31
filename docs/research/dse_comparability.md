# 共同條件下的各方法 DSE：目前差距

研究目標是固定 CNN 工作負載、輸入、硬體候選、PPA 限制／成本及計費方式，由各方法按自己的設計空間探索（DSE）方式產生候選設計。方法本身可以使用不同的狀態、候選生成、內部排程和策略更新；應記錄全部成本，不把它們硬改成同一種策略。

共同的硬體候選和設計表達必須先明確。若某方法原生可搜尋的配置超出共同表達，應記錄能力限制；不能把近似操作視為原版操作，也不能只比較各論文用不同成本模型產生的數字。

| 方法 | 當前實作 | 與目標的差距 |
| --- | --- | --- |
| RACE-RL | 完整 CNN 聯合 action、REINFORCE、共同分析式 PPA | 作為本專案方法；需凍結實驗與硬體限制 |
| Gemini 原版 | 論文 artifact 已能原生執行並匯出 mapping | 原版圖、核心、DRAM、分組、排程與成本不同；沒有同工作負載的完整雙向轉換及共同評估器 callback，內層評估也尚未共同計費 |
| Gemini 改編 | 固定 profile、最多兩層 coupled action、六類局部操作與退火 | 僅是受限近似，不保留原版 DP 分組、core permutation、DRAM 選擇或 intra-core 窮舉；不能稱為原版 Gemini DSE |
| Chiplet-Gym 原版 | 作者 PPO、SA、硬體封裝成本模型的獨立啟動入口 | 原 action 是 64／128 晶粒及封裝參數，不能直接套用 CNN 切分；原預訓練模型的輸入／輸出不匹配 |
| Chiplet-Gym 改編 | SB3 PPO 與 SA 規則，共同 CNN evaluator，64 次預算的一半分配各方法 | 單 PPO＋單 SA、動作轉換、狀態、reward、rollout 和預算分配均與原版不同；不能稱為原版完整 DSE |

`run_cnn_comparison_suite.py` 目前是這些受限改編的探索性比較，不能支持「原版 Gemini／Chiplet-Gym 在相同情況下比較」的結論。要完成目標，需要實際建立可轉換的共同設計表示及 workload，對每個方法核對搜尋空間、候選生成和 evaluator 邊界，並驗證原版方法在該表示下可執行。若找不到語意等價的轉換，應擴充共同表示或將不相容的配置明列為不可比較，不應靜默取模、替換操作或填入未驗證假設。

已新增 `tools/common_dse_evaluator.py`：它允許搜尋方法直接提出完整 `CNNDesign`，不必使用 RACE 的兩層 action；同一設計使用相同 PPA 評估器，無效與快取請求均計入預算。小型測試核對了此介面與 RACE 環境對同一設計的回饋相同。`tools/common_dse_trial.py` 已串接 RACE、Gemini 受限空間改編與 Chiplet-Gym 全層 action 改編，檢查各方法 workload、硬體、限制與成本檔的身分一致；三方法各自消耗相同提案預算。短流程只測試串接，並非成效實驗。

**仍不能當作原版 DSE 比較**：`CNNDesign` 限制 profile、核心配置和記憶體語意；Gemini 受限版沒有原版 DP 圖分組、任意核心群組、DRAM 指派及內層窮舉，Chiplet-Gym 使用重新訓練的單 PPO＋單 SA 及 CNN action，而非原版預訓練多代理／多次 SA。個別方法的搜尋空間、reward、隨機取樣和訓練成本也需按論文方法再核對。`common_dse_trial.py` 目前僅供短測，尚不是多模型多 seed 的正式批次入口。

## 共同條件批次入口

`tools/run_common_dse_suite.py` 在共同 CNN、硬體、PPA 成本、限制與提案次數下，分別執行全域 Random、局部 Random、局部 SA、Gemini 受限改編、Chiplet-Gym PPO＋SA 的全層與局部兩種改編，以及 RACE-RL。局部方法從上一次接受的設計產生候選。各試次檢查同一硬體／workload／成本／需求身分；無效及快取提案都計入預算。PPO 的訓練提案也計入。RACE 每回合重置 cache，其他方法的共同評估器在整次試驗共用 cache；局部 Chiplet-Gym 也按 episode 重置 cache。實際評估次數及時間因此也須公開。

Gemini 改編以可表示的 stripe 初始設計、局部修改及依當前成本調整的退火溫度搜尋；受限於共同表示，六種操作只能近似原版操作，沒有原始分組及窮舉。Chiplet-Gym 改編使用 SB3 PPO 與作者 SA 的接受方式，預算分給各一個 PPO 和 SA；PPO 對每個實作算子提出完整設計。共同 action、單代理／單 SA、訓練資料與原版不同，不使用不相容的預訓練權重。保留的部分和差異都記在每次結果及程式碼；不可宣稱執行了原版完整系統。

短流程與續跑測試通過；長時間批次尚未執行。需先有 `research_exploration_20261030` 的模型限制檔以及 [Chiplet-Gym 依賴](chiplet_gym_common.md)。在專案根目錄執行：

```sh
.venv/bin/python tools/run_common_dse_suite.py \
  --limits-dir experiments/runs/research_exploration_20261030 \
  --out experiments/runs/common_dse_study_v2 \
  --seeds 311 312 313 314 315 --budget 64 --rl-episode-size 16
```

四偏好 × 三 CNN × 五新 seed 共 60 組，每個方法每組 64 次提案。Ctrl-C 停止後，使用相同命令加 `--resume`；未完成試次保留並從頭重跑。程式、依賴 metadata、成本與條件變更會拒絕續跑。結果在 `experiments/runs/common_dse_study_v2/report.md` 和各試次 JSON。成本仍是未校準的分析式設定；模型限制也只由少量探索得到，尚未完成難度驗證。v1 的舊 SA 實作有錯且方法集合不同，僅保留為探索資料，不與 v2 合併。完整首輪分析見 [共同 DSE 首輪結果與研究修正](common_dse_analysis_20261030.md)。
