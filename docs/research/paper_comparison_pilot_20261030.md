# CNN 論文比較：探索紀錄（2026-10-30）

本紀錄是流程與耗時探索，不是預先註冊的正式比較，也不能證明方法優劣。硬體時序、成本與 PPA 均為未經實體校準的分析式情境。CNN 使用隨機權重及單次量化校準，未驗證模型準確率。

## 論文基準查核

* [Gemini, HPCA 2024](https://people.iiis.tsinghua.edu.cn/~gaomy/pubs/gemini.hpca24.pdf) 有[公開程式](https://github.com/SET-Scheduling-Project/GEMINI-HPCA2024)，提供不同硬體與跨層 mapping 搜尋；其可行設計、時序與成本模型並不等同本專案。尚未完成原版執行或與本環境的可信轉換，不能把 generic SA 標作 Gemini。
* [Chiplet-Gym](https://arxiv.org/html/2406.00858) 的 [PPO 程式](https://github.com/KFM135/chiplet-optimizer/blob/main/RL_64_chiplet.py) 使用不同的 MultiDiscrete 硬體／封裝配置，與本專案的 CNN 算子切分動作不同。論文另結合多個 PPO agent 和 SA；單獨比較 PPO 或 SA 都不等於重現其完整系統。後續已加入 `chiplet_gym_ppo_adapted`，使用相近 PPO 超參數與平坦特徵，但動作、reward 與訓練批次不同；細節見 [執行說明](cnn_comparison_execution.md)。原始完整系統尚未接入或驗證。

兩者都不能直接拿原論文不同硬體下的 PPA 數字與本模型比較。正式比較需固定原版來源、確認授權和運行，再記錄每項適配與獨立驗證；若無法完成，不得冒稱已比較 paper baseline。

## 限制探索

三模型均為 64×64，固定模型及校準 seed=7，限制探索 seed=101，每個模型隨機提出 16 個完整設計。物理有效數：ResNet18 16、MobileNetV2 16、VGG16 12。以有效樣本各指標的中位數正規化後排序，取排序中位設計，功耗／延遲／面積各乘 1.1／1.2／1.1，形成候選限制；同模型四種偏好使用相同限制。

| 模型 | 功耗上限 W | 延遲上限 s | 面積上限 mm² |
| --- | ---: | ---: | ---: |
| ResNet18 | 0.717271 | 0.109905 | 49.5 |
| MobileNetV2 | 0.708292 | 0.068493 | 49.5 |
| VGG16 | 0.793384 | 0.671712 | 49.5 |

它們只是在現行基準成本下有探索樣本作可行見證；16 次抽樣不足以證明限制為中等難度或代表性，尚未凍結為正式任務。探索與比較成本要記錄；選擇方法不能根據 RACE-RL 是否勝出。

## 首次串接測試

基準成本、平衡偏好、搜尋 seed=301；random、generic SA、REINFORCE 各 16 次完整提案（包括快取和失敗）。REINFORCE 每 4 次提案更新一次，包含全部訓練成本；各方法均以當次找到的最佳完整設計為結果。SA 使用相同隨機交易提案及 Metropolis 接受規則，尚未調參，**不是論文版 SA**。

| 模型 | Random | Generic SA | REINFORCE |
| --- | ---: | ---: | ---: |
| ResNet18 | 0.3924 | 0.4058 | 0.6862 |
| MobileNetV2 | 0.4318 | 0.4888 | 0.6066 |
| VGG16 | 0.6742 | 0.6503 | 0.5987 |

表中為符合限制設計的加權成本，越低越好。此處每格只有一個 seed，三個方法均找到達標設計；尚無四種偏好、論文基準、公平調參、多 seed、固定時間比較、信賴區間或正式統計檢驗。前兩個模型 RL 較差、第三個較好；不能據此宣稱一般優勢。測試中另有其他本機工作，耗時不作受控 benchmark。

完整工作紀錄位於 `experiments/runs/research_exploration_20261030/` 和 `experiments/runs/research_pilot_20261030/`（執行輸出依專案規則不納入 Git）。這一批探索入口 SHA-256 為 `1a820b905b02edecf4b8c5b54b3753278b3966e0f78549beadb677ae010f1d3f`，比較入口 SHA-256 為 `a465150411887f3c6116d3a07ed6bd888a507b5e54462cba4b352ea1fa184437`；後續入口新增自動記錄自身 hash。歷史紀錄不得重標為新版本結果。

## 尚需完成

1. 檢查論文 baseline 原版與許可，建立可審核的改編及其驗證。
2. 設計並執行獨立預算的限制難度檢查，凍結限制與正式 seeds。
3. 建立等調參預算及全部訓練與搜尋成本的完整帳本，檢查 episode 重置、快取、PPA 計價一致性。
4. 跑四種偏好、多 seed 以及選定成本敏感度，報告全部失敗和不利結果，並分析代表性設計。
