# CNN 多種子比較：終端機執行

這個入口目前是**探索性比較**，不是論文正式結果：`gemini_mapping_adapted` 與 `chiplet_gym_common_adapted` 把論文方法轉到共同 CNN 動作空間，均不是原始完整系統；SA 是通用方法。現有限制是少量探索得到的候選值，尚未完成難度驗證。PPA 與 timing 沒有實體校準；CNN 是隨機權重和單一校準輸入。

Gemini 改編採用局部修改及退火接受機制，共用 RACE 的完整設計提案與 PPA。五類操作分別對應四種固定分割 profile、單層位置起點、相依兩層的位置交換、單晶粒／round-robin 分散、activation／weight 駐留，另加全域 chiplet 數量修改。與 [Gemini 原文](https://arxiv.org/pdf/2312.16436) 的任意 core group／核心交換／跨層資源分配／多 DRAM 選擇並不等價；此改編沒有 DP 圖分組、任意 core permutation、內層 tiling 與 loop-order 窮舉、Gemini 架構引擎或其原始成本函數。每個提案包括無效及快取命中均計入預算；本基準無訓練成本，也未完成超參數調整。

Chiplet-Gym 比較使用作者採用的 Stable-Baselines3 2.1 PPO 實作，以及作者 SA 程式的鄰域步長與接受規則。共同預算的一半分配給重新訓練的一個 PPO、另一半給一個 SA；原論文的多個預訓練代理／多次 SA、原本的封裝動作與成本函數並未保留。動作轉換、超參數與限制見 [Chiplet-Gym 共同環境](chiplet_gym_common.md)。

## 執行

以下命令在專案根目錄執行。先確認 `experiments/runs/research_exploration_20261030/` 中三個模型的 `requirements.json` 存在，且 [Chiplet-Gym 依賴](chiplet_gym_common.md)已安裝；當前限制來自 [首次探索](paper_comparison_pilot_20261030.md)。四種偏好、三模型、五個搜尋 seed，共 60 次試驗；每次五個方法各自最多 64 次完整提案，REINFORCE 每 16 次提案更新一次。CNN 權重／校準 seed 固定為 7；訓練提案均計入 64 次預算。

```sh
cd /Users/hsuyuchia/Documents/RACE-RL
.venv/bin/python tools/run_cnn_comparison_suite.py \
  --limits-dir experiments/runs/research_exploration_20261030 \
  --out experiments/runs/cnn_comparison_common_v1 \
  --seeds 301 302 303 304 305 --budget 64 --rl-episode-size 16
```

執行時間依硬體與模型而異；尚無受控全批次時間量測，不能保證 ETA。執行中可用 Ctrl-C 中斷；保留原命令，附加 `--resume` 續跑。若一個試次中斷，其局部資料會更名保存，該試次從頭執行；已完成試次不重跑。程式、成本檔、限制、seed 或其他設定改變時拒絕續跑，須使用新的輸出資料夾。

報告在 `experiments/runs/cnn_comparison_common_v1/report.md`；`results.json` 保存所有完成試次的摘要，`runs/` 保存每次完整結果及最佳設計／成本回饋。沒有找到達標解的試次會保留。輸出目錄依專案規則不納入 Git，應另外保存。舊 `cnn_comparison_batch` 在執行期間程式版本發生變更，不能恢復或與本批次合併；保留原始資料作探索紀錄。

## 限制

這批程式沒有自動公平調參、正式統計比較、論文原版實作或跨成本情境報告。不同種子可用來探索有利案例，但只呈現有利 seed 不能代表整體方法的表現；完整報告保留所有結果。比較器會重置每個 RL episode 的環境與快取，而通用搜尋方法使用一個連續 episode，這是需要研究與揭露的差異。
