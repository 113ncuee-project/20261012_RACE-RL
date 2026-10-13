# 比較模型紀錄

正式模型為 **REINFORCE**。PPO 已退出正式 GUI／CLI 搜尋路徑，保留為必須明確指定的研究比較模型；目前沒有實驗證明 PPO 在本專案優於 REINFORCE。

| 模型 | 身分 | 更新方式 | Critic | 入口 |
| --- | --- | --- | --- | --- |
| REINFORCE | 正式版本 | 實際 reward-to-go policy gradient，一回合一次 | 無 | GUI、`race_rl.cli search-mesh`、`race_rl.training.RACETrainer` |
| PPO Actor–Critic | 比較模型 | clipped probability ratio、value loss，一回合四 epochs | 有 | `race_rl.comparisons.ppo.PPOTrainer`，工具必須 `--comparison-model ppo` |

PPO 是近端策略優化算法，參考[原始論文](https://arxiv.org/abs/1707.06347)。其 Critic 預測累積 reward，不產生候選 PPA。這仍不能作為本題比較優勢的證據。

## 現行公平比較契約

兩模型使用相同 workload、PPA 條件、顆數集合、action supports、encoder／Actor 結構、起始 Actor 參數、環境、種子、float64 與 action／PPA 預算。兩者都在 reset 零 PPA 評估，採取 action 後才調用同一完整環境。PPO 的 Critic 與額外更新成本必須另外報告，不能當成相同參數數量或相同訓練時間。

正式模型示例：

```sh
python tools/run_hypothesis_pilot.py --variants full_dense --out experiments/runs/formal_trial
```

PPO 比較模型示例（只作研究比較，不變更 GUI）：

```sh
python tools/run_hypothesis_pilot.py --comparison-model ppo --variants full_dense --out experiments/runs/comparison_models/ppo_trial
```

輸出 protocol 記錄 algorithm、model_role、updates_per_episode、initial_ppa_evaluations=0、來源 hash 與 budget。報告所有預先宣告 seeds；訓練和凍結搜尋分開計費。短 smoke 只驗證可執行性，不能用來判定優越性；需要多種工作負載、需求與多種 seed 的正式配對實驗。Checkpoint schema 2 阻止 PPO 權重誤載入正式 REINFORCE。

## 0.7.0 歷史 PPO 資料

位於 `comparison_models/ppo_0_7_0/`。保留原始資料與來源 hash，不重新標成新版成果。該版起始設計會先評估一次，observation schema 4，PPO 及初始 reference 的定義不同；不能與新契約直接做公平數字比較。舊 H1/H2/H3 pilot 比的是 state／action／reward ablations，並沒有比較 PPO 與 REINFORCE；核心假說沒有成立的證據。

舊 GUI 已在切換正式版本時停止，其中尚未完成的長搜尋只保留為未完成 PPO 比較執行資料。新環境與正式策略的驗證見 `evidence/`，當前狀態見[研究進度](research_progress.md)。

現行 PPO 的兩種子 smoke 位於 `comparison_models/ppo_current_smoke/`。它與正式模型使用相同的零預評估、聯合動作及完整環境契約，只驗證參數更新／凍結／輸出。沒有 algorithm superiority 的結論；正式 H1/H2/H3 配對 pilot 只使用 REINFORCE。
