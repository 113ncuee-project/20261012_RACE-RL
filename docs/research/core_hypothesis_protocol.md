# 核心假說的新正式驗證契約

正式模型為 REINFORCE；PPO 只以 `--comparison-model ppo` 明確啟用。所有試驗執行前寫入 protocol，記錄 workload／hash、限制／權重／固定尺度、顆數集合、state／action／reward variants、seeds、algorithm、model_role、更新次数、float64、gamma、步數／新 PPA 評估額度及 frozen evaluation 契約。

H1 比較完整 graph 與 compact／DNN／hardware／no-attribution／no-criticality。H2 比較 coupled 與 independent，所有 count／partition 必須仍可到達，不能因一次差 mapping 刪掉架構。H3 比較 dense、terminal 與 potential；potential 的 terminal 為零且 gamma=1。

reset 零 PPA 評估，第一次 action 才計費。候選生成與 Actor 零 Oracle 呼叫。所有 variants 共用完整環境；每回合 action requests 和新評估預算相同，正式配對關閉 GUI 的提前停止。Frozen rollout 不更新參數，費用與訓練分開。所有 seed 都報告，不做最好 seed 選擇；支持數、時間與參數數量同時揭露。

工具 `tools/run_hypothesis_pilot.py` 預設正式 REINFORCE，一回合一次更新；PPO 比較模型有 Critic、四 epochs，不能宣稱相同參數／更新成本。兩者新的環境契約相同。

本次短 smoke 只驗證流程，沒有正式 CNN 泛化或假說接受。0.7.0 歷史 PPO 的實驗見[比較模型](comparison_models.md)，起始計費與目標定義不同，不能直接與新契約做公平比較。

本次已按此契約執行 10 variants × 3 seeds、scope=2、1–16 顆，2 回合訓練＋1 回合凍結、每回合最多 8 個完整 action。實際試驗結果與未成立的結論見[假說結果](hypothesis_results.md)。
