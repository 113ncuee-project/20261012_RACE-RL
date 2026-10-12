# 目前研究風險

1. **物理模型未校準。** 單顆 area／SRAM density、float64 throughput／energy、外部 DRAM／HBM service 與 IO geometry 尚無校準 profile；compute die area 不含外部記憶體或封裝。
2. **目前只有一個 tiny Toy 的配對假說 pilot。** 三個 policy seeds、每 trial 訓練 16／凍結 8 次 calls；不是充分學習、完整 CNN 或 held-out 泛化證據。
3. **沒有全域最優／架構淘汰依據。** Best observed mapping 不等於該顆數 optimum。接受較差 current、重訪與探索下限能避免硬剪枝，但有限預算仍會錯失優解。
4. **壓縮 observation 未證明完整 Markov sufficiency。** Cache metadata 未逐筆編碼，沒有 history/RNN，Tile bounds 未直接逐維輸入 neural features。
5. **Action 候選有支援邊界與解碼成本。** Connected scope 仍由 bounded enumeration 列出，不是 learned incremental expansion；不支援 channel reduction／sharding／不規則 partitions。
6. **H2 control 的 policy support 不同。** Independent 與 coupled 的分支／候選數及成本不同，必須一起報，不能只比較最好 seed。
7. **H3 coefficient 還未 validation 選值。** 本次 lambda_c=0.1 為事前固定探索值；proxy 是最長單一 critical noncompute service／makespan。
8. **統計只能探索描述。** 三個 seeds 的 bootstrap CI 不等於正式多重比較或顯著性驗收。State×action×reward interactions 尚未執行。
9. **新聯合環境尚缺完整 heuristic baselines。** Greedy／SA 的硬體 rollback／dispatch／global budgets 必須另驗證；不能直接套用固定硬體 helper 宣稱公平比較。
10. **排程保守、封裝抽象。** Spill-all、無 shared reads、effective external interface、atomic path reservation 不是精確 packet simulator 或商用封裝。
11. **跨裝置／正式 test 未完成。** 当前 CPU deterministic；GPU/MPS、跨平台相同浮點、回合中途 resume、獨立 model-family test 尚未驗收。

必要下一步是更有辨識度的 workload instances、充分 training/search budgets、多起始 mapping／seeds、外部服務與物理係數敏感度，以及正式 protocol；不應為迎合假說而改動已執行結果。
