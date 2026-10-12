# 核心假說 pilot：實際結果

本次為 compute-only mesh＋外部記憶體版本。10 variants×3 seeds，480 次訓練及 240 次凍結搜尋評估；完整 query／budget／來源身份核對通過。所有 policy 實際更新，凍結階段權重皆不變；每組參數量相同為 4,986。

這是單一 Toy／三種子的開發實驗，不是正式 CNN 泛化或顯著性驗收。初始硬體、mapping、reference、budget、source／profile 均相同。完整事前設定見 [protocol](core_hypothesis_protocol.md)，原始數據見 [trial records](evidence/core_hypotheses/trial_results.json)。

| Variant | Seed 0 J | Seed 1 J | Seed 2 J | Mean J |
| --- | ---: | ---: | ---: | ---: |
| full_dense | 1.000000 | 1.000000 | 0.955518 | 0.985173 |
| compact_dense | 1.000000 | 1.000000 | 0.955518 | 0.985173 |
| dnn_dense | 1.000000 | 1.000000 | 0.955518 | 0.985173 |
| hardware_dense | 1.000000 | 1.000000 | 0.955518 | 0.985173 |
| no_attribution_dense | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| no_criticality_dense | 1.000000 | 1.000000 | 0.955518 | 0.985173 |
| independent_dense | 1.000000 | 0.980060 | 1.000000 | 0.993353 |
| full_terminal | 1.000000 | 1.000000 | 0.955518 | 0.985173 |
| full_best_potential | 1.000000 | 1.000000 | 0.955518 | 0.985173 |
| full_bottleneck_potential | 1.000000 | 1.000000 | 0.955518 | 0.985173 |

J 為 best verified EDP／固定初始 EDP，越低越好。不是 classification accuracy。

## 假說判讀

- **H1：未成立。** Full 相對 compact／DNN／hardware／without criticality 的最終成績相同；相對 without attribution 只有一個種子改善、另兩個相同。這是有限的探索訊號，不是一般關聯 state 優勢的證明。
- **H2：未成立。** Coupled 對 independent 一勝、一負、一平；平均 left-minus-right J = -0.008181，探索性 paired bootstrap interval 約 [-0.044482, 0.019940]，跨零。不能宣稱 coupled 普遍更有效。
- **H3：未成立。** Terminal／dense／best potential／bottleneck potential 在本次凍結搜尋的最終 EDP 相同，沒有看到 reward shaping 的搜尋收益。

只有 3 seeds，10,000 次 bootstrap 只描述 seed variability；沒有正式 p-value、多重比較驗收或 interaction 實驗。不因結果不利修改已完成規則。分析見 [analysis.json](evidence/core_hypotheses/analysis.json)。

## 下一個驗收點

增加不同 dependency／memory bottleneck 的 workload instances、training／search budgets、多起始 mapping 與獨立 seeds；觀察 variant 是否可穩定分化。正式 CNN split 與物理參數敏感度／校準需事前固定，不能以本次開發資料當作 held-out test。
