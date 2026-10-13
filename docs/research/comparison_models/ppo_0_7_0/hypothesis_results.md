# 核心假說：現行 PPA pilot 結果

固定平均功耗上限 0.1 W、延遲上限 1 ms、運算 die area 上限 16 mm²、PPA 平均。10 variants × 3 seeds，實際 480 次訓練與 240 次凍結搜尋評估。這是單一 Toy 的開發試驗。

| Variant | Seed 0 J | Seed 1 J | Seed 2 J | Mean J |
| --- | ---: | ---: | ---: | ---: |
| full_dense | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| compact_dense | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| dnn_dense | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| hardware_dense | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| no_attribution_dense | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| no_criticality_dense | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| independent_dense | 1.000000 | 0.998664 | 1.000000 | 0.999555 |
| full_terminal | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| full_best_potential | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| full_bottleneck_potential | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

J 是 best feasibility-first PPA merit／固定 initial merit，越低越好。並非直接的加權成本比例；原始 PPA 與 weighted cost 另存於紀錄。

H1／H2／H3 尚未有支持證據。三 seeds、短訓練與單一 Toy 不足以建立正式優勢、否定一般假說或宣稱跨 CNN 泛化。所有 variants／seeds 完整報告，不挑最好 seed。

比較統計見 [analysis](evidence/core_hypotheses/analysis.json)，評分對接與任務辨識力見 [核對報告](scoring_review.md)。只有探索性 bootstrap 描述，沒有正式 p-value 或多重比較驗收。

GUI 的 1–16 顆完整集合與此 pilot 的 2／4／…／16 子集合不同；此試驗不能當作完整 GUI 空間的收斂驗證。
