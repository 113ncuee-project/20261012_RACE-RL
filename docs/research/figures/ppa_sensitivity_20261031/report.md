# RACE-RL PPA 限制敏感度探索

每個模型與情境，圖形使用同一個 seed；此 seed 由平衡偏好結果距離中位數最近選取。其他偏好使用相同 seed，並保留未達標結果。

| 情境 | 模型 | 偏好 | 達標數 | 與平衡共同達標 | 偏好目標低於平衡 | 目標下降中位數 |
|---|---|---|---:|---:|---:|---:|
| reference_v2 | resnet18 | 平衡 | 5/5 | 5 | — | — |
| reference_v2 | resnet18 | 功耗優先 | 5/5 | 5 | 0 | 0.00% |
| reference_v2 | resnet18 | 效能優先 | 5/5 | 5 | 1 | 0.00% |
| reference_v2 | resnet18 | 面積優先 | 5/5 | 5 | 0 | 0.00% |
| reference_v2 | mobilenet_v2 | 平衡 | 5/5 | 5 | — | — |
| reference_v2 | mobilenet_v2 | 功耗優先 | 5/5 | 5 | 1 | 0.00% |
| reference_v2 | mobilenet_v2 | 效能優先 | 5/5 | 5 | 1 | 0.00% |
| reference_v2 | mobilenet_v2 | 面積優先 | 5/5 | 5 | 0 | 0.00% |
| reference_v2 | vgg16 | 平衡 | 5/5 | 5 | — | — |
| reference_v2 | vgg16 | 功耗優先 | 5/5 | 5 | 3 | 1.44% |
| reference_v2 | vgg16 | 效能優先 | 5/5 | 5 | 3 | 2.48% |
| reference_v2 | vgg16 | 面積優先 | 5/5 | 5 | 2 | 0.00% |
| exploratory_latency_x2 | resnet18 | 平衡 | 5/5 | 5 | — | — |
| exploratory_latency_x2 | resnet18 | 功耗優先 | 5/5 | 5 | 0 | 0.00% |
| exploratory_latency_x2 | resnet18 | 效能優先 | 5/5 | 5 | 0 | 0.00% |
| exploratory_latency_x2 | resnet18 | 面積優先 | 5/5 | 5 | 0 | 0.00% |
| exploratory_latency_x2 | mobilenet_v2 | 平衡 | 5/5 | 5 | — | — |
| exploratory_latency_x2 | mobilenet_v2 | 功耗優先 | 5/5 | 5 | 1 | 0.00% |
| exploratory_latency_x2 | mobilenet_v2 | 效能優先 | 5/5 | 5 | 0 | 0.00% |
| exploratory_latency_x2 | mobilenet_v2 | 面積優先 | 5/5 | 5 | 0 | 0.00% |
| exploratory_latency_x2 | vgg16 | 平衡 | 5/5 | 5 | — | — |
| exploratory_latency_x2 | vgg16 | 功耗優先 | 5/5 | 5 | 2 | 0.00% |
| exploratory_latency_x2 | vgg16 | 效能優先 | 5/5 | 5 | 3 | 16.70% |
| exploratory_latency_x2 | vgg16 | 面積優先 | 5/5 | 5 | 0 | 0.00% |
| exploratory_latency_x2_area_x05 | resnet18 | 平衡 | 5/5 | 5 | — | — |
| exploratory_latency_x2_area_x05 | resnet18 | 功耗優先 | 5/5 | 5 | 0 | 0.00% |
| exploratory_latency_x2_area_x05 | resnet18 | 效能優先 | 5/5 | 5 | 0 | 0.00% |
| exploratory_latency_x2_area_x05 | resnet18 | 面積優先 | 5/5 | 5 | 0 | 0.00% |
| exploratory_latency_x2_area_x05 | mobilenet_v2 | 平衡 | 5/5 | 5 | — | — |
| exploratory_latency_x2_area_x05 | mobilenet_v2 | 功耗優先 | 5/5 | 5 | 2 | 0.00% |
| exploratory_latency_x2_area_x05 | mobilenet_v2 | 效能優先 | 5/5 | 5 | 0 | 0.00% |
| exploratory_latency_x2_area_x05 | mobilenet_v2 | 面積優先 | 5/5 | 5 | 0 | 0.00% |
| exploratory_latency_x2_area_x05 | vgg16 | 平衡 | 5/5 | 5 | — | — |
| exploratory_latency_x2_area_x05 | vgg16 | 功耗優先 | 5/5 | 5 | 1 | 0.00% |
| exploratory_latency_x2_area_x05 | vgg16 | 效能優先 | 5/5 | 5 | 3 | 22.01% |
| exploratory_latency_x2_area_x05 | vgg16 | 面積優先 | 5/5 | 5 | 0 | 0.00% |

正百分比表示該目標相對同 seed 平衡結果下降；零或負值亦完整保留。不同情境限制與評分改變，跨情境須閱讀原始 PPA。資料是模型估計，不是實體量測。
