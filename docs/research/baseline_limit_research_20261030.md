# 比較基準與 PPA 限制初步研究

## 文獻與方法

- GAMMA（ICCAD 2020）使用針對 DNN 硬體 mapping 的領域專用遺傳演算法；公開原始碼與 MAESTRO 成本模型：https://par.nsf.gov/servlets/purl/10321972 、https://github.com/maestro-project/gamma 。本專案新增的是一般 GA，並非 GAMMA 重現，不能以它宣稱勝過 GAMMA。
- Timeloop 支援 random、random_pruned、hybrid 等 mapping 搜尋；它的 mapper 是既定硬體下的 mapping 搜尋，不等於本專案完整硬體與 mapping 聯合搜尋：https://timeloop.csail.mit.edu/v4/input-formats/mapper 。
- Reagen 等人的 2017 年研究使用貝氏最佳化比較 grid 與 genetic algorithm；BO 是有文獻依據的較強基準，並非保證 RACE 會勝：https://vlsiarch.eecs.harvard.edu/publications/case-efficient-accelerator-design-space-exploration-bayesian-optimization 。目前尚未實作 BO。
- DiGamma 以領域專用 GA 聯合最佳化硬體與 mapping：https://research.nvidia.com/publication/2022-03_digamma-domain-aware-genetic-algorithm-hw-mapping-co-optimization-dnn 。如要宣稱贏過此方法，必須另行接入並驗證原方法。

## 新增一般 GA

`tools/common_dse_genetic.py` 對共用 CNNDesign 執行種群 8、競標選擇 3、逐層交配與既有局部 mutation；所有初始、無效與快取命中提案均計入預算。這是普通的 GA baseline，不是 GAMMA、DiGamma 原版，並不是保證易勝的基準。`--methods` 現可選擇方法，與原始 Gemini／Chiplet-Gym baseline 是否存在無關。為完整呈現證據，既有 v2 Chiplet-Gym 結果保留，不以新方法替換舊結果。

單一 smoke：ResNet18、balanced、seed 401、每方法 8 次提案；GA 成本 0.517163、RACE 成本 0.513568，兩者皆達標。這只是串接與預算檢查，不是效果結論。`experiments/runs/ga_smoke_20261030/analysis.json` 記錄核對資料。

## 已有候選的限制敏感度

`tools/analyze_ppa_limit_sensitivity.py` 將 v2 原本 RACE 60 組 trace 的每筆實際物理有效候選，重新套用原始、延遲上限 2 倍、延遲上限 2 倍加面積上限 0.5 倍三種條件，以及各偏好權重，找出該舊 trace 中達標成本最低者。所列中位數分別取 5 個 seed 的各指標，見 `experiments/runs/ga_smoke_20261030/offline_limit_sensitivity.json`。

這是事後、探索性、受原訓練條件所限的重評，並非調整限制後重新訓練或比較新方法；不能用來挑設定當作確認性結論。限制同時控制硬約束和成本正規化，因此更動限制改變搜尋任務本身，不能只調圖形後宣稱實驗改善。

- ResNet18：原設定效能優先中位面積 10 mm²；延遲上限 2 倍後，舊 trace 重評降為 5 mm²，效能優先與平衡更接近，並非固定會讓偏好差異變大。
- MobileNetV2：上述三種設定的四偏好中位面積均為 5 mm²；僅放寬延遲無法憑此證明面積優先的區別。
- VGG16：延遲上限 2 倍後，平衡中位約 0.257 W、0.863 s、5 mm²；效能優先約 0.410 W、0.559 s、15 mm²。面積／功耗的下降伴隨延遲增加，顯示可能的取捨，但它們是離線重評的探索性結果。

功耗與面積可能同時隨晶粒數減少，或一種設計同時在多個指標較好。這時四種偏好未必有四種不同結果；應分析 Pareto 前緣、設計空間和共同 seed 的變化，不可靠坐標或單一 seed 製造差異。以目標平台或預先聲明的設計範圍指定需求；若用探索資料擬定限制，另用新 seed 與外部工作負載檢驗。

## 長時間實驗

先用原 v2 限制；三模型、四偏好、全新 seed 321–325，各方法 64 次提案。`genetic` 是一般 GA；`gemini_common_restricted` 是受限改編。不要把此批和 v2 直接合併成同一方法集合或假裝與原論文直接對比。

```bash
cd /Users/hsuyuchia/Documents/RACE-RL
PYTHONPATH=src .venv/bin/python tools/run_common_dse_suite.py \
  --limits-dir configs/common_dse_limits/reference_v2 \
  --out experiments/runs/common_dse_baseline_ga_v1 \
  --models resnet18 mobilenet_v2 vgg16 \
  --preferences balanced power performance area \
  --seeds 321 322 323 324 325 --budget 64 --rl-episode-size 16 \
  --methods random local_random sa genetic gemini_common_restricted reinforce
```

中斷後以相同指令加 `--resume` 繼續；修改原始碼或參數後無法續跑。全部完成後：

```bash
PYTHONPATH=src .venv/bin/python tools/analyze_common_dse.py \
  experiments/runs/common_dse_baseline_ga_v1 \
  --out experiments/runs/common_dse_baseline_ga_v1/analysis.json
```

另一探索條件可將 `--limits-dir` 改成 `configs/common_dse_limits/exploratory_latency_x2`，並使用新的 `--out` 資料夾及新 seed。此設定只供敏感度實驗，不是已證實具實際意義的硬體需求。兩種條件均需對所有方法使用相同限制及預算，並完整報告所有 seed、可行率與成本。
