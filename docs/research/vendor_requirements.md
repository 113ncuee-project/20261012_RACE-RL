# 廠商需求輸入與 PPA 評分

廠商在搜尋開始前輸入三項限制值，並選擇一種偏好。輸入驗證通過後，這些條件固定於同一任務內；chiplet 顆數、切分與 mapping 改變時不會重設限制或正規化尺度。搜尋由一個 RL policy 執行。

## 起始輸入

| 項目 | 定義 | 方向 |
| --- | --- | --- |
| 第一個 P：Power | 整次推論的平均功耗上限，W；包含目前模型計帳的 compute、local memory、external memory、network 及 static power | 越低越好 |
| 第二個 P：Performance | 單次推論的延遲上限；互動輸入用 ms，JSON 與 API 用 s | 越低越好 |
| A：Area | 運算晶粒面積總和上限，mm² | 越低越好 |

目前的功耗是 `energy_j / latency_s`，不是尖峰功耗。Performance 以 batch=1 推論延遲定義，沒有當成穩態吞吐率。Area 不包含封裝、外部 DRAM、interposer 或 I/O die。廠商输入的單位必須與這些定義一致；物理係數仍未校準，結果是分析式研究估計。

## 五種偏好

權重順序固定為 **功耗、效能（延遲）、面積**。

| 顯示選項 | 設定名稱 | 權重 |
| --- | --- | --- |
| PPA 平均 | `balanced` | 1/3、1/3、1/3 |
| 偏重功耗 | `power` | 0.6、0.2、0.2 |
| 偏重效能 | `performance` | 0.2、0.6、0.2 |
| 偏重面積 | `area` | 0.2、0.2、0.6 |
| 使用者自訂 | `custom` | 輸入三個非負有限數，總和需大於零；系統正規化 |

60/20/20 是目前產品預設，不是研究證明的最佳權重。自訂允許零權重；即使某項權重是零，該項硬限制仍必須達標。平均是正規化 PPA 的算術平均，不會直接把 W、s、mm² 相加。

## 評分與達標判定

廠商模式要求三個上限完整輸入；也必須明确選偏好。對完整 PPA 任務，固定尺度就是三個上限：

`C(D) = w_power × power/max_power + w_performance × latency/max_latency + w_area × area/max_area`

C 越低越好。每個候選都由 oracle 重新算原始 PPA，權重只影響排序，不改寫原始量測。物理合法且三項均不超標，才是廠商可行設計。達標判定包含等於上限。

RL 搜尋可經過物理合法、但暫時 PPA 超標的設計，避免起始設計不達標就無法開始。物理失敗不更新 current，但仍計入查詢。1–16 顆均可提出，不用 PPA 限制排除 resize 候選；一次差的功耗、延遲、mapping 或 partition 不永久排除該顆數。

0.9.0 用必要讀取、靜態功耗與面積下界檢查整個任務是否已被證明不可能，零 oracle 評估。若證明不可能，CLI／GUI 說明原因並停止；通過下界只表示未被排除。廠商限制原樣保留。終端批次的模型別研究限制由先導量測憑證衍生，不等同廠商需求，見[研究紀錄](model_suite_20261014.md)。

用於 reward 的搜尋 merit 與廠商加權成本 C 分開記錄：

- 達標設計：`M = C/(1+C)`，範圍為 (0,1)。
- 未達標設計：`M = 2 + Σmax(metric/limit−1,0) + C/(1+C)`，大於 2。

這保證任何達標設計都優先於未達標設計；達標集合內依 C 排序。未達標集合依 merit 改善進度，並非已達標結果。起始不測試 PPA；已測設計 `J=M/(1+M)`，未知最佳檔案使用 sentinel J=1。Dense reward 為 `log(J_best_before/J_best_after)`。內部 best 是已評估的最佳搜尋進度；對外只有達標時才輸出 `best_*`。

自訂零權重造成精確同分時，保留在全部 PPA 皆不差、至少一項更好的候選。這只改善同分 archive，不改變主要權重。

每回合的 best merit 單調改善；current 可以較差，或再次超標。沒有達標方案時輸出 `NO_FEASIBLE_DESIGN_FOUND`、違規項目與 `closest_*`，best 欄位為 null、達標 Pareto 集合為空，不把未達標設計冒充最佳可交付設計。

研究 API 可省略部分限制，以事先宣告的 SI 尺度 1 W／1 s／1 mm² 作缺少項目的尺度，供開發檢查使用。這是內部測試模式；廠商搜尋入口不允許缺項。

## RL 看得到需求

Observation schema 5 增加 16 維 task features：三個正規化權重、三個限制存在旗標、三個固定尺度的 log、三個 current/scale 的 log、三個 current/limit 的 log，以及 best 是否達標；Global 另外加入有無有效 PPA 與前一 reward。尺度與比值的 log 除以 30。這些特徵與 graph/global features 一起輸入共享 encoder，影響無 Critic 的 Actor。Global features 的功耗欄位使用平均功耗。

同一個 policy 可接收不同需求，但目前每個訓練實驗仍是固定任務；尚未以多需求訓練或 held-out 偏好驗證，不能宣稱一次訓練就能泛化到任意廠商要求。

## 使用方法

網頁操作見 [Python 本機 GUI](gui.md)。

互動入口依序輸入三項限制，再從 1–5 選偏好；選 5 時輸入自訂權重：

```sh
.venv/bin/python -m race_rl.cli search-mesh --interactive --out experiments/runs/vendor_search
```

可先編輯 `configs/vendor_requirements.json`，再啟動搜尋。檔案中的數字只是示例，不代表真實廠商需求：

```sh
.venv/bin/python -m race_rl.cli search-mesh --requirements configs/vendor_requirements.json --out experiments/runs/vendor_search
```

命令列也可完整指定：

```sh
.venv/bin/python -m race_rl.cli search-mesh \
  --max-power-w 1 --max-latency-s 0.1 --max-area-mm2 16 \
  --preference custom --ppa-weights 5 3 2 \
  --out experiments/runs/custom_search
```

互動、檔案與命令列三種輸入方式互斥，避免不明確的覆寫。所有輸入在載入 DNN 與啟動 oracle 前驗證。`vendor_report.json` 包含需求、選项、正規化權重、固定尺度、原始 PPA、達標狀態與違規原因；`candidate_metrics.json` 提供所有已評估候選的相同評分。
