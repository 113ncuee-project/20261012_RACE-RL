# 七模型訓練基準與研究修正（2026-10-14）

原始批次 `20261013T164333383414Z` 完成七模型各 60 回合、每回合 64 次互動、輸入 64×64、float64、scope=2、單一 REINFORCE／無 Critic、seed=7。凍結前後測各 seed=1007、64 次互動；PPA 平均，上限 1 W／100 ms／16 mm²。此批次為 0.8.0 的診斷基準，不將後續任務限制回套到原始成效。

台北時間從 2026-10-14 00:43:33 到 10:49:13，累計執行 10:05:39。模型時間包含載入、驗證、訓練、測試、輸出，不只是策略訓練時間。

執行平台：MacBookAir10,1、Apple M1、8 logical CPU、8 GiB RAM、arm64、macOS 27.0.1（26A434）；Python 3.12.14、torch 2.14.1、torchvision 0.29.1、numpy 2.5.4、matplotlib 3.11.2。計算使用 CPU／torch_threads=1／float64／deterministic。這份平台快照在完成後收集，沒有訓練開始時的版本凍結、CPU 使用率、峰值記憶體或降頻紀錄。原始來源 hashes 在修改前與當時本機程式一致；核對時 HEAD 為 8d08d194，不冒稱它就是開始時記錄的 commit。

| 模型 | 全模型耗時 | 訓練最佳顆數 | 功耗 W | 延遲 ms | 面積 mm² | 凍結策略 J 相對改善 |
|---|---|---:|---:|---:|---:|---:|
| resnet18 | 00:46:03 | 8 | 0.013302 | 452.72 | 8 | +0.000% |
| resnet34 | 01:35:06 | 8 | 0.013021 | 873.70 | 8 | -0.233% |
| resnet50 | 02:53:53 | 6 | 0.011474 | 1092.34 | 6 | +0.000% |
| vgg16 | 00:50:16 | 2 | 0.007913 | 4368.22 | 2 | +0.000% |
| vgg19 | 01:02:51 | 2 | 0.007694 | 5098.42 | 2 | -0.237% |
| mobilenet_v2 | 02:26:05 | 1 | 0.006547 | 191.94 | 1 | +0.950% |
| alexnet | 00:31:22 | 2 | 0.009194 | 1362.62 | 2 | -0.253% |

全部選中方案物理合法，但延遲超過 100 ms。獨立 audit 核對 26,880 次訓練互動、26,599 次新評估、281 次快取、420 次參數更新；評分、reward、state 銜接與計費均一致。前後測為 1 改善、3 持平、3 退步，只有單一 training/evaluation seed，不能證明學習有效或無效。

24,897／26,880（92.6%）reward 為零或 ≤1e−12；平均第一步占回合 return 89.8%。11,427 步改善當前 J，9,444 步改善卻未突破歷史最佳，故未得正 reward。這是 dense best-improvement 的機制，不是評分算錯。`return=-log(best_J)`，不能把 return 與 best_J 當兩項獨立學習證據。reset 的 J=1 是未量測 sentinel；不採用審查建議的 reset 預評估。

僅參數必要讀取下界（兩埠合計 2 GB/s）就排除 ResNet50、VGG16、VGG19、AlexNet 的 100 ms 目標。另三模型未被此下界排除，不表示可行。功耗 1 W 遠高於取樣值；面積上限 16 mm² 又等於最大顆數的面積，這次硬限制主要由延遲決定。數值 PASSED 只驗證算子解析／計算，不驗證 silicon PPA。

審查的 critical-path 百分比另經獨立核對：從包含資源相依的 predecessors 取一條最長鏈，確認其 duration 總和等於 makespan。memory_read 包含外部、路由和本地服務，不能全部解釋成 DRAM 本身時間。詳細數值在 audit.json；[硬體與資料流圖](hardware_model.md) 列明模型假設與缺失。

## 已採用的修正

0.9.0 的預設終端任務使用下列「先導結果衍生、已量測可行設計可作憑證」的合成限制。功耗取憑證值 ×1.10 向上取至 0.001 W；延遲 ×1.20 向上取至 0.01 s；面積為憑證顆數＋1 mm²。限制於新訓練前凍結，不是廠商需求，也不是物理校準。使用先導結果選任務可能產生任務選擇偏差，因此新實驗只作已知可行情境的學習診斷，不聲稱所有需求的泛化。

| 模型 | 功耗上限 W | 延遲上限 ms | 面積上限 mm² |
|---|---:|---:|---:|
| resnet18 | 0.015 | 550 | 9 |
| resnet34 | 0.015 | 1050 | 9 |
| resnet50 | 0.013 | 1320 | 7 |
| vgg16 | 0.009 | 5250 | 3 |
| vgg19 | 0.009 | 6120 | 3 |
| mobilenet_v2 | 0.008 | 240 | 2 |
| alexnet | 0.011 | 1640 | 3 |

七份憑證合計約 35 KB，保存完整 workload／hardware／mapping 與出處 SHA256，另作七次「先選完整設計，再送環境」重算全部通過。此驗證在 RL 之外，沒有 reset／Actor 預評估；不把憑證方案給 Actor 或作初始化，不按一次差的 mapping 排除顆數。`--requirements` 仍完全採廠商輸入，不會暗中放寬。CLI／GUI 只用必要讀取、靜態功耗與面積下界提示已證明不可能的要求；其餘情況仍需環境實測。

預設每模型仍 60×64 訓練互動、seed=7，前後各 10 個配對 evaluation seeds=1007–1016，且還原 Python／NumPy／Torch RNG、不更新 policy。每模型總 5,120 次互動，七模型 35,840 次，比舊版 27,776 次多 29%，不是四倍，也不是耗時保證。evaluation seed 反映同一 trained policy 的抽樣變異，不能替代多個 training seeds。

中途每 10 回合測試、等預算 random legal-action walk 與 simulated annealing 已做成明確選項；預設不加這些成本。兩種比較每模型各 60×64 requests，失敗和 cache 都算 request，另報 new evaluation 數；不把 3,840 requests 寫成 3,840 新評估。SA 每回合溫度 0.2→0.01，完整 proposal 評估後才接受／拒絕，無免費鄰居 lookahead。訓練最佳、凍結前後測與比較最佳分開保存；額外診斷查詢／時間不送給策略，仍算整體成本。

加入 normalized entropy、clip 前梯度範數、非零 reward 步数與首步比例，不改 dense REINFORCE 梯度。現行策略本來已有 5% uniform mixture；entropy coefficient=0 不等於沒有探索。保留 dense 預設，提供 terminal reward／learning rate／entropy 的顯式消融參數；沒有證據時不宣稱新公式或 PPO 比較好。

目前仍只有 H-only（2D 為 feature）retile，spill-all、合成硬體、float64 workload；尚未提供完整 W/C/reduction/dataflow 或 224×224 搜尋時間證據。擴充前先定義並驗證硬體服務／重用模型，分析瓶頸與探索支援規模。詳細討論項目見硬體文件。

## 執行與證據

完整七模型正式搜尋由使用者在終端啟動；沒有代為跑新的長訓練：

```sh
.venv/bin/python tools/train_model_suite.py
```

選用完整比較與中途診斷會增加預算，不能當成訓練器變慢的四倍證據：

```sh
.venv/bin/python tools/train_model_suite.py --comparisons random sa --evaluation-interval 10
```

同一任務要取得獨立訓練穩定性時，分別以 `--seed 7/17/27/37/47` 建立五份獨立結果；勿把 50 個 evaluation rollouts 當作 50 個訓練樣本。Reward 消融用 `--reward terminal`，需全新目錄、相同任務與 requests 才能比較。

[獨立核對](evidence/model_suite_20261014/audit.json)、[平台快照](evidence/model_suite_20261014/platform.json)、[各模型時間](evidence/model_suite_20261014/model_times.json)、[可行設計重算](evidence/model_suite_20261014/feasibility_validation.json)。原始本機資料仍保留於 experiments/runs/model_suite/training/20261013T164333383414Z；不複製每個候選帳本至 Git。

Windows 8d08d194 的失敗由使用者提供日誌確認為 `WinError 32`：test_model_suite.py 的 SQLite 連線未關，導致 TemporaryDirectory 無法刪除 history.sqlite。測試及報告讀取器改成明確 closing，另加入關閉回歸與 CI 日誌 artifact。雲端新 commit 尚未推送驗證，不能保證所有 Windows 檢查已通過。
