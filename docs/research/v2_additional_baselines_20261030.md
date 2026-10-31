# 第二輪既有結果補跑基準

## 範圍

先只補新基準，三角圖不改。使用 v2 的 3 個 CNN、4 個偏好、搜尋種子 311–315、原模型種子 7、原硬體與 PPA 條件，每個基準每組 64 次提案。直接讀取對應舊 RACE-RL 記錄，不重新訓練 RACE-RL，也不覆寫舊批次。

每組開始時，程式重建 v2 封存的 Python 評估環境及已儲存 workload，核對原始碼 SHA-256、舊 RACE 記錄 SHA-256 和共同任務 identity。以獨立評估器重新評估舊 RACE 最佳設計作一次見證核對；這個核對不送入新方法、不算搜尋預算，也不是重跑 RACE 搜尋。封存缺少 `gemmini_default.json`，目前使用專案內該資源，並記錄雜湊；硬體 identity 和見證指標均須吻合，否則停止。當年 Python 套件版本未完整封存，單一設計見證不等於完全證明跨環境重現。

## 三個新增項目

1. `genetic`：一般 GA，種群 8，競標 3，逐層交配與局部 mutation；不是 GAMMA 或 DiGamma 原始實作。
2. `bayesopt_gp_ei`：以固定核高斯過程與 expected improvement 選擇提案，一般離散 BO 基準；局部候選池 32、初始 4 提案，物理無效點的代理分數為 10。不是特定論文完整重現。
3. `timeloop_hybrid_restricted`：採隨機 profile 分組、組內列舉有限的配置和駐留方案的受限啟發式。不是 Timeloop 原版：原版的 index factorization、loop permutation 和 mapspace 不在現有 CNNDesign 表示中，不能依據本項宣稱優於 Timeloop 論文或原版工具。

已下載 NVIDIA 官方 Timeloop 原始碼到 `third_party/timeloop`，commit `32370826fdf1aa3c8deb0c93e6b2a2fc7cf053aa`；原版搜尋程式在 `src/search/hybrid.cpp`。要使用原版還需設計並驗證原生 mapspace 與共用 CNNDesign／評估器之間的映射，以及處理原版 C++ 依賴；目前沒有把本受限方法冒稱為原版。Timeloop 的一般 random 已有 v2 共用空間中的 `random` 基準，但同樣不是 Timeloop 原版 mapper。

## 執行

```bash
cd /Users/hsuyuchia/Documents/RACE-RL
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  tools/run_v2_additional_baselines.py \
  --out experiments/runs/common_dse_v2_extra_baselines
```

此命令處理 60 個舊試次，各新增 3 種方法；三者各 64 次提案，完整試次會逐筆寫入、逐步更新 `report.md` 和 `results.json`。原始 `common_dse_study_v2` 只讀，RACE 不重新訓練。中斷後使用相同命令加 `--resume`；來源檔、實作、Python／套件版本或參數改變時拒絕續跑。每個方法都使用相同條件與種子，但不同演算法使用不同隨機程序，不保證相同候選順序。

查看結果時，先比較是否均達標，再比較加權成本；同一組的 `comparable` 只有兩者達標且預算相等時才為真。煙霧測試採較小預算，不能和舊 RACE 視為正式配對或用來主張優劣。不同偏好的加權成本不要跨偏好直接比較平均；完整保留所有種子與失敗結果。
