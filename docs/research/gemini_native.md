# Gemini HPCA 2024 原生程式接入

本專案保存 [Gemini 論文指定的 Zenodo artifact](https://zenodo.org/records/10207613)，而非目前已更新的 GitHub 分支。原始壓縮檔、授權和 SHA-256 見 [第三方來源](../../THIRD_PARTY_NOTICES.md)。`tools/gemini_native.py` 驗證來源、解壓、套用 C++ 可攜性修正並編譯；修正替換 `std::bitset` 非標準尋找方法與 `std::__gcd`，並在搜尋完成後加入唯讀 mapping 匯出。原始搜尋演算法不變。編譯輸出不放 Git。

## 執行原版引擎

```sh
cd /Users/hsuyuchia/Documents/RACE-RL
.venv/bin/python tools/gemini_native.py prepare --jobs 4
.venv/bin/python tools/gemini_native.py run \
  --inputs 0 1 2 2 2 1 30 1 1 1 8 73728 8 512 512 73728 \
  --out experiments/runs/gemini_native_smoke --timeout 10
```

這只是縮小版原生串接測試，不是論文完整實驗或 RACE-RL 比較。輸出包含原生結果、標準輸出、錯誤紀錄與執行身份；狀態只有在程式成功且輸出包含 Gemini mapping 結果時才是 `COMPLETED`。重跑請選新輸出資料夾。原 artifact 至少需要 30 rounds，否則原程式有除零風險。長時間論文實驗由使用者另行啟動。

`gemini_mapping.json` 包含原模型的 layer 圖、每層輸出 tile 的實際座標與核心、分組、partition 及原版 DRAM 設定。它不含內部 loop order 或完整 K blocking。`tools/gemini_bridge_audit.py` 可對照本專案 workload，檢查輸出覆蓋與已知不相容；目前不會產生假裝等價的 CNNDesign。

`tools/gemini_ownership_bridge.py` 可以在明確指定 layer／operator 與 core-to-chiplet 對應後，逐一檢查本專案四種 blocking profile 的輸出 tile 矩形與 owner，找出完全一致的候選。這只檢查單層 ownership；K blocking、算子語意、記憶體、排程與硬體並沒有被推論，也不會輸出 PPA 或宣稱整個設計等價。不同輸出 shape、缺少核心對應或找不到完全相同的 tile 都不能視為轉接成功。

```sh
.venv/bin/python tools/gemini_bridge_audit.py \
  --mapping experiments/runs/gemini_native_smoke/gemini_mapping.json \
  --workload experiments/runs/research_exploration_20261030/vgg16/workload.json \
  --out experiments/runs/gemini_native_smoke/bridge_audit.json
```

2026-10-30 的 2×2 核心短測匯出 84 個 tile；每層 tile 覆蓋檢查通過。與 64×64 VGG16 比較時，原版 VGG19 圖只有 21 個卷積／pooling 算子，RACE VGG16 圖有 40 個算子；圖、輸出 shape、核心／晶粒和記憶體介面都不同，檢查如實輸出 `NOT_COMPARABLE`。結果位於 `experiments/runs/gemini_bridge_exploration_20261030/`，此目錄不納入 Git。

16 個輸入依序為 `mm nn xx yy ss batch rounds ff xcut ycut serdes_lane dram_bw noc_bw mac_dim ul3 tops`，定義與單位以原版 `dse.sh` 與 `src/main.cpp` 為準。`nn=1` 指原版 VGG19，`nn=2` 指原版 ResNet50；不等同目前測試的 VGG16、ResNet18、MobileNetV2。上游以系統時間做部分隨機 seed，其他 generator 有各自的固定 seed；目前會保留原始行為並記錄 stdout，不宣稱可指定同一 seed 重現完整實驗。

## 與 RACE-RL 比較的界線

原生 Gemini 用自己的模型圖、layer grouping、架構、mapping 搜尋和成本函數；它的內建模型與本專案 workload、成本、延遲及晶粒定義並未對齊。其原生輸出不能直接插入 `run_cnn_comparison_suite.py` 或宣稱在相同 PPA 下勝負。

Gemini 的分組、核心交換、跨層交換／分配及 DRAM 位置動作，無法一對一表達為現有 `CNNDesign` 的單起點／round-robin 層配置；原始系統也包含內層 tiling 窮舉，評估預算尚未接入共用計費。完成共同比較需要共同 workload 與硬體表示、可驗證的雙向 mapping 轉換，以及所有內層評估的記帳。任何受限適配都須標示為改編版及其能力損失。

原 artifact 的 `wafer_util` 使用 C++ 鏈式比較 `0 < area <= 25000000`；C++ 語意與數學區間不同。為保留原版本接入沒有修改它，編譯選項抑制該診斷。這會影響成本可信度，不能把原生數值當作已校準晶片結果。
