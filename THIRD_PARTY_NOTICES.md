# 第三方來源

`third_party/rapidchiplet-main/` 是目前評估器使用的原生 [RapidChiplet](https://github.com/spcl/rapidchiplet) 依賴。來源 lock 指向 `17176f92c91eda629a25bc11bf7959d6b963a23b`，執行時核對四個核心模組 SHA-256。原附 BookSim／netrace 授權文件保留，不替第三方重新授權。

新研究程式位於 `src/race_rl/`。目前沒有指定新程式的公開授權，不宣稱整個 repository 具有統一的 permissive license。

## Gemmini 公開配置參考

`third_party/gemmini-reference/` 保存 [Berkeley Gemmini](https://github.com/ucb-bar/gemmini) commit `8c3f9923a44a2fe2c7930587be297d6d4f8c09ca` 的 7 個只讀參考檔，包括原始 `LICENSE`、README、配置與相關硬體原始碼。原始 UC Regents 著作權、授權條件與聲明均保留；沒有替第三方重新授權。固定 URL 與 SHA-256 見 `gemmini.lock.json`。

本專案的新分析式模型參考官方 defaultConfig 的架構參數；GB／LB／RF 層級為本專案另定的研究擴充，不是官方 defaultConfig 的原始組織。它沒有編譯或執行這份 RTL 快照，不代表 Berkeley 的驗證或背書。來源可信度、模型假設與未校準項目見 [Gemmini 硬體模型](docs/research/gemmini_hardware.md)。

完整 CNN 的 gather、GB／LB／RF 組織及 FP32 vector 是本專案研究擴充，非 Gemmini 官方 RTL。數值參照使用已安裝的 PyTorch／Torchvision／NumPy API；不下載或重授權 pretrained weights。

## Gemini HPCA 2024 實驗 artifact

`third_party/gemini-hpca2024/GEMINI_AE.zip` 是 [Zenodo 10207613](https://zenodo.org/records/10207613) 的論文封存版，SHA-256 為 `9726a34f501a4e8ea71569795b955121493b2ce9e0e3f251776b38bdb661be60`。著作權為 2023 Tsinghua ArchipLab，BSD 3-Clause 原始授權保留在該目錄 `LICENSE` 及壓縮檔內。`tools/gemini_native.py` 對解壓副本套用 C++ 可攜性修正並加入唯讀 mapping 匯出，原壓縮檔保持不變；這不代表作者認可或背書本專案。
