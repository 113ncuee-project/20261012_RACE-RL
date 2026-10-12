# Compute-only mesh 與外部記憶體

實體 mesh 只有運算 chiplet。按運算顆數的最接近正方形整數分解排列，row-major 填入 c0…cN-1，只連上下左右相鄰格點。16 顆為 4×4；質數顆數會形成 1×N。

外部服務 dram0／dram1 分別接到 c0／cN-1。它們沒有 die 幾何、不計入 physical chiplet count／compute die area、不進入原生 RapidChiplet chiplet list，也不作路由中繼。神經 state 使用獨立 memory-interface relation 表示這些連接。

| 設定 | 當前探索值 |
| --- | --- |
| 單顆 compute 幾何 | 1×1 mm，間距 1 mm |
| 單顆 compute capacity | 32 MiB |
| 單顆 compute rate | 1e9 modeled operations/s |
| Local／mesh-link bandwidth | 1e9 B/s |
| 每個外部服務 capacity／effective bandwidth | 1 GiB／1e9 B/s |
| Compute 與外部服務 idle power | 各 0.001 W |

這些都未經製程／實測校準。32 MiB 與 1 mm² 未作 SRAM density 耦合，外部介面也不重現某個 HBM controller／PHY。area 只代表 compute die 加總，不能當作完整系統或封裝面積。

資料從外部服務以 effective bandwidth 讀取／寫回，經 gateway-to-compute 的 mesh path 傳輸，再計入 local memory service。即使 consumer 正在 gateway、mesh hops=0，external/local service 與外部動態能量仍為正。外部 static power 另計一次，不會因從 native mesh 移除而免費。

有效外部讀寫 coefficient 吸收未細分的網格外存取／介面成本；不另捏造外部 die 位置、連線長度或 HBM latency。實體 mesh 的 bandwidth／delay／area／idle power 使用已核對的原生 RapidChiplet APIs。容量檢查涵蓋 compute 與外部服務 endpoints 的 lifetimes／peaks。

目前為 spill-all、serial-node-tile admission；所有中間結果寫回，跨 Tile shared reads 關閉，沒有跨層常駐、packet arbitration 或 multicast tree 模擬。

商用設計另有運算、I/O／控制器與 HBM 的封裝層次，例如 [AMD MI300](https://rocm.docs.amd.com/en/develop/reference/gpu-arch/mi300.html)、[NVIDIA Blackwell Ultra](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/) 與 [Intel Ponte Vecchio](https://download.intel.com/newsroom/2021/client-computing/intel-architecture-day-2021-fact-sheet.pdf)。本模型只是分開 compute mesh 與外部 memory service 的研究抽象，沒有宣稱是這些產品的 replica。
