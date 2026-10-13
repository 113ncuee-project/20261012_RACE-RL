# 評分對接與科學意義

本文件對應 0.8.0 正式 REINFORCE。舊 PPO 評分 audit 屬於[比較模型歷史資料](comparison_models.md)，不得視為目前正式版本的驗證結果。

三項 metrics 依序為平均功耗、單次推論延遲、compute die 面積。三項使用廠商限制值正規化；完整 PPA 必須由所選 action 執行後的整個環境計算，不預估候選，不依切分／單層的局部分數篩選架構。

C=Σw_i metric_i/limit_i。達標 M=C/(1+C)；超標 M=2+Σexcess+C/(1+C)，保證任何達標方案優先。零權重可能产生同分，tie 用 Pareto dominance 避免保留被支配方案。物理合法但 PPA 超標可作為探索 current；best 是已測最佳進度。

訓練使用 J=M/(1+M)，對所有有效 M 嚴格單調，所以不改變任何已測設計排序。未測檔案 sentinel J=1，不是假 PPA；第一次 action 才建立 best。Dense reward 的 episodic 總和為 log(1/J_final)。研究 API 缺限制時使用固定 SI 尺度 1 W、1 s、1 mm²；不同任務不因此自動公平，正式需求輸入三項皆完整。

等權表示相對限制的權重相同，不保證不同項對搜索變化的貢獻相同。限制尺度、可達範圍、物理係數、風險分布与成功率都要報告。單調 reward 不保證隨機策略的期望成本等價，也不保證 RL 收斂或全域最優。REINFORCE 與 PPO 優劣尚未被實驗判定。

目前係數未校準，功耗是 accounted energy/makespan 的平均值，面積不含封裝／外部 DRAM／I/O。科學比較需同一環境、任務、action 空間、預先宣告 seed、完整步數與評估／訓練成本，以及獨立凍結搜尋；不能挑最佳 seed。PPO 的額外 Critic 和四次更新須揭露。

來源：[加權多目標方法](https://link.springer.com/article/10.1007/s00158-009-0460-7)、[potential reward shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)、[少量 RL seeds 的不確定性](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)。

本次新正式 pilot 已重算 705 筆原始評估、90 個輸出檔案群與 208 個唯一物理有效設計，audit 通過。此 workload 全部觀測設計達標，因此不能測試尋找可行解的能力；同一候選池五偏好仍選同一方案，不能宣稱偏好辨識或跨需求泛化已成立。後驗重算不是另外五套訓練。
