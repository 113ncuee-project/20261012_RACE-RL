# PPA 評分對接、公平性與研究意義核對

2026-10-12，0.7。結論：**目前評分適合明確定義的廠商限制及偏好任務，程式與受控 pilot 對接通過；現有實驗仍不足以支持正式核心假說、跨需求泛化或產業 PPA 結論。**

## 對接與數值核對

從 30 筆配對 trial 的 720 次原始 oracle 紀錄，重算 90 個訓練／凍結回合的輸出。原始平均功耗等於能量除以 makespan；限制、權重、固定尺度、加權成本、達標狀態及最佳 merit 全部核對通過。另有單位一致性、邊界等於上限、自訂零權重、未達標起始可改善、暫時超標可探索、無達標結果命名，以及同一 policy 的需求特徵測試。

五種偏好對同一批 187 個唯一候選重新評分，選出三種不同設計。這證明目前候選的排序可隨權重變動，**不是五種偏好各自訓練的 RL 成效**。原始候選池來自 balanced 搜尋，不能把這次事後重評當作公平的多偏好演算法競賽。

自訂權重含零時，分數可能相同而有一個設計在全部 PPA 指標皆不差、至少一項更好。已加入精確同分下的 Pareto dominance 判定，保留較好的候選，不改變使用者主要權重。有限探索仍不保證找到未走訪的優解。

## 評分定義是否適當

`C = Σw_i × metric_i/limit_i` 表示相對於廠商上限的資源使用或延遲比例。所有單位消去，任務內尺度固定、不隨搜尋更新，也不以測試集最優值重新正規化。更换上限會同时改變可行集合及偏好的交換尺度；這是目前契約的明確選擇。

「PPA 平均」表示三個**上限使用比例**各占 1/3，不代表三項在現有候選中的數值範圍或分數貢獻一定相等。若面積比例高、延遲比例很低，平均權重仍可能以面積差異為主要排序來源。不能為讓 RL 結果更好看而事後調整尺度。加權和是可解釋的偏好模型，但權重解釋取決於指標與尺度，且不保證生成完整非凸 Pareto 前緣；這與 [Marler 與 Arora 的權重分析](https://link.springer.com/article/10.1007/s00158-009-0460-7) 所討論的限制一致。

60/20/20 是明示的介面預設，不是校準出的廠商效用模型。真實使用前應以代表性設計確認預設是否符合決策者偏好；自訂可覆寫。Power 目前為平均功耗，Performance 为單次推論延遲，Area 為 compute die 總和，不能當作峰值功耗、流水線吞吐或完整封裝面積。平均功耗需與其測量區間、負載及系統邊界一起解讀；[SPEC 的方法文件](https://www.spec.org/power/docs/SPEC-Power_and_Performance_Methodology.pdf) 強調這些條件，但本專案並未執行 SPEC 或矽片量測。

## Reward 與廠商目標

Reward 使用 feasibility-first merit M，讓達標設計始終優先。達標集合內 `M=C/(1+C)` 嚴格維持 C 的設計排序。Dense reward 的整回合總和為 `log(M_initial/M_best_final)`；terminal reward 亦為相同最終總和。Potential 取終止值零、gamma=1，加入 potential difference 後整回合只差固定起始常數。這提供同一初始任務下 H3 的目標一致性，其形式可參考 [Ng 等人的 potential shaping 理論](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)。

這是逐個設計的單調排序，不代表隨機 policy 的期望 C 與期望 log reward 完全等價。不同成功率、風險分布或初始任務不能僅靠 normalized merit 比較。PPO 的近似更新、entropy regularization、有限預算及壓縮 observation 也不因 potential 形式而保證最優。

## 實驗公平性

現有 10 variants × 3 seeds 使用同一 W、起始 H/M、限制、尺度、權重、每回合 8 次評估上限及同樣 initial 計費規則。四捨五入或最好 seed 沒有用於比較。所有組使用 5,242 個參數、相同 PPO 設定，訓練確有更新、凍結搜尋沒有更新，training 與 frozen costs 分開記錄。

H2 的 coupled／independent support 不同，兩組參數相同與 oracle budget 相同不代表每步控制器成本相同。紀錄保留 action support、requests、結構生成時間與 wall time。尚無完整聯合環境 Greedy／SA 基線，不能宣稱 RL 已優於這些方法。

## 實際發現與科學意義

目前合成需求為 0.1 W、1 ms、16 mm²、balanced。187 個唯一候選全部達標，可行率為 100%。初始成本貢獻為功耗 32.64%、延遲 13.15%、面積 54.20%；全部達標候選中面積貢獻約 49.77–80.28%，延遲約 2.33–17.09%。這個任務偏好較小顆數具有契約上的合理性，但對模型切分、mapping、達標能力及 H1/H2/H3 的辨識力不足。

本次 full／state ablations／reward variants 的三個 frozen outcomes 均為 1；independent 僅一個 seed 小幅改善。沒有證據支持 H1／H2／H3。三 seeds、單一 Toy 與短訓練只能提供開發診斷；少量 runs 的不確定性不能忽略，見 [Agarwal 等人的 RL 評估研究](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html)。

下一輪正式研究需事前固定多組需求與獨立 workloads：寬鬆全可行、只有部分可行、確定無解；至少包含運算與通訊瓶頸差異。訓練／validation／test 的 workload 與需求分離，係數只在 validation 選擇，報可行率、首次達標評估數、原始 PPA、可行集合中的 C、相對基準 gap、分開的訓練／搜尋 costs 及不確定性。平均結果不能讓失敗 trial 因 null 成本被排除。這些是待執行的驗收條件，不能把目前資料改標為已完成。

重算證據見 `evidence/core_hypotheses/scoring_audit.json` 與 `audit.json`；工具為 `tools/audit_ppa_scoring.py`。
