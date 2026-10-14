# 目前研究進度

版本 0.9.0。正式版本仍為 **REINFORCE，沒有 PPO、沒有 Critic**；PPO 只保留為[比較模型](comparison_models.md)。廠商 PPA 限制、五種偏好、compute-only mesh、Agent 自行選 1–16 顆、Python 本機 GUI 與搜尋結果圖保留。

七模型 0.8.0 的 60 回合正式執行已完成，總耗時 10:05:39；原始評分／reward／狀態／計費核對通過，但全部未達 100 ms，凍結策略前後測只有 1 改善、3 持平、3 退步。見[實際基準、平台與修正](model_suite_20261014.md)。0.9.0 預設模型別任務具已量測可行憑證，前後各 10 evaluation seeds；中途測試與等預算比較明確選用，不預設增加四倍預算。新完整長訓練尚未執行。

已畫出[現行硬體與資料流](hardware_model.md)。目前 H-only retile、spill-all 與未校準 PPA 尚不足以稱為完整 DSE；擴展 W/C/reduction/dataflow、224×224 及加速前，先釐清微架構、精度與記憶體重用。

## 已實作

reset 不評估 PPA、不建立原生後端。起始 observation 只有結構、需求及未知量 mask。Agent 一次提出一個完整 action，完整環境計算 PPA、reward 與新 observation；下一個 action 使用新狀態與上一個實際 reward。schema 5 支援未知 PPA 與 last reward。

正式策略根據已完成的 trajectory 做一次 REINFORCE reward-to-go 更新；沒有 value predictor、PPO ratio／clip 或多個 PPO epochs。PPA 固定尺度來自需求，J=M/(1+M)，初始最佳檔案為空。顆數不依 PPA 先被排除。GUI 已移除每回合評估／互動次數欄位；完整聯合 action 立即送環境，起始不評估。正式聯合範圍預設最多 2 個相依 Tile。

## 驗證範圍

自動測試核對 reset／Actor 零評估、一步一次環境入口、reward 與 PPA 回到 observation、快速 mask 與逐候選編譯的 action 支援及順序一致、聯合動作中間零評估／一次完整回饋、正式 REINFORCE 梯度、Actor 更新與凍結不更新、PPO 模型隔離、checkpoint 不能跨算法、全部動作失敗不輸出假最佳設計。最新 160 個自動測試通過；流程 audit 位於 `evidence/`。

現行小型 smoke 與 GUI 測試只證明流程可執行；尚未建立 PPO 或 REINFORCE 的比較優勢，也未證明收斂。0.7.0 PPO 的完整 CNN 短搜尋與 H1/H2/H3 pilot 已移到比較模型紀錄，不能充當新正式模型的成果。

## 未完成的研究

新正式模型已完成 10 variants × 3 seeds 的小型配對 pilot：訓練 469／凍結 236 次新評估，152 次多 Tile 聯合動作；全部訓練確有更新、凍結不更新。結果尚不支持 H1／H2／H3，見[實際结果](hypothesis_results.md)。仍需多工作負載、更多 seeds 和有辨識度的 barrier cases。

物理係數仍未校準，功耗為推論平均值，面積只含 compute dies；現在是分析式環境，尚未接實際晶片量測或 cycle-accurate 大型模擬器。Windows 有程式與 CI 設定，尚未於本次本機執行驗證。

## 本次 GUI 實測

ResNet-18、64×64、1 回合、預設內部 64 次完整互動、scope=2、PPA 限制 1 W／1000 ms／16 mm²。總耗時 54.99 秒：動作生成合計 5.16 秒（平均 0.081 秒）、環境 step 合計 24.38 秒、單次 REINFORCE 更新 16.47 秒，其餘為載入／數值驗證／輸出。63 次新評估＋1 次快取，23 次多 Tile 聯合修改；選出 2 顆達標設計，網頁直接顯示 272 筆實測資料傳輸。此測試不能外推 224×224、所有 CNN 或最長訓練時間。

GUI 已移除每回合評估／動作次數欄位；完整 action 生成後立即交給環境。內部仍有限訓練預算以避免無限搜尋，與「組成 action」的成本不同。

## 推送後 CI

commit 8c33e2b 的 Ubuntu／macOS workflow 全部成功；Windows 在原生來源雜湊核對失敗。已重現 Git 的 CRLF 換行轉換並新增來源 LF 規則與回歸測試，本機 135 個測試通過。修正後 Windows 雲端 CI 尚待推送驗證，見[CI 紀錄](ci_verification.md)。

## GUI 啟動修正

Mac／Windows 啟動器改為自動選空閒連接埠，並以專案身分重用已執行的 GUI。Mac 實測避開 8765／8766 後使用 8767，第二次啟動正常重用；142 個完整本機測試通過。此修正只影響 GUI 啟動，不改變 RL／PPA 契約。

## 循序模型訓練與紀錄精簡

已實作終端入口 tools/train_model_suite.py，預設依序七模型各 60 回合、輸入 64；PPA／演算法／scope／更新規則沿用正式設定，停用提前停止以跑滿研究預算。新增相同預算凍結策略前後測，不更新權重並還原 RNG。輸出同一批次資料夾，包含實測進度／時間、逐回合曲線、PPA／達標判定與 HTML 報告；來源一致才能接續完整回合。

GUI／CLI 預設壓縮的精簡 SQLite；逐 action feedback 保留，最終完整選中設計保留，所有候選帳本只在明確 full 模式輸出。測試確認兩種紀錄下動作、PPA、reward、policy 及 RNG 一致。0.8.0 的 60 回合資料已保存；0.9.0 新條件的完整訓練待使用者執行，不以 smoke 宣稱收斂或成效。

依使用者要求，在驗證後刪除舊的大型訓練 raw（約 4.55 GiB）；歷史小型證據保留，舊 raw manifest 的路徑已不再存在。清理紀錄見 evidence/training_output_cleanup.json，流程驗證見 evidence/model_suite_validation.json。

終端進度已改為固定單行刷新，按中文字寬對齊並依視窗寬度縮排；完成／失敗才留下摘要行。重新導向輸出不含控制碼。此修改只涉及顯示，不改訓練、PPA 或檔案紀錄頻率。
