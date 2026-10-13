# 0.7.0 PPO 歷史比較資料

這些檔案保留原始數據與來源 hash，僅為使用者指定保留的 PPO 比較模型紀錄。此處文中「目前／正式／schema 4／初始評估占額度」指 0.7.0 歷史版本，不代表現行正式版本。

現行正式版本為無 PPO／Critic 的 REINFORCE，reset 零評估，見 ../../training_process.md。PPO 的現行可執行比較模組共享新的零預評估環境，見 ../../comparison_models.md。

舊 pilot 不包含 PPO 對 REINFORCE 的比較；舊回合會預評估起始設計，不能直接與新的 action-triggered 契約做公平數字比較。歷史 GUI 圖和短 CNN 結果不是新正式模型的成果。尚未完成的舊長搜尋已停止；raw 檔案位於忽略的 experiments/runs/comparison_models/ppo_0_7_0/。
