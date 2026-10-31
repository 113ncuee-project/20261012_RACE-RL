# Chiplet-Gym 共同 CNN 環境

目標是將 [Chiplet-Gym 論文](https://arxiv.org/html/2406.00858)的搜尋方法放在與 RACE-RL 相同的 CNN、硬體、限制和 PPA 評估器下。原版 [程式碼](https://github.com/KFM135/chiplet-optimizer) revision `6e327348310f927c997efd86b8d3ec2b8986e1f7` 的 PPO 使用 Stable-Baselines3 2.1.0，SA 來源採用步長 10、溫度分子 400 的接受規則。這份 repository 未附 `requirements.txt` 或授權檔；本專案沒有重新散布其原始碼或模型。

## 共同環境的改動

`chiplet_gym_common_adapted` 使用相同版本 Stable-Baselines3 的 PPO、兩層 64 單元 MLP、episode 長度 2、learning rate 0.0003、10 epochs、gamma 0.99、GAE 0.95、clip 0.2、entropy 0.1、value coefficient 0.5；SA 保留作者程式的候選擾動與接受規則。有限的比較預算平均分給一個重新訓練的 PPO 和一個 SA，所有訓練、無效與快取提案都計數。

原始論文處理 64／128 晶粒、HBM 與封裝組合，輸入／輸出與目前 CNN 算子無一對一關係；原版模型權重不能直接使用。此處 action 改為 chiplet 數量、CNN anchor 與最多兩層的 profile／駐留／layout／目標；target 對選定晶粒數取 modulo，若只選到一層，第二組動作不使用。狀態為十項 RACE 全域／需求特徵，reward、評估器與限制均同 RACE，PPO rollout／batch 受共同預算限制而縮短。原文的多個預訓練 agents 和多次 SA 未執行。不能把它稱為原版完整系統，也不能以它的結果推論原論文全部設計能力。

原始完整系統的獨立執行入口是 `tools/chiplet_gym_native.py`，它使用作者原本的成本模型，不列入共同 PPA 比較。為使作者程式在目前機器啟動，入口在執行副本修正模型目錄名、SA 子程序轉向與 SA 輸出解析，並記錄修改；演算法本身未變。

## 依賴

目前這台機器已在 `.cache/chiplet-gym-deps` 安裝依賴；不影響專案主要 `.venv`。若快取被清除，在專案根目錄執行：

```sh
.venv/bin/python -m pip install --target .cache/chiplet-gym-deps --no-deps \
  'gym==0.26.2' 'gymnasium==0.29.1' 'stable-baselines3==2.1.0' \
  'shimmy==0.2.1' 'farama-notifications==0.0.6' 'gym_notices==0.1.0' \
  'cloudpickle==3.1.2' 'pandas==2.2.3' 'python-dateutil==2.9.0.post0' \
  'pytz==2025.2' 'tzdata==2025.2' 'six==1.17.0' 'numpy==1.26.4'
```

共同 CNN 程式使用專案既有 NumPy；原版獨立執行入口把上述快取放在 Python 路徑之前。實際版本在試次紀錄中保存。這些版本是為目前專案環境準備的實驗設定，不是作者提供的鎖定清單。
