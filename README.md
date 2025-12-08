# 專案介紹
使用DQN訓練模型操控pacman，在遊戲中吃豆、閃避鬼取得高分


## 專案功能
使用pygame建立pacman遊戲包含:pacman移動、鬼、牆壁地圖、點數、遊戲結束DQN深度強化學習Agent

## 效能
模型效能CNN DQN，用CNN去處理影像狀態(像素畫面)，並用DQN做強化學習決策遊戲環境效能:pacman環境使用pygame，FPS大約在30~60
訓練效率:300~1000 episodes(局數)，500K~2M steps(步數)

在3分鐘內能躲避鬼並吃完所有的點點

在3分鐘內取得最的分數(吃鬼會加分)
## 介面
外部： 遊戲世界，包含整張地圖、牆壁、點點、能量球、四個鬼、pacman位置

內部：DQN Agent，包含CNN DQN、Agent(更新規則) 
## 驗收


# 系統分析
<img width="1269" height="711" alt="image" src="https://github.com/user-attachments/assets/111ea678-efd7-4190-99fa-4077a5f1352c" />

- 1:遊戲環境模組(pacman_core.py)
    1. 地圖解析:載入pacman地圖、牆、點、能量球
    2. pacman控制:更新位置、碰撞、吃豆子
    3. 鬼(ghost)AI FSM:chase/scatter/frightened/eaten/respawn
    4. 事件判定:被鬼吃、吃鬼、吃能量球、得分
    5. 遊戲重置:關卡清零、重新初始化角色位置
    6. reward 設計:用於強化學習訓練的回報

- 2:強化學習包裝(pacman_env_from_core.py)
    1. 將pacman_core包裝成RL:像gym的rest()、step()
    2. CNN輸入前處理:resize, grayscale, normalization
    3. 鬼(ghost)AI FSM:chase/scatter/frightened/eaten/respawn
    4. 狀態堆疊（optional）:多 frame state
    5. 遊戲重置:關卡清零、重新初始化角色位置
    6. 介面統一:讓 Agent 可以直接讀取 state
    
- 3:CNN Q-Network模型
    1. cnn_dqn.py（Q-Network）
        1. 使用 CNN 萃取遊戲畫面特徵
        2. FC層輸出4個動作的Q-values
        3. forward()用於推論
- 4:DQN 訓練模型(train_full_dqn.py)
    1. (DQN邏輯)
        1. ε-greedy 探索:隨訓練逐漸降低 ε
        2. 記憶回放（Replay Buffer）:儲存 state transition
        3. loss 計算:MSELoss
        4. optimize:更新 Q-network
        5. target network 同步:fixed Q-target
    2. 模型儲存:bast model、last model
    3. tensorBoard紀錄:loss、reword、epsilon
    4. 儲存至Replay Buffer
- 5:Replay Buffer(replay_buffer.py)
    1. 儲存transition
    2. 控制最大容量
    3. 隨機抽樣batch
    4. 協助訓練穩定收斂


