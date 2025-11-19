# src/envs/pacman_env_full.py

import numpy as np
from src.game.pacman_core import (
    LEVEL, build_world, ROWS, COLS,
    Player, Ghost
)

# 動作：0=上, 1=下, 2=左, 3=右
ACTION_TO_DIR = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}

class PacmanEnvFull:
    """
    使用 pacman_core 的完整大地圖與鬼 AI，
    封裝成 RL 用的環境：
      - reset() -> state
      - step(action) -> (state, reward, done, info)
      - state shape: (1, ROWS, COLS)，給 CNN 用
    """

    def __init__(self, max_steps=2000):
        self.max_steps = max_steps
        self.action_space_n = 4
        self.reset()

    def reset(self):
        self.ticks = 0
        self.steps = 0

        # 從 pacman_core 的工具建世界
        self.dots, self.power, self.homes, self.G_area = build_world()

        # 玩家（座標和 pacman_core 主程式一樣）
        self.player = Player(23, 14, speed=2)
        self.player.dir = (0, 0)
        self.player.want = (0, 0)

        # 四隻鬼
        self.ghosts = [
            Ghost(self.homes[0][0], self.homes[0][1], None, self.homes[0]),
            Ghost(self.homes[1][0], self.homes[1][1], None, self.homes[1]),
            Ghost(self.homes[2][0], self.homes[2][1], None, self.homes[2]),
            Ghost(self.homes[3][0], self.homes[3][1], None, self.homes[3]),
        ]

        # 初始隨便給方向（避免卡死）
        seeds = [(-1,0), (1,0), (0,1), (0,-1)]
        for g, d in zip(self.ghosts, seeds):
            g.dir = d

        return self.get_state()

    def get_state(self):
        """
        把整張地圖轉成一個 2D grid，再加一個 channel 維度給 CNN：
        shape = (1, ROWS, COLS)
        編碼：
          0 = 空
          1 = 牆
          2 = 豆子
          3 = 大力丸
          4 = Pacman
          5 = 鬼（一般）
          6 = 鬼（驚嚇）
        """
        grid = np.zeros((ROWS, COLS), dtype=np.float32)

        # 牆
        for r, row in enumerate(LEVEL):
            for c, ch in enumerate(row):
                if ch == "#":
                    grid[r, c] = 1

        # 豆子/大力丸
        for (r, c) in self.dots:
            grid[r, c] = 2
        for (r, c) in self.power:
            grid[r, c] = 3

        # 鬼
        for g in self.ghosts:
            if g.state == "frightened":
                grid[g.r, g.c] = 6
            else:
                grid[g.r, g.c] = 5

        # Pacman 要最後蓋上去
        grid[self.player.r, self.player.c] = 4

        # 加 channel 維度
        return grid[None, :, :]   # shape: (1, H, W)

    def step(self, action: int):
        """
        單一步驟：
          - 用 action 控制 Pacman
          - 移動鬼
          - 計算 reward / done
        """
        self.ticks += 1
        self.steps += 1

        # 控制玩家
        if action in ACTION_TO_DIR:
            self.player.want = ACTION_TO_DIR[action]
            self.player.try_turn(self.player.want)
        # 前方如果不是牆就一步一步走
        if not self.player.forward_blocked():
            self.player.step()
        self.player.warp()

        reward = -0.01  # 每步小懲罰，鼓勵快一點通關
        done = False

        # 吃豆子
        if self.player.at_center():
            if (self.player.r, self.player.c) in self.dots:
                self.dots.remove((self.player.r, self.player.c))
                reward += 1.0
            # 吃大力丸
            if (self.player.r, self.player.c) in self.power:
                self.power.remove((self.player.r, self.player.c))
                reward += 5.0
                # 所有鬼進入驚嚇狀態一段時間（這裡設 5 秒，FPS=60）
                frightened_ticks = 60 * 5
                for g in self.ghosts:
                    g.set_frightened(frightened_ticks)

        # 鬼的行為
        for g in self.ghosts:
            # 根據 pacman_core 中的邏輯，更新模式與速度
            g.update_mode_scatter_chase(self.ticks)
            g.speed = g.speed_now()

            # 目標點：簡單用玩家位置（你之後可以照原作再細分每隻鬼的目標）
            target = (self.player.r, self.player.c)
            # corner_rc 暫時用 (1,1) 之類，因為真正 scatter corner 已經在 pacman_core main 裡定義
            g.choose_dir(target_rc=target, corner_rc=(1, 1), G_area=self.G_area)
            if not g.forward_blocked():
                g.step()
            g.warp()
            g.tick_state()

            # 碰撞判定
            if (g.r, g.c) == (self.player.r, self.player.c):
                if g.state == "frightened":
                    # 玩家吃鬼
                    reward += 10.0
                    g.state = "eaten"
                elif g.state not in ("eaten", "respawn"):
                    # 玩家被鬼吃掉
                    reward -= 10.0
                    done = True

        # 所有豆子被吃光 → 通關
        if len(self.dots) == 0:
            reward += 20.0
            done = True

        # 避免無限遊戲：步數上限
        if self.steps >= self.max_steps:
            done = True

        return self.get_state(), reward, done, {}
