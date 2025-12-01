# src/envs/pacman_env_from_core.py

import numpy as np

from src.game.pacman_core import (
    LEVEL, ROWS, COLS,
    Player, Ghost,
    build_world, center_xy, passable, in_ghost_house,
    FPS,  # 直接用 pacman_core 裡的 FPS
)

print(">>> Using PacmanCoreEnv from:", __file__)


class PacmanCoreEnv:
    """
    強化學習用 Pac-Man 環境：
    - 遊戲規則完全沿用 pacman_core.py（Player, Ghost, build_world, FPS）
    - 多通道 state：7×H×W
    - reward 加上引導：吃豆、接近豆、遠離鬼、探索新格子、死亡懲罰
    """

    NUM_CHANNELS = 7  # 0~6 共 7 個 channel

    def __init__(self, max_steps=2000):
        self.max_steps = max_steps
        self.ticks = 0
        self.score = 0
        self.action_space_n = 4  # 上下左右

        # 給 reward shaping 用
        self.visited = np.zeros((ROWS, COLS), dtype=bool)
        self.prev_dot_dist = None
        self.prev_ghost_dist = None

    # ---------------- reset：開新局 ----------------
    def reset(self):
        self.ticks = 0
        self.score = 0

        self.dots, self.power, self.home_list, self.G_area = build_world()

        self.player = Player(23, 14, speed=2)
        self.player.dir = (0, 0)

        self.ghosts = [
            Ghost(self.home_list[0][0], self.home_list[0][1], (255, 0, 0),    self.home_list[0]),
            Ghost(self.home_list[1][0], self.home_list[1][1], (255,105,180), self.home_list[1]),
            Ghost(self.home_list[2][0], self.home_list[2][1], (0,255,255),   self.home_list[2]),
            Ghost(self.home_list[3][0], self.home_list[3][1], (255,165,0),   self.home_list[3]),
        ]

        seeds = [(-1,0), (1,0), (0,1), (0,-1)]
        for g, d in zip(self.ghosts, seeds):
            g.dir = d

        self.frightened_global = 0
        self.corners = [(1,1), (1,COLS-2), (ROWS-2,1), (ROWS-2,COLS-2)]

        # reward shaping 狀態重新初始化
        self.visited[:, :] = False
        self.prev_dot_dist = None
        self.prev_ghost_dist = None

        return self._get_state()

    # ---------------- helper：距離計算 ----------------
    def _nearest_dot_dist(self, r, c):
        if not self.dots:
            return 0.0
        return min(abs(r - dr) + abs(c - dc) for (dr, dc) in self.dots)

    def _nearest_ghost_dist(self, r, c):
        if not self.ghosts:
            return 99.0
        return min(abs(r - g.r) + abs(c - g.c) for g in self.ghosts)

    # ---------------- state：多通道 grid ----------------
    def _get_state(self):
        """
        回傳 shape = (C, H, W)，C=7：

        ch0：牆
        ch1：小豆
        ch2：大力丸
        ch3：玩家
        ch4：鬼（正常狀態：chase / scatter）
        ch5：鬼（frightened）
        ch6：全域 frightened 強度（同一值鋪滿整張圖，0~1）
        """
        C = self.NUM_CHANNELS
        grid = np.zeros((C, ROWS, COLS), dtype=np.float32)

        # ch0: 牆
        for r, row in enumerate(LEVEL):
            for c, ch in enumerate(row):
                if ch == "#":
                    grid[0, r, c] = 1.0

        # ch1: 小豆
        for (r, c) in self.dots:
            grid[1, r, c] = 1.0

        # ch2: 大力丸
        for (r, c) in self.power:
            grid[2, r, c] = 1.0

        # ch3: 玩家
        grid[3, self.player.r, self.player.c] = 1.0

        # ch4~5: 鬼
        for g in self.ghosts:
            if g.state == "frightened":
                grid[5, g.r, g.c] = 1.0
            else:
                grid[4, g.r, g.c] = 1.0

        # ch6: frightened_global normalized（0~1，鋪滿全圖）
        max_fright = FPS * 6  # 跟大力丸時間一致
        if max_fright > 0:
            v = max(0.0, min(1.0, self.frightened_global / max_fright))
            grid[6, :, :] = v

        return grid

    # ---------------- step：模擬一個 frame ----------------
    def step(self, action):
        """
        action：0上 1下 2左 3右
        流程對齊 pacman_core.main：
        - 玩家移動
        - 吃豆 / 大力丸
        - 鬼更新（模式、速度、方向、warp、respawn、tick_state）
        - 碰撞
        - frightened_global 倒數
        - reward shaping
        """
        self.ticks += 1
        reward = 0.0
        done = False
        info = {"score": self.score}

        # --- 事前距離（給 shaping 用） ---
        prev_dot_dist = self._nearest_dot_dist(self.player.r, self.player.c)
        prev_ghost_dist = self._nearest_ghost_dist(self.player.r, self.player.c)

        # ---- 玩家移動 ----
        if action == 0:
            self.player.want = (-1, 0)
        elif action == 1:
            self.player.want = (1, 0)
        elif action == 2:
            self.player.want = (0, -1)
        elif action == 3:
            self.player.want = (0, 1)

        if self.player.want != (0, 0):
            self.player.try_turn(self.player.want)
        if self.player.forward_blocked():
            self.player.snap()
        else:
            self.player.step()
        self.player.warp()

        # ---- 吃豆 / 大力丸 ----
        if self.player.at_center():
            pos = (self.player.r, self.player.c)

            # 探索新格子
            if not self.visited[self.player.r, self.player.c]:
                self.visited[self.player.r, self.player.c] = True
                reward += 0.05  # 鼓勵走新路

            if pos in self.dots:
                self.dots.remove(pos)
                reward += 10.0
                self.score += 10

            if pos in self.power:
                self.power.remove(pos)
                reward += 50.0
                self.score += 50
                self.frightened_global = FPS * 6
                for g in self.ghosts:
                    g.set_frightened(self.frightened_global)

        # ---- 鬼 ----
        for i, g in enumerate(self.ghosts):
            g.update_mode_scatter_chase(self.ticks)
            g.speed = g.speed_now()

            target = (self.player.r, self.player.c)
            g.choose_dir(target, self.corners[i], self.G_area)

            if g.forward_blocked():
                g.snap()
            else:
                g.step()
            g.warp()

            if g.state == "respawn" and g.at_center() and not in_ghost_house(g.r, g.c, self.G_area):
                g.state = "chase"

            # 碰撞
            if (g.r, g.c) == (self.player.r, self.player.c):
                if g.state == "frightened":
                    reward += 200.0
                    self.score += 200
                    g.state = "eaten"
                elif g.state not in ("eaten", "respawn"):
                    reward -= 200.0
                    self.score += reward
                    done = True
                    info["reason"] = "dead"
                    return self._get_state(), reward, done, info

            g.tick_state()

        # frightened_global 倒數
        if self.frightened_global > 0:
            self.frightened_global -= 1

        # ---- reward shaping：接近豆 / 遠離鬼 / 時間懲罰 ----

        # 1) 接近最近豆子
        if self.dots:
            dot_dist = self._nearest_dot_dist(self.player.r, self.player.c)
            if prev_dot_dist is not None:
                delta = prev_dot_dist - dot_dist
                if delta > 0:
                    # 更接近豆
                    reward += 0.2
                elif delta < 0:
                    # 離豆更遠一點，給小懲罰
                    reward -= 0.05

        # 2) 遠離危險鬼 / 在驚嚇狀態時靠近鬼
        ghost_dist = self._nearest_ghost_dist(self.player.r, self.player.c)
        if prev_ghost_dist is not None:
            delta_g = ghost_dist - prev_ghost_dist

            if self.frightened_global <= 0:
                # 正常狀態：離危險鬼太近 → 懲罰
                if ghost_dist < 4:
                    reward -= 0.5
                # 遠離鬼一點 → 小獎勵
                if delta_g > 0 and ghost_dist < 8:
                    reward += 0.1
                elif delta_g < 0 and ghost_dist < 8:
                    reward -= 0.1
            else:
                # 驚嚇狀態：靠近鬼 → 小獎勵，鼓勵去吃鬼
                if delta_g < 0 and ghost_dist < 8:
                    reward += 0.2

        # 3) 時間懲罰：避免無聊亂晃
        reward -= 0.05  # 每步小懲罰，鼓勵快點解決

        # ---- 結束條件 ----
        if len(self.dots) == 0:
            reward += 1000.0
            self.score += 1000
            done = True
            info["reason"] = "all_clear"

        if self.ticks >= self.max_steps and not done:
            done = True
            info["reason"] = "max_steps"

        info["score"] = self.score
        return self._get_state(), reward, done, info

    def render(self):
        # 訓練時通常不畫畫
        pass
