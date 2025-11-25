# src/envs/pacman_env_from_core.py

import numpy as np
import torch
from src.game.pacman_core import (
    LEVEL, ROWS, COLS,
    Player, Ghost,
    build_world, center_xy, passable, in_ghost_house
)
print(">>> Using PacmanCoreEnv from:", __file__)

FPS = 60

class PacmanCoreEnv:
    def __init__(self, max_steps=2000):
        self.max_steps = max_steps
        self.ticks = 0
        self.score = 0
        self.action_space_n = 4

    def reset(self):
        self.ticks = 0
        self.score = 0

        # world
        self.dots, self.power, self.home_list, self.G_area = build_world()

        # 玩家與鬼
        self.player = Player(23, 14, speed=2)
        self.player.dir = (0, 0)

        self.ghosts = [
            Ghost(self.home_list[0][0], self.home_list[0][1], (255,0,0),    self.home_list[0]),
            Ghost(self.home_list[1][0], self.home_list[1][1], (255,105,180),self.home_list[1]),
            Ghost(self.home_list[2][0], self.home_list[2][1], (0,255,255),  self.home_list[2]),
            Ghost(self.home_list[3][0], self.home_list[3][1], (255,165,0),  self.home_list[3]),
        ]

        # 初始方向（跟 pacman_core 一樣）
        seeds = [(-1,0), (1,0), (0,1), (0,-1)]
        for g, d in zip(self.ghosts, seeds):
            g.dir = d

        self.frightened_global = 0
        self.corners = [(1,1),(1,COLS-2),(ROWS-2,1),(ROWS-2,COLS-2)]

        return self._get_state()

    def _get_state(self):
        """ 建立 1×H×W 的 grid（跟 pacman_core main() 相同編碼） """
        grid = np.zeros((ROWS, COLS), dtype=np.float32)

        for r, row in enumerate(LEVEL):
            for c, ch in enumerate(row):
                if ch == "#":
                    grid[r, c] = 1

        for (r, c) in self.dots:  grid[r, c] = 2
        for (r, c) in self.power: grid[r, c] = 3

        # 鬼
        for g in self.ghosts:
            grid[g.r, g.c] = 5

        # 玩家
        grid[self.player.r, self.player.c] = 4

        # shape: (1, ROWS, COLS)
        return grid[np.newaxis, :, :]

    def step(self, action):
        """ action：0上 1下 2左 3右 """
        self.ticks += 1
        reward = 0

        # ---- Player move ----
        if action == 0:  self.player.want = (-1,0)
        if action == 1:  self.player.want = (1,0)
        if action == 2:  self.player.want = (0,-1)
        if action == 3:  self.player.want = (0,1)

        if self.player.want != (0,0):
            self.player.try_turn(self.player.want)
        if self.player.forward_blocked():
            self.player.snap()
        else:
            self.player.step()
        self.player.warp()

        # 吃豆 / 大力丸
        if self.player.at_center():
            if (self.player.r, self.player.c) in self.dots:
                self.dots.remove((self.player.r, self.player.c))
                reward += 10
                self.score += 10

            if (self.player.r, self.player.c) in self.power:
                self.power.remove((self.player.r, self.player.c))
                reward += 50
                self.score += 50
                self.frightened_global = FPS * 6
                for g in self.ghosts:
                    g.set_frightened(self.frightened_global)

        # ---- Ghosts ----
        for i, g in enumerate(self.ghosts):

            # 模式切換（frightened/eaten/respawn 除外）
            g.update_mode_scatter_chase(self.ticks)

            # 狀態影響速度
            g.speed = g.speed_now()
            target = (self.player.r, self.player.c)

            # 選方向（含鬼屋內優先往上，邏輯在 Ghost.choose_dir 裡）
            g.choose_dir(target, self.corners[i], self.G_area)

            # 前方牆處理
            if g.forward_blocked():
                g.snap()
            else:
                g.step()

            g.warp()

            # respawn：離開鬼屋後切回 chase（對齊 pacman_core.main）
            if g.state == "respawn" and g.at_center() and not in_ghost_house(g.r, g.c, self.G_area):
                g.state = "chase"

            # 碰撞
            if (g.r, g.c) == (self.player.r, self.player.c):
                if g.state == "frightened":
                    # 吃鬼
                    reward += 200
                    self.score += 200
                    g.state = "eaten"
                elif g.state not in ("eaten", "respawn"):
                    # 只有在正常狀態被撞才死亡（跟遊戲一致）
                    reward -= 200
                    self.score += reward
                    return self._get_state(), reward, True, {"score": self.score, "reason": "dead"}

            # 鬼自己的驚嚇 / eaten / respawn 計時
            g.tick_state()

        # global frightened 計時（目前只有同步用，不影響邏輯）
        if self.frightened_global > 0:
            self.frightened_global -= 1

        # --- end conditions ---
        done = False
        info = {"score": self.score}

        # 全清
        if len(self.dots) == 0:
            reward += 1000
            self.score += 1000
            done = True
            info["reason"] = "all_clear"

        # 超過步數
        if self.ticks >= self.max_steps:
            done = True
            info["reason"] = "max_steps"

        return self._get_state(), reward, done, info

    def render(self):
        # 你可以用 pygame 畫，也可以 print grid 做 debug
        pass
