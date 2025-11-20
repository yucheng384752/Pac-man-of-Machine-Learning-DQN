# src/envs/pacman_env_from_core.py

import numpy as np
import torch
from src.game.pacman_core import LEVEL, ROWS, COLS, Player, Ghost, build_world, center_xy, passable

FPS = 60

class PacmanCoreEnv:
    def __init__(self, max_steps=2000):
        self.max_steps = max_steps
        self.ticks = 0
        self.score = 0

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

        # 初始方向
        seeds = [(-1,0), (1,0), (0,1), (0,-1)]
        for g, d in zip(self.ghosts, seeds):
            g.dir = d

        self.frightened_global = 0
        self.corners = [(1,1),(1,COLS-2),(ROWS-2,1),(ROWS-2,COLS-2)]

        return self._get_state()

    def _get_state(self):
        """ 建立 C×H×W 的 grid（跟 pacman_core 完全相同的狀態） """
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

        reward = 0

        # 吃豆
        if self.player.at_center():
            if (self.player.r, self.player.c) in self.dots:
                self.dots.remove((self.player.r, self.player.c))
                reward += 10

            if (self.player.r, self.player.c) in self.power:
                self.power.remove((self.player.r, self.player.c))
                reward += 50
                self.frightened_global = FPS*6
                for g in self.ghosts:
                    g.set_frightened(self.frightened_global)

        # ---- Ghosts ----
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
            g.tick_state()

            # 碰撞
            if (g.r, g.c) == (self.player.r, self.player.c):
                if g.state == "frightened":
                    reward += 200
                    g.state = "eaten"
                else:
                    return self._get_state(), reward - 200, True, {}

        # --- end conditions ---
        done = False
        if len(self.dots) == 0:
            reward += 1000
            done = True

        if self.ticks >= self.max_steps:
            done = True

        return self._get_state(), reward, done, {}

    def render(self):
        # 你可以用 pygame 或 print 簡單版
        pass
