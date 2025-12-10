# src/envs/pacman_env_from_core.py

import numpy as np

from src.game.pacman_core import (
    LEVEL, ROWS, COLS,
    Player, Ghost,
    build_world, center_xy, passable, in_ghost_house,
    FPS,
)

print(">>> Using PacmanCoreEnv from:", __file__)


class PacmanCoreEnv:
    """
    強化學習環境（完整同步 pacman_core）
    - 同步地圖、玩家、鬼的行為
    - 提供 reset() / step() / state
    """

    def __init__(self, max_steps=2000):
        self.max_steps = max_steps
        self.ticks = 0
        self.score = 0
        self.action_space_n = 4  # 上下左右

    # ---------------- reset ----------------
    def reset(self):
        self.ticks = 0
        self.score = 0

        # world
        self.dots, self.power, self.home_list, self.G_area = build_world()

        # 玩家與鬼（與 pacman_core 相同）
        self.player = Player(23, 14, speed=2)
        self.player.dir = (0, 0)

        self.ghosts = [
            Ghost(self.home_list[0][0], self.home_list[0][1], (255, 0, 0), self.home_list[0]),
            Ghost(self.home_list[1][0], self.home_list[1][1], (255,105,180), self.home_list[1]),
            Ghost(self.home_list[2][0], self.home_list[2][1], (0,255,255), self.home_list[2]),
            Ghost(self.home_list[3][0], self.home_list[3][1], (255,165,0), self.home_list[3]),
        ]

        seeds = [(-1,0), (1,0), (0,1), (0,-1)]
        for g, d in zip(self.ghosts, seeds):
            g.dir = d

        self.frightened_global = 0
        self.corners = [(1,1), (1,COLS-2), (ROWS-2,1), (ROWS-2,COLS-2)]

        return self._get_state()

    # ---------------- 狀態 ----------------
    def _get_state(self):
        grid = np.zeros((ROWS, COLS), dtype=np.float32)

        for r, row in enumerate(LEVEL):
            for c, ch in enumerate(row):
                if ch == "#":
                    grid[r, c] = 1

        for (r, c) in self.dots:
            grid[r, c] = 2
        for (r, c) in self.power:
            grid[r, c] = 3

        for g in self.ghosts:
            grid[g.r, g.c] = 5

        grid[self.player.r, self.player.c] = 4

        return grid[np.newaxis, :, :]

    # ---------------- 判斷死路 ----------------
    def _is_dead_end(self, r, c):
        open_paths = 0
        for d in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + d[0], c + d[1]
            if passable(nr, nc):
                open_paths += 1
        return open_paths == 1

    # ---------------- step（核心） ----------------
    def step(self, action):

        self.ticks += 1
        reward = 0.0
        done = False
        info = {}

        # --------- 記錄舊狀態 ---------
        old_r, old_c = self.player.r, self.player.c
        old_dot_count = len(self.dots)
        old_min_ghost_dist = min(abs(old_r - g.r) + abs(old_c - g.c) for g in self.ghosts)

        # --------- 1) 玩家移動 ---------
        if action == 0: self.player.want = (-1,0)
        elif action == 1: self.player.want = (1,0)
        elif action == 2: self.player.want = (0,-1)
        elif action == 3: self.player.want = (0,1)

        if self.player.want != (0,0):
            self.player.try_turn(self.player.want)

        if self.player.forward_blocked():
            self.player.snap()
        else:
            self.player.step()

        self.player.warp()

        # --------- 2) 每步懲罰 ---------
        reward -= 0.05

        # --------- 3) 吃豆 / 大力丸 ---------
        if self.player.at_center():
            pos = (self.player.r, self.player.c)

            if pos in self.dots:
                self.dots.remove(pos)
                self.score += 10
                reward += 3    # RL reward

            if pos in self.power:
                self.power.remove(pos)
                self.score += 50
                reward += 10
                self.frightened_global = FPS * 6
                for g in self.ghosts:
                    g.set_frightened(self.frightened_global)

        # --------- 4) 鬼行為 ---------
        for i, g in enumerate(self.ghosts):

            g.update_mode_scatter_chase(self.ticks)
            g.speed = g.speed_now()
            target = (self.player.r, self.player.c)

            g.choose_dir(target, self.corners[i], self.G_area)

            if g.forward_blocked(): g.snap()
            else: g.step()

            g.warp()

            if g.state == "respawn" and g.at_center() and not in_ghost_house(g.r, g.c, self.G_area):
                g.state = "chase"

            # ---- 碰撞 ----
            if (g.r, g.c) == (self.player.r, self.player.c):

                if g.state == "frightened":
                    self.score += 200
                    reward += 50
                    g.state = "eaten"

                elif g.state not in ("eaten","respawn"):
                    reward -= 300
                    done = True
                    info["reason"] = "dead"
                    info["score"] = self.score
                    return self._get_state(), reward, done, info

            g.tick_state()

        if self.frightened_global > 0:
            self.frightened_global -= 1

        # --------- 5) 靠近鬼懲罰 ---------
        for g in self.ghosts:
            dist = abs(self.player.r - g.r) + abs(self.player.c - g.c)
            if g.state != "frightened":
                if dist <= 2: reward -= 3
                elif dist <= 4: reward -= 1

        # --------- 6) dot 減少獎勵 ---------
        if len(self.dots) < old_dot_count:
            reward += 2

        # --------- 7) 死路懲罰 ---------
        if self._is_dead_end(self.player.r, self.player.c):
            reward -= 2

        # --------- 8) 遠離鬼 = 小獎勵 ---------
        new_min_dist = min(abs(self.player.r - g.r) + abs(self.player.c - g.c) for g in self.ghosts)
        if new_min_dist > old_min_ghost_dist:
            reward += 0.3

        # --------- 結束條件 ---------
        if len(self.dots) == 0:
            reward += 1500
            self.score += 1000
            done = True
            info["reason"] = "all_clear"

        if self.ticks >= self.max_steps and not done:
            reward -= 50
            done = True
            info["reason"] = "max_steps"

        info["score"] = self.score
        return self._get_state(), reward, done, info
