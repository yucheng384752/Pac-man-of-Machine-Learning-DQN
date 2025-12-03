# src/envs/pacman_env_from_core.py

import numpy as np

from src.game.pacman_core import (
    LEVEL, ROWS, COLS,
    Player, Ghost,
    build_world, center_xy, passable, in_ghost_house,
    FPS,  # 直接用 pacman_core 裡的 FPS，保持同步
)

print(">>> Using PacmanCoreEnv from:", __file__)



class PacmanCoreEnv:
    """
    強化學習用的 Pac-Man 環境：
    - 規則、地圖、玩家、鬼的 AI 全部來自 pacman_core.py
    - 只額外提供：reset() / step() / state / reward / done / info
    """

    def __init__(self, max_steps=2000):
        self.max_steps = max_steps
        self.ticks = 0          # 對應 pacman_core.main() 裡的 ticks
        self.score = 0
        self.action_space_n = 4  # 上下左右

    # ---------------- reset：等同開新局 ----------------
    def reset(self):
        self.ticks = 0
        self.score = 0

        # world（完全沿用 pacman_core.build_world）
        self.dots, self.power, self.home_list, self.G_area = build_world()

        # 玩家與鬼（起始位置＆方向與 pacman_core.main 完全一致）
        self.player = Player(23, 14, speed=2)
        self.player.dir = (0, 0)

        self.ghosts = [
            Ghost(self.home_list[0][0], self.home_list[0][1], (255, 0, 0),    self.home_list[0]),
            Ghost(self.home_list[1][0], self.home_list[1][1], (255,105,180), self.home_list[1]),
            Ghost(self.home_list[2][0], self.home_list[2][1], (0,255,255),   self.home_list[2]),
            Ghost(self.home_list[3][0], self.home_list[3][1], (255,165,0),   self.home_list[3]),
        ]

        # 初始分流方向（完全照 pacman_core.main）
        seeds = [(-1,0), (1,0), (0,1), (0,-1)]
        for g, d in zip(self.ghosts, seeds):
            g.dir = d

        # 全域驚嚇時間（對齊 pacman_core 的 frightened_global）
        self.frightened_global = 0

        # 四角散開目標（與 pacman_core.main 裡 corners 一致）
        self.corners = [(1,1), (1,COLS-2), (ROWS-2,1), (ROWS-2,COLS-2)]

        return self._get_state()

    # ---------------- state：照 pacman_core.main 畫 grid ----------------
    def _get_state(self):
        """
        建立 1×H×W 的 grid（與 pacman_core.main() 建 grid 的邏輯同步）
        編碼：
        - 0: 空
        - 1: 牆
        - 2: 小豆
        - 3: 大力丸
        - 4: 玩家
        - 5: 鬼（只標記一種數值，不分顏色）
        """
        grid = np.zeros((ROWS, COLS), dtype=np.float32)

        # 牆
        for r, row in enumerate(LEVEL):
            for c, ch in enumerate(row):
                if ch == "#":
                    grid[r, c] = 1

        # 豆 / 大力丸
        for (r, c) in self.dots:
            grid[r, c] = 2
        for (r, c) in self.power:
            grid[r, c] = 3

        # 鬼
        for g in self.ghosts:
            grid[g.r, g.c] = 5

        # 玩家
        grid[self.player.r, self.player.c] = 4

        # shape = (1, H, W) → 給 CnnDQN(rows=H, cols=W)
        return grid[np.newaxis, :, :]

    # ---------------- step：模擬一個「遊戲 frame」 ----------------
    def step(self, action):
        """
        action：0上 1下 2左 3右
        整個流程盡量對齊 pacman_core.main() 的 while 迴圈：
        - 玩家移動
        - 吃豆 / 吃大力丸
        - 鬼更新（模式、速度、方向、warp、respawn、tick_state）
        - 碰撞判定
        - frightened_global 倒數
        """
        self.ticks += 1
        reward = 0.0
        # 存活獎勵：每活一步 +0.2（鼓勵不要死）
        reward += 0.2

        # ---- 玩家移動（與 pacman_core.main 相同）----
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

        # ---- 吃豆 / 大力丸（完全同 main，只是多了 reward 與 self.score）----
        if self.player.at_center():
            pos = (self.player.r, self.player.c)
            if pos in self.dots:
                self.dots.remove(pos)
                reward += 10
                self.score += 10
            if pos in self.power:
                self.power.remove(pos)
                reward += 50
                self.score += 50
                self.frightened_global = FPS * 6
                for g in self.ghosts:
                    g.set_frightened(self.frightened_global)

        # ---- 鬼（對齊 pacman_core.main 的順序）----
        done = False
        info = {"score": self.score}

        for i, g in enumerate(self.ghosts):
            # 模式切換（fright/eaten/respawn 除外）
            g.update_mode_scatter_chase(self.ticks)

            # 狀態對應速度
            g.speed = g.speed_now()

            # 目標/方向選擇（含屋內優先往上）
            target = (self.player.r, self.player.c)
            g.choose_dir(target, self.corners[i], self.G_area)

            # 前方牆處理
            if g.forward_blocked():
                g.snap()
            else:
                g.step()
            g.warp()

            # respawn：離開鬼屋後切回 chase（與 main 同）
            if g.state == "respawn" and g.at_center() and not in_ghost_house(g.r, g.c, self.G_area):
                g.state = "chase"

            # 碰撞（與 main 邏輯一致，只是加上 reward / done）
            if (g.r, g.c) == (self.player.r, self.player.c):
                if g.state == "frightened":
                    # 吃鬼
                    reward += 200
                    self.score += 200
                    g.state = "eaten"    # 自行導航回家
                elif g.state not in ("eaten", "respawn"):
                    # 玩家死亡（遊戲版是 running=False，這裡改成 done=True）
                    reward -= 200
                    self.score += reward
                    done = True
                    info["reason"] = "dead"
                    # 死亡當下就結束本步驟，直接回傳（對應 while 跳出）
                    return self._get_state(), reward, done, info

            # 計時器衰減
            g.tick_state()

        # frightened_global 倒數（與 main 相同）
        if self.frightened_global > 0:
            self.frightened_global -= 1

        # ---- 結束條件（episode done）----
        # 全清
        if len(self.dots) == 0:
            reward += 1000
            self.score += 1000
            done = True
            info["reason"] = "all_clear"

        # 超過最大步數（避免卡超久）
        if self.ticks >= self.max_steps and not done:
            done = True
            info["reason"] = "max_steps"

        info["score"] = self.score
        return self._get_state(), reward, done, info

    def render(self):
        # 目前留空，訓練時通常不畫畫
        # 如果你要 debug，也可以在這裡 print() 一點資訊
        pass
