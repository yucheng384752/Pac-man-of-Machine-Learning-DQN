# src/game/pacman_core.py
# v0.3.2
# Fix:
# - passable() 加入空白格 ' '，避免鬼屋（空格）被視為不可走，鬼卡住
# - 鬼在鬼屋時優先往上離開（chase/scatter/respawn），確保會陸續出屋
# - 保持：驚嚇模式、被吃→回家→respawn→出屋，格點制移動、左右隧道、7秒散開/追逐切換

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT)

import pygame
import random
import torch
import numpy as np

from src.agents.cnn_dqn import CnnDQN   # ← 你的 CNN 模型

# ---------------- 基本設定 ----------------
TILE = 20
FPS = 60

LEVEL = [
    "############################",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#o####.#####.##.#####.####o#",
    "#.####.#####.##.#####.####.#",
    "#..........................#",
    "#.####.##.########.##.####.#",
    "#.####.##.########.##.####.#",
    "#......##....##....##......#",
    "######.##### ## #####.######",
    "     #.##### ## #####.#     ",
    "######.##          ##.######",
    "      .## ###GG### ##.      ",
    "######.## #      # ##.######",
    "      .   #      #   .      ",
    "######.## #      # ##.######",
    "      .## ######## ##.      ",
    "######.##          ##.######",
    "#............##............#",
    "#.####.#####.##.#####.####.#",
    "#o..##................##..o#",
    "###.##.##.########.##.##.###",
    "#......##....##....##......#",
    "#.##########.##.##########.#",
    "#..........................#",
    "############################",
]

ROWS, COLS = len(LEVEL), len(LEVEL[0])
W, H = COLS * TILE, ROWS * TILE

# 顏色
BLUE   = (0, 60, 255)
YELLOW = (255, 255, 0)
WHITE  = (255, 255, 255)
RED    = (255,   0,   0)
PINK   = (255, 105, 180)
CYAN   = (0,   255, 255)
ORANGE = (255, 165,   0)
SCARED = (0,    0, 160)

# ---------------- 輔助 ----------------
def passable(r, c):
    if 0 <= r < ROWS and 0 <= c < COLS:
        ch = LEVEL[r][c]
        return ch in " .oG" or ch == " "   # ← 加入空白格
    return False

def center_xy(r, c):
    return c*TILE + TILE//2, r*TILE + TILE//2

def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def in_ghost_house(r, c, G_area):
    return (r, c) in G_area

# ---------------- 基底移動（格點制） ----------------
class GridMover:
    def __init__(self, r, c, speed=2):
        self.r, self.c = r, c
        self.x, self.y = center_xy(r, c)
        self.dir  = (0, 0)  # (dr, dc)
        self.want = (0, 0)
        self.speed = speed

    def at_center(self):
        cx, cy = center_xy(self.r, self.c)
        return abs(self.x - cx) <= 1 and abs(self.y - cy) <= 1

    def snap(self):
        self.x, self.y = center_xy(self.r, self.c)

    def step(self):
        self.x += self.dir[1] * self.speed
        self.y += self.dir[0] * self.speed
        self.r = round((self.y - TILE//2) / TILE)
        self.c = round((self.x - TILE//2) / TILE)

    def try_turn(self, d):
        if not self.at_center():
            return False
        nr, nc = self.r + d[0], self.c + d[1]
        if passable(nr, nc):
            self.dir = d
            return True
        return False

    def forward_blocked(self):
        if not self.at_center():
            return False
        nr, nc = self.r + self.dir[0], self.c + self.dir[1]
        return not passable(nr, nc)

    def warp(self):
        # 左右傳送通道（位於最外層可走格時）
        if self.at_center():
            if self.c <= 0 and passable(self.r, COLS-2):
                self.c = COLS-2
                self.snap()
            elif self.c >= COLS-1 and passable(self.r, 1):
                self.c = 1
                self.snap()

# ---------------- 玩家 ----------------
class Player(GridMover):
    pass

# ---------------- 鬼 ----------------
class Ghost(GridMover):
    # 狀態：chase / scatter / frightened / eaten / respawn
    def __init__(self, r, c, color, home_rc):
        super().__init__(r, c, speed=2)
        self.color = color
        self.state = "chase"
        self.fright_ticks = 0
        self.home = home_rc     # 回家座標（在 G 區任一格）
        self.respawn_ticks = 0  # 復活停留/出屋計時
        self.base_speed = 2

    def set_frightened(self, ticks):
        if self.state != "eaten":  # 回家路上不受影響
            self.state = "frightened"
            self.fright_ticks = ticks

    def update_mode_scatter_chase(self, ticks):
        if self.state in ("frightened", "eaten", "respawn"):
            return
        # 每 7 秒切換 scatter/chase
        if (ticks // (7*FPS)) % 2 == 0:
            self.state = "scatter"
        else:
            self.state = "chase"

    def speed_now(self):
        if self.state == "frightened":
            return max(1, self.base_speed - 1)  # 驚嚇稍慢
        if self.state == "eaten":
            return self.base_speed + 1          # 回家加速
        return self.base_speed

    def choose_dir(self, target_rc, corner_rc, G_area):
        if not self.at_center():
            return

        # 若在鬼屋且可出屋，優先往上
        if in_ghost_house(self.r, self.c, G_area) and self.state in ("chase", "scatter", "respawn"):
            if passable(self.r - 1, self.c):
                self.dir = (-1, 0)
                return

        # 可走方向
        options = []
        for d in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = self.r + d[0], self.c + d[1]
            if passable(nr, nc):
                options.append(d)
        if not options:
            return

        # 避免原路返回（若有其他選項）
        back = (-self.dir[0], -self.dir[1])
        if back in options and len(options) > 1:
            options.remove(back)

        # 目標點
        if self.state == "chase":
            goal = target_rc
        elif self.state == "scatter":
            goal = corner_rc
        elif self.state == "frightened":
            goal = (random.randint(0, ROWS-1), random.randint(0, COLS-1))
        elif self.state == "eaten":
            goal = self.home
        elif self.state == "respawn":
            # 往家上方離開
            up = (-1, 0)
            if up in options:
                self.dir = up
                return
            goal = (self.home[0]-3, self.home[1])  # 家上方幾格
        else:
            goal = target_rc

        best = min(options, key=lambda d: manhattan((self.r + d[0], self.c + d[1]), goal))
        self.dir = best

    def tick_state(self):
        # 處理計時器與狀態切換
        if self.state == "frightened":
            self.fright_ticks -= 1
            if self.fright_ticks <= 0:
                self.state = "chase"
        elif self.state == "eaten":
            # 抵達 home → 進入 respawn
            if (self.r, self.c) == self.home and self.at_center():
                self.state = "respawn"
                self.respawn_ticks = FPS  # 在屋內停留 1 秒
        elif self.state == "respawn":
            self.respawn_ticks -= 1
            # 真正切回 chase 的時機放在主循環：離開 G 區後

# ---------------- 世界 ----------------
def build_world():
    dots = set()
    power = set()
    ghost_homes = []
    G_area = set()
    for r, row in enumerate(LEVEL):
        for c, ch in enumerate(row):
            if ch == ".":
                dots.add((r, c))
            elif ch == "o":
                power.add((r, c))
            elif ch == "G":
                ghost_homes.append((r, c))
                G_area.add((r, c))
            elif ch == " ":
                # 空白也視為可走，但不放豆子
                pass
    # 至少要有一個 home
    if not ghost_homes:
        ghost_homes = [(14, 14)]
    while len(ghost_homes) < 4:
        ghost_homes.append(ghost_homes[0])
    return dots, power, ghost_homes, G_area

# ---------------- 主程式（改成直接使用 PacmanCoreEnv） ----------------
def main():
    global W, H
    import pygame
    import torch
    import numpy as np
    from src.envs.pacman_env_from_core import PacmanCoreEnv  # ✅ 用訓練環境

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Pac-Man RL (CnnDQN from PacmanCoreEnv)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    # === 建立環境（跟訓練用的一樣） ===
    env = PacmanCoreEnv(max_steps=2000)
    state = env.reset()   # shape: (C, H, W)
    C, H, W = state.shape

    # === 建立 CNN-DQN 並載入模型（跟 train_full_dqn.py 一致） ===
    device = torch.device("cpu")
    action_dim = env.action_space_n

    model = CnnDQN(
        action_dim=action_dim,
        in_channels=C,
        rows=H,
        cols=W
    ).to(device)

    # 這裡你可以選要載入 best 還是 last
    model_path = "models/full_dqn_last.pt"
    if not os.path.exists(model_path):
        print("Model not found:", model_path)
        print("請先訓練模型（scripts/train_full_dqn.py）再執行。")
        pygame.quit()
        sys.exit()

    print("Loading model from:", model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    running = True

    while running:
        # 跟訓練時一樣，環境每 step 就是一個 frame
        clock.tick(FPS)

        # 處理關閉事件
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        # === 1) 用當前 state 推論動作 ===
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)  # (1, C, H, W)
            q_values = model(x)
            action = int(torch.argmax(q_values, dim=1).item())

        # === 2) 丟進環境 step（完全跟訓練時一樣） ===
        next_state, reward, done, info = env.step(action)
        state = next_state

        # === 3) 畫畫面（用 env 裡的資料） ===
        screen.fill((0, 0, 0))

        # 牆
        for r, row in enumerate(LEVEL):
            for c, ch in enumerate(row):
                if ch == "#":
                    pygame.draw.rect(screen, BLUE, (c*TILE, r*TILE, TILE, TILE))

        # 小豆
        for (r, c) in env.dots:
            pygame.draw.circle(screen, WHITE, center_xy(r, c), 3)

        # 大力丸
        for (r, c) in env.power:
            pygame.draw.circle(screen, WHITE, center_xy(r, c), 6, 1)

        # 玩家（從 env.player 拿）
        pygame.draw.circle(
            screen,
            YELLOW,
            center_xy(env.player.r, env.player.c),
            TILE//2 - 2
        )

        # 鬼（從 env.ghosts 拿）
        ghost_colors = [RED, PINK, CYAN, ORANGE]
        for idx, g in enumerate(env.ghosts):
            col = SCARED if g.state == "frightened" else ghost_colors[idx]
            pygame.draw.circle(screen, col, center_xy(g.r, g.c), TILE//2 - 2)

        # UI：顯示 score / ticks / reason
        info_text = f"Score: {info.get('score', 0)}"
        if "reason" in info:
            info_text += f" ({info['reason']})"

        txt = font.render(info_text, True, WHITE)
        screen.blit(txt, (10, 10))

        pygame.display.flip()

        # 若 episod 結束，就退出（你也可以改成 reset 再來一局）
        if done:
            print("Episode finished. info =", info)
            running = False

    pygame.quit()
    sys.exit()



if __name__ == "__main__":
    main()
