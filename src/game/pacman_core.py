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
import pygame, sys, random
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
                self.c = COLS-2; self.snap()
            elif self.c >= COLS-1 and passable(self.r, 1):
                self.c = 1; self.snap()

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

# ---------------- 主程式 ----------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Pac-Man v0.3.2 (ghost house fix)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    dots, power, home_list, G_area = build_world()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_dim = 4
    model = CnnDQN(action_dim, ROWS, COLS).to(device)
    model.load_state_dict(torch.load("models/full_dqn_best.pt", map_location=device))
    model.eval()

    # 玩家與鬼
    player = Player(23, 14, speed=2)
    player.dir = (0, 0)

    ghosts = [
        Ghost(home_list[0][0], home_list[0][1], RED,    home_list[0]),
        Ghost(home_list[1][0], home_list[1][1], PINK,   home_list[1]),
        Ghost(home_list[2][0], home_list[2][1], CYAN,   home_list[2]),
        Ghost(home_list[3][0], home_list[3][1], ORANGE, home_list[3]),
    ]
    # 初始分流方向
    seeds = [(-1,0), (1,0), (0,1), (0,-1)]
    for g, d in zip(ghosts, seeds):
        g.dir = d

    corners = [(1,1), (1,COLS-2), (ROWS-2,1), (ROWS-2,COLS-2)]  # 四角散開目標
    score = 0
    frightened_global = 0
    running = True
    ticks = 0

    while running:
        dt = clock.tick(FPS); ticks += 1
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
                    # ---------------- AI 控制 ----------------
        # 產生目前環境的 grid 狀態 (1, H, W)
        grid = np.zeros((ROWS, COLS), dtype=np.float32)

        for r, row in enumerate(LEVEL):
            for c, ch in enumerate(row):
                if ch == "#": grid[r, c] = 1

        for (r, c) in dots:  grid[r, c] = 2
        for (r, c) in power: grid[r, c] = 3

        # 鬼
        for g in ghosts:
            grid[g.r, g.c] = 5

        # 玩家
        grid[player.r, player.c] = 4

        state = torch.tensor(grid[None, None, :, :], dtype=torch.float32).to(device)

        with torch.no_grad():
            q_values = model(state)
            action = torch.argmax(q_values).item()

        # action → 方向
        if action == 0:  player.want = (-1, 0)
        if action == 1:  player.want = (1, 0)
        if action == 2:  player.want = (0,-1)
        if action == 3:  player.want = (0, 1)


        # ---- 玩家 ----
        if player.want != (0,0):
            player.try_turn(player.want)
        if player.forward_blocked():
            player.snap()
        else:
            player.step()
        player.warp()

        # 吃豆/大力丸
        if player.at_center():
            if (player.r, player.c) in dots:
                dots.remove((player.r, player.c)); score += 10
            if (player.r, player.c) in power:
                power.remove((player.r, player.c)); score += 50
                frightened_global = FPS * 6
                for g in ghosts:
                    g.set_frightened(frightened_global)

        # ---- 鬼 ----
        for i, g in enumerate(ghosts):
            # 模式切換（fright/eaten/respawn 除外）
            g.update_mode_scatter_chase(ticks)

            # 狀態對應速度
            g.speed = g.speed_now()

            # 目標/方向選擇（含屋內優先往上）
            target = (player.r, player.c)
            g.choose_dir(target, corners[i], G_area)

            # 前方牆處理
            if g.forward_blocked():
                g.snap()
            else:
                g.step()
            g.warp()

            # respawn：離開鬼屋後切回 chase
            if g.state == "respawn" and g.at_center() and not in_ghost_house(g.r, g.c, G_area):
                g.state = "chase"

            # 碰撞
            if (g.r, g.c) == (player.r, player.c):
                if g.state == "frightened":
                    score += 200
                    g.state = "eaten"   # 自行導航回家
                elif g.state not in ("eaten", "respawn"):
                    running = False     # 玩家死亡（簡化處理）

            # 計時器衰減
            g.tick_state()

        if frightened_global > 0:
            frightened_global -= 1

        # ---- 繪圖 ----
        screen.fill((0,0,0))
        # 牆
        for r,row in enumerate(LEVEL):
            for c,ch in enumerate(row):
                if ch == "#":
                    pygame.draw.rect(screen, BLUE, (c*TILE, r*TILE, TILE, TILE))
        # 豆與大力丸
        for (r,c) in dots:
            pygame.draw.circle(screen, WHITE, center_xy(r,c), 3)
        for (r,c) in power:
            pygame.draw.circle(screen, WHITE, center_xy(r,c), 6, 1)

        # 玩家
        pygame.draw.circle(screen, YELLOW, center_xy(player.r, player.c), TILE//2 - 2)

        # 鬼
        cols = [RED, PINK, CYAN, ORANGE]
        for idx, g in enumerate(ghosts):
            col = SCARED if g.state == "frightened" else cols[idx]
            pygame.draw.circle(screen, col, center_xy(g.r, g.c), TILE//2 - 2)

        # UI
        txt = font.render(f"Score: {score}", True, WHITE)
        screen.blit(txt, (10, 10))
        pygame.display.flip()

    pygame.quit(); sys.exit()

if __name__ == "__main__":
    main()
