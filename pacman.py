# src/game/pacman_core.py
# v0.1.0
# 可執行 Pac-Man 遊戲，玩家可移動、鬼會追逐、可吃豆子

import pygame
import random
import sys
import math

# ----------------------------
# 基本設定
# ----------------------------
pygame.init()
FPS = 60
TILE = 20
ROWS, COLS = 31, 28
WIDTH, HEIGHT = COLS * TILE, ROWS * TILE
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pac-Man (Clean Fixed Version)")
clock = pygame.time.Clock()

# 顏色
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
PINK = (255, 105, 180)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)

# ----------------------------
# 地圖
# ----------------------------
# 用簡化的迷宮（28x31 格）
level_map = [
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

# ----------------------------
# 物件定義
# ----------------------------
class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dir = pygame.Vector2(0, 0)
        self.want_dir = pygame.Vector2(0, 0)
        self.speed = 2

    def update(self, walls):
        # 嘗試轉向
        if not self.collides(self.want_dir, walls):
            self.dir = self.want_dir

        # 移動
        new_pos = pygame.Vector2(self.x, self.y) + self.dir * self.speed
        if not self.collides(self.dir, walls):
            self.x, self.y = new_pos

        # 通道穿越
        if self.x < -TILE:
            self.x = WIDTH
        elif self.x > WIDTH:
            self.x = -TILE

    def collides(self, direction, walls):
        next_rect = pygame.Rect(self.x + direction.x * self.speed,
                                self.y + direction.y * self.speed,
                                TILE, TILE)
        return any(next_rect.colliderect(w) for w in walls)

    def draw(self, surf):
        pygame.draw.circle(surf, YELLOW, (int(self.x + TILE/2), int(self.y + TILE/2)), TILE//2 - 2)

class Ghost:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.speed = 1.5
        self.dir = random.choice([(1,0),(-1,0),(0,1),(0,-1)])

    def update(self, player, walls):
        # 鬼 AI：朝玩家方向前進
        px, py = player.x, player.y
        dx, dy = px - self.x, py - self.y
        if abs(dx) > abs(dy):
            new_dir = (math.copysign(1, dx), 0)
        else:
            new_dir = (0, math.copysign(1, dy))

        # 隨機微調避免卡死
        if random.random() < 0.02:
            new_dir = random.choice([(1,0),(-1,0),(0,1),(0,-1)])

        next_rect = pygame.Rect(self.x + new_dir[0]*self.speed,
                                self.y + new_dir[1]*self.speed,
                                TILE, TILE)
        if not any(next_rect.colliderect(w) for w in walls):
            self.dir = new_dir

        self.x += self.dir[0]*self.speed
        self.y += self.dir[1]*self.speed

        # 通道穿越
        if self.x < -TILE:
            self.x = WIDTH
        elif self.x > WIDTH:
            self.x = -TILE

    def draw(self, surf):
        pygame.draw.circle(surf, self.color, (int(self.x+TILE/2), int(self.y+TILE/2)), TILE//2 - 1)

# ----------------------------
# 建立地圖資料
# ----------------------------
walls = []
dots = []

for row, line in enumerate(level_map):
    for col, ch in enumerate(line):
        if ch == "#":
            walls.append(pygame.Rect(col*TILE, row*TILE, TILE, TILE))
        elif ch == ".":
            dots.append(pygame.Rect(col*TILE+TILE/3, row*TILE+TILE/3, TILE/3, TILE/3))

# 玩家與鬼初始位置
player = Player(14*TILE, 23*TILE)
ghosts = [
    Ghost(13*TILE, 14*TILE, RED),
    Ghost(14*TILE, 14*TILE, PINK),
    Ghost(13*TILE, 15*TILE, CYAN),
    Ghost(14*TILE, 15*TILE, ORANGE),
]

font = pygame.font.SysFont("Consolas", 24)
score = 0
running = True
game_over = False

# ----------------------------
# 主遊戲迴圈
# ----------------------------
while running:
    clock.tick(FPS)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                player.want_dir = pygame.Vector2(-1, 0)
            elif event.key == pygame.K_RIGHT:
                player.want_dir = pygame.Vector2(1, 0)
            elif event.key == pygame.K_UP:
                player.want_dir = pygame.Vector2(0, -1)
            elif event.key == pygame.K_DOWN:
                player.want_dir = pygame.Vector2(0, 1)
            elif event.key == pygame.K_ESCAPE:
                running = False
                sys.exit()

    if not game_over:
        player.update(walls)
        for g in ghosts:
            g.update(player, walls)

        # 吃豆子
        for d in dots[:]:
            if pygame.Rect(player.x, player.y, TILE, TILE).colliderect(d):
                dots.remove(d)
                score += 10

        # 被鬼抓
        for g in ghosts:
            if pygame.Rect(player.x, player.y, TILE, TILE).colliderect(pygame.Rect(g.x, g.y, TILE, TILE)):
                game_over = True
                break

        # 全部豆子吃完
        if not dots:
            game_over = True

    # --- 繪圖 ---
    screen.fill(BLACK)
    for w in walls:
        pygame.draw.rect(screen, BLUE, w)
    for d in dots:
        pygame.draw.rect(screen, WHITE, d)
    player.draw(screen)
    for g in ghosts:
        g.draw(screen)

    # UI
    if game_over:
        msg = "YOU WIN!" if not dots else "GAME OVER"
        txt = font.render(f"{msg}  Score: {score}", True, WHITE)
        screen.blit(txt, (WIDTH/2 - txt.get_width()/2, HEIGHT/2 - 20))
    else:
        txt = font.render(f"Score: {score}", True, WHITE)
        screen.blit(txt, (10, 10))

    pygame.display.flip()

pygame.quit()
