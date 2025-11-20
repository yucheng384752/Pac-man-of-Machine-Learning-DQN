import os, sys, random, time
import numpy as np
import torch
import pygame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
print("Python Path Added:", ROOT)

from envs.pacman_env_from_core import PacmanEnvFull
from src.agents.cnn_dqn import CnnDQN
from src.game.pacman_core import (
    LEVEL, ROWS, COLS, TILE,
    BLUE, YELLOW, WHITE, RED, PINK, CYAN, ORANGE, SCARED,
    center_xy
)

def draw_world(screen, env, font):
    screen.fill((0, 0, 0))

    # 牆壁
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

    # 人物（Pac-Man）
    pygame.draw.circle(screen, YELLOW, center_xy(env.player.r, env.player.c), TILE//2 - 2)

    # 鬼
    ghost_colors = [RED, PINK, CYAN, ORANGE]
    for idx, g in enumerate(env.ghosts):
        col = SCARED if g.state == "frightened" else ghost_colors[idx]
        pygame.draw.circle(screen, col, center_xy(g.r, g.c), TILE//2 - 2)

    # 分數 or Debug
    txt = font.render(f"Step: {env.steps}", True, WHITE)
    screen.blit(txt, (10, 10))

    pygame.display.flip()


def main():
    model_path = "models/full_dqn_last.pt"

    if not os.path.exists(model_path):
        print("Model not found:", model_path)
        print("請先訓練模型再執行。")
        return

    print("Loading trained model:", model_path)

    env = PacmanEnvFull(max_steps=3000)
    state = env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, H, W = state.shape

    # 建 CNN 網路
    q = CnnDQN(env.action_space_n, rows=H, cols=W).to(device)
    q.load_state_dict(torch.load(model_path, map_location=device))
    q.eval()

    print("Model loaded. Starting simulation...")

    pygame.init()
    screen = pygame.display.set_mode((COLS*TILE, ROWS*TILE))
    pygame.display.set_caption("Pac-Man AI (Double DQN)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 20)

    running = True

    while running:
        clock.tick(20)   # 20 FPS 視覺比較好看

        # handle quit
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 推論 action
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action = int(q(x).argmax(dim=1).item())

        next_state, reward, done, info = env.step(action)
        state = next_state

        draw_world(screen, env, font)

        if done:
            print("Episode finished.")
            running = False

    pygame.quit()


if __name__ == "__main__":
    main()
