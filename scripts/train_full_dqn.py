# scripts/train_full_dqn.py

import os, sys, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
print("Python Path Added:", ROOT)

from src.envs.pacman_env_from_core import PacmanCoreEnv
from src.agents.cnn_dqn import CnnDQN
from src.rl.replay_buffer import ReplayBuffer


class Config:
    gamma = 0.99
    lr = 1e-4
    batch_size = 64
    buffer_size = 200_000
    start_learning = 10_000
    target_update_every = 2000
    max_steps = 500_000

    eps_start = 1.0
    eps_end = 0.01
    eps_decay_steps = 150_000

    save_best = "models/full_dqn_best.pt"
    save_last = "models/full_dqn_last.pt"


def epsilon_by_step(step, cfg: Config):
    return float(cfg.eps_end + (cfg.eps_start - cfg.eps_end) * math.exp(-step / cfg.eps_decay_steps))


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    writer = SummaryWriter("logs/pacman_full")

    cfg = Config()
    env = PacmanCoreEnv(max_steps=2000)

    device = torch.device("cpu")
    print("Using device:", device)

    s0 = env.reset()
    _, H, W = s0.shape
    action_dim = env.action_space_n

    q = CnnDQN(action_dim, rows=H, cols=W).to(device)
    target_q = CnnDQN(action_dim, rows=H, cols=W).to(device)
    target_q.load_state_dict(q.state_dict())
    target_q.eval()

    opt = optim.Adam(q.parameters(), lr=cfg.lr)
    criterion = nn.SmoothL1Loss()

    buf = ReplayBuffer(cfg.buffer_size)

    global_step = 0
    best_return = -1e9

    state = s0
    ep_return = 0.0
    ep_len = 0
    episode = 0

    while global_step < cfg.max_steps:
        global_step += 1
        ep_len += 1

        eps = epsilon_by_step(global_step, cfg)
        if random.random() < eps:
            action = random.randint(0, action_dim - 1)
        else:
            with torch.no_grad():
                x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action = int(q(x).argmax(dim=1).item())

        next_state, reward, done, _ = env.step(action)

        buf.push(state, action, reward, next_state, done)
        state = next_state
        ep_return += reward

        writer.add_scalar("epsilon", eps, global_step)

        if len(buf) >= cfg.start_learning:
            s, a, r, ns, d = buf.sample(cfg.batch_size)

            s  = torch.tensor(s,  dtype=torch.float32, device=device)
            ns = torch.tensor(ns, dtype=torch.float32, device=device)
            a  = torch.tensor(a,  dtype=torch.int64,   device=device).unsqueeze(1)
            r  = torch.tensor(r,  dtype=torch.float32, device=device).unsqueeze(1)
            d  = torch.tensor(d,  dtype=torch.float32, device=device).unsqueeze(1)

            q_sa = q(s).gather(1, a)

            with torch.no_grad():
                a_star = q(ns).argmax(dim=1, keepdim=True)
                q_target = target_q(ns).gather(1, a_star)
                y = r + (1 - d) * cfg.gamma * q_target

            loss = criterion(q_sa, y)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()

            writer.add_scalar("loss", loss.item(), global_step)

            if global_step % cfg.target_update_every == 0:
                target_q.load_state_dict(q.state_dict())

        if done:
            episode += 1
            writer.add_scalar("episode_return", ep_return, episode)

            if ep_return > best_return:
                best_return = ep_return
                torch.save(q.state_dict(), cfg.save_best)

            if episode % 10 == 0:
                print(f"[Step {global_step}] Episode {episode} | Return {ep_return:6.2f} | Eps {eps:.3f} | Buffer {len(buf)}")

            state = env.reset()
            ep_return = 0.0
            ep_len = 0

    torch.save(q.state_dict(), cfg.save_last)
    print("Training Finished. Best return:", best_return)

    writer.close()


if __name__ == "__main__":
    main()
