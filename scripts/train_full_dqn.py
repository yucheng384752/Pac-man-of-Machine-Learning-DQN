# scripts/train_full_dqn.py

import os, sys, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# -------------------------------------------------
# 確保從任何目錄執行都能正確導入模組與模型路徑
# -------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
print("Python Path Added:", ROOT)

MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

from src.envs.pacman_env_from_core import PacmanCoreEnv
from src.agents.cnn_dqn import CnnDQN
from src.rl.replay_buffer import ReplayBuffer


# -------------------------------------------------
# 設定參數
# -------------------------------------------------
class Config:
    gamma = 0.99
    lr = 1e-4
    batch_size = 64
    buffer_size = 200_000
    start_learning = 10_000
    target_update_every = 2000
    max_steps = 1500_000

    eps_start = 1.0
    eps_end = 0.01
    eps_decay_steps = 150_000

    save_best = os.path.join(MODEL_DIR, "full_dqn_best.pt")
    save_last = os.path.join(MODEL_DIR, "full_dqn_last.pt")


def epsilon_by_step(step, cfg: Config):
    eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * math.exp(-1.0 * step / cfg.eps_decay_steps)
    return max(cfg.eps_end, eps)


# -------------------------------------------------
# 主訓練邏輯
# -------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env = PacmanCoreEnv()
    action_dim = env.action_space_n
    in_channels = 1  # 若未堆疊 frame，保持 1
    H, W = 26, 28    # 若你的 state shape 不同，可改這裡

    q = CnnDQN(action_dim, in_channels, H, W).to(device)
    target_q = CnnDQN(action_dim, in_channels, H, W).to(device)
    optimizer = optim.Adam(q.parameters(), lr=Config.lr)
    buffer = ReplayBuffer(Config.buffer_size)
    writer = SummaryWriter("logs/pacman_full")

    start_step = 0
    best_return = -1e9

    # -------------------------------------------------
    # 嘗試載入舊模型 (支援多種格式)
    # -------------------------------------------------
    if os.path.exists(Config.save_last):
        print(f"Loading last checkpoint: {Config.save_last}")
        checkpoint = torch.load(Config.save_last, map_location=device)

        try:
            if isinstance(checkpoint, dict):
                if "q_state_dict" in checkpoint:
                    # ✅ 新版完整續訓格式
                    q.load_state_dict(checkpoint["q_state_dict"])
                    target_q.load_state_dict(checkpoint["target_q_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    start_step = checkpoint.get("step", 0)
                    best_return = checkpoint.get("best_return", -1e9)
                    print(f"✅ Resumed from step {start_step}, best_return={best_return}")

                elif "q" in checkpoint:
                    # ✅ 舊版完整格式
                    print("Detected legacy checkpoint with full components. Loading...")
                    q.load_state_dict(checkpoint["q"])
                    target_q.load_state_dict(checkpoint["target_q"])
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    start_step = checkpoint.get("global_step", 0)
                    best_return = checkpoint.get("best_return", -1e9)
                    print(f"✅ Resumed from legacy checkpoint, step={start_step}")

                else:
                    # ✅ 最舊版：只有模型權重
                    print("Detected weight-only file. Loading model weights only.")
                    q.load_state_dict(checkpoint)
                    target_q.load_state_dict(checkpoint)
            else:
                print("Detected raw state_dict. Loading...")
                q.load_state_dict(checkpoint)
                target_q.load_state_dict(checkpoint)
        except Exception as e:
            print("⚠️ Failed to load checkpoint:", e)
    else:
        print("No last checkpoint found. Start fresh training.")

    # -------------------------------------------------
    # 主訓練迴圈
    # -------------------------------------------------
    step = start_step
    while step < Config.max_steps:
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            eps = epsilon_by_step(step, Config)

            # 自動修正輸入維度
            x = torch.tensor(state, dtype=torch.float32, device=device)
            if x.ndim == 2:
                x = x.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            elif x.ndim == 3:
                x = x.unsqueeze(0)  # (1,C,H,W)

            if random.random() < eps:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    action = int(q(x).argmax(dim=1).item())

            next_state, reward, done, info = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1

            # ----------------------------
            # 學習更新
            # ----------------------------
            if len(buffer) > Config.start_learning:
                s, a, r, ns, d = buffer.sample(Config.batch_size)
                s = torch.tensor(s, dtype=torch.float32, device=device)
                ns = torch.tensor(ns, dtype=torch.float32, device=device)
                    # 若狀態尚未有 channel 維度，自動補上 (batch, 1, H, W)
                if s.ndim == 3:
                    s = s.unsqueeze(1)
                    ns = ns.unsqueeze(1)
                a = torch.tensor(a, dtype=torch.long, device=device)
                r = torch.tensor(r, dtype=torch.float32, device=device)
                d = torch.tensor(d, dtype=torch.float32, device=device)

                q_values = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q = target_q(ns).max(1)[0]
                    target = r + Config.gamma * (1 - d) * max_next_q

                loss = nn.MSELoss()(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                writer.add_scalar("loss", loss.item(), step)

            # ----------------------------
            # 更新 target network
            # ----------------------------
            if step % Config.target_update_every == 0:
                target_q.load_state_dict(q.state_dict())

        # 每回合記錄
        writer.add_scalar("return", total_reward, step)
        print(f"[Step {step}] Episode | Return {total_reward:.2f} | Eps {eps:.3f}")

        # ----------------------------
        # 儲存 checkpoint
        # ----------------------------
        torch.save({
            "step": step,
            "best_return": best_return,
            "q_state_dict": q.state_dict(),
            "target_q_state_dict": target_q.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, Config.save_last)
        print(f"Checkpoint saved: {Config.save_last}")

        if total_reward > best_return:
            best_return = total_reward
            torch.save({
                "step": step,
                "best_return": best_return,
                "q_state_dict": q.state_dict(),
                "target_q_state_dict": target_q.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, Config.save_best)
            print(f"🎯 New Best Return: {best_return:.2f} | Saved best model.")


if __name__ == "__main__":
    main()
