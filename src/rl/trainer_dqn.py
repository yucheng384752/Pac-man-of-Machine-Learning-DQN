import numpy as np, torch, torch.nn as nn, torch.optim as optim
from dataclasses import dataclass
from src.rl.replay_buffer import ReplayBuffer
from src.agents.dqn_agent import DQN, act_eps_greedy

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 128
    buffer_size: int = 100_000
    start_learning: int = 5_000
    target_update_every: int = 1000
    max_steps: int = 500_000
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_decay_steps: int = 50_000
    huber: bool = True
    double_dqn: bool = True
    save_best: str = "models/dqn_pacman_best.pt"
    save_last: str = "models/dqn_pacman_last.pt"

class DQNTrainer:
    def __init__(self, env, cfg: DQNConfig = DQNConfig(), device=None):
        self.env, self.cfg = env, cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        s0 = env.reset(); self.state_dim = len(s0); self.action_dim = env.action_space_n
        self.q = DQN(self.state_dim, self.action_dim).to(self.device)
        self.t = DQN(self.state_dim, self.action_dim).to(self.device); self.t.load_state_dict(self.q.state_dict()); self.t.eval()
        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.criterion = nn.SmoothL1Loss() if cfg.huber else nn.MSELoss()
        self.buf = ReplayBuffer(cfg.buffer_size)
        self.global_step, self.best_return = 0, -1e9

    def epsilon(self):
        c = self.cfg
        return float(c.eps_end + (c.eps_start - c.eps_end) * np.exp(-self.global_step / c.eps_decay_steps))

    def _target_q(self, next_states):
        with torch.no_grad():
            if self.cfg.double_dqn:
                # Double-DQN: a* = argmax_a Q(s', a; θ), evaluate with target Q(s', a*; θ-)
                a_star = self.q(next_states).argmax(dim=1, keepdim=True)
                return self.t(next_states).gather(1, a_star)
            else:
                return self.t(next_states).max(dim=1, keepdim=True)[0]

    def _learn_step(self):
        s, a, r, ns, d = self.buf.sample(self.cfg.batch_size)
        s  = torch.tensor(s,  dtype=torch.float32).to(self.device)
        a  = torch.tensor(a,  dtype=torch.int64).unsqueeze(1).to(self.device)
        r  = torch.tensor(r,  dtype=torch.float32).unsqueeze(1).to(self.device)
        ns = torch.tensor(ns, dtype=torch.float32).to(self.device)
        d  = torch.tensor(d,  dtype=torch.float32).unsqueeze(1).to(self.device)

        q_sa = self.q(s).gather(1, a)
        y = r + (1.0 - d) * self.cfg.gamma * self._target_q(ns)
        loss = self.criterion(q_sa, y)

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        return float(loss.item())

    def train(self):
        import os
        os.makedirs("models", exist_ok=True)
        c = self.cfg
        s = self.env.reset()
        ep_ret = 0.0
        ep_len = 0
        ep_count = 1

        print("Start Training")
        print(f"State dim = {self.state_dim}, Action dim = {self.action_dim}")
        print(f"Max steps = {c.max_steps}, Start learning after = {c.start_learning} steps")
        print("-" * 60)

        while self.global_step < c.max_steps:
            self.global_step += 1
            ep_len += 1

            a = act_eps_greedy(self.q, s, self.epsilon(), self.device)
            ns, r, done, info = self.env.step(a)

            self.buf.push(s, a, r, ns, done)
            s, ep_ret = ns, ep_ret + r

            loss = None
            if len(self.buf) >= c.start_learning:
                loss = self._learn_step()

                if self.global_step % c.target_update_every == 0:
                    self.t.load_state_dict(self.q.state_dict())

            # 每 1000 steps 印出訓練進度
            if self.global_step % 1000 == 0:
                print(
                    f"[Step {self.global_step:6}] "
                    f"Episode {ep_count:4} | "
                    f"Return {ep_ret:6.2f} | "
                    f"Loss {loss if loss is not None else 0:.4f} | "
                    f"Eps {self.epsilon():.3f} | "
                    f"Buffer {len(self.buf)}"
                )

            # Episode 結束
            if done:
                if ep_ret > self.best_return:
                    self.best_return = ep_ret
                    torch.save(self.q.state_dict(), c.save_best)

                print(
                    f"Episode {ep_count} finished: "
                    f"Return = {ep_ret:.2f}, Steps = {ep_len}, "
                    f"Best = {self.best_return:.2f}"
                )

                s = self.env.reset()
                ep_ret = 0.0
                ep_len = 0
                ep_count += 1

        torch.save(self.q.state_dict(), c.save_last)
        print("Training Finished. Best return:", self.best_return)
