import torch
import torch.nn as nn
from typing import Optional

class DQN(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )
    def forward(self, x):
        return self.net(x)

@torch.no_grad()
def act_eps_greedy(qnet: DQN, state, epsilon: float, device: Optional[torch.device]=None):
    import numpy as np, torch
    if np.random.rand() < epsilon:
        return np.random.randint(qnet.net[-1].out_features)
    x = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device or "cpu")
    return int(qnet(x).argmax(dim=1).item())
