# src/agents/cnn_dqn.py

import torch
import torch.nn as nn

class CnnDQN(nn.Module):
    """
    輸入：  (B, 1, H, W)   # H = ROWS, W = COLS
    輸出：  (B, action_dim)
    """

    def __init__(self, action_dim: int, rows: int, cols: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 大約縮半
            nn.ReLU(),
        )

        # 粗估一下 conv 後大小，安全起見先跑一遍 dummy 再改也可以
        # 這裡直接算：行列各約縮成一半
        h2 = rows // 2
        w2 = cols // 2
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * h2 * w2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        # x: (B, 1, H, W)
        x = x / 6.0  # 簡單 normalize 到 0~1
        x = self.conv(x)
        x = self.fc(x)
        return x
