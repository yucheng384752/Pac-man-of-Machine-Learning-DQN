# src/agents/cnn_dqn.py

import torch
import torch.nn as nn


class CnnDQN(nn.Module):
    def __init__(self, action_dim, in_channels, rows, cols):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 動態計算 flatten 大小
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, rows, cols)
            conv_out = self.conv(dummy)
            n_flat = conv_out.view(1, -1).size(1)

        self.head = nn.Sequential(
            nn.Linear(n_flat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        q = self.head(x)
        return q
