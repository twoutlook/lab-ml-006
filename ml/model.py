"""Cost-to-go 網路。

這個專案跟系列前五個最大的差別：**這裡沒有 policy，也沒有 Q 值。**

網路只做一件事：看一個局面，猜「離解開還有幾步」。一個輸入，一個純量輸出。
它不告訴你要轉哪一面——那是搜尋的工作（ml/search.py）。

    h(s) ≈ 這個局面的最短步數

為什麼不學 policy？因為魔術方塊的動作序列長、而且錯一步就前功盡棄。
一個 90% 正確的 policy 走二十步，全對的機率是 0.9^20 ≈ 12%。
但一個「大概準」的距離估計，配上 A* 搜尋，錯了可以退回來重試。
把「判斷好壞」跟「決定要走哪」分開，是 DeepCubeA 的核心想法。

架構照 DeepCubeA 縮小：兩層全連接進到主幹寬度，接幾個 residual block，最後出一個數字。
用 residual 是因為這個網路要學的是一個很不平滑的函數——
差一步的兩個局面，貼紙可能完全不一樣，但答案只差 1。

BatchNorm 在訓練時很重要（輸入是 one-hot，尺度差很大），
但匯出到瀏覽器時會被摺進前一層的線性層裡（見 export_policy.py），
所以 web/nn.js 只需要 linear + relu + 殘差相加，不用實作 BN。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.fc1 = nn.Linear(h, h)
        self.bn1 = nn.BatchNorm1d(h)
        self.fc2 = nn.Linear(h, h)
        self.bn2 = nn.BatchNorm1d(h)

    def forward(self, x):
        y = torch.relu(self.bn1(self.fc1(x)))
        y = self.bn2(self.fc2(y))
        return torch.relu(x + y)


class ValueNet(nn.Module):
    def __init__(self, obs_size: int, hidden: int = 256, blocks: int = 2):
        super().__init__()
        self.obs_size = obs_size
        self.hidden = hidden
        self.n_blocks = blocks
        self.fc1 = nn.Linear(obs_size, hidden * 2)
        self.bn1 = nn.BatchNorm1d(hidden * 2)
        self.fc2 = nn.Linear(hidden * 2, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(blocks)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.bn1(self.fc1(x)))
        h = torch.relu(self.bn2(self.fc2(h)))
        for b in self.blocks:
            h = b(h)
        return self.head(h).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def fold_bn(lin: nn.Linear, bn: nn.BatchNorm1d):
    """把 BatchNorm 摺進前面那個線性層，變成一個等價的 Linear。

        BN(Wx + b) = gamma * (Wx + b - mean) / sqrt(var + eps) + beta
                   = (gamma/s) W x + [ (b - mean) * gamma/s + beta ]

    推論時 BN 只是「乘一個常數再加一個常數」，所以可以直接吸進權重裡。
    好處是瀏覽器端不必再實作一次 BN——少一份會不一致的東西。
    """
    s = (bn.running_var + bn.eps).sqrt()
    scale = bn.weight / s
    w = lin.weight * scale[:, None]
    b = (lin.bias - bn.running_mean) * scale + bn.bias
    return w, b
