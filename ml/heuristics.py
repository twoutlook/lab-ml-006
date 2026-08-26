"""三種「還要幾步」的估計法，放在同一個介面下，才好互相比。

    h_net    學出來的網路。看整顆方塊，但不保證準，實測有 23% 的局面高估。
    h_pdb    只看角塊的精確距離表（ml/corners.py 建的）。只看一部分，
             但**保證不高估**——因為解開整顆方塊的每一步也都在轉角塊。
    h_max    兩個取大的。指引更強，但只要其中一個會高估，整體就會高估。

這三個放在一起量，問的是一個很具體的問題：
**一張「精確但只看一部分」的表，跟一個「看全部但不精確」的網路，哪個有用？**

而 h_pdb 還有第二個用途，比當 heuristic 更重要：它是**下界**。
3x3x3 沒有正確答案，所以「解比最短解長幾步」本來量不出來；
有了下界就夾得出來——最短解 d 一定滿足 h_pdb(s) <= d <= 找到的解長度。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from cube import Cube, N_COLORS


class Heuristic:
    """吃 (N, S) 的貼紙陣列，回傳 (N,) 的「還要幾步」估計。"""

    name = "?"
    admissible = False        # 保證不高估嗎？決定 A* 還有沒有最短解保證

    def __call__(self, states: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NetHeuristic(Heuristic):
    name = "學出來的網路"
    admissible = False

    def __init__(self, net, device, chunk: int = 200000):
        self.net, self.device, self.chunk = net, device, chunk
        net.eval()

    @torch.no_grad()
    def __call__(self, states):
        out = np.empty(len(states), dtype=np.float32)
        for i in range(0, len(states), self.chunk):
            x = torch.from_numpy(states[i:i + self.chunk]).to(self.device).long()
            x = F.one_hot(x, N_COLORS).float().flatten(1)
            out[i:i + self.chunk] = self.net(x).cpu().numpy()
        return out


class CornerPDB(Heuristic):
    """只看角塊的精確距離。這是 3x3x3 上第一個保證不高估的估計。"""

    name = "角塊距離表（精確）"
    admissible = True

    def __init__(self, cube: Cube, dist=None):
        import corners
        self.corners = corners
        self.cube = cube
        self.dist = corners.load() if dist is None else dist

    def __call__(self, states):
        return self.dist[self.corners.index_of(self.cube, states)].astype(np.float32)


class EdgePDB(Heuristic):
    """只看 12 條邊裡的 6 條，精確距離。Korf 三張表的第二、三張。

    跟角塊那張一樣是下界（解開整顆的每一步也都在動這 6 條邊），
    所以三張取最大仍然保證不高估——而且比任何一張單獨用都緊。
    """

    admissible = True

    def __init__(self, cube: Cube, half: int, dist=None):
        import edges
        self.edges = edges
        self.cube = cube
        self.half = half
        self.tracked = edges.TRACKED[half]
        self.dist = edges.load(half) if dist is None else dist
        self.name = f"邊塊距離表 {half}（精確）"

    def __call__(self, states):
        return self.dist[self.edges.index_of(self.cube, states, self.tracked)].astype(np.float32)


class MaxHeuristic(Heuristic):
    """取大的。兩個都是下界的話取大仍是下界；只要有一個會高估，取大就會高估。"""

    def __init__(self, *hs: Heuristic):
        self.hs = hs
        self.name = " ∪ ".join(h.name for h in hs)
        self.admissible = all(h.admissible for h in hs)

    def __call__(self, states):
        out = self.hs[0](states)
        for h in self.hs[1:]:
            out = np.maximum(out, h(states))
        return out


class ScaledHeuristic(Heuristic):
    """把估計值乘一個係數。

    乘 0.9 之類的可以把一個會高估的 heuristic 往回壓，換回一點最短解的保證；
    代價是估計變小、搜尋要展開更多節點。這個取捨在 2x2x2 上量得出來
    （那邊有正確答案），拿到 3x3x3 就只能靠角塊表當下界去夾。
    """

    def __init__(self, h: Heuristic, k: float):
        self.h, self.k = h, k
        self.name = f"{h.name} × {k}"
        self.admissible = h.admissible and k <= 1.0

    def __call__(self, states):
        return self.h(states) * self.k


def korf_bound(cube: Cube) -> "MaxHeuristic":
    """Korf 的三張 pattern database 取最大 —— 目前最緊的、保證不高估的下界。"""
    h = MaxHeuristic(CornerPDB(cube), EdgePDB(cube, 0), EdgePDB(cube, 1))
    h.name = "三張表取大（角塊 + 兩半邊塊）"
    return h
