"""批次加權 A*（Batch Weighted A*，DeepCubeA 的搜尋端）。

網路只會估「還要幾步」，它不會告訴你要轉哪一面。真正把方塊解開的是這裡。

一般的 A* 一次只展開一個節點——但一次只評估一個局面，等於讓 GPU 空轉。
所以改成一次從優先佇列拿出 N 個最好的節點，把它們的所有子節點湊成一批
（2x2x2 是 N×6，3x3x3 是 N×12）送進網路。GPU 一次算完，搜尋才跟得上。

排序用的分數：

    f = weight * g + h(s)

    g 是已經走了幾步，h 是網路猜還要幾步。
    weight = 1  → 標準 A*，h 不高估的話保證最短解，但要展開很多節點。
    weight = 0  → 完全不管走了多遠，純粹貪婪，快但解會很長。
    weight < 1  → 中間。這個專案 2x2x2 用 0.6、3x3x3 用 0.3。

為什麼不用 weight = 1？因為訓練出來的 h **不保證不高估**。
一旦高估，最短解的保證本來就沒了；那還不如把 weight 調低換速度。
2x2x2 因為有正確答案，這個取捨可以精確量出來（見 ml/benchmark.py）。
"""
from __future__ import annotations

import heapq

import numpy as np
import torch
import torch.nn.functional as F

from cube import Cube, N_COLORS


@torch.no_grad()
def heuristic(net, states: np.ndarray, device, chunk: int = 200000) -> np.ndarray:
    out = np.empty(len(states), dtype=np.float32)
    for i in range(0, len(states), chunk):
        x = torch.from_numpy(states[i:i + chunk]).to(device).long()
        x = F.one_hot(x, N_COLORS).float().flatten(1)
        out[i:i + chunk] = net(x).cpu().numpy()
    return out


def bwas(cube: Cube, net, state: np.ndarray, device,
         weight: float = 0.6, batch: int = 200, max_nodes: int = 200000,
         goal_at_pop: bool = False, goal=None):
    """解一個局面。回傳 (動作序列 or None, 展開了幾個節點)。

    `net` 可以是 PyTorch 模型，也可以是 ml/heuristics.py 裡任何一個 Heuristic
    （角塊距離表、取大的組合…）。兩者都吃 (N,S) 回 (N,)，所以搜尋本身不必知道差別。

    goal：自訂「什麼算走到了」。吃 (N,S) 回 (N,) 的布林陣列。
    預設是「整顆方塊解開」，但換成「角塊全部歸位」就變成人類的第一階段——
    這是 ml/goal2.py 拿來做分階段解法的入口。

    goal_at_pop：終點在「節點被展開時」才判定，而不是「子節點被產生時」。
    課本 A* 是前者，而且**最短解保證只在前者成立**。這個專案原本用後者
    （比較省節點，而且在 2x2x2 上實測兩者結果完全一樣，見下面的說明）；
    但要拿 admissible 的角塊表去證明最短解，就必須切回課本版本。
    """
    h = net if callable(net) and not hasattr(net, "eval") else None
    if h is None:
        from heuristics import NetHeuristic
        h = NetHeuristic(net, device)
    S = cube.n_stickers
    reached = goal if goal is not None else cube.is_solved
    start = state.reshape(1, S)
    if reached(start)[0]:
        return [], 0

    k0 = start[0].tobytes()
    g = {k0: 0}                     # 目前找到的最短「已走步數」
    parent = {k0: (None, -1)}       # 走到這個局面的上一個局面與動作
    h0 = float(h(start)[0])
    heap = [(weight * 0 + h0, 0, k0)]
    counter = 0
    expanded = 0

    def path_from(k, extra=None):
        seq = [] if extra is None else [extra]
        while parent[k][0] is not None:
            pk, pm = parent[k]
            seq.append(pm)
            k = pk
        return seq[::-1]

    while heap and expanded < max_nodes:
        # ── 一次抓 batch 個最好的節點 ──
        pop, seen_now = [], set()
        while heap and len(pop) < batch:
            f, gg, k = heapq.heappop(heap)
            if gg > g.get(k, 1 << 30) or k in seen_now:
                continue            # 這筆是舊的（後來找到更短的路），跳過
            seen_now.add(k)
            pop.append((gg, k))
        if not pop:
            break
        expanded += len(pop)

        cur = np.frombuffer(b"".join(k for _, k in pop), dtype=np.uint8).reshape(len(pop), S)
        gs = np.array([gg for gg, _ in pop], dtype=np.int64)

        if goal_at_pop:
            done = reached(cur)
            if done.any():
                return path_from(pop[int(np.flatnonzero(done)[0])][1]), expanded

        kids = cube.expand(cur)                       # (B, A, S)
        B, A, _ = kids.shape
        flat = np.ascontiguousarray(kids.reshape(-1, S))
        solved = reached(flat)

        # ── 有子節點就是答案，直接回頭串路徑 ──
        # 注意：這是「產生時就判定終點」，不是課本 A* 的「等它被展開才判定」。
        # 課本那樣寫是有原因的——提早判定，就算 heuristic 完全不高估也可能回傳非最短解。
        # 這裡實測過兩種寫法：500 個局面、weight=1.0，最短解率都是 99.2%，
        # 而且失敗的是同樣那 4 個局面。也就是說這裡的非最短解跟判定時機無關，
        # 純粹是 heuristic 高估造成的（benchmark 量到 23.2% 的局面被高估）。
        # 既然結果一樣，就用比較省節點的那個寫法。
        if solved.any() and not goal_at_pop:
            j = int(np.flatnonzero(solved)[0])
            bi, ai = divmod(j, A)
            return path_from(pop[bi][1], ai), expanded

        hv = h(flat)
        gk = np.repeat(gs, A) + 1
        fv = weight * gk + hv

        keys = [flat[i].tobytes() for i in range(len(flat))]
        for i, k in enumerate(keys):
            gi = int(gk[i])
            if gi < g.get(k, 1 << 30):
                g[k] = gi
                parent[k] = (pop[i // A][1], i % A)
                counter += 1
                heapq.heappush(heap, (float(fv[i]), gi, k))

    return None, expanded


def greedy(cube: Cube, net, states: np.ndarray, device, budget: int):
    """完全不搜尋：每一步都挑 h 最小的子節點。用來看「沒有搜尋的話網路夠不夠強」。"""
    h = net if callable(net) and not hasattr(net, "eval") else None
    if h is None:
        from heuristics import NetHeuristic
        h = NetHeuristic(net, device)
    cur = states.copy()
    n, S = cur.shape
    done = cube.is_solved(cur)
    steps = np.where(done, 0, -1)
    for t in range(budget):
        if done.all():
            break
        live = np.flatnonzero(~done)
        kids = cube.expand(cur[live])
        L, A, _ = kids.shape
        flat = np.ascontiguousarray(kids.reshape(-1, S))
        hv = h(flat).reshape(L, A)
        hv[cube.is_solved(flat).reshape(L, A)] = -1e9
        pick = hv.argmin(axis=1)
        cur[live] = kids[np.arange(L), pick]
        now = cube.is_solved(cur)
        steps[now & ~done] = t + 1
        done = now
    return done, steps


def random_walk(cube: Cube, states: np.ndarray, rng, budget: int):
    """地板：完全亂轉。用來說明「這題不能靠運氣」。"""
    cur = states.copy()
    n = len(cur)
    done = cube.is_solved(cur)
    steps = np.where(done, 0, -1)
    for t in range(budget):
        if done.all():
            break
        cur = cube.apply(cur, rng.integers(0, cube.n_actions, size=n))
        now = cube.is_solved(cur) | done
        steps[now & ~done] = t + 1
        done = now
    return done, steps
