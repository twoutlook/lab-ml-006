"""IDA* —— 用角塊表求 3x3x3 的**精確**最短解。

    python ml/idastar.py --depth 12 --n 5

為什麼不用前面那個 A*？因為它撐不到有趣的地方。
A* 每展開一個節點要做十二次「把狀態轉成 54 bytes 的鍵、查字典、丟進堆積」，
實測每個節點約 765 微秒——打亂 10 步要 6.8 秒，11 步就跑不完了。
而網路真正開始失準的區域在最短解 15 步以上，A* 根本走不到那裡。

IDA*（iterative deepening A*）把 open/closed list 整個拿掉：
用深度優先搜尋，配一個 f 值上限，超過就回頭；一輪走完沒找到就把上限提高。
代價是同一個節點會被重複走到，好處是每個節點只剩下「轉一步 + 查一次表」。

這裡再加上兩個加速：

1. **角塊座標增量更新。** 不必從貼紙重新算角塊在哪——
   perm 和 ori 各自有一張移動表（見 ml/corners.py 的說明），
   走一步就是兩次陣列查表，h 就是 dist[perm * 2187 + ori]。

2. **operator.itemgetter 套用轉動。** 轉一步等於照置換表重排 54 個元素。
   用 Python 迴圈做要幾微秒，itemgetter 是 C 寫的，快一個數量級。

剪枝三條，都是安全的（不會剪掉最短解）：
  ・不走上一步的反手（U 之後不走 U'）
  ・同一面不連轉三次（U U U = U'，一定不是最短解的一部分）
  ・對面可交換（U 和 D 誰先誰後結果一樣），所以固定一個順序，只走其中一種
"""
from __future__ import annotations

import argparse
import time
from operator import itemgetter

import numpy as np

import corners
from cube import Cube

INF = 1 << 30


class OptimalSolver:
    def __init__(self, cube: Cube, dist=None):
        self.cube = cube
        self.dist = np.asarray(corners.load() if dist is None else dist)
        self.perm_tbl, self.ori_tbl = corners.build_move_tables(cube)
        self.getters = [itemgetter(*p.tolist()) for p in cube.perms]
        self.goal = tuple(int(v) for v in cube.goal)
        self.A = cube.n_actions
        self.inv = [int(v) for v in cube.inverse_move]
        self.face = [m // 2 for m in range(self.A)]      # 0=U 1=D 2=F 3=B 4=L 5=R
        # 對面成對：(U,D) (F,B) (L,R)。同一對之間可以交換順序，固定成小的先走。
        self.opposite = [f ^ 1 for f in range(6)]
        self.nodes = 0

    def h(self, p, o):
        return int(self.dist[p * corners.N_ORI + o])

    def solve(self, state, max_depth: int = 20):
        """回傳 (最短解, 展開節點數)。走到 max_depth 還沒解開就回 (None, nodes)。"""
        s = tuple(int(v) for v in np.asarray(state).ravel())
        idx = int(corners.index_of(self.cube, np.asarray(state).reshape(1, -1))[0])
        p, o = divmod(idx, corners.N_ORI)
        self.nodes = 0
        if s == self.goal:
            return [], 0
        bound = self.h(p, o)
        path: list[int] = []
        while bound <= max_depth:
            t = self._dfs(s, p, o, 0, bound, -1, 0, path)
            if t == -1:
                return path[:], self.nodes
            if t >= INF:
                return None, self.nodes
            bound = t
        return None, self.nodes

    def _dfs(self, s, p, o, g, bound, last, run, path):
        """回傳 -1 代表找到了；否則回傳這條路上超過 bound 的最小 f（下一輪的新 bound）。"""
        f = g + self.h(p, o)
        if f > bound:
            return f
        if s == self.goal:
            return -1
        self.nodes += 1
        nxt = INF
        lf = self.face[last] if last >= 0 else -1
        for m in range(self.A):
            if last >= 0:
                if m == self.inv[last]:
                    continue                          # 反手
                mf = self.face[m]
                if mf == lf and run >= 2:
                    continue                          # 同一面連三次
                if mf == self.opposite[lf] and mf < lf:
                    continue                          # 對面可交換，固定順序
            s2 = self.getters[m](s)
            p2 = int(self.perm_tbl[p, m])
            o2 = int(self.ori_tbl[o, m])
            path.append(m)
            r = self._dfs(s2, p2, o2, g + 1, bound,
                          m, run + 1 if self.face[m] == lf else 1, path)
            if r == -1:
                return -1
            path.pop()
            if r < nxt:
                nxt = r
        return nxt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--max-depth", type=int, default=20)
    a = ap.parse_args()
    cube = Cube(3)
    sv = OptimalSolver(cube)
    rng = np.random.default_rng(a.seed)
    st, _ = cube.scramble(np.full(a.n, a.depth), rng)
    print(f"打亂 {a.depth} 步 × {a.n} 顆，用角塊表 + IDA* 求精確最短解")
    tot = []
    for i in range(a.n):
        t0 = time.time()
        seq, n = sv.solve(st[i], a.max_depth)
        dt = time.time() - t0
        if seq is None:
            print(f"  #{i}: {a.max_depth} 步以內找不到（展開 {n:,}，{dt:.1f}s）")
            continue
        cur = st[i:i + 1].copy()
        for m in seq:
            cur = cube.apply(cur, np.array([m]))
        assert cube.is_solved(cur)[0], "IDA* 回傳的解轉不回去"
        tot.append((len(seq), n, dt))
        print(f"  #{i}: 最短 {len(seq):>2} 步 · 展開 {n:>10,} · {dt:>7.1f}s · {cube.moves_to_str(seq)}")
    if tot:
        L = np.array(tot)
        print(f"\n平均 {L[:, 0].mean():.2f} 步 · {L[:, 1].mean():,.0f} 節點 · {L[:, 2].mean():.1f}s"
              f" · 每秒 {L[:, 1].sum() / L[:, 2].sum():,.0f} 個節點")


if __name__ == "__main__":
    main()
