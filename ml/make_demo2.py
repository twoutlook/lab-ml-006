"""GOAL002 的影片要用的資料 -> web/demo2.json。

    python ml/make_demo2.py

錄影頁不做推論也不跑搜尋，只會播。所有東西在這裡先算好：

  ・一顆示範方塊的兩種解法：一次解完 vs 先解角再解完
    （後者是這支影片最重要的畫面——看得到角塊先歸位、邊塊還亂著）
  ・沿路每一步的「角塊還要幾步」與「網路猜還要幾步」
  ・角塊的哪幾片貼紙（畫面要把非角塊的淡掉）
  ・goal2.json 的彙總數字
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import corners
from cube import Cube
from benchmark import load_net, verify
from heuristics import CornerPDB, NetHeuristic
from search import bwas
from goal2 import solve_corners_greedy
from idastar import OptimalSolver

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "web" / "demo2.json"


def trace(cube, hs, start, seq):
    """沿著一串動作走，每一站記下各個估計值。"""
    cur = start.reshape(1, -1).copy()
    states = [cur[0].copy()]
    for m in seq:
        cur = cube.apply(cur, np.array([m]))
        states.append(cur[0].copy())
    arr = np.array(states, dtype=np.uint8)
    return {k: [round(float(v), 3) for v in h(arr)] for k, h in hs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--depth", type=int, default=25)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cube = Cube(3)
    net, _ = load_net(3, dev)
    pdb, nh = CornerPDB(cube), NetHeuristic(net, dev)
    cfg = dict(cube.cfg["search"], maxNodes=400_000)
    rng = np.random.default_rng(a.seed)

    g2 = json.loads((HERE / "checkpoints" / "goal2.json").read_text(encoding="utf-8"))

    # ── 示範局面：挑一顆兩種解法都成功的 ──
    for _ in range(40):
        st, sq = cube.scramble(np.array([a.depth]), rng)
        t0 = time.time()
        one, one_nodes = bwas(cube, nh, st[0], dev, cfg["weight"], cfg["batch"], cfg["maxNodes"])
        one_ms = (time.time() - t0) * 1000
        if one is None:
            continue
        s1, mid = solve_corners_greedy(cube, pdb, st[0])
        t0 = time.time()
        s2, two_nodes = bwas(cube, nh, mid, dev, cfg["weight"], cfg["batch"], cfg["maxNodes"])
        two_ms = (time.time() - t0) * 1000
        if s2 is not None:
            break
    assert one is not None and s2 is not None, "抽不到兩種解法都成功的局面"
    assert verify(cube, st[0], one) and verify(cube, mid, s2)

    hs = {"pdb": pdb, "net": nh}
    demo = {
        "scramble": [int(m) for m in sq[0] if m >= 0],
        "oneShot": {"seq": [int(m) for m in one], "nodes": int(one_nodes),
                    "ms": round(one_ms), "trace": trace(cube, hs, st[0], one)},
        # trace 要蓋住「畫面實際會播的整串」＝ 打亂 + 第一階段 + 第二階段。
        # 只存解法那一段的話，錄影頁的索引會提早夾到尾端，
        # 變成方塊還亂著卻顯示「還要 0 步」。
        "staged": {"stage1": [int(m) for m in s1], "stage2": [int(m) for m in s2],
                   "nodes": int(two_nodes), "ms": round(two_ms),
                   "trace": trace(cube, hs, cube.solved(1)[0],
                                  [int(m) for m in sq[0] if m >= 0] + list(s1) + list(s2)),
                   "corner_done_at": len(s1)},
        "scrambleTrace": trace(cube, hs, cube.solved(1)[0], [int(m) for m in sq[0] if m >= 0]),
    }
    print(f"示範：打亂 {len(demo['scramble'])} 步")
    print(f"  一次解完 {len(one)} 步 / {one_nodes:,} 節點 / {one_ms:.0f} ms")
    print(f"  先解角 {len(s1)} 步 + 其餘 {len(s2)} 步 = {len(s1) + len(s2)} 步 / "
          f"{two_nodes:,} 節點 / {two_ms:.0f} ms")

    # ── 畫面要把非角塊的貼紙淡掉，所以要知道哪幾片是角塊 ──
    demo["cornerStickers"] = sorted(int(v) for v in corners.SLOTS.ravel())

    # ── IDA* vs A*：同一批局面、同一個 heuristic，只換搜尋 ──
    sv = OptimalSolver(cube, pdb.dist)
    ida, astar = [], []
    st2, _ = cube.scramble(np.full(5, 10), rng)
    for i in range(5):
        t0 = time.time(); seq, n = sv.solve(st2[i]); ida.append((len(seq), n, time.time() - t0))
        t0 = time.time(); seq2, n2 = bwas(cube, pdb, st2[i], dev, 1.0, 1, 3_000_000, goal_at_pop=True)
        astar.append((len(seq2), n2, time.time() - t0))
        assert len(seq) == len(seq2), "IDA* 跟 A* 給的最短解長度不一樣"
    f = lambda rs, k: float(np.mean([r[k] for r in rs]))
    demo["idaVsAstar"] = {
        "depth": 10, "n": 5,
        "ida": {"nodes": f(ida, 1), "sec": f(ida, 2), "nps": f(ida, 1) / f(ida, 2)},
        "astar": {"nodes": f(astar, 1), "sec": f(astar, 2), "nps": f(astar, 1) / f(astar, 2)},
    }
    print(f"  IDA*  {f(ida, 1):,.0f} 節點 / {f(ida, 2):.2f}s = {f(ida, 1) / f(ida, 2):,.0f} 節點/秒")
    print(f"  A*    {f(astar, 1):,.0f} 節點 / {f(astar, 2):.2f}s = {f(astar, 1) / f(astar, 2):,.0f} 節點/秒")

    demo["pdb"] = g2["corner_pdb"]
    demo["e1"] = g2["e1"]
    demo["e2"] = {k: v for k, v in g2["e2"].items() if k != "records"}
    demo["e2"]["by_depth"] = [
        {"depth": d, "n": len([r for r in g2["e2"]["records"] if r["depth"] == d]),
         "mean_true": round(float(np.mean([r["true"] for r in g2["e2"]["records"] if r["depth"] == d])), 2),
         "mean_nodes": int(np.mean([r["nodes"] for r in g2["e2"]["records"] if r["depth"] == d]))}
        for d in sorted({r["depth"] for r in g2["e2"]["records"]})]
    demo["e3"] = {k: v for k, v in g2["e3"].items() if k != "rows"}
    demo["e4"] = g2["e4"]

    OUT.write_text(json.dumps(demo, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"\n寫出 {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
