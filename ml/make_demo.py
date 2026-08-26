"""把影片與圖文版要用的所有數字，湊成一份 web/demo.json。

    python ml/make_demo.py

影片頁（web/record.js）不做任何推論、也不跑搜尋——它只會播。
所有東西在這裡先算好：示範局面、搜尋找到的解、每一步網路猜幾步、
展開了幾個節點、benchmark 的表、距離分布。

這樣做有三個好處：
  1. 錄影頁不必載 3x3x3 那份 21 MB 的權重
  2. 同樣的 demo.json 一定錄出同樣的影片
  3. 影片裡的數字跟 benchmark 是同一份，不會有人抄錯
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cube import Cube
from benchmark import load_net
from search import bwas, random_walk, heuristic

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "web" / "demo.json"

SIZE_INFO = {
    "2": {"states": "3,674,160", "statesNote": "= 7! × 3⁶（固定一顆角之後）",
          "moves": "3 面 · 6 種轉法", "solvable": "可以，18 秒", "exact": True,
          "solvableNote": "整張最短距離表算得出來 — 這就是正確答案"},
    "3": {"states": "43,252,003,274,489,856,000", "statesNote": "四千三百二十五京",
          "moves": "6 面 · 12 種轉法", "solvable": "不可能", "exact": False,
          "solvableNote": "全部列出來的話，一個局面一奈秒也要一千三百七十億年"},
}


def h_along(cube, net, device, start, seq):
    """回傳沿著這串動作走，每一站網路猜幾步（含起點）。"""
    states = [start.copy()]
    cur = start.reshape(1, -1).copy()
    for m in seq:
        cur = cube.apply(cur, np.array([m]))
        states.append(cur[0].copy())
    arr = np.array(states, dtype=np.uint8)
    h = heuristic(net, arr, device)
    h[cube.is_solved(arr)] = 0.0          # 解開的局面照定義就是 0
    return [round(float(v), 3) for v in h]


def make_case(cube, net, device, depth, weights, rng, exact=None):
    states, seqs = cube.scramble(np.array([depth]), rng)
    scramble = [int(m) for m in seqs[0] if m >= 0]
    solved0 = cube.solved(1)[0]
    case = {
        "scramble": scramble,
        "hScramble": h_along(cube, net, device, solved0, scramble),
        "runs": {},
    }
    if exact is not None:
        case["exact"] = int(exact)
    for w in weights:
        cfg = cube.cfg["search"]
        import time
        t0 = time.time()
        seq, nodes = bwas(cube, net, states[0], device, w, cfg["batch"], cfg["maxNodes"])
        ms = (time.time() - t0) * 1000
        if seq is None:
            case["runs"][str(w)] = {"seq": [], "h": [], "nodes": nodes, "ms": round(ms), "failed": True}
            continue
        case["runs"][str(w)] = {
            "seq": [int(m) for m in seq],
            "h": h_along(cube, net, device, states[0], seq)[1:],
            "nodes": int(nodes), "ms": round(ms),
        }
    return case, states[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--depth2", type=int, default=13, help="示範用的 2x2x2 打亂步數")
    ap.add_argument("--depth3", type=int, default=14, help="示範用的 3x3x3 打亂步數")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(a.seed)

    demo = {"sizeInfo": SIZE_INFO, "defaultWeight": {"2": "0.6", "3": "0.3"}}

    # ── 2x2x2 的正確答案分布 ──
    from bfs import load, lookup
    keys, dist = load()
    demo["hist"] = [int((dist == d).sum()) for d in range(int(dist.max()) + 1)]
    demo["total2"] = int(len(dist))

    b2 = json.loads((HERE / "checkpoints" / "benchmark-2x2.json").read_text(encoding="utf-8"))
    b3 = json.loads((HERE / "checkpoints" / "benchmark-3x3.json").read_text(encoding="utf-8"))
    demo["bench2"] = {"n": b2["n"], "rows": b2["rows"]}
    demo["bench3"] = {"n": b3["n"], "by_depth": b3["by_depth"],
                      "weight": b3["weight"], "max_nodes": b3["max_nodes"],
                      "random_walk_solve_rate": b3["random_walk_solve_rate"],
                      "trained_iters": b3.get("trained_iters"), "n_params": b3.get("n_params")}
    demo["heuristic"] = b2["heuristic"]
    demo["greedy2"] = next(r["solve_rate"] for r in b2["rows"] if "貪婪" in r["name"])
    demo["weights"] = [{"w": float(r["name"].split("=")[1].rstrip("）")),
                        "mean_nodes": r["mean_nodes"], "mean_len": r["mean_len"],
                        "optimal_rate": r["optimal_rate"], "solve_rate": r["solve_rate"],
                        "ms": r["ms_per_cube"]}
                       for r in b2["rows"] if r["name"].startswith("加權 A*")]

    # ── 示範局面 ──
    demo["cases"] = {}
    demo["randomWalk"] = {}
    demo["randomStats"] = {}
    for size, depth in ((2, a.depth2), (3, a.depth3)):
        cube = Cube(size)
        net, _ = load_net(size, device)
        weights = [0.6] if size == 2 else [0.3]
        exact = None
        if size == 2:
            # 示範局面的挑選規則（寫死在這裡，不是挑到好看為止）：
            #   從固定 seed 依序抽，取第一個「真實距離 >= 11 且 A* 剛好給出最短解」的局面。
            # 為什麼要這兩個條件？距離要夠遠，示範才有看頭；解要是最短解，
            # 才不會出現「旁白說八成是最短解、畫面上這個偏偏不是」的錯搭。
            # 統計數字（解開率、最短解率、節點數）全部來自 ml/benchmark.py 的均勻抽樣，
            # 跟這個示範局面無關——這裡挑的只是要放進影片的那一顆。
            import time
            picked = None
            for k in range(400):
                s, sq = cube.scramble(np.array([depth]), rng)
                d = int(lookup(keys, dist, cube.pack(s))[0])
                if d < 11:
                    continue
                cfg = cube.cfg["search"]
                t0 = time.time()
                seq, nodes = bwas(cube, net, s[0], device, weights[0], cfg["batch"], cfg["maxNodes"])
                ms = round((time.time() - t0) * 1000)
                if picked is None:                     # 保底：真的挑不到就用第一個符合距離的
                    picked = (s, sq, d, seq, nodes, ms, k)
                if seq is not None and len(seq) == d:
                    picked = (s, sq, d, seq, nodes, ms, k)
                    break
            s, sq, d, seq, nodes, ms, tries = picked
            scramble = [int(m) for m in sq[0] if m >= 0]
            case = {"scramble": scramble,
                    "hScramble": h_along(cube, net, device, cube.solved(1)[0], scramble),
                    "exact": d, "runs": {str(weights[0]): {
                        "seq": [int(m) for m in seq],
                        "h": h_along(cube, net, device, s[0], seq)[1:],
                        "nodes": int(nodes), "ms": ms}},
                    "selected": "第一個真實距離>=11 且 A* 剛好給出最短解的局面（示範用）"}
            demo["cases"]["2"] = [case]
            print(f"2x2x2 示範（抽了 {tries + 1} 個）：打亂 {depth} 步，真實最短 {d} 步，"
                  f"A* 解 {len(seq)} 步 / {nodes} 節點")
        else:
            case, _ = make_case(cube, net, device, depth, weights, rng)
            demo["cases"]["3"] = [case]
            r = case["runs"]["0.3"]
            print(f"3x3x3 示範：打亂 {depth} 步，A* 解 {len(r['seq'])} 步 / {r['nodes']} 節點")

        # 亂轉：畫面上要播的那一串，以及大樣本的實測
        demo["randomWalk"][str(size)] = [int(x) for x in
                                         rng.integers(0, cube.n_actions, size=400)]
        n = 1000
        st, _ = cube.scramble(np.full(n, cube.cfg["scrambleMax"]), rng)
        ok, _ = random_walk(cube, st, np.random.default_rng(a.seed + 5), 10000)
        demo["randomStats"][str(size)] = {"n": n, "solved": int(ok.sum()),
                                          "solve_rate": float(ok.mean()), "budget": 10000}
        print(f"  {size}x{size}x{size} 亂轉 10,000 步：{int(ok.sum())}/{n} 解開")

    # ── 純展示用的打亂（只給畫面看，不會拿去搜尋）──
    # 標題那一幕要看起來「真的被打亂了」。示範用的解題局面不能太深，
    # 不然搜尋的節點數會爆掉；這兩件事分開，各拿各的。
    demo["displayScramble"] = {}
    for size, d in ((2, 13), (3, 25)):
        c = Cube(size)
        _, sq = c.scramble(np.array([d]), np.random.default_rng(a.seed + 11))
        demo["displayScramble"][str(size)] = [int(m) for m in sq[0] if m >= 0]

    # ── DAVI 說明用：從解開往回走幾步的小圖 ──
    c2 = Cube(2)
    w = c2.solved(1)
    walk = [[int(v) for v in w[0]]]
    r2 = np.random.default_rng(7)
    for _ in range(5):
        w = c2.apply(w, r2.integers(0, c2.n_actions, size=1))
        walk.append([int(v) for v in w[0]])
    demo["walkDemo"] = walk

    OUT.write_text(json.dumps(demo, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"\n寫出 {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
