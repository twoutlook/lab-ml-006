"""大樣本實測，結果寫成 json 給影片與圖文版讀。

    python ml/benchmark.py --size 2 --n 1000
    python ml/benchmark.py --size 3 --n 100

2x2x2 的局面是從全部 3,674,160 個裡**均勻抽**的，不是「亂轉 k 步」——
亂轉 k 步抽出來的分布會偏向簡單的局面（轉 14 步不代表真的離 14 步遠）。
既然整張距離表都算出來了，就該用它來抽樣。

而且因為有正確答案，這裡能問一個大部分強化學習專案問不出來的問題：
**它給的解，是不是最短的那一條？**

3x3x3 沒有正確答案，只能照打亂深度分組，報解開率與步數。
這個差別本身就是這個專案想講的事。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cube import Cube
from model import ValueNet
from search import bwas, greedy, random_walk

HERE = Path(__file__).resolve().parent
CONFIG_OVERRIDE: dict = {}


def load_net(size, device, ckpt=None):
    p = Path(ckpt) if ckpt else HERE / "checkpoints" / f"best-{size}x{size}.pt"
    ck = torch.load(p, map_location=device, weights_only=False)
    net = ValueNet(ck["obs_size"], ck["hidden"], ck["blocks"]).to(device)
    net.load_state_dict(ck["net"])
    net.eval()
    return net, ck


def verify(cube, state, seq):
    s = state.reshape(1, -1).copy()
    for m in seq:
        s = cube.apply(s, np.array([m]))
    return bool(cube.is_solved(s)[0])


def run_astar(cube, net, states, device, weight, batch, max_nodes, truth=None):
    solved, lens, nodes, opt = [], [], [], []
    t0 = time.time()
    for i in range(len(states)):
        seq, exp = bwas(cube, net, states[i], device, weight, batch, max_nodes)
        nodes.append(exp)
        if seq is None:
            solved.append(False)
            continue
        assert verify(cube, states[i], seq), "搜尋回傳的解轉不回去 — 搜尋或引擎有錯"
        solved.append(True)
        lens.append(len(seq))
        if truth is not None:
            opt.append(len(seq) == int(truth[i]))
    solved = np.array(solved)
    out = {"solve_rate": float(solved.mean()),
           "mean_len": float(np.mean(lens)) if lens else None,
           "median_len": float(np.median(lens)) if lens else None,
           "max_len": int(np.max(lens)) if lens else None,
           "mean_nodes": float(np.mean(nodes)),
           "median_nodes": float(np.median(nodes)),
           "ms_per_cube": (time.time() - t0) / len(states) * 1000}
    if truth is not None:
        out["optimal_rate"] = float(np.mean(opt)) if opt else 0.0
        out["excess"] = float(np.mean(np.array(lens) - truth[solved])) if lens else None
    return out


def bench2(args, device):
    from bfs import load
    cube = Cube(2)
    net, ck = load_net(2, device, args.ckpt)
    keys, dist = load()
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(keys), size=args.n, replace=False)
    states = cube.unpack(keys[idx])
    truth = dist[idx].astype(np.int64)
    cfg = dict(cube.cfg["search"], **CONFIG_OVERRIDE)
    budget = cube.cfg["godNumber"] * 20

    print(f"2x2x2 · {args.n} 個局面（從 3,674,160 個裡均勻抽）· "
          f"正確答案平均 {truth.mean():.2f} 步")

    rows = []
    ok, steps = random_walk(cube, states, np.random.default_rng(args.seed + 1), budget)
    rows.append({"name": "亂轉 random", "solve_rate": float(ok.mean()),
                 "mean_len": float(steps[ok].mean()) if ok.any() else None,
                 "optimal_rate": None,       # 一次都沒解開，這一格沒有意義
                 "mean_nodes": float(budget), "note": f"預算 {budget} 步"})
    print(f"  亂轉：解開 {ok.mean():.1%}")

    ok, steps = greedy(cube, net, states, device, budget)
    gl = steps[ok]
    rows.append({"name": "貪婪（只用網路，不搜尋）", "solve_rate": float(ok.mean()),
                 "mean_len": float(gl.mean()) if ok.any() else None,
                 "optimal_rate": float((gl == truth[ok]).mean()) if ok.any() else 0.0,
                 "mean_nodes": float(cube.n_actions),
                 "note": "每一步展開一層挑 h 最小；最短解率是在它解得開的那些裡面算的"})
    print(f"  貪婪：解開 {ok.mean():.1%}")

    for w in args.weights:
        r = run_astar(cube, net, states, device, w, cfg["batch"], cfg["maxNodes"], truth)
        r["name"] = f"加權 A*（weight={w}）"
        r["batch"] = cfg["batch"]
        rows.append(r)
        print(f"  A* w={w}: 解開 {r['solve_rate']:.1%} 最短解 {r['optimal_rate']:.1%} "
              f"平均 {r['mean_len']:.2f} 步 展開 {r['mean_nodes']:.0f} 節點 {r['ms_per_cube']:.0f} ms")

    rows.append({"name": "最短解（全狀態 BFS，沒有 ML）", "solve_rate": 1.0,
                 "mean_len": float(truth.mean()), "optimal_rate": 1.0,
                 "mean_nodes": float(len(dist)), "note": "整張距離表 33 MB"})

    # 精確度剖析：對整個狀態空間量 heuristic 的誤差
    prof = heuristic_profile(cube, net, keys, dist, device)

    # 批次大小 × weight 的對照。批次是為了餵飽 GPU，照理說不該影響結果，
    # 但 2x2x2 的整棵搜尋樹才一兩千個節點——批次一大，每一輪就把佇列整個彈光，
    # 搜尋退化成 BFS，weight 完全沒作用。這張表就是拿來看那個轉折點的。
    abl = []
    sub, subtruth = states[:args.ablation], truth[:args.ablation]
    for b in args.batches:
        for w in args.weights:
            r = run_astar(cube, net, sub, device, w, b, cfg["maxNodes"], subtruth)
            r.update(batch=b, weight=w)
            abl.append(r)
            print(f"  對照 batch={b:>3} w={w:.1f}: 最短解 {r['optimal_rate']:.1%} "
                  f"平均 {r['mean_len']:.2f} 步 展開 {r['mean_nodes']:.0f} 節點")

    return {"size": 2, "n": args.n, "seed": args.seed, "rows": rows,
            "batch_ablation": {"n": len(sub), "rows": abl},
            "truth_mean": float(truth.mean()), "god_number": cube.cfg["godNumber"],
            "state_count": int(len(dist)), "heuristic": prof,
            "trained_iters": ck.get("iter"), "n_params": net.n_params()}


@torch.no_grad()
def heuristic_profile(cube, net, keys, dist, device, n=200000, seed=99):
    """這個專案最想給的那張圖：h 和正確答案差多少，以及每個真實距離上的平均預測。"""
    from search import heuristic
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(keys), size=min(n, len(keys)), replace=False)
    states = cube.unpack(keys[idx])
    truth = dist[idx].astype(np.float32)
    h = heuristic(net, states, device)
    err = h - truth
    by = []
    for d in range(int(dist.max()) + 1):
        m = truth == d
        if m.sum() < 30:
            continue
        by.append({"d": d, "n": int(m.sum()), "mean_h": round(float(h[m].mean()), 3),
                   "std_h": round(float(h[m].std()), 3),
                   "over": round(float((h[m] > d + 1e-6).mean()), 4)})
    return {"n": len(idx), "mae": round(float(np.abs(err).mean()), 4),
            "bias": round(float(err.mean()), 4),
            "over_rate": round(float((err > 1e-6).mean()), 4),
            "within_1": round(float((np.abs(err) <= 1).mean()), 4),
            "by_distance": by}


def bench3(args, device):
    cube = Cube(3)
    net, ck = load_net(3, device, args.ckpt)
    rng = np.random.default_rng(args.seed)
    cfg = dict(cube.cfg["search"], **CONFIG_OVERRIDE)
    depths = args.depths or [3, 5, 8, 10, 12, 15, 20, 26, 30]
    print(f"3x3x3 · 每個深度 {args.n} 個局面 · 沒有正確答案可比，只能報解開率與步數")
    by = []
    for d in depths:
        states, _ = cube.scramble(np.full(args.n, d), rng)
        r = run_astar(cube, net, states, device, cfg["weight"], cfg["batch"], cfg["maxNodes"])
        r["depth"] = d
        by.append(r)
        ml = f"{r['mean_len']:.2f}" if r["mean_len"] is not None else "—"
        print(f"  打亂 {d:>2} 步：解開 {r['solve_rate']:>6.1%}  平均 {ml:>6} 步  "
              f"展開 {r['mean_nodes']:>8.0f} 節點  {r['ms_per_cube']:>6.0f} ms")
    # 亂轉對照組固定 1000 個。--n 是給搜尋用的（一顆要好幾秒），亂轉很便宜，
    # 兩者綁在一起的話這個「0%」會變成只有幾十個樣本，撐不起那個結論。
    n_rw = 1000
    ok, steps = random_walk(cube, cube.scramble(np.full(n_rw, cube.cfg["scrambleMax"]), rng)[0],
                            np.random.default_rng(args.seed + 1), 10000)
    print(f"  （{n_rw} 個局面亂轉 10,000 步當對照：解開 {ok.mean():.2%}）")
    return {"size": 3, "n": args.n, "seed": args.seed, "by_depth": by,
            "random_walk_solve_rate": float(ok.mean()), "random_walk_n": n_rw,
            "weight": cfg["weight"], "max_nodes": cfg["maxNodes"],
            "trained_iters": ck.get("iter"), "n_params": net.n_params(),
            "depth_profile": ck.get("eval")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2, choices=[2, 3])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--weights", type=float, nargs="*", default=[0.0, 0.3, 0.6, 1.0])
    ap.add_argument("--depths", type=int, nargs="*", default=None)
    ap.add_argument("--batches", type=int, nargs="*", default=[1, 5, 20, 200],
                    help="批次大小對照組（只有 2x2x2 會跑）")
    ap.add_argument("--ablation", type=int, default=120, help="對照組用幾個局面")
    ap.add_argument("--max-nodes", type=int, default=None, help="蓋掉 config 的節點上限")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.max_nodes:
        CONFIG_OVERRIDE["maxNodes"] = a.max_nodes
    res = bench2(a, device) if a.size == 2 else bench3(a, device)
    p = HERE / "checkpoints" / f"benchmark-{a.size}x{a.size}.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n寫出 {p}")


if __name__ == "__main__":
    main()
