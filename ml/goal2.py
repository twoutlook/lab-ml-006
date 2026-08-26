"""GOAL002 的四組實驗：把「人類先解角、再解邊」變成可量測的東西。

    python ml/goal2.py --quick     # 小樣本，幾分鐘
    python ml/goal2.py             # 正式樣本，約 1~2 小時

輸出 ml/checkpoints/goal2.json，影片與圖文版都讀它。

起點是一個很老的觀察：1974 年 Ernő Rubik 自己第一次解開他的方塊，用的是
corners-first；1981 年 Waterman method 是純 CF 派。而 Korf 的最優解演算法
把方塊拆成三個 pattern database——角塊、6 條邊、另外 6 條邊。
**人腦拿這個拆法記手順，機器拿它當下界。同一個直覺，兩種用途。**

這裡做的是第一階段（角塊），因為：
  ・8 個角只有 8! × 3^7 = 88,179,840 個狀態，可以整個 BFS 出來（ml/corners.py）
  ・「只把角塊轉回去要幾步」是「整顆方塊要幾步」的下界，而且保證不高估
  ・所以它同時是一個 admissible 的 heuristic，和一把量尺

四組實驗：

  E1  當 heuristic 比一比：角塊表 vs 學出來的網路 vs 兩個取大的
  E2  **3x3x3 第一次有正確答案**：用角塊表 + 課本 A* 證明淺局面的最短解，
      再拿它去量學出來的網路差多少（就像 2x2x2 那張圖）
  E3  深局面的落差夾擠：下界 <= 最短解 <= 找到的解
  E4  人類的分階段解法值多少：先解角再解完 vs 一次解完
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
from heuristics import CornerPDB, NetHeuristic, MaxHeuristic
from search import bwas
from idastar import OptimalSolver

HERE = Path(__file__).resolve().parent
OUT = HERE / "checkpoints" / "goal2.json"
CPU = torch.device("cpu")


def cached(name, fn, force=False):
    """每一段實驗各自存一個檔，跑過的就沿用。

    這幾段加起來要跑快一小時，而且中途被中斷過一次——整批重來太貴。
    每段各自用獨立的亂數種子（見 main() 的 rng_for），所以「跳過前面幾段」
    不會改變後面幾段抽到的局面，續跑出來的結果跟一次跑完完全一樣。
    """
    fp = HERE / "checkpoints" / f"goal2-{name}.json"
    if fp.exists() and not force:
        print(f"  （沿用 {fp.name}）")
        return json.loads(fp.read_text(encoding="utf-8"))
    v = fn()
    fp.write_text(json.dumps(v, ensure_ascii=False), encoding="utf-8")
    return v


def solve_corners_greedy(cube, pdb, state):
    """把角塊解開。不用搜尋——角塊表就是這個子目標的精確距離，
    所以每一步挑一個讓距離減 1 的轉法就好，而且保證是最短的。"""
    cur = state.reshape(1, -1).copy()
    seq = []
    d = int(pdb(cur)[0])
    while d > 0:
        kids = cube.expand(cur)[0]
        hv = pdb(kids)
        m = int(np.argmin(hv))
        assert hv[m] == d - 1, "角塊表不一致：找不到讓距離減 1 的轉法"
        cur = kids[m:m + 1]
        seq.append(m)
        d -= 1
    return seq, cur[0]


def e1_heuristics(cube, pdb, nh, n, rng, cfg):
    """同一批隨機方塊，換三種估計法，其他都不變。"""
    st, _ = cube.scramble(np.full(n, cube.cfg["scrambleMax"]), rng)
    rows = []
    for h in (pdb, nh, MaxHeuristic(pdb, nh)):
        lens, nodes, ok = [], [], 0
        t0 = time.time()
        for i in range(n):
            seq, e = bwas(cube, h, st[i], CPU, cfg["weight"], cfg["batch"], cfg["maxNodes"])
            nodes.append(e)
            if seq is None:
                continue
            assert verify(cube, st[i], seq)
            ok += 1
            lens.append(len(seq))
        rows.append({"name": h.name, "admissible": bool(h.admissible),
                     "solve_rate": ok / n,
                     "mean_len": float(np.mean(lens)) if lens else None,
                     "mean_nodes": float(np.mean(nodes)),
                     "ms_per_cube": (time.time() - t0) / n * 1000,
                     "mean_h": float(np.mean(h(st)))})
        print(f"  {h.name:<28} 解開 {ok / n:>5.0%}  "
              f"平均 {rows[-1]['mean_len'] if lens else float('nan'):>5.1f} 步  "
              f"展開 {np.mean(nodes):>9,.0f} 節點  h 平均 {rows[-1]['mean_h']:.2f}"
              f"  {'（不高估）' if h.admissible else ''}")
    return rows


def e2_ground_truth(cube, pdb, nh, plan, seed0, max_depth=20):
    """用角塊表 + IDA* 求出精確最短解，再量網路差多少。

    最短解保證要兩個條件：heuristic 不高估（角塊表符合，而且它還是 consistent 的），
    以及搜尋本身照 f 值走到底（IDA* 的 iterative deepening 就是在做這件事）。

    為什麼不用前面那個 A*？因為它每個節點約 765 微秒，打亂 11 步就跑不完。
    IDA* 拿掉 open/closed list、角塊座標增量更新，實測每秒 8~13 萬個節點——
    快 100 倍，才走得到「網路開始失準」的那個區域。細節在 ml/idastar.py。

    plan 是 [(打亂步數, 要幾顆), ...]：越深越貴（14 步一顆約 100 秒），所以越深抽越少。
    """
    sv = OptimalSolver(cube, pdb.dist)
    recs = []
    for d, cnt in plan:
        def run(d=d, cnt=cnt):
            # 每個深度自己一組亂數，這樣少跑幾個深度也不會影響其他深度
            r = np.random.default_rng(seed0 + 1000 + d)
            st, _ = cube.scramble(np.full(cnt, d), r)
            out, t0 = [], time.time()
            for i in range(cnt):
                seq, e = sv.solve(st[i], max_depth)
                if seq is None:
                    continue
                assert verify(cube, st[i], seq)
                out.append({"depth": int(d), "true": len(seq),
                            "h_net": float(nh(st[i:i + 1])[0]),
                            "h_pdb": float(pdb(st[i:i + 1])[0]), "nodes": int(e)})
            print(f"  打亂 {d:>2} 步：{len(out)}/{cnt} 顆求出精確最短解，"
                  f"平均 {np.mean([o['true'] for o in out]):.2f} 步，"
                  f"展開 {np.mean([o['nodes'] for o in out]):,.0f} 節點，"
                  f"{(time.time() - t0) / cnt:.1f} 秒/顆", flush=True)
            return out
        recs += cached(f"e2-d{d}", run)
    return recs


def profile(recs, key):
    """照真實最短解分組，看估計值怎麼跟著動——跟 2x2x2 那張圖同一個做法。"""
    out = []
    for d in sorted({r["true"] for r in recs}):
        g = [r for r in recs if r["true"] == d]
        if len(g) < 5:
            continue
        v = np.array([r[key] for r in g], dtype=np.float64)
        out.append({"d": int(d), "n": len(g), "mean": round(float(v.mean()), 3),
                    "std": round(float(v.std()), 3),
                    "over": round(float((v > d + 1e-6).mean()), 4)})
    return out


def e3_squeeze(cube, pdb, nh, n, rng, cfg):
    """深局面沒有正確答案，但夾得出來：下界 <= 最短解 <= 找到的解。"""
    st, _ = cube.scramble(np.full(n, cube.cfg["scrambleMax"]), rng)
    rows = []
    for i in range(n):
        seq, _ = bwas(cube, nh, st[i], CPU, cfg["weight"], cfg["batch"], cfg["maxNodes"])
        if seq is None:
            continue
        assert verify(cube, st[i], seq)
        rows.append({"lower": float(pdb(st[i:i + 1])[0]), "found": len(seq)})
    lo = np.array([r["lower"] for r in rows])
    up = np.array([r["found"] for r in rows])
    print(f"  {len(rows)} 顆：下界平均 {lo.mean():.2f}、找到的解平均 {up.mean():.2f}、"
          f"落差上界平均 {(up - lo).mean():.2f} 步")
    return {"n": len(rows), "mean_lower": float(lo.mean()), "mean_found": float(up.mean()),
            "mean_gap_upper": float((up - lo).mean()), "rows": rows}


def e4_staged(cube, pdb, nh, n, rng, cfg):
    """人類的分階段：先把角塊解開，再解完剩下的。跟一次解完比。"""
    st, _ = cube.scramble(np.full(n, cube.cfg["scrambleMax"]), rng)
    one, two = [], []
    t0 = time.time()
    for i in range(n):
        seq, e = bwas(cube, nh, st[i], CPU, cfg["weight"], cfg["batch"], cfg["maxNodes"])
        if seq is not None:
            assert verify(cube, st[i], seq)
            one.append({"len": len(seq), "nodes": e})
    t_one = (time.time() - t0) / n * 1000

    t0 = time.time()
    for i in range(n):
        s1, mid = solve_corners_greedy(cube, pdb, st[i])
        seq, e = bwas(cube, nh, mid, CPU, cfg["weight"], cfg["batch"], cfg["maxNodes"])
        if seq is None:
            continue
        assert verify(cube, mid, seq)
        two.append({"stage1": len(s1), "stage2": len(seq),
                    "len": len(s1) + len(seq), "nodes": e})
    t_two = (time.time() - t0) / n * 1000

    f = lambda rs, k: float(np.mean([r[k] for r in rs])) if rs else None
    res = {"n": n,
           "one_shot": {"solved": len(one), "mean_len": f(one, "len"),
                        "mean_nodes": f(one, "nodes"), "ms": t_one},
           "staged": {"solved": len(two), "mean_len": f(two, "len"),
                      "mean_stage1": f(two, "stage1"), "mean_stage2": f(two, "stage2"),
                      "mean_nodes": f(two, "nodes"), "ms": t_two}}
    print(f"  一次解完：{len(one)}/{n} 顆，平均 {f(one, 'len'):.2f} 步，"
          f"展開 {f(one, 'nodes'):,.0f} 節點，{t_one:.0f} ms")
    print(f"  先解角再解完：{len(two)}/{n} 顆，平均 {f(two, 'len'):.2f} 步"
          f"（角 {f(two, 'stage1'):.2f} + 其餘 {f(two, 'stage2'):.2f}），"
          f"第二段展開 {f(two, 'nodes'):,.0f} 節點，{t_two:.0f} ms")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--force", action="store_true", help="不理會已存檔的分段結果，整批重跑")
    a = ap.parse_args()
    q = a.quick
    cube = Cube(3)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, ck = load_net(3, dev)
    pdb = CornerPDB(cube)
    nh = NetHeuristic(net, dev)
    cfg = dict(cube.cfg["search"], maxNodes=400_000)
    # 每段一個獨立的種子。這樣「跳過已經跑過的段」不會改變其他段抽到的局面。
    rng_for = lambda k: np.random.default_rng(a.seed + k)

    res = {"corner_pdb": {"states": corners.N_STATES,
                          "max": int(pdb.dist.max()), "mean": float(pdb.dist.mean()),
                          "hist": [int((np.asarray(pdb.dist) == d).sum())
                                   for d in range(int(pdb.dist.max()) + 1)]}}
    print(f"角塊表：{corners.N_STATES:,} 個狀態，上帝之數 {res['corner_pdb']['max']}，"
          f"平均 {res['corner_pdb']['mean']:.3f} 步")

    print("\nE1 三種估計法當 heuristic（同一批隨機方塊）")
    res["e1"] = cached("e1", lambda: e1_heuristics(cube, pdb, nh, 15 if q else 60, rng_for(1), cfg))

    print("\nE2 用角塊表證明最短解 —— 3x3x3 第一次有正確答案")
    # 越深越貴：打亂 12 步約 3 秒一顆、14 步約 100 秒一顆，所以深的抽少一點。
    plan = [(4, 5), (8, 5), (12, 3)] if q else [
        (2, 40), (4, 40), (6, 40), (8, 40), (9, 40), (10, 40),
        (11, 30), (12, 25), (13, 15), (14, 8)]
    recs = e2_ground_truth(cube, pdb, nh, plan, a.seed)
    err = np.array([r["h_net"] - r["true"] for r in recs])
    res["e2"] = {"n": len(recs), "records": recs,
                 "mae": round(float(np.abs(err).mean()), 4),
                 "bias": round(float(err.mean()), 4),
                 "over_rate": round(float((err > 1e-6).mean()), 4),
                 "within_1": round(float((np.abs(err) <= 1).mean()), 4),
                 "by_true_net": profile(recs, "h_net"),
                 "by_true_pdb": profile(recs, "h_pdb")}
    print(f"  網路 vs 精確答案：平均差 {res['e2']['mae']:.2f} 步、"
          f"{res['e2']['within_1'] * 100:.1f}% 在 1 步內、{res['e2']['over_rate'] * 100:.1f}% 高估")

    print("\nE3 深局面的落差夾擠")
    res["e3"] = cached("e3", lambda: e3_squeeze(cube, pdb, nh, 15 if q else 60, rng_for(3), cfg))

    print("\nE4 人類的分階段解法值多少")
    res["e4"] = cached("e4", lambda: e4_staged(cube, pdb, nh, 15 if q else 60, rng_for(4), cfg))

    res["net"] = {"params": net.n_params(), "iters": ck.get("iter")}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n寫出 {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
