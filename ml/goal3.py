"""三張表 vs 一張表：量「下界緊一點」到底值多少。

角塊表平均 10.67，三張取大平均 10.78——只多 0.1 步，看起來不痛不癢。
但 IDA* 要展開的節點數是 (真正距離 - 下界) 的**指數**，
所以這 0.1 步在搜尋上是好幾個數量級。這支程式就是去量那個數量級。

    python ml/goal3.py            # 有快取，跑過的階段不重跑
    python ml/goal3.py --force scale
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from cube import Cube
from heuristics import CornerPDB, EdgePDB, korf_bound
from idastar import OptimalSolver

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "out"


def cached(name, fn, force=()):
    fp = HERE / "checkpoints" / f"goal3-{name}.json"
    if fp.exists() and name not in force:
        return json.loads(fp.read_text(encoding="utf-8"))
    v = fn()
    fp.write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding="utf-8")
    return v


def bounds(cube, seed):
    """三張表各自的值分佈，以及取大之後緊了多少。"""
    rng = np.random.default_rng(seed)
    hs = {"corners": CornerPDB(cube), "edges0": EdgePDB(cube, 0),
          "edges1": EdgePDB(cube, 1), "max3": korf_bound(cube)}
    st, _ = cube.scramble(np.full(20000, 30), rng)
    out = {}
    for k, h in hs.items():
        v = h(st)
        out[k] = {"name": h.name, "mean": float(v.mean()), "max": int(v.max()),
                  "hist": np.bincount(v.astype(int), minlength=15).tolist()}
    v = np.stack([hs[k](st) for k in ("corners", "edges0", "edges1")])
    best = v.max(0)
    out["winner"] = {"corners": int((v[0] >= best).sum()),
                     "edges": int((v[1:].max(0) >= best).sum()),
                     "edges_only": int((v[1:].max(0) > v[0]).sum()), "n": len(st)}
    return out


def scale(cube, seed, depths, n, budget):
    """同一批方塊，兩種下界，量節點數怎麼隨深度長。

    一張表在某個深度跑到超時之後就不再用它——那件事本身就是結果。
    每一顆都給 deadline，不然一顆跑起來就收不了工（深度 14 只用角塊表就是這樣）。
    """
    rng = np.random.default_rng(seed)
    sv = {"corners": OptimalSolver(cube, use_edges=False),
          "max3": OptimalSolver(cube, use_edges=True)}
    rows = []
    for d in depths:
        st, _ = cube.scramble(np.full(n, d), rng)
        row = {"depth": d, "n": n}
        for k in list(sv):
            nodes, secs, lens, done = [], [], [], True
            for i in range(n):
                t0 = time.time()
                seq, nd = sv[k].solve(st[i], 20, deadline=budget - sum(secs))
                dt = time.time() - t0
                if seq is None:                       # 時間到，這一顆沒解完
                    secs.append(dt)
                    done = False
                    break
                cur = st[i:i + 1].copy()
                for m in seq:
                    cur = cube.apply(cur, np.array([m]))
                assert cube.is_solved(cur)[0], "回傳的解轉不回去"
                nodes.append(nd); secs.append(dt); lens.append(len(seq))
                if sum(secs) > budget:
                    done = False
                    break
            if not nodes:                             # 一顆都沒跑完
                print(f"  打亂 {d:>2} 步 · {k:<7} · 一顆都跑不完（{budget:.0f}s 上限）", flush=True)
                del sv[k]
                continue
            row[k] = {"nodes": float(np.mean(nodes)), "secs": float(np.mean(secs[:len(nodes)])),
                      "len": float(np.mean(lens)), "cubes": len(nodes), "complete": done,
                      # 每一顆都留著：兩種下界跑完的顆數可能不一樣，
                      # 「省幾倍」只能在兩邊都跑完的那幾顆上算，不然是拿不同方塊在比
                      "per_cube": [int(v) for v in nodes]}
            print(f"  打亂 {d:>2} 步 · {k:<7} · {len(nodes)} 顆 · "
                  f"平均 {np.mean(lens):.2f} 步 · {np.mean(nodes):>13,.0f} 節點 · "
                  f"{np.mean(secs[:len(nodes)]):>8.2f}s"
                  f"{'' if done else '  （超時，只跑了這幾顆）'}", flush=True)
            if not done:
                del sv[k]
        rows.append(row)
    return rows


def common_ratio(c, m):
    """只在兩種下界都跑完的那幾顆上比。顆數不一樣的話，直接比平均是拿不同方塊在比。"""
    if not c or not m:
        return None
    n = min(len(c.get("per_cube", [])), len(m.get("per_cube", [])))
    if not n:
        return None
    a, b = np.mean(c["per_cube"][:n]), np.mean(m["per_cube"][:n])
    return float(a / b) if b else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=606)
    ap.add_argument("--force", nargs="*", default=[])
    ap.add_argument("--budget", type=float, default=900, help="每個深度每種下界最多花幾秒")
    a = ap.parse_args()
    cube = Cube(3)
    force = set(a.force)

    print("\n[1/2] 三張表的值分佈（20000 個隨機局面）")
    b = cached("bounds", lambda: bounds(cube, a.seed), force)
    for k in ("corners", "edges0", "edges1", "max3"):
        print(f"  {b[k]['name']:<28} 平均 {b[k]['mean']:6.3f}  最大 {b[k]['max']:>2}")
    w = b["winner"]
    print(f"  取大時角塊表就是最大值的比例：{w['corners'] / w['n'] * 100:.1f}%")
    print(f"  邊塊表比角塊表更大（真的有貢獻）的比例：{w['edges_only'] / w['n'] * 100:.1f}%")

    print("\n[2/2] IDA* 節點數 vs 打亂深度")
    s = cached("scale",
               lambda: scale(cube, a.seed + 1, [8, 10, 11, 12, 13, 14, 15, 16], 8, a.budget),
               force)
    print(f"\n  {'打亂':>4} {'最短解':>7} {'角塊表節點':>16} {'三張表節點':>14} {'省幾倍':>9}")
    for r in s:
        c, m = r.get("corners"), r.get("max3")
        if not m:
            continue
        ratio = f"{common_ratio(c, m):,.0f}×" if common_ratio(c, m) else "—"
        cn = (f"{c['nodes']:,.0f}" + ("" if c["complete"] else "*")) if c else "跑不完"
        print(f"  {r['depth']:>4} {m['len']:>7.2f} {cn:>16} {m['nodes']:>14,.0f} {ratio:>9}")
    print("  * 代表在時間上限內只跑完一部分方塊")

    OUT.mkdir(exist_ok=True)
    for r in s:
        r["ratio"] = common_ratio(r.get("corners"), r.get("max3"))
    (OUT / "goal3.json").write_text(
        json.dumps({"bounds": b, "scale": s}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n寫出 {OUT / 'goal3.json'}")


if __name__ == "__main__":
    main()
