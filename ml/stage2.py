"""先解角，再解邊——代價是什麼？

人類的 corners-first 是先把八顆角轉回去，再處理邊。這支程式量兩件事：

  1. **繞路多遠——精確值。** 兩個階段都求到最短：第一階段沿著角塊精確表走，
     第二階段用三張表跑 IDA*。跟直接求出的最短解一比，就是精確的繞路。
     只有淺的局面做得到（第二階段很貴），所以另外再補一組只給下界的深局面。

  2. **為什麼第二階段會變難。** 角塊一歸位，角塊表就永遠讀 0，
     三張表裡只剩兩張在說話。heuristic 掉了多少，量得出來。

    python ml/stage2.py
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

import corners
from cube import Cube
from heuristics import CornerPDB, EdgePDB, korf_bound
from idastar import OptimalSolver

HERE = pathlib.Path(__file__).resolve().parent


class CornerStage:
    """第一階段：只把八顆角轉回去，而且是最短的。

    角塊表是**精確**距離，所以不必搜尋——站在距離 d 的地方，
    一定有鄰居的距離是 d-1，一路走下去就是一條最短解。
    整批一起走，4000 顆方塊也只要幾秒。
    """

    def __init__(self, cube: Cube):
        self.cube = cube
        self.pdb = CornerPDB(cube)
        self.dist = np.asarray(self.pdb.dist)
        self.perm_tbl, self.ori_tbl = corners.build_move_tables(cube)

    def _d(self, p, o):
        return self.dist[p.astype(np.int64) * corners.N_ORI + o]

    def solve(self, states):
        """回傳 (每顆的步數, 角塊歸位後的局面)。"""
        cube, n = self.cube, len(states)
        idx = corners.index_of(cube, states)
        p, o = idx // corners.N_ORI, idx % corners.N_ORI
        cur, steps = states.copy(), np.zeros(n, int)
        while True:
            d = self._d(p, o)
            live = d > 0
            if not live.any():
                break
            mv = np.full(n, -1)
            for m in range(cube.n_actions):
                nxt = self._d(self.perm_tbl[p, m], self.ori_tbl[o, m])
                mv[live & (mv < 0) & (nxt == d - 1)] = m
            act = mv >= 0
            assert act[live].all(), "精確表上找不到距離少 1 的鄰居——不可能"
            cur[act] = cube.apply(cur[act], mv[act])
            steps[act] += 1
            idx = corners.index_of(cube, cur)
            p, o = idx // corners.N_ORI, idx % corners.N_ORI
        assert (self.pdb(cur) == 0).all(), "第一階段沒把角塊解完"
        return steps, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=303)
    ap.add_argument("--n-exact", type=int, default=24, help="求直接最短解的方塊數")
    ap.add_argument("--n-full", type=int, default=6, help="兩階段都求最短的方塊數")
    ap.add_argument("--full-depths", type=int, nargs="*", default=[8, 9, 10])
    ap.add_argument("--full-budget", type=float, default=1800, help="第二階段每顆最多幾秒")
    ap.add_argument("--n-bound", type=int, default=4000, help="只取下界的方塊數")
    ap.add_argument("--budget", type=float, default=240, help="每個深度求精確解最多幾秒")
    a = ap.parse_args()

    cube = Cube(3)
    rng = np.random.default_rng(a.seed)
    st1 = CornerStage(cube)
    kb, cp = korf_bound(cube), CornerPDB(cube)
    e0, e1 = EdgePDB(cube, 0), EdgePDB(cube, 1)
    sv = OptimalSolver(cube, use_edges=True)

    # ── 0. 精確的繞路：兩個階段都求到最短 ──────────────────────
    print("\n[1/4] 精確的繞路（兩個階段都是最短的）")
    print(f"  {'打亂':>4} {'#':>2} {'直接最短解':>11} {'階段1':>7} {'階段2':>7} "
          f"{'合計':>6} {'繞路':>7} {'階段2節點':>14} {'秒':>8}")
    exact = []
    for d in a.full_depths:
        s, _ = cube.scramble(np.full(a.n_full, d), rng)
        n1, after = st1.solve(s)
        for i in range(len(s)):
            t0 = time.time()
            opt, _ = sv.solve(s[i], 20)
            seq, nodes = sv.solve(after[i], 20, deadline=a.full_budget)
            dt = time.time() - t0
            if seq is None:
                print(f"  {d:>4} {i:>2} {len(opt):>11} {n1[i]:>7} {'跑不完':>7} "
                      f"{'—':>6} {'—':>7} {nodes:>14,} {dt:>8.1f}")
                exact.append({"depth": d, "optimal": len(opt), "stage1": int(n1[i]),
                              "stage2": None, "nodes": nodes, "secs": dt})
                continue
            chk = after[i:i + 1].copy()
            for m in seq:
                chk = cube.apply(chk, np.array([m]))
            assert cube.is_solved(chk)[0], "第二階段的解轉不回去"
            tot = n1[i] + len(seq)
            print(f"  {d:>4} {i:>2} {len(opt):>11} {n1[i]:>7} {len(seq):>7} "
                  f"{tot:>6} {tot - len(opt):>+7} {nodes:>14,} {dt:>8.1f}")
            exact.append({"depth": d, "optimal": len(opt), "stage1": int(n1[i]),
                          "stage2": len(seq), "total": int(tot),
                          "detour": int(tot - len(opt)), "nodes": nodes, "secs": dt})
    ok = [r for r in exact if r["stage2"] is not None]
    if ok:
        det = np.array([r["detour"] for r in ok], float)
        print(f"  {len(ok)}/{len(exact)} 顆兩階段都求到最短，平均繞路 {det.mean():+.2f} 步"
              f"（最少 {det.min():+.0f}、最多 {det.max():+.0f}）")

    # ── 1. 有證明的繞路：淺局面，直接最短解算得出來 ──────────────
    print("\n[2/4] 更深一點：直接最短解還算得動，但第二階段只能給下界")
    print(f"  {'打亂':>4} {'直接最短解':>11} {'階段1':>7} {'階段2下界':>11} "
          f"{'先解角至少':>11} {'至少繞路':>9}")
    detour = []
    for d in (8, 10, 11, 12, 13):
        s, _ = cube.scramble(np.full(a.n_exact, d), rng)
        opt, spent = [], 0.0
        for i in range(len(s)):
            t0 = time.time()
            seq, _ = sv.solve(s[i], 20)
            spent += time.time() - t0
            assert seq is not None, "20 步以內找不到最短解"
            opt.append(len(seq))
            if spent > a.budget:
                break
        s = s[:len(opt)]
        opt = np.array(opt, float)
        n1, after = st1.solve(s)
        lo2 = kb(after)
        row = {"depth": d, "cubes": len(opt), "optimal": float(opt.mean()),
               "stage1": float(n1.mean()), "stage2_lower": float(lo2.mean()),
               "cf_lower": float((n1 + lo2).mean()),
               "detour": float((n1 + lo2 - opt).mean())}
        detour.append(row)
        print(f"  {d:>4} {row['optimal']:>11.2f} {row['stage1']:>7.2f} "
              f"{row['stage2_lower']:>11.2f} {row['cf_lower']:>11.2f} "
              f"{row['detour']:>+9.2f}")
    print("  「至少繞路」是有證明的下限：階段 1 是最短的，階段 2 用下界，兩個加起來仍是下界。")

    # ── 2. 深局面只能給下界 ────────────────────────────────────
    print(f"\n[3/4] 深局面（{a.n_bound} 顆，兩邊都只能給下界）")
    print(f"  {'打亂':>4} {'階段1':>7} {'階段2下界':>11} {'先解角至少':>11}")
    deep = []
    for d in (18, 25, 30):
        s, _ = cube.scramble(np.full(a.n_bound, d), rng)
        n1, after = st1.solve(s)
        lo2 = kb(after)
        deep.append({"depth": d, "stage1": float(n1.mean()),
                     "stage2_lower": float(lo2.mean()), "cf_lower": float((n1 + lo2).mean())})
        print(f"  {d:>4} {n1.mean():>7.2f} {lo2.mean():>11.2f} {(n1 + lo2).mean():>11.2f}")

    # ── 3. 為什麼第二階段會變難 ────────────────────────────────
    print("\n[4/4] 角塊一歸位，三張表就只剩兩張在說話")
    s, _ = cube.scramble(np.full(a.n_bound, 30), rng)
    n1, after = st1.solve(s)
    before = {"corners": float(cp(s).mean()), "edges0": float(e0(s).mean()),
              "edges1": float(e1(s).mean()), "max3": float(kb(s).mean())}
    post = {"corners": float(cp(after).mean()), "edges0": float(e0(after).mean()),
            "edges1": float(e1(after).mean()), "max3": float(kb(after).mean())}
    print(f"  {'':<10} {'角塊表':>8} {'邊表 0':>8} {'邊表 1':>8} {'取大':>8}")
    for label, r in (("打亂的局面", before), ("角塊歸位後", post)):
        print(f"  {label:<10} {r['corners']:>8.2f} {r['edges0']:>8.2f} "
              f"{r['edges1']:>8.2f} {r['max3']:>8.2f}")
    print(f"  下界掉了 {before['max3'] - post['max3']:.2f} 步。"
          f"IDA* 的節點數是 (真正距離 − 下界) 的指數，所以第二階段貴得多——"
          f"上面 [1/4] 那些秒數就是代價。")

    out = HERE.parent / "out" / "stage2.json"
    out.write_text(json.dumps({"exact": exact, "detour": detour, "deep": deep,
                               "heuristic": {"before": before, "after": post}},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n寫出 {out}")


if __name__ == "__main__":
    main()
