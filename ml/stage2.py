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

import torch

import corners
from benchmark import load_net
from cube import Cube
from heuristics import CornerPDB, EdgePDB, MaxHeuristic, korf_bound
from idastar import OptimalSolver
from search import bwas

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

    def best_endpoint(self, state, edge_h, slack: int = 2, beam: int = 3000):
        """在所有「角塊歸位」的終點裡，挑一個讓第二階段最好走的。

        上面那個 solve() 是**貪心**的：隨便挑一個距離少 1 的鄰居走下去。
        角塊會歸位，但落在哪個邊塊局面完全看運氣——這就是為什麼第二階段
        又長又難算。

        關鍵是：角塊歸位的終點**有非常多個**。角塊最短解通常不只一條，
        再放寬 slack 步就更多。這些終點的角塊都一樣（都歸位了），
        差別只在邊塊。挑一個邊塊下界最小的，第二階段就從近得多的地方開始。

        多花幾步在第一階段，換第二階段好走——這是這個拆法真正該有的樣子。

        回傳 (第一階段的走法, 終點局面, 那個終點的邊塊下界)。
        """
        cube = self.cube
        cur = np.asarray(state).reshape(1, -1).copy()
        d0 = int(self.dist[corners.index_of(cube, cur)[0]])
        budget = d0 + slack
        seq = np.zeros((1, 0), dtype=np.int8)
        best = None                              # (分數, 走法, 終點, 邊塊下界)
        if d0 == 0:
            return [], cur[0].copy(), float(edge_h(cur)[0])

        for g in range(budget):
            m = cube.n_actions
            rep = np.repeat(cur, m, axis=0)
            mv = np.tile(np.arange(m), len(cur))
            ch = cube.apply(rep, mv)
            cs = np.concatenate(
                [np.repeat(seq, m, axis=0), mv[:, None].astype(np.int8)], axis=1)

            # 還走得到嗎：剩下的預算夠不夠把角塊轉回去
            cd = self.dist[corners.index_of(cube, ch)]
            keep = (g + 1) + cd <= budget
            ch, cs, cd = ch[keep], cs[keep], cd[keep]
            if not len(ch):
                break
            _, u = np.unique(ch, axis=0, return_index=True)
            ch, cs, cd = ch[u], cs[u], cd[u]

            eh = edge_h(ch)
            done = cd == 0
            if done.any():
                # 分數就是「兩個階段合起來至少要幾步」的下界
                score = (g + 1) + eh[done]
                k = int(np.argmin(score))
                if best is None or score[k] < best[0]:
                    best = (float(score[k]), cs[done][k].copy(),
                            ch[done][k].copy(), float(eh[done][k]))

            # 排名留下 beam 個。角塊已經歸位的也留著——
            # 預算還有剩的話，繞開再轉回來可能落在更好的邊塊局面。
            rank = (g + 1) + cd + eh
            order = np.argsort(rank, kind="stable")[:beam]
            cur, seq = ch[order], cs[order]

        assert best is not None, "beam 裡找不到角塊歸位的終點——不可能"
        return [int(v) for v in best[1]], best[2], best[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=303)
    ap.add_argument("--n-exact", type=int, default=24, help="求直接最短解的方塊數")
    ap.add_argument("--n-full", type=int, default=6, help="兩階段都求最短的方塊數")
    ap.add_argument("--full-depths", type=int, nargs="*", default=[8, 9, 10])
    ap.add_argument("--full-budget", type=float, default=1800, help="第二階段每顆最多幾秒")
    ap.add_argument("--n-bound", type=int, default=4000, help="只取下界的方塊數")
    ap.add_argument("--budget", type=float, default=240, help="每個深度求精確解最多幾秒")
    ap.add_argument("--slack", type=int, default=2, help="第一階段可以比角塊最短解多走幾步")
    ap.add_argument("--beam", type=int, default=2000, help="挑終點時每層留幾個候選")
    ap.add_argument("--deep-depths", type=int, nargs="*", default=[20, 25, 30])
    ap.add_argument("--n-deep", type=int, default=4, help="深局面每個深度幾顆")
    ap.add_argument("--n-bound-pick", type=int, default=60, help="下界對照每個深度幾顆")
    ap.add_argument("--net-nodes", type=int, default=400_000, help="網路求解器的節點上限")
    a = ap.parse_args()

    cube = Cube(3)
    rng = np.random.default_rng(a.seed)
    st1 = CornerStage(cube)
    kb, cp = korf_bound(cube), CornerPDB(cube)
    e0, e1 = EdgePDB(cube, 0), EdgePDB(cube, 1)
    edge_lo = MaxHeuristic(e0, e1)          # 角塊歸位之後，就只剩這兩張在說話
    sv = OptimalSolver(cube, use_edges=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, _ = load_net(3, dev)

    # ── 0. 兩種第一階段的對照 ───────────────────────────────────
    #
    # 貪心：隨便走一條角塊最短解，落在哪個邊塊局面看運氣。
    # 挑過：在所有「角塊歸位」的終點裡（含放寬 slack 步的），
    #       挑一個邊塊下界最小的。多花幾步換第二階段好走。
    #
    print(f"\n[1/4] 第一階段怎麼收尾，決定第二階段有多難"
          f"（挑終點：放寬 {a.slack} 步、beam {a.beam}）")
    print(f"  {'打亂':>4} {'#':>2} {'最短解':>7} │ {'貪心 s1':>7} {'貪心 s2':>7} "
          f"{'合計':>5} {'節點':>12} {'秒':>7} │ {'挑過 s1':>7} {'挑過 s2':>7} "
          f"{'合計':>5} {'節點':>12} {'秒':>7}")
    exact = []
    for d in a.full_depths:
        s, _ = cube.scramble(np.full(a.n_full, d), rng)
        n1, after = st1.solve(s)
        for i in range(len(s)):
            opt, _ = sv.solve(s[i], 20)
            row = {"depth": d, "optimal": len(opt)}

            for tag, first, endpt in (
                    ("greedy", int(n1[i]), after[i]),
                    ("best", *(lambda r: (len(r[0]), r[1]))(
                        st1.best_endpoint(s[i], edge_lo, slack=a.slack, beam=a.beam)))):
                t0 = time.time()
                seq, nodes = sv.solve(endpt, 20, deadline=a.full_budget)
                dt = time.time() - t0
                r = {"stage1": first, "nodes": nodes, "secs": dt, "stage2": None}
                if seq is not None:
                    chk = np.asarray(endpt).reshape(1, -1).copy()
                    for m in seq:
                        chk = cube.apply(chk, np.array([m]))
                    assert cube.is_solved(chk)[0], "第二階段的解轉不回去"
                    r["stage2"] = len(seq)
                    r["total"] = first + len(seq)
                    r["detour"] = r["total"] - len(opt)
                row[tag] = r

            f = lambda r: (f"{r['stage1']:>7} {r['stage2']:>7} {r['total']:>5} "
                           f"{r['nodes']:>12,} {r['secs']:>7.1f}"
                           if r["stage2"] is not None else
                           f"{r['stage1']:>7} {'跑不完':>7} {'—':>5} "
                           f"{r['nodes']:>12,} {r['secs']:>7.1f}")
            print(f"  {d:>4} {i:>2} {len(opt):>7} │ {f(row['greedy'])} │ {f(row['best'])}",
                  flush=True)
            exact.append(row)

    for tag, label in (("greedy", "貪心"), ("best", "挑過")):
        ok = [r for r in exact if r[tag]["stage2"] is not None]
        if not ok:
            continue
        det = np.array([r[tag]["detour"] for r in ok], float)
        nd = np.array([r[tag]["nodes"] for r in ok], float)
        print(f"  {label}：{len(ok)}/{len(exact)} 顆算得完，平均繞路 {det.mean():+.2f} 步"
              f"（最少 {det.min():+.0f}、最多 {det.max():+.0f}）、"
              f"第二階段平均 {nd.mean():,.0f} 節點")

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

    # ── 2. 深局面：第二階段求不到最短，但求得到「一個解」 ──────────
    #
    # 打亂 20 步以上的方塊，第二階段用 IDA* 求最短是不可能的。
    # 但這個專案本來就有一支「找得到解、但不保證最短」的求解器——
    # 上一篇那個學出來的網路配批次加權 A*。用它，深局面的總步數就量得出來。
    #
    print(f"\n[3/4] 深局面：第二階段改用網路求解（找得到解，但不保證最短）")
    print(f"  {'打亂':>4} {'#':>2} │ {'貪心 s1':>7} {'貪心 s2':>7} {'合計':>5} {'節點':>10} "
          f"│ {'挑過 s1':>7} {'挑過 s2':>7} {'合計':>5} {'節點':>10} │ {'省':>5}")
    deep = []
    net_cfg = dict(cube.cfg["search"])
    for d in a.deep_depths:
        s, _ = cube.scramble(np.full(a.n_deep, d), rng)
        n1, after = st1.solve(s)
        for i in range(len(s)):
            row = {"depth": d}
            b_seq, b_end, _ = st1.best_endpoint(s[i], edge_lo, slack=a.slack, beam=a.beam)
            for tag, first, endpt in (("greedy", int(n1[i]), after[i]),
                                      ("best", len(b_seq), b_end)):
                seq, nodes = bwas(cube, net, endpt, dev, net_cfg["weight"],
                                  net_cfg["batch"], a.net_nodes)
                r = {"stage1": first, "stage2": None if seq is None else len(seq),
                     "nodes": int(nodes)}
                if seq is not None:
                    chk = np.asarray(endpt).reshape(1, -1).copy()
                    for m in seq:
                        chk = cube.apply(chk, np.array([m]))
                    assert cube.is_solved(chk)[0], "第二階段的解轉不回去"
                    r["total"] = first + len(seq)
                row[tag] = r
            g, b = row["greedy"], row["best"]
            save = (f"{g['total'] - b['total']:>+5}"
                    if g.get("total") and b.get("total") else f"{'—':>5}")
            f = lambda r: (f"{r['stage1']:>7} {r['stage2']:>7} {r['total']:>5} "
                           f"{r['nodes']:>10,}" if r["stage2"] is not None else
                           f"{r['stage1']:>7} {'找不到':>7} {'—':>5} {r['nodes']:>10,}")
            print(f"  {d:>4} {i:>2} │ {f(g)} │ {f(b)} │ {save}", flush=True)
            deep.append(row)

    for tag, label in (("greedy", "貪心"), ("best", "挑過")):
        ok = [r for r in deep if r[tag].get("total")]
        if ok:
            t = np.array([r[tag]["total"] for r in ok], float)
            nd = np.array([r[tag]["nodes"] for r in ok], float)
            print(f"  {label}：{len(ok)}/{len(deep)} 顆解得開，"
                  f"平均合計 {t.mean():.2f} 步、第二階段平均 {nd.mean():,.0f} 節點")

    # ── 2b. 下界的全貌（便宜，可以跑很多顆）────────────────────
    print(f"\n[3b/4] 兩種第一階段的下界對照（每個深度 {a.n_bound} 顆，只查表不搜尋）")
    print(f"  {'打亂':>4} {'貪心 s1':>8} {'貪心邊下界':>11} {'貪心合計下界':>13} "
          f"│ {'挑過 s1':>8} {'挑過邊下界':>11} {'挑過合計下界':>13}")
    bounds = []
    for d in (18, 25, 30):
        s, _ = cube.scramble(np.full(a.n_bound_pick, d), rng)
        n1, after = st1.solve(s)
        lo_g = edge_lo(after)
        n1b, lo_b = [], []
        for i in range(len(s)):
            seq, _, lo = st1.best_endpoint(s[i], edge_lo, slack=a.slack, beam=a.beam)
            n1b.append(len(seq)); lo_b.append(lo)
        n1b, lo_b = np.array(n1b, float), np.array(lo_b, float)
        row = {"depth": d, "cubes": len(s),
               "greedy": {"stage1": float(n1.mean()), "lower": float(lo_g.mean()),
                          "total_lower": float((n1 + lo_g).mean())},
               "best": {"stage1": float(n1b.mean()), "lower": float(lo_b.mean()),
                        "total_lower": float((n1b + lo_b).mean())}}
        bounds.append(row)
        print(f"  {d:>4} {n1.mean():>8.2f} {lo_g.mean():>11.2f} {(n1 + lo_g).mean():>13.2f} "
              f"│ {n1b.mean():>8.2f} {lo_b.mean():>11.2f} {(n1b + lo_b).mean():>13.2f}")

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
                               "bounds": bounds, "pick": {"slack": a.slack, "beam": a.beam},
                               "heuristic": {"before": before, "after": post}},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n寫出 {out}")


if __name__ == "__main__":
    main()
