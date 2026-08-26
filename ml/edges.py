"""3x3x3 的「只看 6 條邊」距離表 —— Korf 三張 pattern database 的第二、三張。

    python ml/edges.py            # 建兩張表，各約 43 MB
    python ml/edges.py --check    # 只驗證編碼與移動表

第一張是角塊（ml/corners.py）。這裡做剩下的兩張：12 條邊拆成兩半，
每半 6 條，各自整個窮舉。為什麼要拆兩半？因為 12 條邊一起算是
12! × 2^11 = 980,995,276,800 個狀態，數不完；拆成 6 條就只有

    12P6 × 2^6 = 665,280 × 64 = 42,577,920

比角塊那張（88,179,840）還小一半。而「只把這 6 條邊轉回去要幾步」
同樣是「解開整顆要幾步」的下界——解開整顆的每一步也都在動這 6 條邊。

三張表取最大，就是一個更緊的、仍然保證不高估的 heuristic。

── 順帶澄清一件事：3x3x3 沒有「解中心」這個階段 ──

六個面的中心貼紙在 12 種轉法之下**永遠不動**（ml/edges.py --check 會驗給你看）。
它們是釘死的參考點，不是要解的東西。中心會動要到 4x4x4 以上。
所以人類「先角、再邊」講完就沒了，機器這邊的第三段是「邊的另一半」。

── 為什麼這個座標也可以整個 BFS ──

跟角塊同一個道理：把狀態拆成兩個在轉動下各自獨立的座標。

    位置：6 條被追蹤的邊分別在哪個槽（有序的 6 元組）-> 12P6 = 665,280
    翻轉：那 6 條邊各自翻了沒                        -> 2^6 = 64

    新的 pos[k] = M(pos[k])                    只跟位置有關
    新的 flip[k] = flip[k] XOR d_M(pos[k])     翻轉量只跟「原本在哪個槽」有關

第二式成立的關鍵跟角塊一樣：翻轉量是**按槽**記的。
於是可以預先算兩張只有 665,280 × 12 的小表：

    permMove[posRank][m]  轉一步之後的位置編號
    flipMask[posRank][m]  轉一步之後那 6 個 bit 要 XOR 什麼

有了它們，四千兩百萬個狀態的鄰居就是兩次查表，BFS 純向量化，不必解碼。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cube import Cube

HERE = Path(__file__).resolve().parent
N_SLOTS = 12
K = 6                                   # 每張表追蹤幾條邊
N_POS = 665_280                         # 12P6
N_FLIP = 1 << K                         # 64
N_STATES = N_POS * N_FLIP               # 42,577,920

FACES = ["U", "D", "F", "B", "L", "R"]
BASIS = {
    "U": ((0, 1, 0), (1, 0, 0), (0, 0, -1)), "D": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    "F": ((0, 0, 1), (1, 0, 0), (0, 1, 0)), "B": ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
    "L": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)), "R": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
}
_NORMAL_FACE = {(0, 1, 0): 0, (0, -1, 0): 1, (0, 0, 1): 2,
                (0, 0, -1): 3, (-1, 0, 0): 4, (1, 0, 0): 5}


def _points(n=3):
    pts = {}
    for fi, f in enumerate(FACES):
        nor, rt, up = BASIS[f]
        for r in range(n):
            for c in range(n):
                du, dr = (n - 1) - 2 * r, 2 * c - (n - 1)
                pts[fi * n * n + r * n + c] = tuple(
                    nor[k] * n + rt[k] * dr + up[k] * du for k in range(3))
    return pts


def edge_slots():
    """12 個邊塊槽，每個給 2 片貼紙，第一片是「主要」那一片。

    主要那一片的挑法：碰到 U 或 D 的邊，取 U/D 面那一片；
    中層那四條（碰 F/B 與 L/R）取 F/B 面那一片。
    只要這個挑法固定，「翻了沒」就定義得出來，而且是按槽記的——
    這正是翻轉座標能跟位置座標分開的原因。
    """
    pts = _points()
    sign = lambda v: (v > 0) - (v < 0)
    groups = {}
    for i, p in pts.items():
        s = tuple(sign(v) for v in p)
        if s.count(0) != 1:                # 角塊 0 個零、中心 2 個零
            continue
        groups.setdefault(s, []).append(i)
    assert len(groups) == 12 and all(len(v) == 2 for v in groups.values())

    slots = []
    for v in sorted(groups):
        a, b = groups[v]
        fa, fb = a // 9, b // 9
        # U=0 D=1 優先；都不是的話 F=2 B=3 優先
        pri = a if fa in (0, 1) else (b if fb in (0, 1) else (a if fa in (2, 3) else b))
        sec = b if pri == a else a
        slots.append([pri, sec])
    return np.array(slots, dtype=np.int64)      # (12, 2)


SLOTS = edge_slots()
SOLVED_COLORS = SLOTS // 9                       # (12, 2) 解開時每個槽的兩個顏色
_ID = {}
for _i, _c in enumerate(SOLVED_COLORS.tolist()):
    _ID[tuple(sorted(_c))] = _i
assert len(_ID) == 12, "十二條邊的顏色組合有重複，slot 排錯了"

# 兩個顏色排序後 -> 是哪一條邊。查表用，避免在大批次上跑 Python 迴圈。
_EDGE_LUT = np.full(36, -1, dtype=np.int64)
for _k, _i in _ID.items():
    _EDGE_LUT[_k[0] * 6 + _k[1]] = _i


def read_edges(states: np.ndarray):
    """從貼紙陣列讀出 (perm, flip)，兩者都是「按槽」記的。

    perm[n, i] = 第 n 個局面裡，槽 i 放的是哪一條邊
    flip[n, i] = 那條邊翻了沒（主要貼紙有沒有落在主要位置）
    """
    st = states.reshape(1, -1) if states.ndim == 1 else states
    cols = st[:, SLOTS]                          # (N, 12, 2)
    lo = np.minimum(cols[:, :, 0], cols[:, :, 1]).astype(np.int64)
    hi = np.maximum(cols[:, :, 0], cols[:, :, 1]).astype(np.int64)
    perm = _EDGE_LUT[lo * 6 + hi]
    assert (perm >= 0).all(), "出現不存在的邊塊顏色組合 — 貼紙陣列壞了"
    # 翻轉：這條邊的「主要顏色」有沒有落在這個槽的主要貼紙上
    prim_color = SOLVED_COLORS[:, 0][perm]       # (N, 12) 這條邊本來的主要顏色
    flip = (cols[:, :, 0] != prim_color).astype(np.int64)
    return perm, flip


# ── 有序 6 元組（從 12 個槽裡挑，有序）的編碼 ──────────────────
_POPC = np.array([bin(i).count("1") for i in range(1 << N_SLOTS)], dtype=np.int64)


def pos_rank(pos: np.ndarray) -> np.ndarray:
    """pos (N, 6) 是六個相異的槽編號 -> 0..665,279。"""
    pos = np.atleast_2d(pos)
    used = np.zeros(len(pos), dtype=np.int64)
    r = np.zeros(len(pos), dtype=np.int64)
    for i in range(K):
        a = pos[:, i]
        lower = _POPC[used & ((1 << a) - 1)]     # 已用掉且比 a 小的槽有幾個
        r = r * (N_SLOTS - i) + (a - lower)
        used |= 1 << a
    return r


def pos_unrank(r: np.ndarray) -> np.ndarray:
    r = np.atleast_1d(r).astype(np.int64)
    digits = np.empty((len(r), K), dtype=np.int64)
    x = r.copy()
    for i in range(K - 1, -1, -1):
        digits[:, i] = x % (N_SLOTS - i)
        x //= (N_SLOTS - i)
    out = np.empty((len(r), K), dtype=np.int64)
    used = np.zeros(len(r), dtype=np.int64)
    for i in range(K):
        d = digits[:, i]
        # 找第 d 個還沒用掉的槽
        cnt = np.zeros(len(r), dtype=np.int64)
        pick = np.full(len(r), -1, dtype=np.int64)
        for s in range(N_SLOTS):
            free = ((used >> s) & 1) == 0
            hit = free & (cnt == d) & (pick < 0)
            pick = np.where(hit, s, pick)
            cnt = np.where(free & (pick < 0), cnt + 1, cnt)
        out[:, i] = pick
        used |= 1 << pick
    return out


def slot_moves(cube: Cube):
    """從貼紙引擎推出：一次轉動把槽 i 的內容搬到哪個槽，以及要 XOR 的翻轉量。"""
    A = cube.n_actions
    solved = cube.solved(1)
    to = np.empty((A, N_SLOTS), dtype=np.int64)
    dflip = np.empty((A, N_SLOTS), dtype=np.int64)
    for m in range(A):
        p, f = read_edges(cube.apply(solved, np.array([m])))
        for j in range(N_SLOTS):
            to[m, p[0, j]] = j                   # 解開時第 p[j] 條邊本來就在槽 p[j]
            dflip[m, p[0, j]] = f[0, j]
    return to, dflip


def build_move_tables(cube: Cube):
    """位置座標與翻轉遮罩的移動表。兩張表跟「追蹤哪 6 條邊」無關——
    座標記的是「有序的六個槽」，換一組邊只是換起點而已。"""
    to, dflip = slot_moves(cube)
    A = cube.n_actions
    all_pos = pos_unrank(np.arange(N_POS))       # (N_POS, 6)
    perm_tbl = np.empty((N_POS, A), dtype=np.int32)
    flip_tbl = np.empty((N_POS, A), dtype=np.uint8)
    for m in range(A):
        np_ = to[m][all_pos]                     # 新的槽
        perm_tbl[:, m] = pos_rank(np_)
        bits = dflip[m][all_pos]                 # (N_POS, 6) 每條邊要 XOR 的 bit
        flip_tbl[:, m] = (bits << np.arange(K)).sum(axis=1).astype(np.uint8)
    return perm_tbl, flip_tbl


def index_of(cube: Cube, states: np.ndarray, tracked: np.ndarray) -> np.ndarray:
    """貼紙陣列 -> 這半邊的狀態編號 0..42,577,919。"""
    perm, flip = read_edges(states)
    n = perm.shape[0]
    # 每條被追蹤的邊現在在哪個槽（perm 是「槽 -> 邊」，這裡要反過來）
    inv = np.empty_like(perm)
    np.put_along_axis(inv, perm, np.broadcast_to(np.arange(N_SLOTS), perm.shape), axis=1)
    pos = inv[:, tracked]                        # (N, 6)
    fl = np.take_along_axis(flip, pos, axis=1)   # 那六條邊各自翻了沒
    return pos_rank(pos) * N_FLIP + (fl << np.arange(K)).sum(axis=1)


def build(cube: Cube, tracked: np.ndarray, perm_tbl, flip_tbl, verbose=True):
    dist = np.full(N_STATES, 255, dtype=np.uint8)
    start = index_of(cube, cube.solved(1), tracked)[0]
    dist[start] = 0
    frontier = np.array([start], dtype=np.int64)
    hist = [1]
    d = 0
    A = cube.n_actions
    while len(frontier):
        d += 1
        nxt = []
        for lo in range(0, len(frontier), 4_000_000):
            f = frontier[lo:lo + 4_000_000]
            p, fl = f // N_FLIP, (f % N_FLIP).astype(np.uint8)
            for m in range(A):
                ch = perm_tbl[p, m].astype(np.int64) * N_FLIP + (fl ^ flip_tbl[p, m])
                ch = ch[dist[ch] == 255]
                if not len(ch):
                    continue
                ch = np.unique(ch)
                ch = ch[dist[ch] == 255]
                dist[ch] = d
                nxt.append(ch)
        frontier = np.unique(np.concatenate(nxt)) if nxt else np.array([], dtype=np.int64)
        if not len(frontier):
            break
        hist.append(len(frontier))
        if verbose:
            print(f"  深度 {d:>2}: {len(frontier):>10,} 個新狀態   （累計 {sum(hist):>11,}）")
    return dist, hist


def path(half: int) -> Path:
    return HERE / "checkpoints" / f"edges3_{half}.npy"


def load(half: int):
    p = path(half)
    if not p.exists():
        raise SystemExit(f"缺 {p} — 先跑 python ml/edges.py")
    return np.load(p, mmap_mode="r")


TRACKED = (np.arange(0, 6), np.arange(6, 12))


def write_stats(stats):
    sp = HERE / "checkpoints" / "edges-build.json"
    sp.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n寫出 {sp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--stats", action="store_true",
                    help="不建表，直接讀現成的兩張表產出 edges-build.json")
    a = ap.parse_args()
    cube = Cube(3)
    rng = np.random.default_rng(0)

    # ── 先把「中心塊不會動」驗出來，因為常有人以為那是第三個階段 ──
    fixed = [i for i in range(54) if all(cube.perms[m][i] == i for m in range(cube.n_actions))]
    assert fixed == [4, 13, 22, 31, 40, 49], fixed
    print(f"中心塊檢查：{len(fixed)} 片貼紙在 12 種轉法下永遠不動 = 六個面的正中央")
    print("           所以 3x3x3 沒有「解中心」這個階段，中心是釘死的參考點。\n")

    st, _ = cube.scramble(rng.integers(1, 31, size=2000), rng)
    perm, flip = read_edges(st)
    assert (np.sort(perm, axis=1) == np.arange(12)).all(), "邊塊置換不是一個排列"
    assert (flip.sum(axis=1) % 2 == 0).all(), "邊塊總翻轉不是偶數 — 讀錯了"
    print(f"編碼檢查：{len(st)} 個局面，邊塊置換合法、總翻轉恆為偶數")

    perm_tbl, flip_tbl = build_move_tables(cube)
    for h, tracked in enumerate(TRACKED):
        idx = index_of(cube, st, tracked)
        mv = rng.integers(0, cube.n_actions, size=len(st))
        want = index_of(cube, cube.apply(st, mv), tracked)
        p, fl = idx // N_FLIP, (idx % N_FLIP).astype(np.uint8)
        got = perm_tbl[p, mv].astype(np.int64) * N_FLIP + (fl ^ flip_tbl[p, mv])
        assert (want == got).all(), f"第 {h} 半的座標移動表跟貼紙引擎對不起來"
    print(f"移動表檢查：兩半都通過（走一步逐一比對 {len(st)} 個局面）")
    print(f"狀態空間 12P6 × 2^6 = {N_POS:,} × {N_FLIP} = {N_STATES:,}（角塊那張是 88,179,840）")
    if a.check:
        return

    stats = {"states": N_STATES, "n_pos": N_POS, "n_flip": N_FLIP, "halves": []}
    if a.stats:
        # 表已經在了（而且可能被別的程式 mmap 著，覆寫不了），只重算統計。
        for h, tracked in enumerate(TRACKED):
            dist = np.asarray(load(h))
            stats["halves"].append({
                "half": h, "edges": [int(v) for v in tracked], "secs": None,
                "mb": round(path(h).stat().st_size / 1e6), "max": int(dist.max()),
                "mean": float(dist.mean()),
                "hist": [int(v) for v in np.bincount(dist, minlength=int(dist.max()) + 1)]})
            assert sum(stats["halves"][h]["hist"]) == N_STATES
            print(f"  第 {h} 半：平均 {dist.mean():.3f} 步、最遠 {dist.max()}、"
                  f"{stats['halves'][h]['mb']} MB")
        write_stats(stats)
        return

    for h, tracked in enumerate(TRACKED):
        print(f"\n第 {h} 半（邊 {tracked[0]}~{tracked[-1]}）全狀態 BFS…")
        t0 = time.time()
        dist, hist = build(cube, tracked, perm_tbl, flip_tbl)
        assert sum(hist) == N_STATES, f"只走到 {sum(hist):,}，應該是 {N_STATES:,}"
        p = path(h)
        p.parent.mkdir(parents=True, exist_ok=True)
        secs = time.time() - t0
        np.save(p, dist)
        print(f"  全部 {sum(hist):,} 個狀態，最遠 {len(hist) - 1} 步，{secs:.0f}s")
        print(f"  寫出 {p} ({p.stat().st_size / 1e6:.0f} MB)"
              f"　平均 {dist.mean():.3f} 步、中位 {int(np.median(dist))}、最遠 {dist.max()}")
        stats["halves"].append({
            "half": h, "edges": [int(v) for v in tracked], "secs": round(secs, 1),
            "mb": round(p.stat().st_size / 1e6), "max": int(dist.max()),
            "mean": float(dist.mean()), "hist": [int(v) for v in hist]})

    sp = HERE / "checkpoints" / "edges-build.json"
    sp.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n寫出 {sp}")


if __name__ == "__main__":
    main()
