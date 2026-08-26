"""3x3x3 的「只看角塊」距離表 —— 一個精確、而且**保證不高估**的 heuristic。

    python ml/corners.py            # 建表，約 3 分鐘，輸出 88 MB
    python ml/corners.py --check    # 只驗證，不建表

為什麼是角塊？因為這是人類解方塊的第一步。1974 年 Ernő Rubik 自己第一次
解開他的方塊用的就是 corners-first，1981 年的 Waterman method 是純 CF 派。
而機器那邊，Korf 的最優解演算法把方塊拆成三個 pattern database：
角塊、6 條邊、另外 6 條邊。**人腦拿它記手順，機器拿它當下界，是同一個直覺。**

關鍵數字：8 個角塊的狀態數是

    8! × 3^7 = 40,320 × 2,187 = 88,179,840

跟 2x2x2 的 3,674,160 同一個量級，小到可以整個 BFS 出來。
而「只把角塊轉回去要幾步」是「整顆方塊要幾步」的**下界**——
因為解開整顆方塊的每一步也都是在轉角塊，不可能更少。

這件事在這個專案裡的用途有兩個，而且第二個比第一個重要：

1. 當 heuristic。它保證不高估（admissible），所以 A* 的最短解保證是真的回來了——
   而學出來的那個網路有 23% 的局面高估，保證本來就不成立。

2. **當量尺。** 圖文版目前寫著「3x3x3 的解比最短解長幾步，量不出來」。
   有了下界就量得出來了：解 27 步、下界 X 步 -> 落差最多 27-X 步。
   這是 3x3x3 上第一個「精確」的東西。

── 為什麼可以整個 BFS ──

角塊的狀態是「哪顆角在哪個位置」(perm) 加上「每個位置的角轉了幾度」(ori)。
關鍵是這兩個座標在轉動下**各自獨立**：

    新的 perm[M(i)] = perm[i]                  只跟 perm 有關
    新的 ori[M(i)]  = (ori[i] + twist_M(i)) % 3   只跟 ori 有關

第二式成立是因為 ori 是「按位置」記的：一次轉動把位置的內容搬去別的位置，
再加上一個只跟位置與轉法有關的固定扭轉量——跟「哪顆角在那裡」無關。
所以 40,320 × 12 和 2,187 × 12 兩張小表就足以推出全部 88,179,840 個狀態的鄰居，
整個 BFS 可以純向量化，不用一顆一顆算。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cube import Cube

HERE = Path(__file__).resolve().parent
OUT = HERE / "checkpoints" / "corners3.npy"

N_PERM = 40320          # 8!
N_ORI = 2187            # 3^7（第 8 個角的扭轉由「總和 ≡ 0 (mod 3)」決定）
N_STATES = N_PERM * N_ORI

FACES = ["U", "D", "F", "B", "L", "R"]
BASIS = {
    "U": ((0, 1, 0), (1, 0, 0), (0, 0, -1)), "D": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    "F": ((0, 0, 1), (1, 0, 0), (0, 1, 0)), "B": ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
    "L": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)), "R": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
}


def _sticker_points(n=3):
    pts = {}
    for fi, f in enumerate(FACES):
        nor, rt, up = BASIS[f]
        for r in range(n):
            for c in range(n):
                du, dr = (n - 1) - 2 * r, 2 * c - (n - 1)
                pts[fi * n * n + r * n + c] = tuple(
                    nor[k] * n + rt[k] * dr + up[k] * du for k in range(3))
    return pts


def corner_slots():
    """8 個角塊位置，每個給 3 個貼紙編號，順序是固定的。

    順序規則：第一片一定是 U 或 D 面那一片，另外兩片照右手定則排
    （n1 × n2 = n3）。有了固定順序，「扭轉了幾格」才定義得出來。
    """
    pts = _sticker_points()
    sign = lambda v: (v > 0) - (v < 0)
    groups = {}
    for i, p in pts.items():
        s = tuple(sign(v) for v in p)
        if 0 in s:
            continue                      # 邊塊或中心，不是角塊
        groups.setdefault(s, []).append(i)
    assert len(groups) == 8 and all(len(v) == 3 for v in groups.values())

    slots = []
    for v in sorted(groups):              # 固定一個位置編號順序
        vx, vy, vz = v
        ex, ey, ez = (vx, 0, 0), (0, vy, 0), (0, 0, vz)
        # cross(ey, ex) = (0,0,-vy*vx)；要等於 ez 的話 vx*vy*vz 必須是 -1
        rest = [ex, ez] if vx * vy * vz == 1 else [ez, ex]
        order = [ey] + rest
        idx = []
        for nor in order:
            fi = FACES.index({(0, 1, 0): "U", (0, -1, 0): "D", (0, 0, 1): "F",
                              (0, 0, -1): "B", (-1, 0, 0): "L", (1, 0, 0): "R"}[nor])
            idx.append(next(i for i in groups[v] if i // 9 == fi))
        slots.append(idx)
    return np.array(slots, dtype=np.int64)      # (8, 3)


SLOTS = corner_slots()
# 解開時每個角塊位置的三個顏色（= 三個面的序號）。拿來認「這是哪一顆角」。
SOLVED_COLORS = SLOTS // 9                       # (8, 3)
_ID = {tuple(sorted(c)): i for i, c in enumerate(SOLVED_COLORS.tolist())}
assert len(_ID) == 8, "八顆角的顏色組合居然有重複，slot 排錯了"

# 「這三個顏色是哪一顆角」的查表。三個顏色排序後編成一個 0..215 的號碼，
# 直接查陣列。原本是用 8×8 的 Python 迴圈逐一比對，在大批次上慢很多——
# 搜尋每展開一批就要認一次角，這裡快一點整條路都會快。
_CORNER_LUT = np.full(216, -1, dtype=np.int64)
for _k, _i in _ID.items():
    _CORNER_LUT[_k[0] * 36 + _k[1] * 6 + _k[2]] = _i


def read_corners(state: np.ndarray):
    """從貼紙陣列讀出 (perm, ori)。state 可以是 (54,) 或 (N,54)。"""
    single = state.ndim == 1
    st = state.reshape(1, -1) if single else state
    cols = st[:, SLOTS]                          # (N, 8, 3)
    k = np.sort(cols, axis=2).astype(np.int64)   # 顏色排序後就是這顆角的身分證
    perm = _CORNER_LUT[k[:, :, 0] * 36 + k[:, :, 1] * 6 + k[:, :, 2]]
    assert (perm >= 0).all(), "出現不存在的角塊顏色組合 — 貼紙陣列壞了"
    # 扭轉 = U/D 顏色（0 或 1）落在三格中的哪一格
    ori = np.argmax(cols <= 1, axis=2).astype(np.int64)
    return (perm[0], ori[0]) if single else (perm, ori)


def perm_rank(p: np.ndarray) -> np.ndarray:
    """8 個元素的排列 -> 0..40319（Lehmer code）。p 是 (N,8)。"""
    p = np.atleast_2d(p)
    r = np.zeros(len(p), dtype=np.int64)
    for i in range(8):
        smaller = (p[:, i + 1:] < p[:, i:i + 1]).sum(axis=1)
        r = r * (8 - i) + smaller
    return r


def perm_unrank(r: np.ndarray) -> np.ndarray:
    r = np.atleast_1d(r).astype(np.int64)
    out = np.empty((len(r), 8), dtype=np.int64)
    code = np.empty((len(r), 8), dtype=np.int64)
    x = r.copy()
    for i in range(7, -1, -1):
        code[:, i] = x % (8 - i)
        x //= (8 - i)
    for k in range(len(r)):
        pool = list(range(8))
        for i in range(8):
            out[k, i] = pool.pop(code[k, i])
    return out


def ori_rank(o: np.ndarray) -> np.ndarray:
    o = np.atleast_2d(o)
    r = np.zeros(len(o), dtype=np.int64)
    for i in range(7):
        r = r * 3 + o[:, i]
    return r


def ori_unrank(r: np.ndarray) -> np.ndarray:
    r = np.atleast_1d(r).astype(np.int64)
    out = np.zeros((len(r), 8), dtype=np.int64)
    x = r.copy()
    for i in range(6, -1, -1):
        out[:, i] = x % 3
        x //= 3
    out[:, 7] = (-out[:, :7].sum(axis=1)) % 3     # 總扭轉必須是 3 的倍數
    return out


def build_move_tables(cube: Cube):
    """從貼紙引擎推出角塊的位置置換與扭轉量，再展開成兩張座標移動表。"""
    A = cube.n_actions
    solved = cube.solved(1)
    pos_move = np.empty((A, 8), dtype=np.int64)   # 位置 i 的內容轉到位置 pos_move[m,i]
    twist = np.empty((A, 8), dtype=np.int64)
    for m in range(A):
        after = cube.apply(solved, np.array([m]))
        p, o = read_corners(after[0])
        for j in range(8):
            pos_move[m, p[j]] = j                 # 解開時第 p[j] 顆角本來就在位置 p[j]
            twist[m, p[j]] = o[j]

    perm_tbl = np.empty((N_PERM, A), dtype=np.int32)
    all_p = perm_unrank(np.arange(N_PERM))
    for m in range(A):
        np_ = np.empty_like(all_p)
        np_[:, pos_move[m]] = all_p
        perm_tbl[:, m] = perm_rank(np_)

    ori_tbl = np.empty((N_ORI, A), dtype=np.int16)
    all_o = ori_unrank(np.arange(N_ORI))
    for m in range(A):
        no = np.empty_like(all_o)
        no[:, pos_move[m]] = (all_o + twist[m]) % 3
        ori_tbl[:, m] = ori_rank(no)
    return perm_tbl, ori_tbl


def index_of(cube: Cube, states: np.ndarray) -> np.ndarray:
    """貼紙陣列 -> 角塊狀態編號 0..88,179,839。"""
    p, o = read_corners(states)
    return perm_rank(p) * N_ORI + ori_rank(o)


def build(cube: Cube, verbose=True):
    perm_tbl, ori_tbl = build_move_tables(cube)
    A = cube.n_actions
    dist = np.full(N_STATES, 255, dtype=np.uint8)
    start = index_of(cube, cube.solved(1))[0]
    dist[start] = 0
    frontier = np.array([start], dtype=np.int64)
    hist = [1]
    d = 0
    while len(frontier):
        d += 1
        nxt = []
        # 分塊處理，免得一次配置好幾 GB
        for lo in range(0, len(frontier), 4_000_000):
            f = frontier[lo:lo + 4_000_000]
            p, o = f // N_ORI, f % N_ORI
            for m in range(A):
                ch = perm_tbl[p, m].astype(np.int64) * N_ORI + ori_tbl[o, m]
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
            print(f"  深度 {d:>2}: {len(frontier):>11,} 個新狀態   （累計 {sum(hist):>11,}）")
    return dist, hist


def load():
    if not OUT.exists():
        raise SystemExit(f"缺 {OUT} — 先跑 python ml/corners.py")
    return np.load(OUT, mmap_mode="r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只驗證編碼與移動表，不建表")
    a = ap.parse_args()
    cube = Cube(3)
    rng = np.random.default_rng(0)

    # ── 驗證：座標編碼與貼紙引擎必須一致 ──
    st, _ = cube.scramble(rng.integers(1, 31, size=2000), rng)
    p, o = read_corners(st)
    assert (o.sum(axis=1) % 3 == 0).all(), "角塊總扭轉不是 3 的倍數 — 讀錯了"
    assert (np.sort(p, axis=1) == np.arange(8)).all(), "角塊置換不是一個排列 — 讀錯了"
    idx = index_of(cube, st)
    assert (perm_rank(perm_unrank(perm_rank(p))) == perm_rank(p)).all(), "perm rank/unrank 不一致"
    assert (ori_rank(ori_unrank(ori_rank(o))) == ori_rank(o)).all(), "ori rank/unrank 不一致"

    perm_tbl, ori_tbl = build_move_tables(cube)
    # 用座標表走一步，跟用貼紙引擎走一步，結果必須是同一個編號
    mv = rng.integers(0, cube.n_actions, size=len(st))
    want = index_of(cube, cube.apply(st, mv))
    got = perm_tbl[idx // N_ORI, mv].astype(np.int64) * N_ORI + ori_tbl[idx % N_ORI, mv]
    assert (want == got).all(), "座標移動表跟貼紙引擎對不起來"
    print(f"編碼與移動表驗證通過（{len(st)} 個隨機局面，走一步逐一比對）")
    print(f"狀態空間 8! × 3^7 = {N_PERM:,} × {N_ORI:,} = {N_STATES:,}")
    if a.check:
        return

    print("\n開始全狀態 BFS…")
    import time
    t0 = time.time()
    dist, hist = build(cube)
    print(f"\n全部 {sum(hist):,} 個狀態，最遠 {len(hist) - 1} 步，花了 {time.time() - t0:.0f}s")
    assert sum(hist) == N_STATES, f"只走到 {sum(hist):,} 個狀態，應該是 {N_STATES:,}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, dist)
    print(f"寫出 {OUT}  ({OUT.stat().st_size / 1e6:.0f} MB)")
    print(f"平均 {dist.mean():.3f} 步、中位 {int(np.median(dist))} 步、最遠 {dist.max()} 步")


if __name__ == "__main__":
    main()
