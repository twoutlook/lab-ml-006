"""2x2x2 的完整廣度優先搜尋：把全部 3,674,160 個局面的「最短還要幾步」算出來。

    python ml/bfs.py

輸出 ml/checkpoints/dist222.npz（約 33 MB，排好序的 uint64 鍵 + uint8 距離）。

這一份東西是整個專案的地基，但它跟機器學習一點關係也沒有——
它是**正確答案**。有了它才能問出這個專案真正想問的問題：

    訓練出來的那個 heuristic，到底離正確答案多遠？

3x3x3 沒有這種奢侈（4.3 × 10^19 個局面），所以那邊只能量「解得開嗎、幾步」。
先在一個看得到答案的方塊上把方法驗清楚，再放大——這就是為什麼要有 2x2x2。

驗證方式：算出來的距離分布必須跟公開的 2x2x2 四分之一轉（QTM）分布逐項相同。
那張表不是我算的，錯一個數字就代表引擎有問題。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from cube import Cube

HERE = Path(__file__).resolve().parent
OUT = HERE / "checkpoints" / "dist222.npz"

# 公開的 2x2x2 QTM（每次只轉 90 度、固定一顆角）距離分布。
# 來源：Rubik's cube 群論社群長年算出來的結果，總和 3,674,160 = 7! * 3^6。
KNOWN = [1, 6, 27, 120, 534, 2256, 8969, 33058, 114149, 360508,
         930588, 1350852, 782536, 90280, 276]


def build(verbose: bool = True):
    c = Cube(2)
    t0 = time.time()

    frontier = c.solved(1)
    seen = c.pack(frontier)                     # 排好序的 uint64
    all_keys = [seen]
    all_dist = [np.zeros(1, dtype=np.uint8)]
    hist = [1]

    depth = 0
    while len(frontier):
        depth += 1
        kids = c.expand(frontier).reshape(-1, c.n_stickers)
        keys = np.unique(c.pack(kids))
        # 只留沒看過的。seen 是排好序的，所以用 searchsorted 就好，不用 np.isin。
        pos = np.searchsorted(seen, keys)
        pos_clipped = np.minimum(pos, len(seen) - 1)
        fresh = keys[seen[pos_clipped] != keys]
        if not len(fresh):
            break
        all_keys.append(fresh)
        all_dist.append(np.full(len(fresh), depth, dtype=np.uint8))
        hist.append(len(fresh))
        seen = np.union1d(seen, fresh)          # union1d 回傳的就是排好序的
        frontier = c.unpack(fresh)
        if verbose:
            print(f"  深度 {depth:>2}: {len(fresh):>9,} 個新局面   （累計 {len(seen):>9,}）")

    keys = np.concatenate(all_keys)
    dist = np.concatenate(all_dist)
    order = np.argsort(keys)
    keys, dist = keys[order], dist[order]

    if verbose:
        print(f"\n全部 {len(keys):,} 個局面，最遠 {depth - 1} 步，花了 {time.time() - t0:.1f}s")
    return keys, dist, hist


def load():
    if not OUT.exists():
        raise SystemExit(f"缺 {OUT} — 先跑 python ml/bfs.py")
    z = np.load(OUT)
    return z["keys"], z["dist"]


def lookup(keys_sorted, dist, query) -> np.ndarray:
    """查表。query 是 pack 過的 uint64，回傳精確的最短步數。"""
    pos = np.searchsorted(keys_sorted, query)
    if (pos >= len(keys_sorted)).any() or (keys_sorted[np.minimum(pos, len(keys_sorted) - 1)] != query).any():
        raise ValueError("查到不存在的局面 — pack 或引擎不一致")
    return dist[pos]


def main():
    keys, dist, hist = build()

    print("\n距離分布 vs 公開答案：")
    ok = True
    for d in range(max(len(hist), len(KNOWN))):
        got = hist[d] if d < len(hist) else 0
        want = KNOWN[d] if d < len(KNOWN) else 0
        flag = "✓" if got == want else "✗"
        if got != want:
            ok = False
        print(f"  {d:>2} 步  {got:>9,}   應為 {want:>9,}  {flag}")
    total = int(sum(hist))
    print(f"  合計 {total:,}   應為 {sum(KNOWN):,}  {'✓' if total == sum(KNOWN) else '✗'}")
    print(f"        7! × 3^6 = {5040 * 729:,}")
    if not ok:
        raise SystemExit("分布對不上公開答案 — 置換表或 pack 有錯，不要往下做")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, keys=keys, dist=dist)
    print(f"\n寫出 {OUT}  ({OUT.stat().st_size / 1e6:.0f} MB)")
    print(f"平均最短步數 {dist.mean():.3f}，中位 {int(np.median(dist))}，上帝之數 {dist.max()}")


if __name__ == "__main__":
    main()
