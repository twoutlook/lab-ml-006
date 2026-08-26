"""從 3D 幾何算出每一種轉動對貼紙的置換表，寫成 shared/moves.json。

    python ml/gen_moves.py

為什麼要用「算的」而不是手打？
六個面 × 兩個方向 = 12 張置換表，3x3x3 每張 54 個數字。手打一定會錯，
而且錯了不會噴錯——只會讓訓練跑得起來但學不到東西（因為那根本不是魔術方塊）。
所以這裡從立方體的座標推出來，兩邊（JS 與 Python）都讀同一份輸出。

座標系（右手座標，從外面看每一面）：

    +Y = U（上）      +X = R（右）      +Z = F（前）
    -Y = D（下）      -X = L（左）      -Z = B（後）

每一片貼紙用一個整數座標 p 代表它的中心。為了避免出現 0.5，
座標放大兩倍：n 階方塊上，某一軸的第 i 格中心是 2i-(n-1)，
所以格心座標都是整數，而外表面落在 ±n 上。

一次順時針轉動（從方塊外面看那一面）就是繞著該面法向量 a 轉 -90 度：

    p' = a (a·p) - (a × p)

會被轉到的貼紙就是 a·p >= n-1 的那些（面自己的 n² 片，加上四個鄰面的一排）。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 面的順序固定：這個順序就是貼紙編號的順序，改了整個專案都要跟著改。
FACES = ["U", "D", "F", "B", "L", "R"]

# 每一面：法向量、以及「從外面看這一面」時的右方向與上方向。
#   U 從上往下看，畫面的上方是往 B（-Z）
#   D 從下往上看，畫面的上方是往 F（+Z）
#   B 從後面看，畫面的右方是往 L（-X）
#   R 從右邊看，畫面的右方是往 B（-Z）
BASIS = {
    "U": ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    "D": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    "F": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    "B": ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
    "L": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "R": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
}


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def sticker_points(n: int):
    """回傳 {座標: 貼紙編號}。貼紙編號 = 面序號*n² + 列*n + 行（列從上、行從左）。"""
    pts = {}
    for fi, f in enumerate(FACES):
        normal, right, up = BASIS[f]
        for r in range(n):
            for c in range(n):
                du = (n - 1) - 2 * r     # 上下：r=0 在最上面
                dr = 2 * c - (n - 1)     # 左右：c=0 在最左邊
                p = tuple(normal[k] * n + right[k] * dr + up[k] * du for k in range(3))
                pts[p] = fi * n * n + r * n + c
    assert len(pts) == 6 * n * n, "座標撞號了，幾何寫錯"
    return pts


def face_turn(n: int, face: str, pts: dict) -> list[int]:
    """一次順時針轉動的置換表 P：new[k] = old[P[k]]。"""
    a = BASIS[face][0]
    perm = list(range(6 * n * n))
    for p, i in pts.items():
        if dot(a, p) < n - 1:
            continue                     # 不在這一層，不動
        ap = dot(a, p)
        ax = cross(a, p)
        q = (a[0] * ap - ax[0], a[1] * ap - ax[1], a[2] * ap - ax[2])
        j = pts[q]                       # 原本在 i 的顏色，轉完之後在 j
        perm[j] = i
    return perm


def inverse(perm: list[int]) -> list[int]:
    inv = [0] * len(perm)
    for k, v in enumerate(perm):
        inv[v] = k
    return inv


def build(n: int, faces: list[str]):
    """faces = 這個尺寸允許轉哪幾面。每一面給「順時針」與「逆時針」兩個動作。"""
    pts = sticker_points(n)
    names, perms = [], []
    for f in faces:
        cw = face_turn(n, f, pts)
        names.append(f)
        perms.append(cw)
        names.append(f + "'")
        perms.append(inverse(cw))
    return names, perms


def compose(a: list[int], b: list[int]) -> list[int]:
    """先做 a 再做 b。new[k] = old[a[b[k]]]。"""
    return [a[b[k]] for k in range(len(a))]


def order(perm: list[int]) -> int:
    ident = list(range(len(perm)))
    cur, k = perm, 1
    while cur != ident:
        cur = compose(perm, cur)
        k += 1
        if k > 10000:
            raise SystemExit("置換的階算不出來，表一定錯了")
    return k


def main():
    out = {}

    # 2x2x2：把 DBL 那顆角固定住，只轉 U / R / F。
    # 少了這個限制，整顆方塊的旋轉會讓同一個局面有 24 種寫法，狀態空間憑空大 24 倍。
    n2, faces2 = 2, ["U", "R", "F"]
    names2, perms2 = build(n2, faces2)
    out["2"] = {"n": n2, "faces": faces2, "moves": names2, "perms": perms2,
                "n_stickers": 6 * n2 * n2}

    n3, faces3 = 3, ["U", "D", "F", "B", "L", "R"]
    names3, perms3 = build(n3, faces3)
    out["3"] = {"n": n3, "faces": faces3, "moves": names3, "perms": perms3,
                "n_stickers": 6 * n3 * n3}

    # ── 自我檢查（表錯了要在這裡就爆掉，不要等到訓練跑完才發現）──
    for key in ("2", "3"):
        d = out[key]
        for name, p in zip(d["moves"], d["perms"]):
            assert sorted(p) == list(range(len(p))), f"{key} {name} 不是置換"
            assert order(p) == 4, f"{key} {name} 的階不是 4"
    # 3x3x3 的經典檢查：R U 這兩步一直重複，要 105 次才會回到原狀。
    i3 = {m: k for k, m in enumerate(out["3"]["moves"])}
    ru = compose(out["3"]["perms"][i3["U"]], out["3"]["perms"][i3["R"]])
    assert order(ru) == 105, f"(R U) 的階是 {order(ru)}，應該是 105 — 置換表有錯"
    # 2x2x2 只有角塊，同樣的 R U 階是 15。105 = lcm(角 15, 邊 7)，
    # 兩個數字對得起來，等於順便驗了「3x3x3 的角塊部分跟 2x2x2 是同一件事」。
    i2 = {m: k for k, m in enumerate(out["2"]["moves"])}
    ru2 = compose(out["2"]["perms"][i2["U"]], out["2"]["perms"][i2["R"]])
    assert order(ru2) == 15, f"2x2x2 的 (R U) 階是 {order(ru2)}，應該是 15"
    # sexy move (R U R' U') 的階是 6
    seq = [i3["R"], i3["U"], i3["R'"], i3["U'"]]
    acc = list(range(54))
    for m in seq:
        acc = compose(out["3"]["perms"][m], acc)
    assert order(acc) == 6, f"(R U R' U') 的階是 {order(acc)}，應該是 6"

    dst = ROOT / "shared" / "moves.json"
    dst.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"寫出 {dst}  ({dst.stat().st_size / 1024:.0f} KB)")
    for key in ("2", "3"):
        d = out[key]
        print(f"  {key}x{key}x{key}: {d['n_stickers']} 片貼紙, "
              f"{len(d['moves'])} 個動作 {d['moves']}")
    print("檢查通過：每個動作的階都是 4，3x3x3 的 (R U)=105、2x2x2 的 =15，(R U R' U')=6")


if __name__ == "__main__":
    main()
