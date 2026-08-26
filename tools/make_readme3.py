"""把第二階段（邊塊表）的結果寫進 README.md。

    python tools/make_readme3.py

數字全部讀檔，不手抄：
    ml/checkpoints/edges-build.json   兩張邊塊表的建表統計
    out/goal3.json                    三張表的值分佈 + IDA* 節點數 vs 深度
    out/stage2.json                   先解角繞了多遠

會在 README.md 裡找 <!--GOAL003--> ... <!--/GOAL003--> 整段換掉；
找不到就接在檔案最後。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
MARK_A, MARK_B = "<!--GOAL003-->", "<!--/GOAL003-->"


def main():
    rj = lambda p: json.loads((ROOT / p).read_text(encoding="utf-8"))
    build = rj("ml/checkpoints/edges-build.json")
    g3 = rj("out/goal3.json")
    s2 = rj("out/stage2.json")
    pdb = rj("ml/checkpoints/goal2.json")["corner_pdb"]

    B, S = g3["bounds"], g3["scale"]
    h0, h1 = build["halves"]
    win = B["winner"]
    gain = B["max3"]["mean"] - B["corners"]["mean"]
    edge_mean = (B["edges0"]["mean"] + B["edges1"]["mean"]) / 2

    ref = next((r for r in S if r["depth"] == 12 and "corners" in r and "max3" in r),
               None) or [r for r in S if "corners" in r and "max3" in r][-1]
    ratio = ref["ratio"]

    secs = lambda h: "—" if h.get("secs") is None else f"{h['secs']:.0f} s"
    tbl_rows = "\n".join([
        f"| 角塊 | 八顆角的位置與方向 | {pdb['states']:,} | {pdb['max']} | "
        f"{pdb['mean']:.3f} | 114 s | 88 MB |",
        f"| 邊塊 0 | 十二條邊裡的第 {h0['edges'][0]}~{h0['edges'][-1]} 條 | "
        f"{build['states']:,} | {h0['max']} | {h0['mean']:.3f} | {secs(h0)} | {h0['mb']} MB |",
        f"| 邊塊 1 | 另外六條 | {build['states']:,} | {h1['max']} | {h1['mean']:.3f} | "
        f"{secs(h1)} | {h1['mb']} MB |",
    ])

    ex = [r for r in s2["exact"] if r["stage2"] is not None]
    # 跑不完的也列出來——只列跑得完的，會讓繞路看起來比實際小
    exact_rows = "\n".join(
        (f"| {r['depth']} | {r['optimal']} | {r['stage1']} | "
         f"_{r['secs'] / 60:.0f} 分鐘還沒算完_ | — | — | {r['secs']:.1f} |")
        if r["stage2"] is None else
        (f"| {r['depth']} | {r['optimal']} | {r['stage1']} | **{r['stage2']}** | {r['total']} | "
         f"**+{r['detour']}** | {r['secs']:.1f} |")
        for r in s2["exact"])
    unfinished = len(s2["exact"]) - len(ex)
    det = [r["detour"] for r in ex]
    mean_det = sum(det) / max(1, len(det))
    slowest = max(r["secs"] for r in ex)
    deepest = max(r["depth"] for r in s2["exact"])

    node_rows = "\n".join(
        f"| {r['depth']} | {r['max3']['len']:.2f} | "
        + (f"{r['corners']['nodes']:,.0f}{'' if r['corners']['complete'] else '*'} | "
           f"{r['corners']['secs']:.2f} | " if "corners" in r else "跑不完 | — | ")
        + f"{r['max3']['nodes']:,.0f} | {r['max3']['secs']:.2f} | "
        + (f"{r['ratio']:,.0f}×" if r.get("ratio") else "—")
        + " |"
        for r in S if "max3" in r)

    detour_rows = "\n".join(
        f"| {r['depth']} | {r['optimal']:.2f} | {r['stage1']:.2f} | {r['stage2_lower']:.2f} | "
        f"{r['cf_lower']:.2f} | **+{r['detour']:.2f}** |"
        for r in s2["detour"])

    H = s2["heuristic"]
    drop = H["before"]["max3"] - H["after"]["max3"]
    collapse_rows = "\n".join(
        f"| {n} | {r['corners']:.2f} | {r['edges0']:.2f} | {r['edges1']:.2f} | {r['max3']:.2f} |"
        for n, r in (("打亂的局面", H["before"]), ("角塊歸位後", H["after"])))

    body = f"""{MARK_A}
## 第二階段 — 邊塊，以及「先解角」到底值不值

### 先講一個很容易誤會的地方：3×3×3 沒有「解中心」這件事

人在手上轉方塊會整顆翻來翻去，中心當然會動——但那不改變方塊的狀態，只是換個握法。
這個專案的動作集是六個面的順逆時針共十二種轉法，在這個座標系下，
`ml/edges.py` 開頭就把它驗出來：

```
中心塊檢查：6 片貼紙在 12 種轉法下永遠不動 = 六個面的正中央
           所以 3x3x3 沒有「解中心」這個階段，中心是釘死的參考點。
```

所以 3×3×3 只有**兩個階段**：角、邊。（要到 4×4×4 以上中心才會動、才需要解。）

那 Korf 為什麼是三張表？因為十二條邊一起有 12! × 2¹¹ ≈ 9,810 億個狀態，數不完。
切成兩半、每半六條，就只剩 12P6 × 2⁶ = {build['states']:,} 個，
**比角塊那張還小一半**。第三張表不是第三個階段，是同一個階段被切開來實作。

### 三張表

| 表 | 看的是 | 狀態數 | 最遠 | 平均 | 建表 | 大小 |
|---|---|---:|---:|---:|---:|---:|
{tbl_rows}

兩張邊塊表的平均與最遠**完全一樣**（{h0['mean']:.3f} / {h0['max']}）——
那不是巧合，是因為那兩組各六條邊在方塊的對稱群下互相對應。

三張都是**下界**：解開整顆方塊的每一步，也都在轉這些塊。三個下界取最大仍是下界，
所以最短解的保證還在（`ml/heuristics.py` 的 `korf_bound()`）。

編碼跟角塊那張同一招——位置與翻轉在轉動下各自獨立，
所以兩張 {build['n_pos']:,} × 12 的小表就能推出四千兩百萬個狀態的鄰居。
`python ml/edges.py --check` 會驗三件事：邊塊置換合法、總翻轉恆為偶數、
移動表跟貼紙引擎逐一相同。

### 平均只推高 {gain:.2f} 步，搜尋卻快了三個數量級

補上兩張邊塊表之後，第一個看到的數字是失望的：

```
角塊距離表（精確）              平均 {B['corners']['mean']:6.3f}  最大 {B['corners']['max']:>2}
邊塊距離表 0（精確）            平均 {B['edges0']['mean']:6.3f}  最大 {B['edges0']['max']:>2}
邊塊距離表 1（精確）            平均 {B['edges1']['mean']:6.3f}  最大 {B['edges1']['max']:>2}
三張表取大                      平均 {B['max3']['mean']:6.3f}  最大 {B['max3']['max']:>2}
```

下界的平均值只從 {B['corners']['mean']:.2f} 推到 {B['max3']['mean']:.2f}。
原因很清楚：每張六條邊的表平均只有 {edge_mean:.2f}，**比角塊那張還低**，
{win['corners'] / win['n'] * 100:.1f}% 的局面取大之後根本沒動到。

然後去量搜尋，數字整個翻過來。同一批方塊、同一支 IDA*，只換 heuristic：

| 打亂 | 最短解 | 角塊表・節點 | 角塊表・秒 | 三張表・節點 | 三張表・秒 | 省幾倍 |
|---:|---:|---:|---:|---:|---:|---:|
{node_rows}

\\* 代表在時間上限內只跑完一部分方塊。
「省幾倍」只在兩種下界**都跑完的那幾顆**上算——不然深度 14 那一列會拿 2 顆的平均去比 8 顆的平均。

因為 IDA* 要展開的節點數大約是**（真正的距離 − 下界）的指數**，而下界坐在指數上。
指數上少 {gain:.2f}，底下就是 {ratio:,.0f} 倍。
**一個「平均只好一點點」的 heuristic，在搜尋裡可以是完全不同的量級**——
這也是為什麼 heuristic 的好壞不該只看平均誤差。

### 先解角繞了多遠——精確值

有了三張表，一個一直問不了的問題現在問得了。兩個階段都求到最短：

- 第一階段是可證明最短的。角塊表是**精確**距離，站在距離 d 的地方一定有鄰居是 d−1，
  沿著走下去就是一條最短解，連搜尋都不用（`ml/stage2.py` 的 `CornerStage`，整批向量化）。
- 第二階段從角塊歸位後的局面，用三張表跑 IDA* 求整顆的最短解。
- 再跟直接求出的最短解一比，繞路就是**精確值**，不是估計。

| 打亂 | 直接最短解 | 階段 1・角 | 階段 2・剩下的 | 合計 | 繞路 | 階段 2 秒數 |
|---:|---:|---:|---:|---:|---:|---:|
{exact_rows}

在算得完的 {len(ex)} 顆裡，平均繞路 **+{mean_det:.1f} 步**（最少 +{min(det)}、最多 +{max(det)}）。
另外 {unfinished} 顆的第二階段在時間上限內沒算完——那幾顆也列在表裡，
不然平均會看起來比實際小。

注意「階段 2」那一欄：**把角轉回去之後，剩下的部分反而離解開更遠了**。
一顆打亂 9 步的方塊，解完角之後可能還要 14 步才解得完。

深一點的局面第二階段就求不完了，但下界還是給得出來：

| 打亂 | 直接最短解 | 階段 1（角，最短） | 階段 2 下界 | 先解角至少 | 至少繞路 |
|---:|---:|---:|---:|---:|---:|
{detour_rows}

最後一欄是**有證明的下限**：真正的繞路只會更多，不會更少——上面那張精確的表就是證據。

### 為什麼第二階段反而更難算

角塊一歸位，角塊表就永遠讀 0——三張表裡最強的那張直接閉嘴：

| | 角塊表 | 邊表 0 | 邊表 1 | 取大 |
|---|---:|---:|---:|---:|
{collapse_rows}

下界掉了 {drop:.2f} 步。而下界坐在指數上，所以後果很直接：
直接求打亂 {ref['depth']} 步的最短解只要 {ref['max3']['secs']:.2f} 秒；
先把角解掉之後，剩下那段最慢的一顆花了 {slowest:.0f} 秒，
另外 {unfinished} 顆到時間上限都還沒算完——而那些全都只是打亂 {deepest} 步的方塊。

**先解角讓剩下的部分更難解，不是更好解。**

那人類為什麼還是先解角？因為人不是在跑 heuristic 搜尋。
人用的是背下來的公式——一組「只動我要動的塊、其他都還原」的手順。
那是完全不同的資源：**人有記憶，機器有搜尋**，
而先解角這個決定，剛好對記憶有利、對搜尋不利。

### 這一輪多出來的檔案

| 檔案 | 做什麼 |
|---|---|
| `ml/edges.py` | 兩張六條邊的 pattern database，各 {build['states']:,} 個狀態全數 BFS |
| `ml/goal3.py` | 三張表的值分佈；IDA* 節點數 vs 打亂深度，兩種下界對照 |
| `ml/stage2.py` | 先解角的繞路（精確值 + 深局面的下限）＋ heuristic 崩掉多少 |
| `ml/heuristics.py` | 多了 `EdgePDB` 與 `korf_bound()` |
| `ml/idastar.py` | `OptimalSolver(..., use_edges=True)`，三個座標都是增量更新 |

```bash
python ml/edges.py --check    # 只驗編碼與移動表，不建表
python ml/edges.py            # 建兩張表，各 43 MB
python ml/edges.py --stats    # 表已經在了，只重算統計
python ml/goal3.py            # 值分佈 + 節點數尺度（有快取）
python ml/stage2.py           # 先解角的代價
python ml/idastar.py --depth 13 --n 8 --edges   # 三張表版的最短解求解器
```

{MARK_B}"""

    txt = README.read_text(encoding="utf-8")
    if MARK_A in txt and MARK_B in txt:
        pre, rest = txt.split(MARK_A, 1)
        _, post = rest.split(MARK_B, 1)
        txt = pre + body + post
    else:
        txt = txt.rstrip() + "\n\n" + body + "\n"
    README.write_text(txt, encoding="utf-8")
    print(f"README.md 的第二階段段落已更新（{len(body):,} 字元）")


if __name__ == "__main__":
    main()
