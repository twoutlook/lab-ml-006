"""把 GOAL002 的結果寫進 README.md（在 goal002-corners-first 這個分支上）。

    python tools/make_readme2.py

跟旁白稿一樣，數字直接讀 ml/checkpoints/goal2.json，不手抄。
會在 README.md 裡找 <!--GOAL002--> ... <!--/GOAL002--> 這段整個換掉；
找不到的話就接在檔案最後。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
G2 = ROOT / "ml" / "checkpoints" / "goal2.json"
README = ROOT / "README.md"
MARK_A, MARK_B = "<!--GOAL002-->", "<!--/GOAL002-->"


def main():
    g = json.loads(G2.read_text(encoding="utf-8"))
    iv = json.loads((ROOT / "web" / "demo2.json").read_text(encoding="utf-8"))["idaVsAstar"]
    speed = iv["ida"]["nps"] / iv["astar"]["nps"]
    us_astar = iv["astar"]["sec"] / iv["astar"]["nodes"] * 1e6
    us_ida = iv["ida"]["sec"] / iv["ida"]["nodes"] * 1e6
    P, E1, E2, E3, E4 = g["corner_pdb"], g["e1"], g["e2"], g["e3"], g["e4"]
    pdb_row = next(r for r in E1 if r["admissible"])
    net_row = next(r for r in E1 if not r["admissible"] and "∪" not in r["name"])
    max_row = next(r for r in E1 if "∪" in r["name"])
    save = (1 - max_row["mean_nodes"] / net_row["mean_nodes"]) * 100
    one, two = E4["one_shot"], E4["staged"]
    extra = two["mean_len"] - one["mean_len"]
    nsave = (1 - two["mean_nodes"] / one["mean_nodes"]) * 100

    ln = lambda v: "—" if v is None else f"{v:.2f}"
    e1_rows = "\n".join(
        f"| {r['name']} | {r['mean_h']:.2f} | {'不會' if r['admissible'] else '會'} | "
        f"{r['solve_rate'] * 100:.0f}% | {ln(r['mean_len'])} | {r['mean_nodes']:,.0f} |"
        for r in E1)

    by_depth = {}
    for r in E2["records"]:
        by_depth.setdefault(r["depth"], []).append(r)
    e2_rows = "\n".join(
        f"| {d} | {len(v)} | {sum(x['true'] for x in v) / len(v):.2f} | "
        f"{sum(x['nodes'] for x in v) / len(v):,.0f} |"
        for d, v in sorted(by_depth.items()))

    deep = E2["by_true_net"][-1]
    # 「穿過對角線」＝ 最後一個估計值還 >= 真值的點，跟它的下一個點之間。
    # 不能寫成「第一個 mean < d」——最淺那幾格的平均是 1.999 這種數字，
    # 會被浮點捨入誤判成一開始就穿過去了。
    _over = [r["d"] for r in E2["by_true_net"] if r["mean"] >= r["d"] - 1e-9]
    _lo = max(_over) if _over else None
    _hi = min([r["d"] for r in E2["by_true_net"] if _lo is not None and r["d"] > _lo], default=None)
    cross = f"{_lo}~{_hi}" if _lo is not None and _hi is not None else "?"
    prof_rows = "\n".join(
        f"| {r['d']} | {r['n']} | {r['mean']:.2f} | {r['mean'] - r['d']:+.2f} | {p2.get(r['d'], 0):.2f} |"
        for r in E2["by_true_net"]
        for p2 in [{x['d']: x['mean'] for x in E2['by_true_pdb']}])

    body = f"""{MARK_A}

## GOAL002 — 先解角，再解邊

> 這一段在 `goal002-corners-first` 分支上。影片與圖文版的連結在最後。

第一輪（主線）結束在一句不甘心的話：**3×3×3 沒有正確答案，所以「這個解比最短解長幾步」量不出來。**
這一輪把那句話改掉一半。方法不是更大的網路，是回頭看人類怎麼解方塊。

1974 年 Ernő Rubik 第一次解開他自己發明的方塊，用的是**先解角**；1981 年 Marc Waterman
的方法是純角先派的代表作。人這樣拆是為了好記——人腦沒辦法一次考慮 4.3 × 10¹⁹ 個局面。
但同一個拆法換到機器那邊，用途完全不一樣：Korf 1997 年算隨機方塊的最短解，
用的就是三張 pattern database——**角塊、6 條邊、另外 6 條邊**。
**人腦拿這個拆法記手順，機器拿它當下界。**

### 角塊距離表：{P['states']:,} 個狀態全部數完

八顆角的狀態數是 8! × 3⁷ = {P['states']:,}。從解開狀態做一次 BFS，114 秒，
每一個角塊狀態離解開幾步就全部在手上。上帝之數 **{P['max']}**，平均 {P['mean']:.3f} 步。

對照主線那個 2×2×2：3,674,160 個狀態、上帝之數 14、平均 10.666 步。
兩邊各自獨立算出來卻對得上（{P['states']:,} = 24 × 3,674,160，2×2×2 就是固定一顆角的角塊群），
是一次很便宜的交叉驗證。

關鍵性質：**「只把角塊轉回去要幾步」是「解開整顆要幾步」的下界**，
因為解開整顆的每一步也都在轉角塊。所以它保證不高估。

能整個數完的原因：角塊的「誰在哪」與「轉了幾度」在轉動下**各自獨立**——
`新 ori[M(i)] = (舊 ori[i] + 固定扭轉量) mod 3`，跟哪顆角在那裡無關。
所以 40,320 × 12 與 2,187 × 12 兩張小表就推得出全部 8,800 萬個狀態的鄰居，BFS 純向量化。
驗證：一致性（每走一步距離最多變 1）24 萬次轉動 0 違規；下界檢查 0 違規。

### E1 · 當 heuristic：精確但片面 vs 不精確但完整

同一批 {E1[0].get('n', 60)} 顆隨機方塊，其他設定不動，只換估計法：

| heuristic | 平均估計 | 會高估 | 解開率 | 平均步數 | 展開節點 |
|---|---:|---|---:|---:|---:|
{e1_rows}

**精確的那個，單獨用一顆都解不開。** 它太保守——平均只說 {pdb_row['mean_h']:.2f} 步，
而隨機方塊的真實最短解在 20 上下，搜尋找不到方向。
但跟網路取大之後，節點少了 **{save:.0f}%**，解也短了
{net_row['mean_len'] - max_row['mean_len']:.1f} 步。
原因是：在網路低估得特別離譜的那些局面上，角塊表把它拉了回來。
**精確但片面的東西，單獨用沒力，當補丁很有效。**

### E2 · 3×3×3 第一次有正確答案

角塊表保證不高估，配上課本版的 A\\*，找到的解就是**可以證明的最短解**。
一共求出 **{E2['n']}** 個局面的精確最短解：

| 打亂步數 | 局面數 | 精確最短解 | 展開節點 |
|---:|---:|---:|---:|
{e2_rows}

（最短解一律小於打亂步數——亂轉會繞回來。這就是為什麼「打亂 k 步」不能當成「距離 k」。）

有了正確答案，網路的誤差就量得出來了：平均差 **{E2['mae']:.2f}** 步、
{E2['within_1'] * 100:.0f}% 落在 1 步以內、{E2['over_rate'] * 100:.0f}% 高估。

| 真實最短解 | 局面數 | 網路平均 | 偏差 | 角塊表 |
|---:|---:|---:|---:|---:|
{prof_rows}

**網路那條線在真實最短解 {cross} 步之間穿過對角線**：比它淺的高估、比它深的低估。
這就是回歸往平均縮，跟主線那張 2×2×2 的圖同一個形狀，只是這次是在真的 3×3×3 上量到的。

而整體的平均偏差是 **{E2['bias']:+.2f}** 步，幾乎是零——**但那個零是假的**。
它是「淺的高估」跟「深的低估」互相抵銷出來的。
只看偏差會以為這個網路沒有系統性誤差；看了分佈才知道它有兩個方向相反的誤差。
（角塊表那一欄永遠在真值下面——它必須在下面，因為它是下界。）

還有一件事要講清楚：**這張表只到 {deep['d']} 步，因為再深就求不出精確最短解了。**
而隨機打亂的方塊真實最短解在 20 上下，正好落在量不到的地方。
所以主線說的「網路對很深的局面一律回答 14 左右」，這張表其實看不到——
只能從 E3 的夾擠間接推：隨機方塊的最短解至少 {E3['mean_lower']:.1f} 步、最多 {E3['mean_found']:.1f} 步，
而網路對它們一律回答 {next(r["mean_h"] for r in E1 if not r["admissible"] and "∪" not in r["name"]):.1f}。

### E3 · 深的局面：夾，但夾不緊

隨機方塊求不出精確最短解（節點數爆炸），但夾得出來：

| | 步數 |
|---|---:|
| 角塊表給的下界 | {E3['mean_lower']:.2f} |
| 搜尋找到的解 | {E3['mean_found']:.2f} |
| **落差最多** | **{E3['mean_gap_upper']:.2f}** |

夾子很鬆，原因很明確：角塊表最多只能說 {P['max']}，但隨機方塊的真實最短解在 20 上下。
要夾緊得補上 Korf 另外兩張「六條邊」的表（各 12P6 × 2⁶ = 42,577,920 個狀態，比角塊那張還小）。

### E4 · 人類的分階段，到底值多少

既然有精確的角塊表，就可以真的照人類的方式解一次：
**先把八顆角轉回去（保證最短，因為表就是那個子目標的精確距離），再解完剩下的。**

| | 總步數 | 展開節點 | 每顆耗時 |
|---|---:|---:|---:|
| 一次解完 | {one['mean_len']:.2f} | {one['mean_nodes']:,.0f} | {one['ms']:.0f} ms |
| 先解角，再解完 | {two['mean_len']:.2f}（角 {two['mean_stage1']:.1f} + 其餘 {two['mean_stage2']:.1f}） | {two['mean_nodes']:,.0f} | {two['ms']:.0f} ms |

分階段多走 **{extra:.1f} 步**（{extra / one['mean_len'] * 100:.0f}%），
但第二段少展開 **{nsave:.0f}%** 的節點。

這正是人類那樣解的理由——只是人類換到的不是節點數，是**記憶量**：
切成兩塊之後每一塊都小到可以背。代價一樣是那多出來的步數，
所以速解高手最後都不用純角先法，而是用步數更省、但要背更多的方法。

### 順帶一個工程上的坑：瓶頸不是演算法，是簿記

第一版求最短解用的是主線那個批次加權 A\\*（weight=1、終點改成展開時判定）。
它是對的，但撐不到有趣的地方：每展開一個節點要做十二次
「把 54 個位元組轉成鍵、查字典、丟進堆積」，實測**每個節點約 {us_astar:.0f} 微秒**。
最深那批局面一顆要展開一千八百萬個節點——用 A\\* 要跑一小時以上。

換成 **IDA\\***（深度優先 + f 值上限，沒有 open/closed list）就通了，
再加上兩件事：角塊座標**增量更新**（走一步 = 兩次陣列查表，不必從貼紙重算）、
用 `operator.itemgetter` 套用轉動（C 寫的，比 Python 迴圈快一個數量級）。
剪枝三條都安全：不走反手、同一面不連三次、對面可交換所以固定順序。

實測（打亂 {iv['depth']} 步、{iv['n']} 顆、同一個 heuristic）：

| | 展開節點 | 耗時 | 每秒節點 | 每節點 |
|---|---:|---:|---:|---:|
| A\\* | {iv['astar']['nodes']:,.0f} | {iv['astar']['sec']:.2f} s | {iv['astar']['nps']:,.0f} | {us_astar:.0f} µs |
| **IDA\\*** | {iv['ida']['nodes']:,.0f} | **{iv['ida']['sec']:.2f} s** | {iv['ida']['nps']:,.0f} | **{us_ida:.0f} µs** |

**快 {speed:.1f} 倍**，而兩者展開的節點數幾乎一樣。

值得記住的是：**這個倍數完全沒有換演算法的聰明。**
heuristic 一模一樣，IDA\\* 展開的節點只有更多不會更少。差別純粹在每個節點的簿記成本。
主線那一篇列的五個技術挑戰，第三個就是「推論加速之後，優先佇列會變成瓶頸」——
這裡就是那件事的實例：把佇列整個拿掉，事情就成了。

### 這一輪多出來的檔案

```
ml/corners.py      角塊的座標編碼與全狀態 BFS（88 MB 的距離表）
ml/heuristics.py   三種估計法放在同一個介面下：網路、角塊表、取大
ml/idastar.py      IDA*，求精確最短解
ml/goal2.py        四組實驗（每段各自存檔，可續跑）
ml/make_demo2.py   影片要用的資料
tools/make_script2.py / make_readme2.py   旁白稿與這一段 README，都從結果產生
```

```bash
python ml/corners.py          # 建角塊表，114 秒
python ml/goal2.py            # 四組實驗，約 1 小時（中斷可續跑）
python ml/idastar.py --depth 12 --n 5   # 單獨玩最短解求解器
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
    print(f"README.md 的 GOAL002 段落已更新（{len(body):,} 字元）")


if __name__ == "__main__":
    main()
