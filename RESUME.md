# RESUME — 接下來從哪裡繼續

最後更新：2026-08-26　分支：`goal003-edges`

---

## 一句話

第二階段（邊塊表）已經做完並發布了，**但使用者在發布之後指出我的做法有問題**，
新的實驗證實他是對的——已發布的 artifact 現在有一段結論是錯的，必須改。

---

## 現況：三件事已完成、一件事做到一半

### 已完成並發布

| 東西 | 狀態 |
|---|---|
| `ml/edges.py` 兩張六條邊的 pattern database | ✅ 各 42,577,920 狀態、最遠 11、平均 8.762、43 MB |
| `ml/goal3.py` 三張表的值分佈 + IDA* 節點數尺度 | ✅ `out/goal3.json` |
| `ml/stage2.py` 先解角的代價 | ⚠️ 結論已被推翻，見下 |
| artifact 2（同一個 URL） | ⚠️ 已發布，但內容需要改 |
| 分支 `goal003-edges` | ✅ 已推上 GitHub，commit `55d6b1e` |

artifact 網址：<https://claude.ai/code/artifact/ee508e66-6cf3-472b-a0a3-4ac4a5187c06>

### 站得住腳的發現（不用改）

1. **3×3×3 沒有「解中心」這個階段。** 12 種面轉之下，54 片貼紙有 6 片永遠不動，
   正好是六個面的正中央。所以只有兩個階段：角、邊。Korf 的第三張表不是第三個階段，
   是「邊」這一個階段切兩半實作（12 條邊一起有 9,810 億個狀態，數不完）。
   使用者另外補充得很對：中心不動是**這個座標系**造成的，人手上整顆翻不改變方塊狀態。

2. **兩張邊表把下界平均只推高 0.11 步（10.63 → 10.74），92.6% 的局面根本沒動到，
   但 IDA\* 的節點數掉 167–665 倍。** 因為節點數是（真正距離 − 下界）的指數。
   只用角塊表在打亂 15 步撞牆，三張表推到 16。

3. **角塊一歸位，角塊表就永遠讀 0**，下界從 10.74 掉到 9.15。

---

## 使用者的指正（這是重點）

原文：

> 人先解角是對的，機器先解角是錯的 --- better to re-define the end of stage1 which will
> be starting points of stage2, it's ok it will take few more steps but to split the complex
> to 2 managed stages, so make 2 stages for 3×3×3 corner bench …
> end of stage1 is exactly 只解角塊, then it will be starting point of stage2,
> you need to revise stage2 and don't refuse it to find reasons

**他是對的。** 我原本的 stage 1 是**貪心**的：隨便走一條角塊最短解，
落在哪個邊塊局面完全看運氣。但角塊歸位的終點**有非常多個**
（角塊最短解通常不只一條，再放寬幾步更多），這些終點角塊都一樣，差別只在邊塊。
挑一個邊塊下界最小的，stage 2 就從近得多的地方開始。

所以錯的不是「先解角」，是**「瞎的 stage 1」**。

已經實作：`CornerStage.best_endpoint(state, edge_h, slack, beam)` in `ml/stage2.py`。
用 beam search，限制「剩下的預算夠把角塊轉回去」，排名用 `g + 角塊距離 + 邊塊下界`。

### 已量到的數字

挑終點參數定案：**slack=2、beam=2000**（slack=0 就拿到大部分好處，
slack 超過 2 沒有再變好）。

打亂 25 步、6 顆：

| | 階段 1 | 邊塊下界 | 兩段合計下界 |
|---|---:|---:|---:|
| 貪心 | 10.7 | 9.5 | 20.2 |
| slack=0 | 10.7 | 7.3 | 18.0 |
| slack=2 | 11.0 | 6.8 | **17.8** |

淺方塊（兩階段都求到最短，seed 303）：

| 打亂 | # | 最短解 | 貪心合計 | 挑過合計 |
|---:|---:|---:|---:|---:|
| 8 | 0–5 | 8 | 16 / 18 / 18 / 18 / 20 / 20 | **8**（六顆全部） |
| 9 | 0 | 9 | 15 | **9** |
| 9 | 1 | 9 | 17 | 19 ← **挑壞了** |
| 9 | 2,3,4,5 | 9,9,7,9 | 19 / 21 / 17 / 19 | **9 / 9 / 7 / 9** |
| 10 | 0 | 10 | 跑不完（240s、930 萬節點） | 22（239s） |
| 10 | 1 | 8 | 14 | 14 |

挑過之後大部分方塊 stage 2 是 **0 步、0 節點**——因為 stage 1 直接走到最短解上了。

**兩個必須誠實寫出來的但書：**
- 9 #1 那顆挑壞了（19 > 17）。排名用的是邊塊**下界**不是真距離，偶爾會誤判。
- 淺方塊上 stage 1 常常把整顆解掉，兩階段的拆分其實 collapse 了。
  真正要看兩階段的地方是**深方塊**（stage 1 的預算不夠解完整顆）。

---

## 做到一半：深方塊的實驗

`ml/stage2.py` 的 `[3/4]` 段已經寫好但**還沒跑完**：深方塊（20/25/30 步）的
stage 2 改用網路那支求解器（`search.bwas` + `benchmark.load_net`，找得到解但不保證最短），
這樣深方塊的**真實總步數**才量得出來。`[3b/4]` 是兩種 stage 1 的下界對照（每深度 40 顆）。

重跑指令：

```bash
cd ml
python stage2.py --n-full 6 --full-depths 8 9 10 --full-budget 240 \
                 --n-deep 4 --n-bound-pick 40
```

跑完會寫出 `out/stage2.json`，裡面多了 `bounds` 與 `pick` 兩個欄位。

**注意**：`ml/checkpoints/edges3_*.npy` 用 `mmap` 開，Windows 上被開著的時候
`python ml/edges.py` 會覆寫失敗。要重算建表統計用 `python ml/edges.py --stats`。

---

## 還沒做：三件事

### 1. 改 artifact（必做，目前線上內容有錯）

`tools/artifact2_template.html` 的「先解角的代價」那一節現在的標題是

> **人先解角是對的，機器先解角是錯的**

這句話**站不住了**。要改成大意如下：

> 先解角不是繞路——**瞎著先解角**才是繞路。
> 角塊歸位的終點有非常多個，挑對了，第二階段又短又好算。
> 兩個階段不該對彼此視而不見。

同一節裡「先解角讓剩下的部分更難解，不是更好解」也要改。
要保留的是「角塊歸位之後下界從 10.74 掉到 9.15」這個機制——
那仍然是對的，只是結論不同：正因為下界會塌，**stage 1 的終點才更需要挑**。

改完之後：

```bash
node tools/build_artifact2.mjs
python tools/make_readme3.py
```

然後用 Artifact 工具帶 `url=https://claude.ai/code/artifact/ee508e66-6cf3-472b-a0a3-4ac4a5187c06`
重新發布（同一個網址）。

`tools/make_readme3.py` 的「先解角繞了多遠——精確值」整段也要跟著改。

### 2. 互動台改成兩階段（使用者明確要求）

> so make 2 stages for 3×3×3 corner bench …
> end of stage1 is exactly 只解角塊, then it will be starting point of stage2

現在的互動台只有「只解角塊」（= stage 1 結束）。要加 stage 2。

**技術難點**：瀏覽器沒有邊塊表，也沒有網路（22 MB 塞不進去）。
已經評估過的可行路線：

- 邊塊的移動表**可以在瀏覽器現場生成**——只要送 12 個 slot 置換（12×12 個數字），
  JS 做 665,280 × 12 次 unrank/rank，估計一秒內。
- 然後 BFS 建一張 42,577,920 的表 = 42.5 MB `Uint8Array`。
  記憶體上可行（現有的角塊雙向 BFS 已經配置 88 MB）。JS 純量迴圈估計 10–30 秒，
  要做成延遲載入 + 進度條。
- 有了那張表，stage 2 用 IDA\* 跑。如果 stage 1 的終點挑得好，剩下的距離很短，搜尋會很快。

比較省的替代方案：只建「6 條邊的位置、不管翻轉」那張 665,280 的表（JS 一秒內建完），
下界比較鬆但可能夠用——**還沒驗過，要先量**。

另外 `web/corners3.js` 的 `CornerSolver` 可以參考，那支已經在瀏覽器裡做雙向 BFS。

### 3. 影片（等使用者決定）

前兩篇的慣例是 artifact + YouTube 影片 + 分支各一套。這一輪我只做了 artifact 更新，
**沒有問到使用者要不要新影片**。公開上傳是對外動作，要他點頭才做。

---

## 使用者那邊還沒處理的事

- **兩個 artifact 的 share pin 都還停在舊版本。** 讀者看不到更新，
  兩支 YouTube 影片說明欄的連結對外也是死的。這只有他能在頁面上操作。
- 這一輪要不要新的 YouTube 影片。

---

## 檔案地圖（這一輪新增/改動的）

| 檔案 | 做什麼 |
|---|---|
| `ml/edges.py` | 兩張六條邊的 PDB。`--check` 只驗編碼、`--stats` 只重算統計 |
| `ml/goal3.py` | 三張表值分佈 + IDA\* 節點數尺度。有分段快取，`--force scale` 可重跑 |
| `ml/stage2.py` | `CornerStage.solve()` 貪心、`.best_endpoint()` 挑終點。四段實驗 |
| `ml/heuristics.py` | 多了 `EdgePDB`、`korf_bound()` |
| `ml/idastar.py` | `OptimalSolver(use_edges=True)`、`solve(..., deadline=秒)` |
| `tools/make_readme3.py` | 產生 README 的第二階段段落，數字全部讀檔 |
| `tools/artifact2_template.html` | 加了三節 + `drawNodes()` 對數圖 |
| `tools/build_artifact2.mjs` | 多灌 `out/goal3.json`、`out/stage2.json`、`edges-build.json` |

**沒進版控**（`.gitignore`）：`ml/checkpoints/edges3_*.npy`（各 43 MB，
`python ml/edges.py` 七十秒重建）、`ml/checkpoints/goal3-*.json`（分段快取）。

---

## 踩過的坑（別再踩一次）

1. **`id="gSpeed"` 撞名**。既有的 IDA\* 那一節已經用了，我新加的重複，
   害 IDA\* 的儀表顯示成 665×。已改名 `gEdgeSpeed`。
   加新的 `id` 之前先跑一次重複檢查。
2. **「省幾倍」拿不同方塊在比**。goal3 深度 14 只有 2 顆跑完角塊表、8 顆跑完三張表，
   直接比平均是錯的。已改成 `common_ratio()` 只比共同的前綴。
3. **heredoc 裡的 `\n`** 在 Bash 工具會變成真的換行，寫進 Python f-string 就爆
   `unterminated f-string literal`。改用 Write/Edit 工具寫多行 Python。
4. **本機預覽亂碼**不是 bug。`python -m http.server` 不送 charset，
   而 artifact 的 `<meta charset>` 是發布時由外層 `<head>` 提供的。
   要本機看就自己包一層 `<!doctype html><meta charset="utf-8">`。
5. **背景程序常被殺掉**。長實驗用 `nohup ... &` 加分段快取。
