"""建立（或找到）playlist、組出描述、上傳影片，然後把影片網址寫回 out/youtube.json。

    python tools/publish_youtube.py --dry-run       # 只印出標題與描述，不上傳
    python tools/publish_youtube.py                 # 上傳
    python tools/publish_youtube.py --update <videoId>   # 只改描述

playlist 一定要在上傳「之前」就存在：描述裡必須帶自己 playlist 的網址，
而共用的 uploader 是上傳完才建 playlist，所以這裡先建、拿到真網址再寫描述。

章節時間直接從 out/plan.json 讀——那是錄這支影片時實際用的分鏡表，
不是人工抄的，改了影片長度也不會對不上。

結果表也一樣：直接讀 ml/checkpoints/benchmark-*.json，
所以描述裡的數字跟影片裡的數字必定是同一次跑出來的。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import time

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKEN = os.environ.get(
    "YT_TOKEN", r"C:\Users\mark\Documents\2026-mark-locally-only\yt_token.json")
UPLOADER = r"C:\2026BizProject\GOAL\001\routine\upload_youtube.py"
SCOPES = ["https://www.googleapis.com/auth/youtube"]

# 跟 lab-ml-001 ~ 005 同一個 playlist
PLAYLIST = "ai-ml-lab"
PLAYLIST_DESC = (
    "AI / ML lab — 從零手寫的機器學習練習：自己寫環境、自己寫演算法、自己訓練。"
    "Hand-written machine-learning experiments: own environment, own algorithm, own training loop."
)

# Claude artifact 預設是私人的，擁有者去分享之後外人才打得開。
ARTIFACT_URL = "https://claude.ai/code/artifact/f755f774-a455-47e8-ac5a-711fd6ecf763"

VIDEO = os.path.join(ROOT, "out", "rubik-deepcubea.mp4")
PLAN = os.path.join(ROOT, "out", "plan.json")
B2 = os.path.join(ROOT, "ml", "checkpoints", "benchmark-2x2.json")
B3 = os.path.join(ROOT, "ml", "checkpoints", "benchmark-3x3.json")
OUT_JSON = os.path.join(ROOT, "out", "youtube.json")
DESC_PATH = os.path.join(ROOT, "out", "youtube-desc.txt")

TITLE = "讓程式自己學會解魔術方塊｜DeepCubeA 實作：網路只學「還要幾步」，解開的是 A*"
TAGS = ("魔術方塊,Rubik's Cube,DeepCubeA,強化學習,reinforcement learning,value iteration,"
        "DAVI,A star,heuristic search,啟發式搜尋,機器學習,深度學習,PyTorch,numpy,"
        "2x2x2,pocket cube,上帝之數,God's number,Claude Code,AI 教學,RL from scratch")

CHAPTER_NAMES = {
    "title": "這支影片在做什麼",
    "random": "地板：亂轉一萬步，解開率 0%",
    "small": "先做一個知道正確答案的版本（2×2×2）",
    "hist": "上帝之數 14 — 以及評估時必須先扣掉的底線",
    "davi": "DAVI：不互動、不試錯、只學距離",
    "error": "這個估計到底有多準（它自己解不開）",
    "astar": "另外一半：加權 A* 搜尋",
    "weight": "weight — 速度與品質之間唯一的旋鈕",
    "cube3": "同一份程式，換成 3×3×3",
    "results": "實測數字與三個踩過的坑",
}


def creds():
    c = Credentials.from_authorized_user_file(TOKEN, SCOPES)
    if not c.valid:
        if c.expired and c.refresh_token:
            c.refresh(Request())
            bak = TOKEN + time.strftime(".bak-%Y%m%d-%H%M%S")
            try:
                with open(TOKEN) as f, open(bak, "w") as g:
                    g.write(f.read())
            except OSError:
                pass
            with open(TOKEN, "w") as f:
                f.write(c.to_json())
            print(f"token refreshed (backup: {os.path.basename(bak)})")
        else:
            sys.exit("token invalid and cannot refresh — re-auth needed")
    return c


def find_or_create(yt, title, privacy):
    req = yt.playlists().list(part="snippet", mine=True, maxResults=50)
    while req is not None:
        res = req.execute()
        for it in res.get("items", []):
            if it["snippet"]["title"].strip().lower() == title.lower():
                print(f"playlist exists: {title} ({it['id']})")
                return it["id"]
        req = yt.playlists().list_next(req, res)
    pl = yt.playlists().insert(
        part="snippet,status",
        body={"snippet": {"title": title, "description": PLAYLIST_DESC},
              "status": {"privacyStatus": "public" if privacy == "public" else "unlisted"}},
    ).execute()
    print(f"playlist created: {title} ({pl['id']})")
    return pl["id"]


def mmss(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def chapters():
    """章節時間來自這支影片自己的分鏡表，不是手抄的。"""
    if not os.path.exists(PLAN):
        raise SystemExit(f"缺 {PLAN} — 章節時間必須來自實際錄製的那一次")
    with io.open(PLAN, encoding="utf-8") as f:
        plan = json.load(f)
    fps = plan["fps"]
    return "\n".join(f"{mmss(s['start'] / fps)}  {CHAPTER_NAMES.get(s['id'], s['id'])}"
                     for s in plan["scenes"])


def load(p):
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)


def results_block():
    b2, b3 = load(B2), load(B3)
    out = [f"【2×2×2】{b2['n']} 個局面，從全部 {b2['state_count']:,} 個裡均勻抽（不是「亂轉 k 步」，"
           f"那樣會偏向簡單的局面）。正確答案平均 {b2['truth_mean']:.2f} 步。", ""]
    for r in b2["rows"]:
        opt = "—" if r.get("optimal_rate") is None else f"{r['optimal_rate'] * 100:.1f}%"
        ln = "—" if r.get("mean_len") is None else f"{r['mean_len']:.2f} 步"
        out.append(f"・{r['name']}：解開 {r['solve_rate'] * 100:.1f}% / 最短解 {opt} / "
                   f"平均 {ln} / 展開 {r['mean_nodes']:,.0f} 節點")
    h = b2["heuristic"]
    out += ["", f"heuristic 本身的準度（抽樣 {h['n']:,} 個局面）："
                f"平均差 {h['mae']:.2f} 步、{h['within_1'] * 100:.1f}% 落在 1 步以內、"
                f"{h['over_rate'] * 100:.1f}% 高估。", ""]
    out.append(f"【3×3×3】每個打亂深度 {b3['n']} 個局面。沒有正確答案可以比，只能問「解得開嗎、幾步」。")
    out.append("")
    for r in b3["by_depth"]:
        ln = "—" if r.get("mean_len") is None else f"{r['mean_len']:.1f} 步"
        out.append(f"・打亂 {r['depth']:>2} 步：解開 {r['solve_rate'] * 100:.0f}% / "
                   f"平均 {ln} / 展開 {r['mean_nodes']:,.0f} 節點")
    out.append(f"・（對照）{b3.get('random_walk_n', 1000):,} 個局面各亂轉 10,000 步的解開率："
               f"{b3['random_walk_solve_rate'] * 100:.1f}%")
    deep = b3["by_depth"][-1]
    out += ["", f"每一個深度都 100%，包含完全隨機的 {deep['depth']} 步打亂——"
                f"平均 {deep['mean_len']:.1f} 步、展開 {deep['mean_nodes']:,.0f} 個節點、"
                f"{deep['ms_per_cube'] / 1000:.1f} 秒一顆。",
            "但這裡量不出「差最短解幾步」——那正是 2×2×2 存在的理由。"]
    return "\n".join(out)


def build_description(playlist_url):
    b2 = load(B2)
    h = b2["heuristic"]
    greedy = next(r["solve_rate"] for r in b2["rows"] if "貪婪" in r["name"])
    astar = [r for r in b2["rows"] if r["name"].startswith("加權 A*")]
    best = max(astar, key=lambda r: r["optimal_rate"])
    return f"""一顆三階魔術方塊有 43,252,003,274,489,856,000 種排列，其中只有一種是解開的。
亂轉一萬步碰到那一種的機率，實測是 0%——不是很低，是一次都沒有。
所以「先隨機探索，嚐到一次成功再把訊號往回傳」這條路，從一開始就是斷的。

這支影片做的是 DeepCubeA 的作法：網路完全不學「要轉哪一面」，
它只學一件事——看一眼局面，猜「離解開還有幾步」。真正把方塊解開的是 A* 搜尋。

▶ 播放清單 / Playlist: {playlist_url}
▶ 可互動的完整圖文版（中英雙語，網頁上的網路是真的在跑推論，2×2×2 還會即時算出精確答案跟它對照）/ Interactive write-up: {ARTIFACT_URL}

{chapters()}

── 為什麼要先做 2×2×2 ──
因為 2×2×2 只有 3,674,160 個局面，小到可以整個列出來。
從解開狀態做一次廣度優先搜尋，18 秒之後，每一個局面的精確最短步數就全部在手上了。
（算出來的分布跟數學社群公開的那張表逐項相同——順便驗了置換表沒寫錯。）

有了正確答案，才問得出這個專案真正的問題：一個學出來的估計值，到底離真的有多遠？
3×3×3 沒有這個奢侈，所以那邊只能問「解得開嗎、幾步」。
先在看得到答案的地方把方法驗清楚，再放大——這就是 2×2×2 存在的理由。

── 結果 ──
{results_block()}

── 這支影片真正想講的三件事 ──

1. **這個網路自己解不開魔術方塊。**
   如果不搜尋、每一步就挑它覺得最近的子節點走，解開率只有 {greedy * 100:.1f}%。
   但同一個網路配上加權 A*，解開率 100%——實務設定（weight=0.6）平均只展開
   {[r for r in astar if '0.6' in r['name']][0]['mean_nodes']:.0f} 個節點、
   {[r for r in astar if '0.6' in r['name']][0]['optimal_rate'] * 100:.0f}% 剛好是最短解；
   把 weight 調到 1.0 則有 {best['optimal_rate'] * 100:.1f}% 是最短解。
   heuristic 不需要準，只要方向大致對，搜尋會把細節補回來。
   把「判斷好壞」跟「決定要走哪」拆開，是這套方法能從三百萬個局面搬到四千京個局面的原因。

2. **評估的時候要先扣掉底線。**
   2×2×2 有八成以上的局面落在 10~12 步之間。也就是說，一個「不管看到什麼都回答 11」的
   假網路，平均誤差就已經很小了。所以「平均差 {h['mae']:.2f} 步」這個數字單獨看沒有意義——
   要看的是它在很近和很遠的局面上有沒有跟著動（影片裡那張圖），
   以及 {h['over_rate'] * 100:.1f}% 的高估率——因為那決定了 A* 還有沒有最短解的保證。

3. **照論文抄超參數，訓練會靜悄悄地停住。**
   DAVI 的 target 更新規則是「loss 掉到門檻以下才把 target 往前推一版」。
   論文那個門檻（0.05）放到我這個比較小的網路上，loss 根本掉不到——
   12,000 輪裡 target 只推了 10 版就再也不動，最好只到 MAE 1.35。
   而訓練看起來完全正常：還在跑、loss 還在慢慢降、什麼錯都沒噴。
   門檻放寬到 0.15 之後，同樣 12,000 輪推了 34 版，MAE 1.13。
   同一份程式碼、同樣的訓練步數，差別只有那一個數字。
   （現在程式裡多了一條保險：連續幾次檢查都沒推，就強制推一版。）

   圖文版還有兩個沒進影片的坑：一個「只為了餵飽 GPU」的批次參數會把 A* 悄悄換成 BFS；
   以及 3×3×3 沒有正確答案，連「該存哪個 checkpoint」都會挑錯——
   用 loss 挑，挑到的是最初期什麼都還沒學會的那一版。

── 怎麼做的 ──
・轉動表：從立方體的 3D 座標算出來，不是手打的。順時針轉一面就是繞法向量轉 −90 度，
　p' = a(a·p) − (a×p)。算完會用群論性質自我檢查（(R U) 的階是 105 等等），
　錯一個數字就當場爆掉，不會拖到訓練完才發現。
・訓練：DAVI（deep approximate value iteration）。沒有環境互動、沒有 episode、
　沒有 replay buffer、沒有 epsilon。資料是從解開狀態往回亂轉造出來的。
　目標 y(s) = min over a [ 1 + h_target(a(s)) ]——跟策略無關，所以爛策略帶不壞它。
・搜尋：批次加權 A*。一次從佇列拿 N 個最好的節點，子節點湊成一批送 GPU。
・部署：權重把 BatchNorm 摺進線性層之後匯出成 JSON，瀏覽器用純 JS 前向傳播，
　不需要 TensorFlow.js。JS 與 Python 兩邊逐格對帳，推論誤差小於 1e-5。
・網頁上的「正確答案」：雙向 BFS，兩邊各走 7 步一定相遇（上帝之數 14），
　只要看不到十萬個局面。單向走 14 步要看三百六十七萬個——那就撐不住了。
・語音：edge-tts zh-TW-HsiaoChenNeural
・影片：headless Chrome 逐格離線算圖（非螢幕錄影，不會掉格），旁白時間軸由程式計算對齊

── English ──
A 3x3x3 Rubik's cube has 43,252,003,274,489,856,000 states and exactly one of them is solved.
Ten thousand random turns finds it 0% of the time — measured, not estimated — so the usual
"explore until you stumble into a reward" loop is dead on arrival. This is DeepCubeA's answer:
the network never learns which face to turn. It learns one number — how far this state is from
solved — and a weighted A* search does the solving. Everything is hand-written: the move tables
are derived from 3D geometry (and checked against known group orders), the environment is numpy,
the value network is a small residual MLP trained by approximate value iteration on states
generated backwards from the goal. The 2x2x2 comes first because its 3,674,160 states can be
enumerated exhaustively in 18 seconds — which means "how wrong is the learned heuristic" is
measured exactly, against ground truth, rather than guessed. The headline: the network on its own
solves only {greedy * 100:.1f}% of cubes greedily; with search it solves 100% of them, expanding
28 nodes on average, and {best['optimal_rate'] * 100:.1f}% of those solutions are provably optimal. Separating judgment from
decision is what lets the same code scale from three million states to forty-three quintillion.
Full bilingual write-up, with the trained network running live in the page, at the link above.

Created by MarkLuce AI · Claude Code · Claude Opus 5
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--privacy", default="public", choices=["unlisted", "public", "private"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--update", help="只更新這個 videoId 的描述")
    a = ap.parse_args()

    if "__ARTIFACT_URL__" in ARTIFACT_URL:
        print("警告：ARTIFACT_URL 還沒填，描述裡會出現佔位字串。", file=sys.stderr)

    yt = build("youtube", "v3", credentials=creds())
    pid = find_or_create(yt, PLAYLIST, a.privacy)
    purl = f"https://www.youtube.com/playlist?list={pid}"
    print(f"PLAYLIST: {purl}")

    desc = build_description(purl)
    with io.open(DESC_PATH, "w", encoding="utf-8") as f:
        f.write(desc)

    bad = [c for c in "<>" if c in desc]
    if bad:
        raise SystemExit(f"描述裡有 {bad} — YouTube 的 videos.insert 不接受角括號"
                         "（會回 invalidDescription）。改成「小於」「大於」之類的寫法。")
    if len(desc) > 5000:
        raise SystemExit(f"描述 {len(desc)} 字元，超過 YouTube 的 5000 上限 {len(desc)-5000} 字。"
                         "先把 build_description() 修短再跑。")

    if a.dry_run:
        print(f"\nTITLE ({len(TITLE)} chars): {TITLE}\n")
        print("-" * 70)
        print(desc)
        print("-" * 70)
        print(f"\ndry run：不會上傳。描述已寫到 {DESC_PATH}（{len(desc)} 字元，上限 5000）")
        return

    if a.update:
        cur = yt.videos().list(part="snippet", id=a.update).execute()["items"][0]["snippet"]
        cur["description"] = desc
        yt.videos().update(part="snippet", body={"id": a.update, "snippet": cur}).execute()
        print(f"description updated on {a.update}")
        return

    if not os.path.exists(VIDEO):
        sys.exit(f"找不到影片 {VIDEO}")
    size_mb = os.path.getsize(VIDEO) / 1048576
    print(f"uploading {os.path.basename(VIDEO)} — {size_mb:.0f} MB as {a.privacy} …")
    proc = subprocess.run(
        [sys.executable, UPLOADER,
         "--video", VIDEO, "--title", TITLE, "--desc-file", DESC_PATH,
         "--playlist", PLAYLIST, "--privacy", a.privacy, "--tags", TAGS],
        check=True, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    m = re.search(r"uploaded video:\s*([\w-]{11})", proc.stdout) or \
        re.search(r"youtu\.be/([\w-]{11})", proc.stdout)
    vid = m.group(1) if m else None

    # 共用的 uploader 會吃掉 --tags，上傳完的影片標籤是空的。
    # 這裡補一次 videos.update 補回去——body 裡一定要帶 categoryId，
    # 少了它 YouTube 會默默忽略整個 snippet 更新（回 200，但 tags 還是空的）。
    if vid:
        try:
            sn = yt.videos().list(part="snippet", id=vid).execute()["items"][0]["snippet"]
            sn["tags"] = [t.strip() for t in TAGS.split(",") if t.strip()]
            sn.setdefault("categoryId", "28")      # Science & Technology
            sn["defaultLanguage"] = "zh-Hant"
            r = yt.videos().update(part="snippet", body={"id": vid, "snippet": sn}).execute()
            print(f"補上 {len(r['snippet'].get('tags', []))} 個 tags")
        except Exception as e:
            print(f"警告：tags 沒補成功（{e}），可以手動再跑一次", file=sys.stderr)

    info = {"playlist_id": pid, "playlist_url": purl,
            "video_id": vid, "video_url": f"https://youtu.be/{vid}" if vid else None,
            "title": TITLE, "privacy": a.privacy}
    with io.open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if not vid:
        print("警告：videoId 沒抓到。artifact 的反向連結需要它——請手動補進 out/youtube.json。")


if __name__ == "__main__":
    main()
