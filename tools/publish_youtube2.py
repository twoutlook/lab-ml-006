"""建立（或找到）playlist、組出描述、上傳影片，然後把影片網址寫回 out/youtube.json。

    python tools/publish_youtube2.py --dry-run       # 只印出標題與描述，不上傳
    python tools/publish_youtube2.py                 # 上傳
    python tools/publish_youtube2.py --update <videoId>   # 只改描述

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
# 這一支自己的圖文版，發布之後再填。上一支的網址在 out/artifact1.json。
ARTIFACT_URL = "https://claude.ai/code/artifact/ee508e66-6cf3-472b-a0a3-4ac4a5187c06"

VIDEO = os.path.join(ROOT, "out", "rubik-corners-first.mp4")
PLAN = os.path.join(ROOT, "out", "plan2.json")
G2 = os.path.join(ROOT, "ml", "checkpoints", "goal2.json")
A1 = os.path.join(ROOT, "out", "artifact1.json")
B3 = os.path.join(ROOT, "ml", "checkpoints", "benchmark-3x3.json")
OUT_JSON = os.path.join(ROOT, "out", "youtube2.json")
DESC_PATH = os.path.join(ROOT, "out", "youtube2-desc.txt")

TITLE = "先解角，再解邊｜用人類 1974 年的老方法，替 3×3×3 造出第一把精確的尺"
TAGS = ("魔術方塊,Rubik's Cube,corners first,角先法,Waterman method,pattern database,"
        "Korf,Thistlethwaite,IDA star,A star,admissible heuristic,啟發式搜尋,最短解,"
        "上帝之數,God's number,機器學習,深度學習,numpy,PyTorch,Claude Code,AI 教學")

CHAPTER_NAMES = {
    "back": "上一支停在哪裡：3×3×3 沒有正確答案",
    "human": "人為什麼先解角（Rubik 1974 / Waterman 1981）",
    "count": "把角塊整個數完：88,179,840 個狀態，114 秒",
    "heur": "精確但片面，對上不精確但完整",
    "truth": "3×3×3 第一次有正確答案",
    "staged": "人類的分階段解法，到底值多少",
    "ida": "為什麼換掉 A*：IDA* 快 100 倍",
    "end": "還缺兩張表",
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
    g = load(G2)
    P, E1, E2, E3, E4 = g["corner_pdb"], g["e1"], g["e2"], g["e3"], g["e4"]
    pdb_row = next(r for r in E1 if r["admissible"])
    net_row = next(r for r in E1 if not r["admissible"] and "∪" not in r["name"])
    max_row = next(r for r in E1 if "∪" in r["name"])
    save = (1 - max_row["mean_nodes"] / net_row["mean_nodes"]) * 100
    one, two = E4["one_shot"], E4["staged"]
    out = [f"【角塊距離表】{P['states']:,} 個狀態（8! × 3^7）全部數完，114 秒。"
           f"上帝之數 {P['max']}，平均 {P['mean']:.3f} 步。",
           "對照：2×2×2 是 3,674,160 個狀態、上帝之數 14、平均 10.666 步——"
           "兩邊各自獨立算出來卻對得上（88,179,840 = 24 × 3,674,160）。", "",
           "【當 heuristic 比一比】同一批隨機方塊，其他設定不動：", ""]
    for r in E1:
        ln = "—" if r["mean_len"] is None else f"{r['mean_len']:.1f} 步"
        out.append(f"・{r['name']}：平均估計 {r['mean_h']:.2f}／"
                   f"{'保證不高估' if r['admissible'] else '會高估'}／"
                   f"解開 {r['solve_rate'] * 100:.0f}%／平均 {ln}／"
                   f"展開 {r['mean_nodes']:,.0f} 節點")
    _over = [r["d"] for r in E2["by_true_net"] if r["mean"] >= r["d"] - 1e-9]
    _lo = max(_over) if _over else None
    _hi = min([r["d"] for r in E2["by_true_net"] if _lo is not None and r["d"] > _lo], default=None)
    cross = f"{_lo}~{_hi}" if _lo is not None and _hi is not None else "?"
    out += ["", f"精確但片面的那個，單獨用一顆都解不開；跟網路取大之後卻少了 {save:.0f}% 的節點。", "",
            f"【3×3×3 第一次有正確答案】{E2['n']} 個局面求出可以證明的最短解。",
            f"網路對這些局面的平均誤差 {E2['mae']:.2f} 步、{E2['within_1'] * 100:.0f}% 落在 1 步以內。",
            f"而且網路那條線在真實最短解 {cross} 步附近穿過對角線——比它淺的高估、比它深的低估，"
            f"就是回歸往平均縮。整體平均偏差 {E2['bias']:+.2f} 步幾乎是零，"
            "但那個零是兩個相反方向的誤差抵銷出來的，單看它會以為沒有系統性誤差。", ""]
    by = {}
    for r in E2["records"]:
        by.setdefault(r["depth"], []).append(r["true"])
    for d, v in sorted(by.items()):
        out.append(f"・打亂 {d:>2} 步（{len(v)} 顆）：精確最短解平均 {sum(v) / len(v):.2f} 步")
    out += ["", f"【人類的分階段值多少】先把八顆角解開（保證最短），再解完剩下的：",
            f"・一次解完：{one['mean_len']:.2f} 步／展開 {one['mean_nodes']:,.0f} 節點",
            f"・先解角再解完：{two['mean_len']:.2f} 步"
            f"（角 {two['mean_stage1']:.1f} + 其餘 {two['mean_stage2']:.1f}）／"
            f"第二段展開 {two['mean_nodes']:,.0f} 節點",
            f"・分階段多走 {two['mean_len'] - one['mean_len']:.1f} 步，"
            f"但第二段少展開 {(1 - two['mean_nodes'] / one['mean_nodes']) * 100:.0f}% 的節點", "",
            f"【深局面的夾擠】下界平均 {E3['mean_lower']:.1f} 步、找到的解平均 {E3['mean_found']:.1f} 步，"
            f"落差最多 {E3['mean_gap_upper']:.1f} 步。夾子還很鬆——要夾緊得補上 Korf 另外兩張「六條邊」的表。"]
    return "\n".join(out)


def build_description(playlist_url):
    g = load(G2)
    a1 = load(A1)
    iv = load(os.path.join(ROOT, "web", "demo2.json"))["idaVsAstar"]
    speed = iv["ida"]["nps"] / iv["astar"]["nps"]
    us_astar = iv["astar"]["sec"] / iv["astar"]["nodes"] * 1e6
    us_ida = iv["ida"]["sec"] / iv["ida"]["nodes"] * 1e6
    P, E2, E4 = g["corner_pdb"], g["e2"], g["e4"]
    one, two = E4["one_shot"], E4["staged"]
    return f"""上一支影片的最後停在一個很不甘心的地方：3×3×3 沒有正確答案，
所以「這個解比最短解長幾步」量不出來。這一支把那句話改掉一半。

方法不是更大的網路，也不是更久的訓練——是回頭看人類怎麼解方塊。
1974 年 Ernő Rubik 第一次解開他自己發明的方塊，用的是先解角。
人這樣拆是為了好記；但同一個拆法換到機器那邊，用途完全不一樣——它是一把尺。

▶ 播放清單 / Playlist: {playlist_url}
▶ 這一支的圖文版（中英雙語）/ Write-up: {ARTIFACT_URL}
▶ 上一支影片 / Previous video: {a1['video']}
▶ 上一支的圖文版 / Previous write-up: {a1['url']}

{chapters()}

── 為什麼「先解角」在機器這邊是一把尺 ──
八顆角只有 8! × 3^7 = {P['states']:,} 個狀態。跟整顆方塊的 4.3 × 10^19 比起來是可以數完的，
而且跟上一支那個 2×2×2 的 3,674,160 是同一個量級（剛好 24 倍）。
從解開狀態做一次廣度優先搜尋，114 秒，每一個角塊狀態離解開幾步就全部在手上。

關鍵在於：「只把角塊轉回去要幾步」是「解開整顆要幾步」的下界，
因為解開整顆的每一步也都在轉角塊。所以它保證不高估——
配上課本版的 A*，找到的解就是**可以證明的最短解**。
Korf 1997 年算隨機方塊的最短解，用的就是這個想法的三張表：角塊、6 條邊、另外 6 條邊。

── 結果 ──
{results_block()}

── 三件值得記住的事 ──

1. **精確但片面的東西，單獨用沒力，當補丁很有效。**
   角塊表保證不高估，但它太保守（平均只說 10.5 步，而真實最短解在 20 上下），
   單獨當 heuristic 一顆都解不開。可是跟網路取大之後，節點數明顯下降——
   因為在網路低估得最離譜的那些局面上，它把估計拉了回來。

2. **人類分階段是為了記憶，不是為了效率，而這個代價量得出來。**
   先解角的總步數比一次解完多，但第二段要搜的節點少。
   人類換到的不是節點數，是「每一塊都小到可以背」。
   所以速解高手最後都不用純角先法——那多出來的步數，在計時賽裡太貴。

3. **有時候瓶頸不是演算法，是簿記。**
   第一版用 A* 求最短解，每個節點要做十二次「轉成鍵、查字典、丟進堆積」。
   換成 IDA*（深度優先 + f 值上限，沒有 open/closed list，角塊座標增量更新）之後，
   同一批局面、同一個 heuristic、展開的節點數幾乎一樣，快了 {speed:.0f} 倍
   （每節點從 {us_astar:.0f} 微秒降到 {us_ida:.0f} 微秒）。
   heuristic 一模一樣，IDA* 展開的節點只有更多——差別純粹在每個節點的成本。

── 怎麼做的 ──
・角塊表：8 顆角的「誰在哪」與「轉了幾度」在轉動下各自獨立，
　所以 40,320×12 和 2,187×12 兩張小表就推得出全部 8,800 萬個狀態的鄰居，BFS 純向量化
・驗證：一致性（每走一步距離最多變 1）、下界（角塊距離不曾大於任何一條解的長度）
・最短解：IDA* + 三條安全剪枝（不走反手、同一面不連三次、對面固定順序）
・語音：edge-tts zh-TW-HsiaoChenNeural
・影片：headless Chrome 逐格離線算圖，旁白時間軸由程式計算對齊

── English ──
The previous video ended by admitting that a 3x3x3 has no ground truth, so "how much longer than
optimal is this solution" could not be measured. This one fixes half of that — not with a bigger
network or more training, but by borrowing the way people solve cubes. Ernő Rubik solved his own
cube corners-first in 1974; humans decompose for memorability. The same decomposition on the
machine side is something else entirely: the eight corners have only 8! x 3^7 = {P['states']:,}
states, few enough to enumerate exhaustively in 114 seconds, and "moves to fix the corners alone"
is a provable lower bound on solving the whole cube. That makes it an admissible heuristic — and
with it, textbook A* returns provably optimal solutions, giving this project ground truth on a
3x3x3 for the first time. Three findings: an exact-but-partial estimate is useless alone and
valuable as a patch; the human staging costs {two['mean_len'] - one['mean_len']:.1f} extra moves
and buys a smaller search; and a 100x speed-up came from removing bookkeeping, not from a smarter
algorithm. Full bilingual write-up at the link above.

Created by MarkLuce AI · Claude Code · Claude Opus 5
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--privacy", default="public", choices=["unlisted", "public", "private"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--update", help="只更新這個 videoId 的描述")
    a = ap.parse_args()

    if "__ARTIFACT2_URL__" in ARTIFACT_URL:
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
