"""DAVI — Deep Approximate Value Iteration。訓練 cost-to-go 網路。

    python ml/train.py --size 2          # 2x2x2，約 10 分鐘
    python ml/train.py --size 3          # 3x3x3，約 2 小時

這是整個專案裡最該看懂的一段。它跟前面幾個專案的 DQN 有三個根本差別：

1. **沒有環境互動、沒有 episode、沒有 replay buffer、沒有 epsilon。**
   訓練資料是從「已經解開」的狀態往回亂轉出來的。
   往回轉 d 步的局面，答案一定不超過 d 步——所以資料本身自帶難度標籤，
   而且要多少有多少。不需要 agent 先會玩才有資料可學。

2. **學的是 value，不是 policy。** 網路輸出一個數字：還要幾步。

3. **bootstrap 用的是「展開一層取最小」，不是「照著策略走」。**

       y(s) = min over a  [ 1 + h_target(a(s)) ]      （a(s) 已解開的話那項是 1）

   這就是 value iteration 的更新式，只是 h 從一張表換成一個網路。
   注意它跟策略無關：不管現在的網路多爛，這個 target 都是「所有子節點裡最好的那個 +1」。
   DQN 的 n-step return 會被爛策略帶壞（lab-ml-004 踩過），這裡不會。

target 網路的更新規則也不一樣：不是每 N 步硬換，
而是「等到 loss 掉到門檻以下才換」——代表現在的網路已經把上一版的 target 學會了，
可以把地基往上疊一層。這就是 value iteration 一輪一輪逼近的過程。
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cube import Cube, N_COLORS
from model import ValueNet

HERE = Path(__file__).resolve().parent
CKPT = HERE / "checkpoints"


def encode_torch(states: np.ndarray, device) -> torch.Tensor:
    """one-hot 編碼直接在 GPU 上做。

    在 numpy 做的話，3x3x3 每一輪要搬 300 MB 的 float32 過 PCIe，
    比訓練本身還久。搬 uint8 過去（13 MB）再在 GPU 上展開，快一個數量級。
    """
    x = torch.from_numpy(states).to(device, non_blocking=True).long()
    return F.one_hot(x, N_COLORS).float().flatten(1)


@torch.no_grad()
def eval_exact(net, cube: Cube, keys, dist, device, n=20000, seed=1234):
    """2x2x2 專用：拿精確答案來量這個 heuristic 有多準。

    三個數字：
      mae         平均差幾步
      over        h > h* 的比例（高估。A* 用高估的 heuristic 會失去最短解保證）
      greedy      不搜尋、每一步就照 h 挑最好的子節點，解得開的比例
    """
    from bfs import lookup
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(keys), size=n, replace=False)
    states = cube.unpack(keys[idx])
    truth = dist[idx].astype(np.float32)
    net.eval()
    h = net(encode_torch(states, device)).cpu().numpy()
    mae = float(np.abs(h - truth).mean())
    over = float((h > truth + 1e-6).mean())

    # 貪婪走：純看 h，不搜尋。走到 godNumber*3 步還沒解開就算失敗。
    cur = states[:5000].copy()
    done = cube.is_solved(cur)
    for _ in range(cube.cfg["godNumber"] * 3):
        if done.all():
            break
        live = ~done
        kids = cube.expand(cur[live])                            # (L, A, S)
        L, A, S = kids.shape
        flat = kids.reshape(-1, S)
        hv = net(encode_torch(flat, device)).cpu().numpy().reshape(L, A)
        hv[cube.is_solved(flat).reshape(L, A)] = -1e9             # 解開的子節點永遠最優
        pick = hv.argmin(axis=1)
        cur[live] = flat.reshape(L, A, S)[np.arange(L), pick]
        done = cube.is_solved(cur)
    net.train()
    return {"mae": mae, "over": over, "greedy": float(done.mean())}


@torch.no_grad()
def eval_depth_profile(net, cube: Cube, device, depths=(1, 5, 10, 15, 20, 25, 30), n=2000, seed=1234):
    """3x3x3 沒有正確答案可比，退而求其次：看預測值隨打亂深度怎麼長。

    亂轉 d 步的真實距離 <= d，而且 d 大了之後會飽和在上帝之數附近。
    如果網路的輸出跟著 d 單調上升、最後平掉，至少代表它學到了「難度」這個維度。
    """
    rng = np.random.default_rng(seed)
    net.eval()
    out = {}
    for d in depths:
        if d > cube.cfg["scrambleMax"]:
            continue
        s, _ = cube.scramble(np.full(n, d), rng)
        h = net(encode_torch(s, device)).cpu().numpy()
        out[str(d)] = round(float(h.mean()), 3)
    net.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2, choices=[2, 3])
    ap.add_argument("--iters", type=int, default=None, help="蓋掉 config 的 iters")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--target-loss", type=float, default=None,
                    help="loss 掉到多少才把 target 往前推一版（蓋掉 config）")
    ap.add_argument("--tag", default=None, help="checkpoint 檔名後綴，用來同時留住好幾組設定")
    args = ap.parse_args()

    device = torch.device(args.device)
    cube = Cube(args.size)
    cfg = cube.cfg
    d = cfg["davi"]
    iters = args.iters or d["iters"]
    if args.target_loss is not None:
        d = dict(d, targetUpdateLoss=args.target_loss)
    CKPT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.size}x{args.size}"

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    net = ValueNet(cube.obs_size, cfg["net"]["hidden"], cfg["net"]["blocks"]).to(device)
    target = ValueNet(cube.obs_size, cfg["net"]["hidden"], cfg["net"]["blocks"]).to(device)
    target.load_state_dict(net.state_dict())
    target.eval()
    opt = torch.optim.Adam(net.parameters(), lr=d["lr"])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=d["lrDecay"])

    start_it, updates = 0, 0
    last_ckpt = CKPT / f"last-{tag}.pt"
    if args.resume and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        net.load_state_dict(ck["net"])
        target.load_state_dict(ck["target"])
        opt.load_state_dict(ck["opt"])
        start_it, updates = ck["iter"], ck["updates"]
        print(f"從 {last_ckpt.name} 的第 {start_it} 輪接著練")

    # 2x2x2 的話把正確答案載進來，訓練途中就能量準度
    truth = None
    if args.size == 2:
        try:
            from bfs import load
            truth = load()
            print(f"載入正確答案表：{len(truth[0]):,} 個局面")
        except SystemExit as e:
            print(f"（{e}，訓練照跑，只是沒有準度可以量）")

    log_path = CKPT / f"log-{tag}.csv"
    new_log = not log_path.exists() or not args.resume
    log_f = open(log_path, "w" if new_log else "a", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    if new_log:
        log_w.writerow(["iter", "loss", "updates", "lr", "mae", "over", "greedy", "profile"])

    best = None
    n_params = net.n_params()
    print(f"\n{args.size}x{args.size}x{args.size} [{tag}] · {n_params:,} 個參數 · obs {cube.obs_size} 維 · "
          f"{cube.n_actions} 個動作 · batch {d['batch']} · 打亂 1~{cfg['scrambleMax']} 步 · "
          f"target 門檻 {d['targetUpdateLoss']}")
    print(f"每一輪展開 {d['batch'] * cube.n_actions:,} 個子節點送進 target 網路\n")

    run_loss, t0 = [], time.time()
    stall = 0                      # 連續幾次檢查沒推 target
    for it in range(start_it, iters):
        # ── 1. 資料：從解開狀態往回亂轉 1~K 步 ──
        # 走軌跡取樣（見 cube.scramble_walk），比每一列各走各的快一個數量級
        states, _ = cube.scramble_walk(d["batch"], cfg["scrambleMax"], rng)

        # ── 2. target：展開一層，取 min(1 + h_target(子節點)) ──
        kids = cube.expand(states)                              # (B, A, S)
        B, A, S = kids.shape
        flat = kids.reshape(-1, S)
        solved_kid = torch.from_numpy(cube.is_solved(flat)).to(device).view(B, A)
        with torch.no_grad():
            # target 的前向佔了整輪一半的時間，而它只是用來產生回歸目標，
            # 不參與反向傳播——用 fp16 算快一倍。fp16 的相對精度約 0.05%，
            # 值域最大 30 步的話誤差在 0.015 步以內，比網路本身的誤差小兩個數量級。
            with torch.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                h = target(encode_torch(flat, device)).view(B, A)
            h = h.float()
            h = torch.where(solved_kid, torch.zeros_like(h), h)
            y = (1.0 + h).min(dim=1).values
            # 打亂之後剛好又解開了（例如 R U R' U' 轉六輪）——那一列的答案就是 0
            y = torch.where(torch.from_numpy(cube.is_solved(states)).to(device),
                            torch.zeros_like(y), y)

        # ── 3. 回歸 ──
        pred = net(encode_torch(states, device))
        loss = F.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sched.step()
        run_loss.append(loss.item())

        # ── 4. loss 夠低才把 target 往前推一版 ──
        if (it + 1) % d["checkEvery"] == 0:
            m = float(np.mean(run_loss))
            # 原始規則：loss 夠低才推。但門檻設得比 loss 的下限還低的話，
            # target 會永遠不動，訓練就靜悄悄地停在那裡（2x2x2 用 0.05 就是這樣，
            # 停在第 10 版、MAE 卡在 1.43）。maxStall 是保險絲。
            forced = stall >= d.get("maxStall", 10**9)
            bumped = m < d["targetUpdateLoss"] or forced
            stall = 0 if bumped else stall + 1
            if bumped:
                target.load_state_dict(net.state_dict())
                target.eval()
                updates += 1
            run_loss = []

            ex = eval_exact(net, cube, *truth, device) if truth else {"mae": "", "over": "", "greedy": ""}
            prof = eval_depth_profile(net, cube, device) if args.size == 3 else {}
            el = time.time() - t0
            eta = el / max(1, it + 1 - start_it) * (iters - it - 1)
            extra = (f"mae {ex['mae']:.3f}  高估 {ex['over']:.1%}  貪婪解開 {ex['greedy']:.1%}"
                     if truth else f"預測值 {prof}")
            print(f"[{it + 1:>6}/{iters}] loss {m:.4f}  target 更新 {updates:>3}"
                  f"{' ←強制' if forced else (' ←' if bumped else '   ')}  lr {sched.get_last_lr()[0]:.2e}  {extra}"
                  f"  ({el / 60:.1f}m 已過, 約剩 {eta / 60:.0f}m)")
            log_w.writerow([it + 1, round(m, 5), updates, sched.get_last_lr()[0],
                            ex["mae"], ex["over"], ex["greedy"], json.dumps(prof)])
            log_f.flush()

            # 挑 checkpoint 的標準。
            # 2x2x2 有正確答案，直接用 MAE。
            # 3x3x3 沒有——而這裡有個陷阱：**不能用 loss**。
            # DAVI 訓練最初期的 loss 是最低的，因為那時候 target 網路幾乎是個常數，
            # 「猜一個固定的數字」當然很好擬合。用 loss 挑，會挑到第 2000 輪那個
            # 什麼都還沒學會的版本（實測就是這樣，差點把它當成果出貨）。
            # 改用「打亂最深時的預測值」：DAVI 的估計值是從 0 開始往真值爬的，
            # 爬得越高代表價值往外傳得越遠，這個量才跟訓練進度同方向。
            score = -ex["mae"] if truth else float(prof[str(max(int(k) for k in prof))])
            if best is None or score > best:
                best = score
                torch.save({"net": net.state_dict(), "size": args.size,
                            "obs_size": cube.obs_size, "hidden": cfg["net"]["hidden"],
                            "blocks": cfg["net"]["blocks"], "iter": it + 1,
                            "updates": updates, "eval": ex if truth else prof},
                           CKPT / f"best-{tag}.pt")
            torch.save({"net": net.state_dict(), "target": target.state_dict(),
                        "opt": opt.state_dict(), "size": args.size,
                        "obs_size": cube.obs_size, "hidden": cfg["net"]["hidden"],
                        "blocks": cfg["net"]["blocks"], "iter": it + 1,
                        "updates": updates, "eval": ex if truth else prof},
                       last_ckpt)

    log_f.close()
    print(f"\n完成。target 網路總共往前推了 {updates} 版，花了 {(time.time() - t0) / 60:.1f} 分鐘。")
    print(f"checkpoint: {CKPT / f'best-{tag}.pt'}")


if __name__ == "__main__":
    main()
