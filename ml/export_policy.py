"""把訓練好的 checkpoint 轉成瀏覽器直接讀得動的 policy.json。

    python ml/export_policy.py --size 2
    python ml/export_policy.py --size 3 --digits 3

匯出時做兩件事：

1. **把 BatchNorm 摺進前面的線性層。** 推論時 BN 只是「乘一個常數、加一個常數」，
   可以直接吸進權重裡。這樣 web/nn.js 就不必再實作一次 BN——
   少一份要跟 PyTorch 對齊的東西，就少一個會偷偷不一致的地方。

2. **四捨五入。** 權重存幾位小數直接決定檔案多大：
   5 位約 8 bytes/參數，3 位約 6 bytes。頁面要能在網路上打得開，這個很有感。
   摺完 BN 之後會順便印出「摺前 vs 摺後」的最大誤差，確認沒有摺壞。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cube import Cube, N_COLORS
from model import ValueNet, fold_bn

HERE = Path(__file__).resolve().parent


def dump(w: torch.Tensor, b: torch.Tensor, nd: int):
    return {"w": [[round(float(v), nd) for v in row] for row in w],
            "b": [round(float(v), nd) for v in b]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2, choices=[2, 3])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--digits", type=int, default=4)
    args = ap.parse_args()

    tag = f"{args.size}x{args.size}"
    ckpt = Path(args.ckpt) if args.ckpt else HERE / "checkpoints" / f"best-{tag}.pt"
    out = Path(args.out) if args.out else HERE.parent / "web" / f"policy-{tag}.json"

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = ValueNet(ck["obs_size"], ck["hidden"], ck["blocks"])
    net.load_state_dict(ck["net"])
    net.eval()

    d = args.digits
    spec = {
        "arch": "value_resnet_folded",
        "size": args.size,
        "obs_size": ck["obs_size"],
        "hidden": ck["hidden"],
        "n_blocks": ck["blocks"],
        "trained_iters": ck.get("iter", 0),
        "target_updates": ck.get("updates", 0),
        "n_params": net.n_params(),
        "eval": ck.get("eval"),
        "fc1": dump(*fold_bn(net.fc1, net.bn1), d),
        "fc2": dump(*fold_bn(net.fc2, net.bn2), d),
        "blocks": [{"fc1": dump(*fold_bn(b.fc1, b.bn1), d),
                    "fc2": dump(*fold_bn(b.fc2, b.bn2), d)} for b in net.blocks],
        "head": dump(net.head.weight, net.head.bias, d),
    }

    # ── 摺完 + 四捨五入之後，跟原網路比對 ──
    cube = Cube(args.size)
    rng = np.random.default_rng(7)
    states, _ = cube.scramble(rng.integers(1, cube.cfg["scrambleMax"] + 1, size=512), rng)
    x = torch.from_numpy(cube.encode(states))
    with torch.no_grad():
        ref = net(x).numpy()
    got = forward_folded(spec, cube.encode(states))
    err = float(np.abs(ref - got).max())

    out.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
    mb = out.stat().st_size / 1e6
    print(f"寫出 {out}  ({mb:.2f} MB, {spec['n_params']:,} 參數, {d} 位小數)")
    print(f"  來源 {ckpt.name}：第 {spec['trained_iters']:,} 輪，target 推了 {spec['target_updates']} 版")
    print(f"  摺 BN + 四捨五入後的最大誤差 {err:.4f} 步" +
          ("  ✓" if err < 0.02 else "  ← 太大了，位數要再多一點"))
    if spec["eval"]:
        print(f"  這份權重的 eval: {spec['eval']}")
    if err >= 0.05:
        raise SystemExit("誤差太大，不要拿這份權重去跑網頁")


def forward_folded(spec, x: np.ndarray) -> np.ndarray:
    """用匯出後的數字自己算一次，確認 web/nn.js 該有的行為是對的。"""
    lin = lambda h, L: h @ np.array(L["w"], dtype=np.float32).T + np.array(L["b"], dtype=np.float32)
    relu = lambda h: np.maximum(h, 0)
    h = relu(lin(x, spec["fc1"]))
    h = relu(lin(h, spec["fc2"]))
    for b in spec["blocks"]:
        y = relu(lin(h, b["fc1"]))
        h = relu(h + lin(y, b["fc2"]))
    return lin(h, spec["head"])[:, 0]


if __name__ == "__main__":
    main()
