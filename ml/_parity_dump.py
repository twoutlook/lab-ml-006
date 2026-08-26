"""產生跨語言對帳用的題目：web/_parity_cases.json。

    python ml/_parity_dump.py --size 2

裡面有兩種題目：

1. **轉動**：一串動作，以及 Python 這邊轉出來的貼紙陣列。
   JS 照著轉，結果必須逐格相同。

2. **推論**：同樣那些局面，Python 拿**匯出後的那份 policy.json** 算出來的數字。
   JS 讀同一份權重，必須在 1e-5 以內。

   基準特地取「摺完 BatchNorm、四捨五入之後」的版本，不是原始的 PyTorch 網路。
   原因是那兩件事本來就會造成約 0.006 步的誤差（export_policy.py 會印出來），
   拿原始網路當基準的話，這個對帳量到的是捨入誤差，而不是
   「web/nn.js 的層序或殘差有沒有接錯」——後者才是這裡要抓的東西。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cube import Cube
from model import ValueNet

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2, choices=[2, 3])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=99)
    a = ap.parse_args()

    cube = Cube(a.size)
    rng = np.random.default_rng(a.seed)
    depths = rng.integers(1, cube.cfg["scrambleMax"] + 1, size=a.n)
    states, seqs = cube.scramble(depths, rng)

    cases = [{"moves": [int(m) for m in seqs[i] if m >= 0],
              "state": [int(v) for v in states[i]]} for i in range(a.n)]

    out = {"size": a.size, "move_names": cube.move_names, "cases": cases}

    pol = HERE.parent / "web" / f"policy-{a.size}x{a.size}.json"
    if pol.exists():
        from export_policy import forward_folded
        spec = json.loads(pol.read_text(encoding="utf-8"))
        h = forward_folded(spec, cube.encode(states))
        out["values"] = [round(float(v), 8) for v in h]
        print(f"含 {a.n} 筆推論答案（來自 {pol.name}，摺完 BN、四捨五入之後的權重）")

        ck = HERE / "checkpoints" / f"best-{a.size}x{a.size}.pt"
        if ck.exists():                      # 順帶報一次「摺 + 捨入」本身帶來多少誤差
            c = torch.load(ck, map_location="cpu", weights_only=False)
            net = ValueNet(c["obs_size"], c["hidden"], c["blocks"])
            net.load_state_dict(c["net"]); net.eval()
            with torch.no_grad():
                raw = net(torch.from_numpy(cube.encode(states))).numpy()
            print(f"  （對照：摺 BN + 四捨五入本身的最大誤差 {abs(raw - h).max():.4f} 步）")
    else:
        print(f"（{pol.name} 還沒有，這次只對帳轉動）")

    dst = HERE.parent / "web" / f"_parity_cases-{a.size}x{a.size}.json"
    dst.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"寫出 {dst}（{a.n} 題，打亂 {depths.min()}~{depths.max()} 步）")


if __name__ == "__main__":
    main()
