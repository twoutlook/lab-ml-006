"""把角塊的座標移動表匯出給瀏覽器 -> web/corner_tables.json。

    python ml/export_corner_tables.py

為什麼不直接搬那張 88 MB 的距離表？因為 artifact 的上限是 16 MB。
但其實不用搬——**在瀏覽器裡現算就好**：

角塊的上帝之數是 14，所以從現在這個角塊狀態往外走、
同時從解開狀態往外走，兩邊各走 7 步一定會相遇（雙向 BFS）。
7 步以內的角塊狀態有 1,053,180 個，兩邊加起來兩百萬出頭——
用 typed array 撐得住，而且不必事先知道任何一個距離。

要搬的只有推鄰居用的兩張表：

    perm 40,320 × 12    哪顆角在哪 -> 轉一步之後的編號
    ori   2,187 × 12    每個位置轉了幾度 -> 轉一步之後的編號

兩張都用 uint16 存（40,319 塞得進 uint16），base64 之後約 1.4 MB。
這是 ml/corners.py 那份「角塊的兩個座標各自獨立」的直接後果——
獨立，才有辦法用兩張小表推出全部 8,800 萬個狀態的鄰居。
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

import corners
from cube import Cube

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "web" / "corner_tables.json"


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def main():
    cube = Cube(3)
    perm_tbl, ori_tbl = corners.build_move_tables(cube)
    assert perm_tbl.max() < 65536 and ori_tbl.max() < 65536

    spec = {
        "nPerm": corners.N_PERM,
        "nOri": corners.N_ORI,
        "nStates": corners.N_STATES,
        "nActions": cube.n_actions,
        "moveNames": cube.move_names,
        "godNumber": int(np.load(HERE / "checkpoints" / "corners3.npy", mmap_mode="r").max())
        if (HERE / "checkpoints" / "corners3.npy").exists() else 14,
        # 貼紙 -> 角塊座標要用的東西
        "slots": corners.SLOTS.tolist(),                       # (8,3) 每個角塊位置的三片貼紙
        "cornerLut": corners._CORNER_LUT.astype(np.int8).tolist(),  # 三個顏色排序後 -> 是哪顆角
        "solvedIndex": int(corners.index_of(cube, cube.solved(1))[0]),
        "permTable": b64(perm_tbl.astype(np.uint16)),
        "oriTable": b64(ori_tbl.astype(np.uint16)),
    }
    OUT.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
    mb = OUT.stat().st_size / 1e6
    print(f"寫出 {OUT}  ({mb:.2f} MB)")
    print(f"  perm 表 {perm_tbl.shape}、ori 表 {ori_tbl.shape}、解開的編號 {spec['solvedIndex']:,}")
    print(f"  角塊上帝之數 {spec['godNumber']} -> 雙向 BFS 兩邊各走 {spec['godNumber'] // 2 + 1} 步一定相遇")


if __name__ == "__main__":
    main()
