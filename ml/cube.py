"""魔術方塊的 numpy 引擎。整個檔案沒有 for 迴圈跑單一方塊——全部是批次操作。

狀態就是一個貼紙陣列：`(N, S)` 的 uint8，值是 0~5 的顏色（= 面的序號）。
S 是貼紙數，2x2x2 有 24 片，3x3x3 有 54 片。

轉動就是一次 gather：`new[j] = old[perm[j]]`。
置換表由 ml/gen_moves.py 從幾何算出來，存在 shared/moves.json，
JS 那邊（web/cube.js）讀的是同一份檔案。

為什麼要批次？DAVI 每一輪要對兩萬個狀態各展開所有子節點——
2x2x2 是 12 萬次轉動，3x3x3 是 24 萬次。用 Python 迴圈做這件事，
光是產資料就會比訓練本身還慢。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MOVES_JSON = ROOT / "shared" / "moves.json"
CONFIG_JSON = ROOT / "shared" / "config.json"

CONFIG = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
_MOVES = json.loads(MOVES_JSON.read_text(encoding="utf-8"))

N_COLORS = 6


class Cube:
    """一個尺寸的方塊。所有方法都吃 (N, S) 的批次，不吃單顆。"""

    def __init__(self, size: int = 2):
        key = str(size)
        if key not in _MOVES:
            raise ValueError(f"shared/moves.json 沒有 {size}x{size}x{size}")
        d = _MOVES[key]
        self.size = size
        self.n = d["n"]
        self.move_names: list[str] = d["moves"]
        self.perms = np.array(d["perms"], dtype=np.int64)      # (A, S)
        self.n_actions, self.n_stickers = self.perms.shape
        self.cfg = CONFIG["sizes"][key]
        self.obs_size = self.n_stickers * N_COLORS

        # 解開的樣子：第 f 面的每一片都是顏色 f。
        self.goal = np.repeat(np.arange(6, dtype=np.uint8), self.n * self.n)

        # 每個動作的反動作。動作是成對排的（U, U', R, R', ...），所以 XOR 1 就是反的。
        self.inverse_move = np.arange(self.n_actions, dtype=np.int64) ^ 1
        assert all(self.move_names[i ^ 1] == self._flip(self.move_names[i])
                   for i in range(self.n_actions)), "動作順序不是成對的"

        # 哪些貼紙會動。2x2x2 固定了 DBL 那顆角，它的三片永遠不動。
        moved = np.zeros(self.n_stickers, dtype=bool)
        for a in range(self.n_actions):
            moved |= self.perms[a] != np.arange(self.n_stickers)
        self.movable = np.flatnonzero(moved)
        self.fixed = np.flatnonzero(~moved)

        # 在動作表最後補一個「什麼都不做」的置換。
        # 打亂的時候每一列要轉的步數不一樣，有了它就可以讓短的那些列
        # 去做恆等變換，整批一起 gather——不必每一步都 flatnonzero 挑出還在動的列。
        # 這一招把 3x3x3 打亂 20000 個局面的時間從 84 毫秒壓到 40 毫秒左右。
        self.perms_ext = np.vstack([self.perms, np.arange(self.n_stickers, dtype=np.int64)])
        self.NOOP = self.n_actions

    @staticmethod
    def _flip(name: str) -> str:
        return name[:-1] if name.endswith("'") else name + "'"

    # ── 基本操作 ────────────────────────────────────────────────
    def solved(self, n: int = 1) -> np.ndarray:
        return np.tile(self.goal, (n, 1))

    def is_solved(self, states: np.ndarray) -> np.ndarray:
        return (states == self.goal).all(axis=1)

    def apply(self, states: np.ndarray, moves: np.ndarray) -> np.ndarray:
        """每一列各做各的動作。states (N,S)、moves (N,) -> (N,S)。"""
        return np.take_along_axis(states, self.perms[moves], axis=1)

    def expand(self, states: np.ndarray) -> np.ndarray:
        """把每個狀態的所有子節點都展開。(N,S) -> (N, A, S)。"""
        return states[:, self.perms]

    # ── 打亂 ────────────────────────────────────────────────────
    def scramble(self, depths: np.ndarray, rng: np.random.Generator):
        """從解開狀態往回亂轉。depths (N,) 是每一列要轉幾步。

        會避開「上一步的反動作」——連著轉 R 再 R' 等於沒轉，
        那種樣本會讓「打亂 d 步」跟「真實距離」差更多。
        （但也僅止於此：轉 d 步不保證真實距離就是 d，
        例如 R R R 其實只離 1 步。這件事在評估的時候要記得。）
        """
        n = len(depths)
        maxd = int(depths.max()) if n else 0
        A = self.n_actions

        # 先一次把整張動作表抽出來（N × maxd），再一步一步套上去。
        # 避開反動作的寫法：從 A-1 個選項裡抽，抽到的號碼 >= 禁止的那個就 +1，
        # 這樣剩下的 A-1 個動作機率完全均勻，不用重抽也不用迴圈。
        seqs = np.empty((n, maxd), dtype=np.int64)
        if maxd:
            seqs[:, 0] = rng.integers(0, A, size=n)
            for t in range(1, maxd):
                ban = self.inverse_move[seqs[:, t - 1]]
                r = rng.integers(0, A - 1, size=n)
                seqs[:, t] = r + (r >= ban)
            # 步數不足的那些列，後面補恆等變換
            seqs[np.arange(maxd)[None, :] >= depths[:, None]] = self.NOOP

        states = self.solved(n)
        for t in range(maxd):
            states = np.take_along_axis(states, self.perms_ext[seqs[:, t]], axis=1)
        seqs = np.where(seqs == self.NOOP, -1, seqs)
        return states, seqs

    def scramble_walk(self, n: int, maxd: int, rng: np.random.Generator):
        """訓練用的取樣：走幾條隨機軌跡，把沿路每一站都收下來。

        回傳 (states (n,S), depths (n,))，depths 大致均勻分布在 1..maxd。

        為什麼不用 scramble()？因為那個是「每一列各自從解開狀態走到自己的深度」，
        總共要做 n × maxd 次 gather。3x3x3 一輪就是 60 萬次，佔掉整輪一半的時間。

        改成走 n/maxd 條軌跡、每條走 maxd 步、沿路每一站都當一個樣本，
        gather 次數就只剩 n 次——少 maxd 倍。同一條軌跡上的樣本彼此相關，
        但 DAVI 的 target 是每個狀態各自展開算的，不像 DQN 那樣沿著軌跡回傳，
        所以相關性不會偏掉任何東西。DeepCubeA 原本也是這樣取樣的。
        """
        n_traj = max(1, -(-n // maxd))                 # 無條件進位
        cur = self.solved(n_traj)
        out = np.empty((n_traj * maxd, self.n_stickers), dtype=np.uint8)
        depths = np.empty(n_traj * maxd, dtype=np.int64)
        last = np.full(n_traj, -1, dtype=np.int64)
        A = self.n_actions
        for t in range(maxd):
            if t == 0:
                m = rng.integers(0, A, size=n_traj)
            else:
                ban = self.inverse_move[last]
                r = rng.integers(0, A - 1, size=n_traj)
                m = r + (r >= ban)
            cur = np.take_along_axis(cur, self.perms[m], axis=1)
            last = m
            out[t * n_traj:(t + 1) * n_traj] = cur
            depths[t * n_traj:(t + 1) * n_traj] = t + 1
        if len(out) > n:                                # 只取需要的那些，順序打散
            pick = rng.choice(len(out), size=n, replace=False)
            return out[pick], depths[pick]
        return out, depths

    # ── 給網路吃的編碼 ──────────────────────────────────────────
    def encode(self, states: np.ndarray) -> np.ndarray:
        """每片貼紙 one-hot 成 6 維 -> (N, S*6) 的 float32。

        為什麼不直接餵 0~5 的顏色編號？因為那會暗示「顏色 5 比顏色 1 大」，
        但顏色之間沒有大小關係。one-hot 把這個假的順序拿掉。
        """
        n = states.shape[0]
        out = np.zeros((n, self.n_stickers, N_COLORS), dtype=np.float32)
        np.put_along_axis(out, states[:, :, None].astype(np.int64), 1.0, axis=2)
        return out.reshape(n, self.obs_size)

    # ── 2x2x2 專用：把狀態壓成一個 uint64 ──────────────────────
    def pack(self, states: np.ndarray) -> np.ndarray:
        """21 片會動的貼紙 × 3 bits = 63 bits，剛好塞得進 uint64。

        全狀態 BFS 需要一個「這個局面看過沒有」的鍵。用 uint64 的話
        整張表就是一個排好序的陣列，np.searchsorted 一次查幾百萬筆。
        3x3x3 的 51 片 × 3 = 153 bits 塞不下，所以這個方法只給 2x2x2 用。
        """
        if len(self.movable) * 3 > 63:
            raise ValueError(f"{self.size}x{self.size}x{self.size} 有 {len(self.movable)} 片會動的貼紙，壓不進 uint64")
        v = states[:, self.movable].astype(np.uint64)
        key = np.zeros(states.shape[0], dtype=np.uint64)
        for i in range(len(self.movable)):
            key |= v[:, i] << np.uint64(3 * i)
        return key

    def unpack(self, keys: np.ndarray) -> np.ndarray:
        states = np.tile(self.goal, (len(keys), 1))
        for i, s in enumerate(self.movable):
            states[:, s] = ((keys >> np.uint64(3 * i)) & np.uint64(7)).astype(np.uint8)
        return states

    # ── 好讀的表示法 ────────────────────────────────────────────
    def moves_to_str(self, seq) -> str:
        return " ".join(self.move_names[int(m)] for m in seq if int(m) >= 0)

    def parse(self, text: str) -> np.ndarray:
        idx = {m: i for i, m in enumerate(self.move_names)}
        out = []
        for tok in text.replace("’", "'").split():
            if tok.endswith("2"):          # R2 = R R
                out += [idx[tok[:-1]]] * 2
            else:
                out.append(idx[tok])
        return np.array(out, dtype=np.int64)


if __name__ == "__main__":
    for size in (2, 3):
        c = Cube(size)
        rng = np.random.default_rng(0)
        s, seq = c.scramble(np.full(5, 10), rng)
        # 照著反過來轉回去，一定要回到解開
        back = s.copy()
        for t in range(seq.shape[1] - 1, -1, -1):
            back = c.apply(back, c.inverse_move[seq[:, t]])
        ok = c.is_solved(back).all()
        print(f"{size}x{size}x{size}: {c.n_stickers} 貼紙 / {c.n_actions} 動作 / "
              f"obs {c.obs_size} 維 / 會動的貼紙 {len(c.movable)} 片 / 反轉回原狀 {ok}")
        print(f"   範例打亂: {c.moves_to_str(seq[0])}")
        assert ok
        if size == 2:
            k = c.pack(s)
            assert (c.unpack(k) == s).all(), "pack/unpack 對不起來"
            print(f"   pack 檢查通過，key 範例 {k[0]}")
