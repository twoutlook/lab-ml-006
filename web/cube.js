// 魔術方塊的規則引擎。不碰 DOM，也不畫任何東西。
//
// 跟 ml/cube.py 是同一套規則的兩份實作，但**置換表只有一份**：
// shared/moves.json 由 ml/gen_moves.py 從幾何算出來，兩邊都讀它。
// 這樣「JS 和 Python 會不會不一致」就只剩下「有沒有正確使用那張表」，
// 而那件事 _parity_test.mjs 會逐格比對。
//
// 狀態就是一個長度 S 的 Uint8Array，值 0~5 代表顏色（= 面的序號 U D F B L R）。

export const FACES = ["U", "D", "F", "B", "L", "R"];

// 每一面的法向量、以及從外面看這一面時的右方向與上方向。
// 跟 ml/gen_moves.py 的 BASIS 必須一模一樣——畫圖跟算轉動都靠它。
export const BASIS = {
  U: [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
  D: [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
  F: [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
  B: [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
  L: [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
  R: [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
};

export class Cube {
  constructor(spec) {
    this.size = spec.n;
    this.n = spec.n;
    this.moveNames = spec.moves;
    this.nActions = spec.moves.length;
    this.nStickers = spec.n_stickers;
    this.perms = spec.perms.map((p) => Int32Array.from(p));
    // 動作成對排列（U, U', R, R', …），所以反動作就是 XOR 1
    this.inverseMove = Int32Array.from({ length: this.nActions }, (_, i) => i ^ 1);
    this.goal = new Uint8Array(this.nStickers);
    for (let f = 0; f < 6; f++) this.goal.fill(f, f * this.n * this.n, (f + 1) * this.n * this.n);
    // 這個尺寸有哪些面真的會轉（2x2x2 固定了 DBL 那顆角，只有 U R F）
    this.turnFaces = spec.faces;
    this.geometry = buildGeometry(this.n);
  }

  static async load(url = "../shared/moves.json", size = 2) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`moves.json 讀不到 (${res.status})`);
    const all = await res.json();
    return new Cube(all[String(size)]);
  }

  solved() { return this.goal.slice(); }

  isSolved(s) {
    for (let i = 0; i < s.length; i++) if (s[i] !== this.goal[i]) return false;
    return true;
  }

  // new[j] = old[perm[j]]
  apply(s, m) {
    const p = this.perms[m], out = new Uint8Array(s.length);
    for (let j = 0; j < p.length; j++) out[j] = s[p[j]];
    return out;
  }

  applySeq(s, seq) {
    let cur = s;
    for (const m of seq) cur = this.apply(cur, m);
    return cur;
  }

  // 所有子節點，攤平成一個 (A*S) 的陣列，順序跟 moveNames 一樣
  expand(s) {
    const A = this.nActions, S = s.length;
    const out = new Uint8Array(A * S);
    for (let a = 0; a < A; a++) {
      const p = this.perms[a];
      for (let j = 0; j < S; j++) out[a * S + j] = s[p[j]];
    }
    return out;
  }

  // 打亂：避開上一步的反動作（連著轉 R 再 R' 等於沒轉）
  scramble(depth, rand = Math.random) {
    let s = this.solved();
    const seq = [];
    let last = -1;
    for (let t = 0; t < depth; t++) {
      let m;
      do { m = Math.floor(rand() * this.nActions); } while (m === this.inverseMove[last]);
      s = this.apply(s, m);
      seq.push(m);
      last = m;
    }
    return { state: s, seq };
  }

  // 網路的輸入：每片貼紙 one-hot 成 6 維
  encode(s, out = null) {
    const v = out || new Float32Array(this.nStickers * 6);
    v.fill(0);
    for (let i = 0; i < s.length; i++) v[i * 6 + s[i]] = 1;
    return v;
  }

  toStr(seq) { return seq.map((m) => this.moveNames[m]).join(" "); }
}

// ── 幾何：每片貼紙的四個角，用來畫 3D ──────────────────────────
//
// 座標跟 ml/gen_moves.py 一樣放大兩倍，所以一格寬 2、外表面在 ±n 上。
// 貼紙中心 p 的四個角就是 p ± right ± up。
function buildGeometry(n) {
  const stickers = [];
  for (let fi = 0; fi < 6; fi++) {
    const [normal, right, up] = BASIS[FACES[fi]];
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const du = (n - 1) - 2 * r, dr = 2 * c - (n - 1);
        const p = [0, 1, 2].map((k) => normal[k] * n + right[k] * dr + up[k] * du);
        const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) =>
          [0, 1, 2].map((k) => p[k] + right[k] * a + up[k] * b));
        stickers.push({ face: fi, faceName: FACES[fi], row: r, col: c, center: p, corners, normal });
      }
    }
  }
  return stickers;
}
