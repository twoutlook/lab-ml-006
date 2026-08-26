// 在瀏覽器裡跑 policy.json 的前向傳播。純 JS，不用 TensorFlow.js。
//
// 網路只輸出一個數字：這個局面離解開還有幾步。
//
// BatchNorm 在匯出的時候已經被摺進前一層的線性層了（見 ml/model.py 的 fold_bn），
// 所以這裡只需要三件事：矩陣乘法、ReLU、殘差相加。少一份會不一致的東西。
//
// 權重載進來之後轉成攤平的 Float32Array。巢狀陣列在 JS 裡每一列都是一次
// 指標追蹤，攤平之後純粹是連續記憶體掃描，搜尋時快好幾倍——
// 而搜尋一次要算幾千個局面，這個差別是「秒」等級的。

function flat(layer) {
  const rows = layer.w.length, cols = layer.w[0].length;
  const w = new Float32Array(rows * cols);
  for (let o = 0; o < rows; o++) w.set(layer.w[o], o * cols);
  return { w, b: Float32Array.from(layer.b), rows, cols };
}

// y = W x + b，一次算 n 個樣本。x 是 (n, cols)，y 是 (n, rows)。
function linear(x, n, L, y) {
  const { w, b, rows, cols } = L;
  for (let s = 0; s < n; s++) {
    const xo = s * cols, yo = s * rows;
    for (let o = 0; o < rows; o++) {
      const wo = o * cols;
      let acc = b[o];
      for (let i = 0; i < cols; i++) acc += w[wo + i] * x[xo + i];
      y[yo + o] = acc;
    }
  }
}

const relu = (a, n) => { for (let i = 0; i < n; i++) if (a[i] < 0) a[i] = 0; };

export class ValueNet {
  constructor(spec) {
    this.spec = spec;
    this.obsSize = spec.obs_size;
    this.hidden = spec.hidden;
    this.size = spec.size;
    this.trainedIters = spec.trained_iters || 0;
    this.nParams = spec.n_params || 0;
    this.eval = spec.eval || null;
    this.fc1 = flat(spec.fc1);
    this.fc2 = flat(spec.fc2);
    this.blocks = spec.blocks.map((b) => ({ fc1: flat(b.fc1), fc2: flat(b.fc2) }));
    this.head = flat(spec.head);
    this._cap = 0;
  }

  static async load(url) {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(`${url} 讀不到 (${res.status})`);
    return new ValueNet(await res.json());
  }

  _alloc(n) {
    if (n <= this._cap) return;
    this._cap = n;
    this.bufA = new Float32Array(n * this.hidden * 2);
    this.bufB = new Float32Array(n * this.hidden);
    this.bufC = new Float32Array(n * this.hidden);
    this.bufOut = new Float32Array(n);
  }

  /** x 是攤平的 (n, obsSize) one-hot。回傳長度 n 的 Float32Array：每個局面還要幾步。 */
  forward(x, n) {
    this._alloc(n);
    const H = this.hidden, a = this.bufA, b = this.bufB, c = this.bufC;
    linear(x, n, this.fc1, a); relu(a, n * H * 2);
    linear(a, n, this.fc2, b); relu(b, n * H);
    for (const blk of this.blocks) {
      linear(b, n, blk.fc1, c); relu(c, n * H);
      linear(c, n, blk.fc2, a);              // a 前 n*H 格當暫存
      for (let i = 0; i < n * H; i++) { const v = b[i] + a[i]; b[i] = v > 0 ? v : 0; }
    }
    linear(b, n, this.head, this.bufOut);
    return this.bufOut.subarray(0, n);
  }

  /** 方便版：算單一局面。 */
  value(cube, state) {
    this._x1 = this._x1 || new Float32Array(this.obsSize);
    cube.encode(state, this._x1);
    return this.forward(this._x1, 1)[0];
  }
}
