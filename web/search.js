// 批次加權 A*，瀏覽器版。跟 ml/search.py 是同一個演算法。
//
//   f = weight * g + h(s)
//
// 一次從佇列拿出 batch 個最好的節點，把它們的子節點湊成一批送進網路——
// 對 GPU 是為了餵滿，對純 JS 是為了讓那個三層迴圈跑在連續記憶體上。
//
// 每處理完一批就把控制權交還給瀏覽器（await yieldToUI），
// 否則頁面在搜尋的時候會整個凍住，動畫也停了。

const key = (s) => String.fromCharCode.apply(null, s);
const yieldToUI = () => new Promise((r) => setTimeout(r, 0));

// 最小堆。node = [f, g, key]
class Heap {
  constructor() { this.a = []; }
  get size() { return this.a.length; }
  push(v) {
    const a = this.a; a.push(v);
    let i = a.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (a[p][0] <= a[i][0]) break;
      [a[p], a[i]] = [a[i], a[p]]; i = p;
    }
  }
  pop() {
    const a = this.a, top = a[0], last = a.pop();
    if (a.length) {
      a[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < a.length && a[l][0] < a[m][0]) m = l;
        if (r < a.length && a[r][0] < a[m][0]) m = r;
        if (m === i) break;
        [a[m], a[i]] = [a[i], a[m]]; i = m;
      }
    }
    return top;
  }
}

/**
 * 解一個局面。回傳 { seq, expanded, ms } 或 { seq: null, ... }（放棄）。
 * onProgress({expanded, bestH}) 每一批呼叫一次，用來更新畫面上的節點計數。
 */
export async function solve(cube, net, state, opts = {}) {
  const weight = opts.weight ?? 0.6;
  const batch = opts.batch ?? 100;
  const maxNodes = opts.maxNodes ?? 60000;
  const onProgress = opts.onProgress;
  const t0 = performance.now();
  const S = cube.nStickers, A = cube.nActions;

  if (cube.isSolved(state)) return { seq: [], expanded: 0, ms: 0 };

  const g = new Map(), parent = new Map(), st = new Map();
  const k0 = key(state);
  g.set(k0, 0); parent.set(k0, null); st.set(k0, state);
  const heap = new Heap();
  heap.push([net.value(cube, state), 0, k0]);

  let expanded = 0, bestH = Infinity;

  while (heap.size && expanded < maxNodes) {
    // ── 抓一批最好的節點 ──
    const pop = [];
    const seenNow = new Set();
    while (heap.size && pop.length < batch) {
      const [, gg, k] = heap.pop();
      if (gg > (g.get(k) ?? Infinity) || seenNow.has(k)) continue;   // 舊的，跳過
      seenNow.add(k);
      pop.push([gg, k]);
    }
    if (!pop.length) break;
    expanded += pop.length;

    // ── 展開，湊成一批 ──
    const B = pop.length, M = B * A;
    const kids = new Uint8Array(M * S);
    for (let i = 0; i < B; i++) kids.set(cube.expand(st.get(pop[i][1])), i * A * S);

    // 有子節點已經解開 -> 串路徑回去
    for (let j = 0; j < M; j++) {
      const off = j * S;
      let ok = true;
      for (let t = 0; t < S; t++) if (kids[off + t] !== cube.goal[t]) { ok = false; break; }
      if (!ok) continue;
      const bi = (j / A) | 0, ai = j % A;
      const seq = [ai];
      let k = pop[bi][1];
      while (parent.get(k)) { const [pk, pm] = parent.get(k); seq.push(pm); k = pk; }
      return { seq: seq.reverse(), expanded, ms: performance.now() - t0 };
    }

    // ── 一次算完這批的 h ──
    const xs = new Float32Array(M * net.obsSize);
    for (let j = 0; j < M; j++) {
      const off = j * S;
      for (let t = 0; t < S; t++) xs[j * net.obsSize + t * 6 + kids[off + t]] = 1;
    }
    const h = net.forward(xs, M);

    for (let j = 0; j < M; j++) {
      const bi = (j / A) | 0;
      const gi = pop[bi][0] + 1;
      const child = kids.subarray(j * S, j * S + S);
      const k = key(child);
      if (gi >= (g.get(k) ?? Infinity)) continue;
      g.set(k, gi);
      parent.set(k, [pop[bi][1], j % A]);
      st.set(k, child.slice());
      if (h[j] < bestH) bestH = h[j];
      heap.push([weight * gi + h[j], gi, k]);
    }
    if (onProgress) onProgress({ expanded, bestH });
    await yieldToUI();
  }
  return { seq: null, expanded, ms: performance.now() - t0 };
}

/** 完全不搜尋：每一步挑 h 最小的子節點。用來對照「沒有搜尋會怎樣」。 */
export function greedyStep(cube, net, state) {
  const S = cube.nStickers, A = cube.nActions;
  const kids = cube.expand(state);
  const xs = new Float32Array(A * net.obsSize);
  for (let a = 0; a < A; a++) {
    for (let t = 0; t < S; t++) xs[a * net.obsSize + t * 6 + kids[a * S + t]] = 1;
  }
  const h = net.forward(xs, A);
  let best = 0, bv = Infinity;
  const vals = [];
  for (let a = 0; a < A; a++) {
    let solved = true;
    for (let t = 0; t < S; t++) if (kids[a * S + t] !== cube.goal[t]) { solved = false; break; }
    const v = solved ? -1 : h[a];
    vals.push(v);
    if (v < bv) { bv = v; best = a; }
  }
  return { move: best, values: vals };
}
