// 2x2x2 的**精確**最短解，在瀏覽器裡即時算出來。
//
// 這一段跟機器學習沒有關係，但它是整個頁面最重要的東西：
// 有了它，「網路猜幾步」旁邊才擺得出「其實是幾步」。
//
// 作法是雙向廣度優先：從現在這個局面往外走，同時從解開狀態往外走，
// 兩邊撞在一起就是答案。為什麼一定撞得到？因為 2x2x2 的上帝之數是 14——
// 任何局面最多 14 步能解開，所以兩邊各走 7 步一定會相遇。
//
// 各走 7 步要看幾個局面？7 步以內總共 44,971 個。兩邊加起來不到十萬，
// 純 JS 的 Map 撐得住，通常 100 毫秒內就回來了。
// 單向走 14 步要看 3,674,160 個——那就撐不住了。這就是雙向的意義：
// 指數的一半，不是一半的時間，是開根號。
//
// 3x3x3 沒有這個選項：上帝之數 26（四分之一轉），各走 13 步是天文數字。
// 這正是為什麼那邊只能靠學出來的 heuristic。

// 名字跟 search.js 錯開：build_artifact.mjs 會把兩個檔案串在一起。
const ekey = (s) => String.fromCharCode.apply(null, s);

/**
 * 回傳 { dist, seq }：最短步數，以及其中一條最短解。
 * 走完兩邊各 maxHalf 步還沒撞到就回 null（代表引擎有問題，不該發生）。
 */
export function exactSolve(cube, state, maxHalf = 7) {
  if (cube.isSolved(state)) return { dist: 0, seq: [] };

  // 兩張表都存「怎麼走到這裡」：[上一個局面的 key, 用了哪個動作]
  const F = new Map([[ekey(state), [null, -1, 0]]]);
  const B = new Map([[ekey(cube.goal), [null, -1, 0]]]);
  let fF = [[ekey(state), state]], bF = [[ekey(cube.goal), cube.goal]];
  let dF = 0, dB = 0;

  const step = (frontier, table, depth) => {
    const next = [];
    for (const [pk, s] of frontier) {
      for (let a = 0; a < cube.nActions; a++) {
        const c = cube.apply(s, a);
        const k = ekey(c);
        if (table.has(k)) continue;
        table.set(k, [pk, a, depth]);
        next.push([k, c]);
      }
    }
    return next;
  };

  // 兩邊表裡有沒有共同的局面？有的話那個就是會合點。
  const findMeet = () => {
    const [small, big] = F.size <= B.size ? [F, B] : [B, F];
    let bestK = null, best = Infinity;
    for (const [k, v] of small) {
      const w = big.get(k);
      if (w === undefined) continue;
      const t = v[2] + w[2];
      if (t < best) { best = t; bestK = k; }
    }
    return bestK;
  };

  let meetK = findMeet();
  while (meetK === null && dF + dB < maxHalf * 2) {
    if (fF.length <= bF.length && dF < maxHalf) { dF++; fF = step(fF, F, dF); }
    else if (dB < maxHalf) { dB++; bF = step(bF, B, dB); }
    else if (dF < maxHalf) { dF++; fF = step(fF, F, dF); }
    else break;
    if (!fF.length && !bF.length) break;
    meetK = findMeet();
  }
  if (meetK === null) return null;

  // 起點 -> 會合點：照 F 的紀錄倒著串回去
  const head = [];
  for (let k = meetK; F.get(k)[0] !== null; ) {
    const [pk, a] = F.get(k);
    head.push(a);
    k = pk;
  }
  head.reverse();
  // 會合點 -> 解開：B 記的是「從解開走到這裡」，所以要反過來轉
  const tail = [];
  for (let k = meetK; B.get(k)[0] !== null; ) {
    const [pk, a] = B.get(k);
    tail.push(cube.inverseMove[a]);
    k = pk;
  }
  const seq = head.concat(tail);
  return { dist: seq.length, seq };
}

export const exactDistance = (cube, state, maxHalf = 7) => {
  const r = exactSolve(cube, state, maxHalf);
  return r ? r.dist : null;
};
