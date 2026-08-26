// 在瀏覽器裡算 3×3×3 的**精確**角塊距離，不搬那張 88 MB 的表。
//
// 這一頁的主角是「只把角塊轉回去要幾步」——它是整顆方塊的下界，而且是精確值。
// 表本身有 88,179,840 筆，塞不進 artifact 的 16 MB 上限。但不必塞：
//
//   角塊的上帝之數是 14 -> 從現在的狀態往外走、同時從解開狀態往外走，
//   兩邊各走 7 步一定相遇。7 步以內的角塊狀態有 1,053,180 個，
//   兩邊加起來兩百萬出頭，typed array 撐得住。
//
// 要搬的只有推鄰居用的兩張座標移動表（約 1.4 MB，見 ml/export_corner_tables.py）。
// 能只搬這兩張，是因為角塊的「誰在哪」和「轉了幾度」在轉動下各自獨立。
//
// 造訪紀錄用一個 88 MB 的 Uint8Array，一個 byte 塞兩邊的深度（各一個 nibble，
// 存「深度 + 1」，所以 0 代表沒走過）。有了深度就不必另外存父節點——
// 回頭找路的時候，往深度小 1 的鄰居走就對了。

const NIB_F = 0x0f, NIB_B = 0xf0;

function unb64(s, Type) {
  const bin = atob(s);
  const buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new Type(buf.buffer);
}

export class CornerSolver {
  constructor(spec) {
    this.nOri = spec.nOri;
    this.nStates = spec.nStates;
    this.A = spec.nActions;
    this.moveNames = spec.moveNames;
    this.slots = spec.slots.flat();               // 8×3，攤平
    this.lut = Int8Array.from(spec.cornerLut);
    this.solved = spec.solvedIndex;
    this.godNumber = spec.godNumber;
    this.permT = unb64(spec.permTable, Uint16Array);   // (nPerm, A)
    this.oriT = unb64(spec.oriTable, Uint16Array);     // (nOri,  A)
    this.inv = Int32Array.from({ length: this.A }, (_, i) => i ^ 1);
    this.seen = null;
    this.half = Math.ceil(this.godNumber / 2);
  }

  static async load(url = "corner_tables.json") {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(`${url} 讀不到 (${r.status})`);
    return new CornerSolver(await r.json());
  }

  /** 貼紙陣列 -> 角塊狀態編號。跟 ml/corners.py 的 index_of 同一套編碼。 */
  index(state) {
    const perm = new Int32Array(8), ori = new Int32Array(8);
    for (let j = 0; j < 8; j++) {
      const a = state[this.slots[j * 3]], b = state[this.slots[j * 3 + 1]], c = state[this.slots[j * 3 + 2]];
      // 三個顏色排序後就是這顆角的身分證
      let x = a, y = b, z = c, t;
      if (x > y) { t = x; x = y; y = t; }
      if (y > z) { t = y; y = z; z = t; }
      if (x > y) { t = x; x = y; y = t; }
      perm[j] = this.lut[x * 36 + y * 6 + z];
      // 扭轉 = U/D 的顏色（0 或 1）落在三格裡的哪一格
      ori[j] = a <= 1 ? 0 : (b <= 1 ? 1 : 2);
    }
    let pr = 0;
    for (let i = 0; i < 8; i++) {
      let smaller = 0;
      for (let k = i + 1; k < 8; k++) if (perm[k] < perm[i]) smaller++;
      pr = pr * (8 - i) + smaller;
    }
    let or_ = 0;
    for (let i = 0; i < 7; i++) or_ = or_ * 3 + ori[i];
    return pr * this.nOri + or_;
  }

  step(idx, m) {
    const p = (idx / this.nOri) | 0, o = idx - p * this.nOri;
    return this.permT[p * this.A + m] * this.nOri + this.oriT[o * this.A + m];
  }

  /**
   * 雙向 BFS。回傳 { dist, seq }：角塊的精確最短步數，以及其中一條最短解。
   * onProgress({ seen }) 每展開一層呼叫一次。
   */
  solve(startIdx, onProgress) {
    // 已經解開就直接回，別為了這個配置 88 MB（頁面一載入就會問一次）
    if (startIdx === this.solved) return { dist: 0, seq: [], expanded: 0 };
    if (!this.seen) {
      // 88 MB 的造訪紀錄。桌機沒問題，但記憶體吃緊的裝置可能配置不出來——
      // 接住它，讓呼叫端顯示訊息，不要讓按鈕默默沒反應。
      try {
        this.seen = new Uint8Array(this.nStates);
      } catch (e) {
        this.oom = true;
        return null;
      }
    } else this.seen.fill(0);
    const seen = this.seen, A = this.A;

    seen[startIdx] = 1;                       // fwd 深度 0（存的是深度 + 1）
    seen[this.solved] |= 1 << 4;              // bwd 深度 0

    let fF = Int32Array.of(startIdx), fB = Int32Array.of(this.solved);
    let dF = 0, dB = 0, total = 2, meet = -1, meetF = 0, meetB = 0;

    const expand = (frontier, depth, fwd) => {
      const out = new Int32Array(Math.max(64, frontier.length * A));
      let n = 0;
      const mine = fwd ? NIB_F : NIB_B;
      const theirs = fwd ? NIB_B : NIB_F;
      const shift = fwd ? 0 : 4;
      for (let i = 0; i < frontier.length; i++) {
        const f = frontier[i];
        const p = (f / this.nOri) | 0, o = f - p * this.nOri;
        for (let m = 0; m < A; m++) {
          const c = this.permT[p * A + m] * this.nOri + this.oriT[o * A + m];
          const v = seen[c];
          if (v & mine) continue;                      // 自己這邊走過了
          seen[c] = v | ((depth + 2) << shift);   // 存的是「深度 + 1」，0 保留給「沒走過」
          out[n++] = c;
          if (v & theirs) {                            // 對面來過 -> 相遇
            if (meet < 0) {
              meet = c;
              meetF = fwd ? depth + 1 : ((v & NIB_F) - 1);
              meetB = fwd ? ((v >> 4) - 1) : depth + 1;
            }
          }
        }
      }
      total += n;
      return out.subarray(0, n);
    };

    while (meet < 0 && dF + dB < this.half * 2) {
      if (fF.length <= fB.length) { fF = expand(fF, dF, true); dF++; }
      else { fB = expand(fB, dB, false); dB++; }
      if (onProgress) onProgress({ seen: total });
      if (!fF.length && !fB.length) break;
    }
    if (meet < 0) return null;

    // 回頭找路：往「自己這邊深度小 1」的鄰居走，不必存父節點
    const head = [];
    let cur = meet;
    for (let k = meetF; k > 0; k--) {
      for (let m = 0; m < A; m++) {
        const pred = this.step(cur, this.inv[m]);
        if ((seen[pred] & NIB_F) === k) { head.push(m); cur = pred; break; }
      }
    }
    head.reverse();

    const tail = [];
    cur = meet;
    for (let j = meetB; j > 0; j--) {
      for (let m = 0; m < A; m++) {
        const pred = this.step(cur, this.inv[m]);
        if (((seen[pred] & NIB_B) >> 4) === j) { tail.push(this.inv[m]); cur = pred; break; }
      }
    }
    return { dist: head.length + tail.length, seq: head.concat(tail), expanded: total };
  }
}
