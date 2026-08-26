// 畫方塊。Canvas 2D，沒有任何 3D 函式庫。
//
// 每片貼紙在 cube.js 的 geometry 裡已經有四個 3D 角點了，
// 這裡做的事只有三件：轉動 -> 投影到 2D -> 由遠到近畫（畫家演算法）。
//
// 轉動一面的動畫用 Rodrigues 公式，繞著那一面的法向量轉：
//
//     p' = p cosθ + (a × p) sinθ + a (a·p)(1 - cosθ)
//
// 只有那一層的貼紙要套，其他的原地不動。因為每一格都是獨立的四邊形，
// 轉到一半的樣子會自動長對——不需要為「轉動中」另外寫一套畫法。

const rot = (p, a, th) => {
  const c = Math.cos(th), s = Math.sin(th);
  const d = a[0] * p[0] + a[1] * p[1] + a[2] * p[2];
  const cr = [a[1] * p[2] - a[2] * p[1], a[2] * p[0] - a[0] * p[2], a[0] * p[1] - a[1] * p[0]];
  return [0, 1, 2].map((k) => p[k] * c + cr[k] * s + a[k] * d * (1 - c));
};

// 視角：先繞 Y 轉 yaw，再繞 X 轉 pitch。正交投影，深度就是轉完的 z。
function project(p, yaw, pitch) {
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const x1 = p[0] * cy + p[2] * sy, z1 = -p[0] * sy + p[2] * cy;
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const y2 = p[1] * cp - z1 * sp, z2 = p[1] * sp + z1 * cp;
  return [x1, -y2, z2];
}

function roundQuad(ctx, pts, inset) {
  // 往形心縮一點，縮出來的空隙就是貼紙之間的黑縫
  const cx = (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4;
  const cy = (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4;
  ctx.beginPath();
  pts.forEach(([x, y], i) => {
    const px = x + (cx - x) * inset, py = y + (cy - y) * inset;
    i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
  });
  ctx.closePath();
}

export class CubeView {
  constructor(cube, colors) {
    this.cube = cube;
    this.colors = colors;
    this.yaw = -0.62;      // 約 -35 度，看得到 U / F / R 三面
    this.pitch = 0.48;     // 約 27 度
  }

  /**
   * ctx        canvas 2d context
   * state      Uint8Array，貼紙顏色
   * opts.cx/cy 畫在哪
   * opts.scale 一格多大（像素）
   * opts.anim  轉動中的話 { face: "U", angle: 弧度 }（順時針是負的）
   * opts.dim   要淡掉的貼紙 index Set（用來強調某幾片）
   */
  /** 方塊的黑色本體。轉動中那一層會離開本體，只畫貼紙的話會從縫隙看穿到背景，
   *  看起來像有幾片飛出去了。真的方塊裡面是黑色的塑膠核心，補上就自然了。
   *  作法：把一個略小的立方體八個角投影出來，取凸包（投影後是六邊形）填滿。 */
  drawCore(ctx, cx, cy, scale) {
    const n = this.cube.n * 0.97;
    const pts = [];
    for (const sx of [-1, 1]) for (const sy of [-1, 1]) for (const sz of [-1, 1]) {
      const [X, Y] = project([sx * n, sy * n, sz * n], this.yaw, this.pitch);
      pts.push([cx + X * scale, cy + Y * scale]);
    }
    const gx = pts.reduce((a, p) => a + p[0], 0) / 8;
    const gy = pts.reduce((a, p) => a + p[1], 0) / 8;
    pts.sort((a, b) => Math.atan2(a[1] - gy, a[0] - gx) - Math.atan2(b[1] - gy, b[0] - gx));
    ctx.fillStyle = "#0c0e12";
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.fill();
  }

  draw(ctx, state, opts) {
    const { cx, cy, scale } = opts;
    if (opts.core !== false) this.drawCore(ctx, cx, cy, scale);
    const anim = opts.anim || null;
    const n = this.cube.n;
    const BASIS = this.cube.geometry;
    let axis = null, layerMin = 0;
    if (anim) {
      const b = { U: [0, 1, 0], D: [0, -1, 0], F: [0, 0, 1], B: [0, 0, -1], L: [-1, 0, 0], R: [1, 0, 0] }[anim.face];
      axis = b;
      layerMin = n - 1;
    }

    const quads = [];
    for (let i = 0; i < BASIS.length; i++) {
      const st = BASIS[i];
      let corners = st.corners;
      if (axis) {
        const d = axis[0] * st.center[0] + axis[1] * st.center[1] + axis[2] * st.center[2];
        if (d >= layerMin) corners = corners.map((p) => rot(p, axis, anim.angle));
      }
      const pr = corners.map((p) => project(p, this.yaw, this.pitch));
      // 背面剔除：貼紙的四個角是「從外面看逆時針」排的，
      // 投影到螢幕座標（y 軸朝下）之後就變成順時針，鞋帶公式是負的。
      // 所以「面積 >= 0」代表這片正背對鏡頭，可以不用畫。
      let area = 0;
      for (let k = 0; k < 4; k++) {
        const a = pr[k], b = pr[(k + 1) % 4];
        area += a[0] * b[1] - b[0] * a[1];
      }
      if (area >= 0) continue;
      const depth = (pr[0][2] + pr[1][2] + pr[2][2] + pr[3][2]) / 4;
      quads.push({ i, depth, pts: pr.map(([x, y]) => [cx + x * scale, cy + y * scale]) });
    }
    quads.sort((a, b) => a.depth - b.depth);   // 遠的先畫

    for (const q of quads) {
      const dim = opts.dim && opts.dim.has(q.i);
      ctx.fillStyle = "#0c0e12";
      roundQuad(ctx, q.pts, 0);
      ctx.fill();
      ctx.fillStyle = this.colors[state[q.i]];
      ctx.globalAlpha = dim ? 0.18 : 1;
      roundQuad(ctx, q.pts, 0.12);
      ctx.fill();
      ctx.globalAlpha = 1;
    }
  }
}

// ── 攤平圖：六個面全部看得到，用來確認背面 ────────────────────
//
//        U
//      L F R B
//        D
const NET_POS = { U: [1, 0], L: [0, 1], F: [1, 1], R: [2, 1], B: [3, 1], D: [1, 2] };
// 跟 cube.js 的 FACES 同一份順序。這裡另外取名是因為 tools/build_artifact.mjs
// 會把所有模組串成單一檔案，同名的 const 會重複宣告。
const NET_ORDER = ["U", "D", "F", "B", "L", "R"];

export function drawNet(ctx, cube, state, colors, x0, y0, cell, gap = 2, plate = "#0c0e12") {
  const n = cube.n, fw = n * cell + gap;
  // 底板。白色那一面在淺色背景上會整片消失，墊一層深色才看得出來有六個面。
  if (plate) {
    ctx.fillStyle = plate;
    ctx.beginPath();
    ctx.roundRect(x0 - 6, y0 - 6, 4 * fw + 12 - gap, 3 * fw + 12 - gap, 5);
    ctx.fill();
  }
  for (let fi = 0; fi < 6; fi++) {
    const [gx, gy] = NET_POS[NET_ORDER[fi]];
    for (let r = 0; r < n; r++) {
      for (let c = 0; c < n; c++) {
        const i = fi * n * n + r * n + c;
        ctx.fillStyle = colors[state[i]];
        ctx.fillRect(x0 + gx * fw + c * cell, y0 + gy * fw + r * cell,
                     cell - gap, cell - gap);
      }
    }
  }
}

export const netSize = (cube, cell, gap = 2) => ({
  w: 4 * (cube.n * cell + gap),
  h: 3 * (cube.n * cell + gap),
});
