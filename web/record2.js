// GOAL002 影片的畫面產生器。1920×1080，逐格算，不是螢幕錄影。
//
// 跟第一支（record.js）同一套骨架：__rec.init() 回分鏡表，__rec.renderFrame() 畫一格。
// 差別在內容——這支講的是「先解角」，所以多了兩個東西：
//   ・可以把非角塊的貼紙淡掉（CubeView 的 dim 參數），一眼看出角塊歸位了沒
//   ・分階段解法的動畫：先解角（邊塊還亂著）→ 再解完
//
// 所有數字來自 web/demo2.json（ml/make_demo2.py 產生），沒有一個寫死在這裡。

import { Cube } from "./cube.js";
import { CubeView, drawNet } from "./render.js";

const W = 1920, H = 1080, FPS = 30;
const PAD_TAIL = 12;

const cv = document.getElementById("stage");
const ctx = cv.getContext("2d");

const [CFG, SCRIPT, TIMING, D, MOVES] = await Promise.all([
  fetch("../shared/config.json").then((r) => r.json()),
  fetch("../tools/script2.json").then((r) => r.json()),
  fetch("../out/voice2/timing.json").then((r) => r.json()).catch(() => null),
  fetch("demo2.json").then((r) => r.json()),
  fetch("../shared/moves.json").then((r) => r.json()),
]);

const COLORS = CFG.view.colors;
const CUBE = new Cube(MOVES["3"]);
const VIEW = new CubeView(CUBE, COLORS);
const CORNER = new Set(D.cornerStickers);
const NOT_CORNER = new Set([...Array(54).keys()].filter((i) => !CORNER.has(i)));

const C = {
  bg: "#06080f", fg: "#e6edf7", dim: "#7c8ba4", line: "#1b2436",
  accent: "#38bdf8", good: "#4ade80", warn: "#fbbf24", bad: "#f87171", grey: "#8fa0b8",
};

const fmt = (n) => n.toLocaleString("en-US");
const clamp01 = (t) => Math.max(0, Math.min(1, t));
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - 2 * (1 - t) * (1 - t));

function text(s, x, y, o = {}) {
  const { size = 22, color = C.fg, weight = 400, align = "left", font = "sans" } = o;
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.font = `${weight} ${size}px ${font === "mono" ? "ui-monospace, Consolas, monospace"
    : '"Noto Sans TC", "Microsoft JhengHei", system-ui, sans-serif'}`;
  ctx.fillText(s, x, y);
}
function box(x, y, w, h, fill = "#0d1320") {
  ctx.fillStyle = fill; ctx.strokeStyle = C.line; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.roundRect(x, y, w, h, 12); ctx.fill(); ctx.stroke();
}

// ── 分鏡表 ──────────────────────────────────────────────────
function buildPlan() {
  const byId = Object.fromEntries((TIMING || []).map((t) => [t.id, t.duration]));
  const scenes = [];
  let start = 0;
  for (const sc of SCRIPT.scenes) {
    const dur = byId[sc.id];
    const frames = Math.max(90, Math.round((dur || 20) * FPS) + PAD_TAIL);
    const s = { ...sc, start, frames, voice: dur || null };
    if (sc.kind === "staged" || sc.kind === "corners") Object.assign(s, prep(sc, frames));
    scenes.push(s);
    start += frames;
  }
  return { fps: FPS, totalFrames: start, durationSec: +(start / FPS).toFixed(2), scenes };
}

function prep(sc, frames) {
  const usable = frames - 40;
  if (sc.kind === "corners") {
    const seq = D.scramble;
    return { seq, totalSteps: seq.length, stepsPerFrame: seq.length / (usable * 0.45) };
  }
  const s = D.staged;
  const seq = D.scramble.concat(s.stage1, s.stage2);
  return { seq, totalSteps: seq.length, stepsPerFrame: seq.length / usable,
           scrambleLen: D.scramble.length, cornerAt: D.scramble.length + s.stage1.length,
           trace: s.trace };
}

let PLAN = null, frame = 0;

function chrome(sc, local) {
  ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H);
  text("魔術方塊 × 先解角", 60, 58, { size: 24, color: C.dim, weight: 600 });
  text("lab-ml-006 · GOAL 002", 60, 84, { size: 15, color: "#4a5568", font: "mono" });
  text(sc.chapter || "", 1860, 62, { size: 30, color: C.fg, weight: 600, align: "right" });
  ctx.strokeStyle = C.line; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(60, 104); ctx.lineTo(1860, 104); ctx.stroke();
  const p = (sc.start + local) / PLAN.totalFrames;
  ctx.fillStyle = "#131b2b"; ctx.fillRect(60, 1042, 1800, 3);
  ctx.fillStyle = C.accent; ctx.fillRect(60, 1042, 1800 * p, 3);
}

const opts_cx = () => 560;

function drawCube(state, opts = {}) {
  VIEW.yaw = -0.62 + (opts.local || 0) * 0.0022;
  VIEW.draw(ctx, state, {
    cx: opts.cx ?? 560, cy: opts.cy ?? 470, scale: opts.scale ?? 67,
    anim: opts.anim || null, dim: opts.dim || null,
  });
  if (opts.net !== false) {
    const cell = 22, fw = CUBE.n * cell + 3;
    drawNet(ctx, CUBE, state, COLORS, (opts.cx ?? 560) - 2 * fw, 812, cell, 3, "#0a0d15");
  }
}

function playTo(seq, step) {
  let s = CUBE.solved();
  for (let i = 0; i < step; i++) s = CUBE.apply(s, seq[i]);
  return s;
}
function animAt(seq, step, frac) {
  if (step >= seq.length || frac <= 0.02) return null;
  const nm = CUBE.moveNames[seq[step]];
  return { face: nm[0], angle: (nm.endsWith("'") ? 1 : -1) * ease(frac) * Math.PI / 2 };
}

// ── 場景：只把角塊點亮 ──────────────────────────────────────
function sceneCorners(sc, local) {
  const lead = 30;
  const raw = Math.max(0, (local - lead) * sc.stepsPerFrame);
  const step = Math.min(sc.totalSteps, Math.floor(raw));
  const state = playTo(sc.seq, step);
  // 前半段正常顯示，後半段把非角塊淡掉，看得出「角塊是哪八顆」
  const t = clamp01((local - lead - sc.totalSteps / sc.stepsPerFrame - 20) / 40);
  drawCube(state, { local, anim: animAt(sc.seq, step, raw - step), dim: t > 0.5 ? NOT_CORNER : null });
  if (t > 0.5) text("只看這 24 片貼紙", 560, 172, { size: 28, color: C.warn, weight: 600, align: "center" });

  const x = 1120, w = 740;
  box(x, 200, w, 220);
  text("整顆 3×3×3 有幾種局面", x + 30, 250, { size: 18, color: C.dim });
  text("43,252,003,274,489,856,000", x + 30, 316, { size: 34, color: C.fg, weight: 700, font: "mono" });
  text("四千三百二十五京 — 數不完", x + 30, 366, { size: 19, color: C.bad });

  box(x, 452, w, 220);
  text("只看八顆角呢", x + 30, 502, { size: 18, color: C.dim });
  text(fmt(D.pdb.states), x + 30, 570, { size: 52, color: C.good, weight: 700, font: "mono" });
  text("8! × 3⁷ — 114 秒可以全部數完", x + 30, 618, { size: 19, color: C.good });

  if (t > 0.6) {
    box(x, 704, w, 200);
    text("角塊的上帝之數", x + 30, 754, { size: 18, color: C.dim });
    text(String(D.pdb.max), x + 30, 822, { size: 52, color: C.accent, weight: 700, font: "mono" });
    text(`平均 ${D.pdb.mean.toFixed(3)} 步（2×2×2 是 10.666）`, x + 30, 870, { size: 19, color: C.dim });
  }
}

// ── 場景：分階段解法（這支影片的主角）──────────────────────
function sceneStaged(sc, local) {
  const lead = 30;
  const raw = Math.max(0, (local - lead) * sc.stepsPerFrame);
  const step = Math.min(sc.totalSteps, Math.floor(raw));
  const state = playTo(sc.seq, step);
  const inStage1 = step >= sc.scrambleLen && step < sc.cornerAt;
  const done1 = step >= sc.cornerAt;

  drawCube(state, { local, anim: animAt(sc.seq, step, raw - step),
                    dim: inStage1 ? NOT_CORNER : null });

  text("攤平圖（含邊塊，沒有淡掉）", opts_cx(), 796, { size: 16, color: C.dim, align: "center" });

  let label = "打亂中", col = C.dim;
  if (inStage1) { label = "第一階段：只解角塊"; col = C.warn; }
  else if (done1 && step < sc.totalSteps) { label = "第二階段：解完剩下的"; col = C.accent; }
  else if (step >= sc.totalSteps) { label = "解開了"; col = C.good; }
  text(label, 560, 172, { size: 28, color: col, weight: 600, align: "center" });

  const x = 1120, w = 740;
  const ti = Math.min(step, sc.trace.pdb.length - 1);
  box(x, 180, w, 300);
  text("角塊還要幾步（精確，查表）", x + 30, 228, { size: 18, color: C.dim });
  const cd = Math.max(0, sc.trace.pdb[ti]);
  text(cd.toFixed(0), x + 30, 320, { size: 72, color: cd === 0 ? C.good : C.warn, weight: 700, font: "mono" });
  if (done1) text("角塊歸位 ✓", x + 200, 320, { size: 30, color: C.good, weight: 600 });
  text("網路猜整顆還要幾步（估計）", x + 30, 396, { size: 18, color: C.dim });
  text(Math.max(0, sc.trace.net[ti]).toFixed(2), x + 30, 456, { size: 44, color: C.accent, weight: 700, font: "mono" });

  box(x, 512, w, 390);
  text("這一顆的兩種解法", x + 30, 560, { size: 18, color: C.dim });
  const one = D.oneShot, two = D.staged;
  const rows = [
    ["一次解完", `${one.seq.length} 步`, `${fmt(one.nodes)} 節點`, C.fg],
    ["先解角，再解完", `${two.stage1.length + two.stage2.length} 步`, `${fmt(two.nodes)} 節點`, C.warn],
  ];
  rows.forEach(([a, b, c, col2], i) => {
    const y = 620 + i * 96;
    text(a, x + 30, y, { size: 22, color: col2, weight: 600 });
    text(b, x + 330, y, { size: 30, color: col2, weight: 700, font: "mono", align: "right" });
    text(c, x + w - 30, y, { size: 26, color: C.dim, font: "mono", align: "right" });
  });
  const dl = two.stage1.length + two.stage2.length - one.seq.length;
  text(`分階段多走 ${dl} 步，但第二段少展開 ${Math.round((1 - two.nodes / one.nodes) * 100)}% 的節點`,
       x + 30, 852, { size: 20, color: C.good });
}

// ── 場景：角塊距離分布 ──────────────────────────────────────
function sceneHist(sc, local) {
  const hist = D.pdb.hist, t = clamp01((local - 20) / 60);
  const x0 = 210, y0 = 800, w = 1500, maxV = Math.max(...hist);
  const bw = w / hist.length;
  text("角塊的距離分布（全部 88,179,840 個狀態）", 960, 190,
       { size: 30, color: C.fg, align: "center", weight: 600 });
  text("每一根是「只把角塊轉回去要幾步」的局面數", 960, 230,
       { size: 19, color: C.dim, align: "center" });
  hist.forEach((v, d) => {
    const h = (v / maxV) * 500 * ease(clamp01(t * 1.6 - d * 0.03));
    ctx.fillStyle = d === hist.length - 1 ? C.warn : (d >= 10 && d <= 12 ? C.good : "#2b6f4a");
    ctx.beginPath(); ctx.roundRect(x0 + d * bw + 6, y0 - h, bw - 12, h, 4); ctx.fill();
    text(String(d), x0 + d * bw + bw / 2, y0 + 30, { size: 20, color: C.dim, align: "center", font: "mono" });
  });
  if (t > 0.7) {
    box(1230, 260, 600, 210);
    text("上帝之數（角塊）", 1260, 310, { size: 20, color: C.dim });
    text(String(D.pdb.max), 1260, 388, { size: 62, color: C.warn, weight: 700, font: "mono" });
    text(`最遠的只有 ${fmt(hist[hist.length - 1])} 個`, 1260, 436, { size: 19, color: C.dim });
    box(180, 260, 620, 210);
    text("2×2×2 的對照", 210, 310, { size: 20, color: C.dim });
    text("14 · 10.666", 210, 388, { size: 48, color: C.good, weight: 700, font: "mono" });
    text(`角塊是 ${D.pdb.max} · ${D.pdb.mean.toFixed(3)} — 同一個群`, 210, 436, { size: 19, color: C.dim });
  }
}

// ── 場景：三種 heuristic ────────────────────────────────────
function sceneE1(sc, local) {
  const t = clamp01((local - 18) / 55);
  text("同一批隨機方塊，只換估計法", 960, 186, { size: 32, color: C.fg, align: "center", weight: 700 });
  const rows = D.e1;
  const nm = (r) => r.name.replace("（精確）", "").replace(" ∪ ", " + ");
  rows.forEach((r, i) => {
    const y = 280 + i * 200;
    const tt = ease(clamp01(t * 1.5 - i * 0.18));
    ctx.globalAlpha = tt;
    box(180, y, 1560, 168, r.name.includes("∪") ? "#0f1a26" : "#0d1320");
    text(nm(r), 220, y + 52, { size: 26, color: r.admissible ? C.good : C.accent, weight: 700 });
    text(r.admissible ? "保證不高估" : "會高估", 220, y + 92, { size: 18, color: r.admissible ? C.good : C.warn });
    const cells = [
      ["平均估計", r.mean_h.toFixed(2), C.fg],
      ["解開率", `${Math.round(r.solve_rate * 100)}%`, r.solve_rate > 0.5 ? C.good : C.bad],
      ["平均步數", r.mean_len == null ? "—" : r.mean_len.toFixed(1), C.dim],
      ["展開節點", fmt(Math.round(r.mean_nodes)), C.accent],
    ];
    cells.forEach(([k, v, col], j) => {
      const cx = 800 + j * 240;
      text(k, cx, y + 52, { size: 17, color: C.dim });
      text(v, cx, y + 108, { size: 36, color: col, weight: 700, font: "mono" });
    });
    ctx.globalAlpha = 1;
  });
  if (t > 0.85) {
    const net = rows.find((r) => !r.admissible && !r.name.includes("∪"));
    const mx = rows.find((r) => r.name.includes("∪"));
    text(`精確但片面的，單獨用解開率 ${Math.round(rows.find((r) => r.admissible).solve_rate * 100)}%；`
         + `當補丁卻少了 ${Math.round((1 - mx.mean_nodes / net.mean_nodes) * 100)}% 的節點`,
         960, 940, { size: 24, color: C.warn, align: "center", weight: 600 });
  }
}

// ── 場景：3×3×3 第一次有正確答案 ───────────────────────────
function sceneErr(sc, local) {
  const by = D.e2.by_true_net, bp = D.e2.by_true_pdb;
  const t = clamp01((local - 18) / 55);
  const maxD = Math.max(...by.map((r) => r.d)) + 1;
  const x0 = 300, y0 = 850, w = 880, h = 630;
  const px = (d) => x0 + (w * d) / maxD;
  const py = (v) => y0 - (h * v) / maxD;
  text("網路的估計 vs 精確答案", 960, 178, { size: 32, color: C.fg, align: "center", weight: 700 });
  text(`${fmt(D.e2.n)} 個局面，用角塊表證明出最短解`, 960, 216, { size: 19, color: C.dim, align: "center" });

  ctx.strokeStyle = "#182437"; ctx.lineWidth = 1;
  for (let g = 0; g <= maxD; g += 2) {
    ctx.beginPath(); ctx.moveTo(px(g), y0); ctx.lineTo(px(g), py(maxD)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x0, py(g)); ctx.lineTo(px(maxD), py(g)); ctx.stroke();
    text(String(g), px(g), y0 + 30, { size: 17, color: C.dim, align: "center", font: "mono" });
    text(String(g), x0 - 14, py(g) + 6, { size: 17, color: C.dim, align: "right", font: "mono" });
  }
  ctx.strokeStyle = C.good; ctx.setLineDash([7, 6]); ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(px(0), py(0)); ctx.lineTo(px(maxD), py(maxD)); ctx.stroke();
  ctx.setLineDash([]);
  text("完全準的話會落在這條線上", px(maxD) - 6, py(maxD) + 26, { size: 17, color: C.good, align: "right" });

  const line = (rows, col, lw) => {
    const vis = rows.filter((r) => r.d <= maxD * t + 0.5);
    if (vis.length < 2) return;
    ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.beginPath();
    vis.forEach((r, i) => (i ? ctx.lineTo(px(r.d), py(r.mean)) : ctx.moveTo(px(r.d), py(r.mean))));
    ctx.stroke();
  };
  line(bp, C.grey, 2.5);
  line(by, C.accent, 4);
  text("真正的最短解", (x0 + px(maxD)) / 2, y0 + 66, { size: 20, color: C.dim, align: "center" });
  ctx.save(); ctx.translate(x0 - 96, (y0 + py(maxD)) / 2); ctx.rotate(-Math.PI / 2);
  text("估計值", 0, 0, { size: 20, color: C.dim, align: "center" }); ctx.restore();

  if (t > 0.55) {
    const stats = [
      ["網路的平均誤差", `${D.e2.mae.toFixed(2)} 步`, C.accent],
      ["誤差在 1 步以內", `${(D.e2.within_1 * 100).toFixed(0)}%`, C.accent],
      ["平均偏差", D.e2.bias.toFixed(2), C.warn],
      ["角塊表（灰線）", "永遠在下面", C.grey],
    ];
    stats.forEach(([k, v, col], i) => {
      box(1290, 290 + i * 150, 540, 128);
      text(k, 1320, 340 + i * 150, { size: 19, color: C.dim });
      // 中文的值用等寬字會 fallback 成襯線，看起來像另一份文件
      const cjk = /[^ -]/.test(v.replace(/[%步]/g, ""));
      text(v, 1320, 396 + i * 150,
           { size: cjk ? 34 : 40, color: col, weight: 700, font: cjk ? "sans" : "mono" });
    });
  }
}

// ── 場景：IDA* ──────────────────────────────────────────────
function sceneIda(sc, local) {
  const t = clamp01((local - 18) / 50);
  const iv = D.idaVsAstar;
  text("為什麼換掉 A*", 960, 186, { size: 34, color: C.fg, align: "center", weight: 700 });
  text("同一個 heuristic、同一台機器、同一批局面 — 只換搜尋的寫法",
       960, 232, { size: 21, color: C.dim, align: "center" });
  const rows = [["A*（有 open / closed list）", iv.astar, C.dim],
                ["IDA*（深度優先 + f 值上限）", iv.ida, C.good]];
  const maxNps = Math.max(iv.ida.nps, iv.astar.nps);
  rows.forEach(([nm, r, col], i) => {
    const y = 330 + i * 240;
    const tt = ease(clamp01(t * 1.4 - i * 0.2));
    ctx.globalAlpha = tt;
    box(200, y, 1520, 200);
    text(nm, 240, y + 56, { size: 26, color: col, weight: 700 });
    ctx.fillStyle = "#131d2e";
    ctx.beginPath(); ctx.roundRect(240, y + 92, 900, 34, 8); ctx.fill();
    ctx.fillStyle = col === C.good ? C.good : "#3b4a63";
    ctx.beginPath(); ctx.roundRect(240, y + 92, 900 * (r.nps / maxNps) * tt, 34, 8); ctx.fill();
    text(`${fmt(Math.round(r.nps))} 節點/秒`, 1180, y + 120, { size: 30, color: col, weight: 700, font: "mono" });
    text(`打亂 ${iv.depth} 步：${fmt(Math.round(r.nodes))} 節點 / ${r.sec.toFixed(2)} 秒`,
         1180, y + 62, { size: 19, color: C.dim });
    ctx.globalAlpha = 1;
  });
  if (t > 0.8) {
    text(`快 ${Math.round(iv.ida.nps / iv.astar.nps)} 倍 — 而且 heuristic 一模一樣，展開的節點只有更多。`
         + `差別純粹是每個節點的簿記成本。`,
         960, 900, { size: 24, color: C.warn, align: "center", weight: 600 });
  }
}

// ── 場景：純文字 ────────────────────────────────────────────
function sceneText(sc, local) {
  const lines = sc.lines || [];
  text(sc.title || "", 960, 240, { size: 40, color: C.fg, align: "center", weight: 700 });
  lines.forEach((l, i) => {
    const tt = clamp01((local - 20 - i * 16) / 24);
    ctx.globalAlpha = tt;
    const big = l.startsWith("*");
    text(big ? l.slice(1) : l, 960, 380 + i * 78,
         { size: big ? 34 : 27, color: big ? C.accent : C.dim, align: "center",
           weight: big ? 700 : 400 });
    ctx.globalAlpha = 1;
  });
}

const DISPATCH = { corners: sceneCorners, staged: sceneStaged, hist: sceneHist,
                   e1: sceneE1, err: sceneErr, ida: sceneIda, text: sceneText };

function renderFrame() {
  const sc = PLAN.scenes.find((s) => frame >= s.start && frame < s.start + s.frames)
    || PLAN.scenes[PLAN.scenes.length - 1];
  const local = frame - sc.start;
  chrome(sc, local);
  DISPATCH[sc.kind](sc, local);
  if (local < 10) {
    ctx.fillStyle = C.bg; ctx.globalAlpha = 1 - local / 10;
    ctx.fillRect(0, 0, W, H); ctx.globalAlpha = 1;
  }
  frame++;
}

window.__rec = {
  init() { PLAN = buildPlan(); frame = 0; return PLAN; },
  renderFrame,
  seek(f) { frame = f; },
};
