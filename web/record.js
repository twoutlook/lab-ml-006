// 錄影用的畫面產生器。1920×1080，一格一格算，不是螢幕錄影。
//
// tools/record_video.mjs 會用 headless Chrome 開 record.html，先呼叫 __rec.init()
// 拿到分鏡表（每個場景幾格、從第幾格開始），再一格一格呼叫 __rec.renderFrame()
// 把畫面取出來餵給 ffmpeg。所以：
//   ・不會掉格（跟機器快慢無關）
//   ・同樣的資料一定錄出同樣的影片
//   ・旁白是錄完之後照場景起始時間貼上去的，對時是算出來的
//
// 畫面上所有數字都來自 web/demo.json（由 ml/make_demo.py 產生），
// 沒有一個是寫死在這裡的。重跑訓練、重跑 benchmark，影片的內容就跟著換。

import { Cube } from "./cube.js";
import { CubeView, drawNet } from "./render.js";

const W = 1920, H = 1080, FPS = 30;
const PAD_TAIL = 12;          // 每個場景旁白唸完之後多留幾格

const cv = document.getElementById("stage");
const ctx = cv.getContext("2d");

const [CFG, SCRIPT, TIMING, DEMO, MOVES] = await Promise.all([
  fetch("../shared/config.json").then((r) => r.json()),
  fetch("../tools/script.json").then((r) => r.json()),
  fetch("../out/voice/timing.json").then((r) => r.json()).catch(() => null),
  fetch("demo.json").then((r) => r.json()),
  fetch("../shared/moves.json").then((r) => r.json()),
]);

const COLORS = CFG.view.colors;
const CUBES = { 2: new Cube(MOVES["2"]), 3: new Cube(MOVES["3"]) };
const VIEWS = { 2: new CubeView(CUBES[2], COLORS), 3: new CubeView(CUBES[3], COLORS) };

const C = {
  bg: "#06080f", fg: "#e6edf7", dim: "#7c8ba4", line: "#1b2436",
  accent: "#38bdf8", good: "#4ade80", warn: "#fbbf24", bad: "#f87171",
};

// ── 小工具 ──────────────────────────────────────────────────
const fmt = (n) => n.toLocaleString("en-US");
function text(s, x, y, { size = 22, color = C.fg, weight = 400, align = "left", font = "sans" } = {}) {
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.font = `${weight} ${size}px ${font === "mono" ? "ui-monospace, Consolas, monospace"
    : '"Noto Sans TC", "Microsoft JhengHei", system-ui, sans-serif'}`;
  ctx.fillText(s, x, y);
}
function box(x, y, w, h, fill = "#0d1320", stroke = C.line) {
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, 12);
  ctx.fill();
  ctx.stroke();
}
const lerp = (a, b, t) => a + (b - a) * t;
const clamp01 = (t) => Math.max(0, Math.min(1, t));
// 場景開頭淡入用的緩動
const ease = (t) => (t < 0.5 ? 2 * t * t : 1 - 2 * (1 - t) * (1 - t));

// ── 分鏡表 ──────────────────────────────────────────────────
function buildPlan() {
  const byId = Object.fromEntries((TIMING || []).map((t) => [t.id, t.duration]));
  const scenes = [];
  let start = 0;
  for (const sc of SCRIPT.scenes) {
    const dur = byId[sc.id];
    const frames = Math.max(90, Math.round((dur || 20) * FPS) + PAD_TAIL);
    const s = { ...sc, start, frames, voice: dur || null };
    if (sc.kind === "cube") Object.assign(s, prepCube(sc, frames));
    scenes.push(s);
    start += frames;
  }
  return { fps: FPS, totalFrames: start, durationSec: +(start / FPS).toFixed(2), scenes };
}

// cube 場景：先算好整串要播的動作，再攤到可用的格數上
function prepCube(sc, frames) {
  const cube = CUBES[sc.size];
  const cs = DEMO.cases[String(sc.size)][sc.caseIndex || 0];
  const usable = frames - 30;
  let seq, hvals, phase;
  if (sc.mode === "random") {
    // 亂轉只要「看得出來它在亂轉」就夠了，播太快會糊成一片。
    // 每秒 4 步，剛好一步一眼。真正的一萬步實測數字在右邊的面板上。
    const n = Math.min(DEMO.randomWalk[String(sc.size)].length, Math.round((usable / FPS) * 4));
    seq = DEMO.randomWalk[String(sc.size)].slice(0, n);
    hvals = null;
    phase = "random";
  } else if (sc.mode === "solve") {
    const run = cs.runs[String(sc.weight ?? DEMO.defaultWeight[String(sc.size)])];
    seq = cs.scramble.concat(run.seq);
    hvals = cs.hScramble.concat(run.h);
    phase = "solve";
    return { seq, hvals, phase, scrambleLen: cs.scramble.length, run, exact: cs.exact,
             totalSteps: seq.length, stepsPerFrame: seq.length / usable };
  } else {
    // 展示用的打亂比解題用的深：標題那一幕要看起來真的亂了，
    // 而解題那一幕的深度受限於搜尋的節點預算。
    seq = DEMO.displayScramble[String(sc.size)] || cs.scramble;
    hvals = cs.hScramble;
    phase = "scramble";
  }
  // 打亂 / 亂轉：前 60% 的時間播完，後面停著讓人看清楚
  const play = sc.mode === "random" ? usable : usable * 0.6;
  return { seq, hvals, phase, scrambleLen: seq.length, exact: cs.exact ?? null,
           totalSteps: seq.length, stepsPerFrame: seq.length / play };
}

let PLAN = null, frame = 0;

// ── 外框 ────────────────────────────────────────────────────
function chrome(sc, local) {
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, W, H);
  text("魔術方塊 × DeepCubeA", 60, 58, { size: 24, color: C.dim, weight: 600 });
  text("lab-ml-006", 60, 84, { size: 15, color: "#4a5568", font: "mono" });
  text(sc.chapter || "", 1860, 62, { size: 30, color: C.fg, weight: 600, align: "right" });
  ctx.strokeStyle = C.line;
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(60, 104); ctx.lineTo(1860, 104); ctx.stroke();

  const p = (sc.start + local) / PLAN.totalFrames;
  ctx.fillStyle = "#131b2b"; ctx.fillRect(60, 1042, 1800, 3);
  ctx.fillStyle = C.accent; ctx.fillRect(60, 1042, 1800 * p, 3);
}

// ── 場景：方塊 ──────────────────────────────────────────────
function sceneCube(sc, local) {
  const cube = CUBES[sc.size];
  const lead = 24;                                  // 開頭先靜止幾格
  const step = Math.max(0, Math.min(sc.totalSteps, Math.floor((local - lead) * sc.stepsPerFrame)));
  const frac = Math.max(0, Math.min(1, (local - lead) * sc.stepsPerFrame - step));

  let state = cube.solved();
  for (let i = 0; i < step; i++) state = cube.apply(state, sc.seq[i]);

  // 正在轉的那一步畫成轉到一半
  let anim = null;
  if (step < sc.totalSteps && frac > 0.02) {
    const name = cube.moveNames[sc.seq[step]];
    anim = { face: name[0], angle: (name.endsWith("'") ? 1 : -1) * ease(frac) * Math.PI / 2 };
  }

  // 讓視角一直慢慢轉。方塊本身很多時候是停著的，鏡頭不動的話畫面會死掉。
  const view = VIEWS[sc.size];
  view.yaw = -0.62 + local * 0.0022;
  view.draw(ctx, state, {
    cx: 570, cy: sc.size === 2 ? 470 : 452, scale: sc.size === 2 ? 100 : 67, anim,
  });

  const cell = sc.size === 2 ? 30 : 22;
  const fw = cube.n * cell + 3;
  drawNet(ctx, cube, state, COLORS, 570 - 2 * fw, 812, cell, 3, "#0a0d15");
  text("攤平圖（六個面都看得到）", 570, 796, { size: 16, color: C.dim, align: "center" });

  if (sc.label) text(sc.label, 570, 150, { size: 28, color: C.accent, weight: 600, align: "center" });

  // ── 右邊的面板 ──
  const x = 1120, w = 740;
  if (sc.mode === "random") {
    box(x, 200, w, 190);
    text("已經亂轉", x + 30, 250, { size: 18, color: C.dim });
    text(fmt(step), x + 30, 320, { size: 62, color: C.fg, weight: 700, font: "mono" });
    text("步", x + 30 + ctx.measureText(fmt(step)).width + 16, 320, { size: 24, color: C.dim });
    box(x, 420, w, 190);
    text("解開了嗎", x + 30, 470, { size: 18, color: C.dim });
    text("沒有", x + 30, 540, { size: 62, color: C.bad, weight: 700 });
    box(x, 640, w, 250);
    text("實測：亂轉一萬步", x + 30, 690, { size: 18, color: C.dim });
    const rw = DEMO.randomStats[String(sc.size)];
    text(`${(rw.solve_rate * 100).toFixed(1)}%`, x + 30, 762, { size: 56, color: C.bad, weight: 700, font: "mono" });
    text(`${fmt(rw.n)} 個打亂過的方塊，解開 ${rw.solved} 個`, x + 30, 806, { size: 19, color: C.dim });
    text("隨機探索永遠嚐不到那一次成功", x + 30, 848, { size: 19, color: C.warn });
  } else if (sc.mode === "solve") {
    const solving = step >= sc.scrambleLen;
    const solveStep = Math.max(0, step - sc.scrambleLen);
    box(x, 180, w, 152);
    text(solving ? "搜尋找到的解，正在播" : "打亂中", x + 30, 226, { size: 18, color: C.dim });
    text(`${solving ? solveStep : step} / ${solving ? sc.run.seq.length : sc.scrambleLen} 步`,
         x + 30, 296, { size: 50, color: solving ? C.good : C.fg, weight: 700, font: "mono" });

    box(x, 356, w, 300);
    text("網路猜「還要幾步」vs 真正還要幾步", x + 30, 400, { size: 18, color: C.dim });
    drawHLine(sc, step, x + 30, 420, w - 60, 210);

    box(x, 686, w, 260);
    const cells = [
      ["加權 A* 展開節點", fmt(sc.run.nodes), C.accent],
      ["解的長度", `${sc.run.seq.length} 步`, C.good],
    ];
    if (sc.exact != null) {
      cells.push(["精確最短解", `${sc.exact} 步`, C.good]);
      cells.push([sc.run.seq.length === sc.exact ? "就是最短解" : `比最短多 ${sc.run.seq.length - sc.exact} 步`,
                  "", sc.run.seq.length === sc.exact ? C.good : C.warn]);
    } else {
      cells.push(["狀態空間", "4.3 × 10¹⁹", C.dim]);
      cells.push(["沒有正確答案可比", "", C.warn]);
    }
    cells.forEach(([k, v, col], i) => {
      const cx = x + 30 + (i % 2) * (w / 2 - 10), cy = 736 + Math.floor(i / 2) * 106;
      if (v) {
        text(k, cx, cy, { size: 17, color: C.dim });
        text(v, cx, cy + 48, { size: 38, color: col, weight: 700, font: "mono" });
      } else {
        // 只有一句話的格子（「就是最短解」），不要用小小的標籤字級
        text(k, cx, cy + 34, { size: 30, color: col, weight: 700 });
      }
    });
  } else {
    // 標題 / 介紹尺寸
    const info = DEMO.sizeInfo[String(sc.size)];
    box(x, 220, w, 210);
    text("有幾種排列", x + 30, 268, { size: 18, color: C.dim });
    text(info.states, x + 30, 340, { size: info.states.length > 14 ? 34 : 46, color: C.fg, weight: 700, font: "mono" });
    text(info.statesNote, x + 30, 388, { size: 19, color: C.dim });
    box(x, 460, w, 190);
    text("能轉的面 / 動作數", x + 30, 508, { size: 18, color: C.dim });
    text(info.moves, x + 30, 576, { size: 40, color: C.accent, weight: 700, font: "mono" });
    box(x, 680, w, 210);
    text("能不能把答案整個列出來", x + 30, 728, { size: 18, color: C.dim });
    text(info.solvable, x + 30, 796, { size: 36, color: info.exact ? C.good : C.bad, weight: 700 });
    text(info.solvableNote, x + 30, 844, { size: 19, color: C.dim });
  }
}

// 解到哪裡、網路怎麼看：一條隨著播放長出來的折線
function drawHLine(sc, step, x, y, w, h) {
  const n = sc.hvals.length;
  const maxV = Math.max(...sc.hvals, sc.exact || 0, 14) * 1.1;
  const px = (i) => x + (w * i) / Math.max(1, n - 1);
  const py = (v) => y + h - (h * v) / maxV;

  ctx.strokeStyle = "#16203350"; ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g++) {
    const yy = y + (h * g) / 4;
    ctx.beginPath(); ctx.moveTo(x, yy); ctx.lineTo(x + w, yy); ctx.stroke();
    text(((maxV * (4 - g)) / 4).toFixed(0), x - 10, yy + 6, { size: 14, color: C.dim, align: "right", font: "mono" });
  }
  // 打亂結束的位置
  const sx = px(sc.scrambleLen);
  ctx.strokeStyle = "#2a3852"; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(sx, y); ctx.lineTo(sx, y + h); ctx.stroke();
  ctx.setLineDash([]);
  text("開始解", sx + 8, y + 18, { size: 14, color: C.dim });

  // 真正的最短距離：擺在「開始解」那個位置當對照
  if (sc.exact != null) {
    ctx.fillStyle = C.good;
    ctx.beginPath(); ctx.arc(sx, py(sc.exact), 6, 0, 7); ctx.fill();
    ctx.strokeStyle = C.good; ctx.lineWidth = 1.5; ctx.setLineDash([3, 4]);
    ctx.beginPath(); ctx.moveTo(x, py(sc.exact)); ctx.lineTo(sx, py(sc.exact)); ctx.stroke();
    ctx.setLineDash([]);
    text(`真正 ${sc.exact}`, sx - 10, py(sc.exact) - 12, { size: 17, color: C.good, weight: 700, align: "right", font: "mono" });
  }

  ctx.strokeStyle = C.accent; ctx.lineWidth = 3;
  ctx.beginPath();
  for (let i = 0; i <= Math.min(step, n - 1); i++) {
    const X = px(i), Y = py(sc.hvals[i]);
    i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y);
  }
  ctx.stroke();
  const i = Math.min(step, n - 1);
  ctx.fillStyle = C.accent;
  ctx.beginPath(); ctx.arc(px(i), py(sc.hvals[i]), 6, 0, 7); ctx.fill();
  text(sc.hvals[i].toFixed(2), px(i) + 12, py(sc.hvals[i]) - 10, { size: 20, color: C.accent, weight: 700, font: "mono" });
  text("網路的估計值", x + 6, y + h + 24, { size: 15, color: C.accent });
}

// ── 場景：距離分布 ──────────────────────────────────────────
function sceneHist(sc, local) {
  const hist = DEMO.hist;
  const t = clamp01((local - 20) / 60);
  const x0 = 200, y0 = 820, w = 1520, maxV = Math.max(...hist);
  const bw = w / hist.length;

  text("2×2×2：每個局面離解開幾步（全部 3,674,160 個）", 960, 190,
       { size: 26, color: C.fg, align: "center", weight: 600 });
  text("這張分布跟數學社群公開的表逐項相同 — 順便驗了引擎沒寫錯", 960, 228,
       { size: 19, color: C.dim, align: "center" });

  hist.forEach((v, d) => {
    const hgt = (v / maxV) * 520 * ease(clamp01(t * 1.6 - d * 0.03));
    const isPeak = d === 11 || d === 12;
    ctx.fillStyle = isPeak ? C.accent : (d === 14 ? C.warn : "#2b4a6f");
    ctx.beginPath();
    ctx.roundRect(x0 + d * bw + 6, y0 - hgt, bw - 12, hgt, 4);
    ctx.fill();
    text(String(d), x0 + d * bw + bw / 2, y0 + 30, { size: 20, color: d === 14 ? C.warn : C.dim, align: "center", font: "mono" });
    if (v / maxV > 0.06 && t > 0.6) {
      text(fmt(v), x0 + d * bw + bw / 2, y0 - hgt - 14,
           { size: 16, color: isPeak ? C.accent : C.dim, align: "center", font: "mono" });
    }
  });
  text("最短還要幾步", 960, y0 + 66, { size: 20, color: C.dim, align: "center" });

  if (t > 0.75) {
    const share = ((hist[10] + hist[11] + hist[12]) / DEMO.total2 * 100).toFixed(1);
    box(1210, 250, 620, 200);
    text("10 ~ 12 步的局面佔了", 1240, 300, { size: 20, color: C.dim });
    text(`${share}%`, 1240, 372, { size: 62, color: C.accent, weight: 700, font: "mono" });
    text("所以「都猜 11」就已經很準了 — 這是底線", 1240, 418, { size: 19, color: C.warn });
    box(150, 250, 500, 200);
    text("最遠的局面（上帝之數）", 180, 300, { size: 20, color: C.dim });
    text(`14 步`, 180, 372, { size: 62, color: C.warn, weight: 700, font: "mono" });
    text(`只有 ${fmt(hist[14])} 個`, 180, 418, { size: 19, color: C.dim });
  }
}

// ── 場景：DAVI 說明 ────────────────────────────────────────
function sceneBullets(sc, local) {
  text("DAVI — 訓練資料是「倒著」造出來的", 960, 200,
       { size: 34, color: C.fg, align: "center", weight: 700 });

  // 左邊：資料怎麼來
  const t = clamp01((local - 15) / 50);
  box(120, 270, 800, 620);
  text("① 資料", 160, 322, { size: 22, color: C.accent, weight: 700 });
  text("從解開的狀態往回亂轉 d 步", 160, 366, { size: 24, color: C.fg });
  text("往回 d 步的局面，答案一定 ≤ d 步。", 160, 404, { size: 19, color: C.dim });
  text("資料自帶難度標籤，而且要多少有多少 —", 160, 434, { size: 19, color: C.dim });
  text("不需要 agent 先會玩才有資料可學。", 160, 464, { size: 19, color: C.dim });

  // 一條往回走的示意軌跡
  const cube = CUBES[2];
  const walk = DEMO.walkDemo;
  const shown = Math.max(1, Math.round(t * walk.length));
  for (let i = 0; i < Math.min(shown, walk.length); i++) {
    const st = Uint8Array.from(walk[i]);
    const cx = 205 + i * 128, cy = 608;
    VIEWS[2].draw(ctx, st, { cx, cy, scale: 25 });
    text(i === 0 ? "解開" : `${i} 步`, cx, 706, { size: 17, color: i === 0 ? C.good : C.dim, align: "center", font: "mono" });
  }
  text("← 往回亂轉", 160, 752, { size: 19, color: C.dim });
  text("② 訓練目標：把子節點展開一層，取最小 + 1", 160, 812, { size: 22, color: C.accent, weight: 700 });
  text("y(s) = min over a [ 1 + h_target( a(s) ) ]", 160, 856, { size: 24, color: C.fg, font: "mono" });

  // 右邊：跟 DQN 的差別
  box(970, 270, 830, 620);
  text("跟前面五個專案的 DQN 差在哪", 1010, 322, { size: 22, color: C.accent, weight: 700 });
  const rows = [
    ["沒有環境互動", "不用先會玩才有資料"],
    ["沒有 episode、沒有 replay buffer", "資料是造出來的，不是玩出來的"],
    ["沒有 epsilon", "探索這件事整個消失了"],
    ["學 value，不學 policy", "網路只回答「還要幾步」"],
    ["target 跟策略無關", "永遠取所有子節點裡最好的"],
  ];
  rows.forEach(([a, b], i) => {
    const y = 386 + i * 96;
    const tt = clamp01((local - 20 - i * 14) / 22);
    ctx.globalAlpha = tt;
    ctx.fillStyle = C.accent;
    ctx.beginPath(); ctx.arc(1030, y - 8, 5, 0, 7); ctx.fill();
    text(a, 1054, y, { size: 24, color: C.fg, weight: 600 });
    text(b, 1054, y + 34, { size: 19, color: C.dim });
    ctx.globalAlpha = 1;
  });
}

// ── 場景：準確度 ────────────────────────────────────────────
function sceneError(sc, local) {
  const by = DEMO.heuristic.by_distance;
  const t = clamp01((local - 18) / 55);
  const x0 = 300, y0 = 860, w = 900, h = 640;
  const maxD = 15;
  const px = (d) => x0 + (w * d) / maxD;
  const py = (v) => y0 - (h * v) / maxD;

  text("網路猜的 vs 真正的最短步數", 960, 180, { size: 32, color: C.fg, align: "center", weight: 700 });
  text(`從全部 ${fmt(DEMO.heuristic.n)} 個抽樣的局面`, 960, 218, { size: 19, color: C.dim, align: "center" });

  ctx.strokeStyle = "#182437"; ctx.lineWidth = 1;
  for (let g = 0; g <= maxD; g += 3) {
    ctx.beginPath(); ctx.moveTo(px(g), y0); ctx.lineTo(px(g), py(maxD)); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x0, py(g)); ctx.lineTo(px(maxD), py(g)); ctx.stroke();
    text(String(g), px(g), y0 + 30, { size: 17, color: C.dim, align: "center", font: "mono" });
    text(String(g), x0 - 14, py(g) + 6, { size: 17, color: C.dim, align: "right", font: "mono" });
  }
  // 完美的話應該落在對角線上
  ctx.strokeStyle = C.good; ctx.setLineDash([7, 6]); ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(px(0), py(0)); ctx.lineTo(px(maxD), py(maxD)); ctx.stroke();
  ctx.setLineDash([]);
  text("完全準的話會落在這條線上", px(maxD) - 6, py(maxD) + 26, { size: 17, color: C.good, align: "right" });

  // ±1 標準差的帶子
  const vis = by.filter((r) => r.d <= maxD * t + 0.5);
  if (vis.length > 1) {
    ctx.fillStyle = "rgba(56,189,248,.16)";
    ctx.beginPath();
    vis.forEach((r, i) => { const X = px(r.d), Y = py(r.mean_h + r.std_h); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
    for (let i = vis.length - 1; i >= 0; i--) ctx.lineTo(px(vis[i].d), py(vis[i].mean_h - vis[i].std_h));
    ctx.closePath(); ctx.fill();

    ctx.strokeStyle = C.accent; ctx.lineWidth = 4;
    ctx.beginPath();
    vis.forEach((r, i) => { const X = px(r.d), Y = py(r.mean_h); i ? ctx.lineTo(X, Y) : ctx.moveTo(X, Y); });
    ctx.stroke();
  }
  text("真正的最短步數", (x0 + px(maxD)) / 2, y0 + 66, { size: 20, color: C.dim, align: "center" });
  ctx.save(); ctx.translate(x0 - 66, (y0 + py(maxD)) / 2); ctx.rotate(-Math.PI / 2);
  text("網路猜的", 0, 0, { size: 20, color: C.dim, align: "center" }); ctx.restore();

  if (t > 0.55) {
    const H = DEMO.heuristic;
    const stats = [
      ["平均誤差", `${H.mae.toFixed(2)} 步`, C.accent],
      ["誤差在 1 步以內", `${(H.within_1 * 100).toFixed(1)}%`, C.accent],
      ["高估（h > 真值）", `${(H.over_rate * 100).toFixed(1)}%`, C.warn],
      ["只用它、不搜尋的解開率", `${(DEMO.greedy2 * 100).toFixed(1)}%`, C.bad],
    ];
    stats.forEach(([k, v, col], i) => {
      box(1290, 290 + i * 150, 540, 128);
      text(k, 1320, 340 + i * 150, { size: 19, color: C.dim });
      text(v, 1320, 396 + i * 150, { size: 44, color: col, weight: 700, font: "mono" });
    });
    text("← 這個網路自己解不開魔術方塊", 1290, 940, { size: 22, color: C.bad, weight: 600 });
  }
}

// ── 場景：weight 取捨 ──────────────────────────────────────
function sceneWeight(sc, local) {
  const rows = DEMO.weights;
  const t = clamp01((local - 18) / 55);
  text("weight 是速度與品質之間唯一的旋鈕", 960, 186, { size: 34, color: C.fg, align: "center", weight: 700 });
  text("f = weight × 已走步數 + 網路猜的剩餘步數", 960, 232, { size: 24, color: C.accent, align: "center", font: "mono" });

  const cols = [
    ["展開節點數", (r) => r.mean_nodes, (r) => fmt(Math.round(r.mean_nodes)), C.accent, false],
    ["平均步數", (r) => r.mean_len, (r) => r.mean_len.toFixed(2), C.fg, false],
    ["剛好是最短解", (r) => r.optimal_rate, (r) => `${(r.optimal_rate * 100).toFixed(1)}%`, C.good, true],
  ];
  cols.forEach(([title, get, label, color], ci) => {
    const x = 180 + ci * 570, w = 480;
    box(x, 300, w, 610);
    text(title, x + w / 2, 352, { size: 24, color: C.dim, align: "center", weight: 600 });
    const maxV = Math.max(...rows.map(get));
    rows.forEach((r, i) => {
      const y = 400 + i * 122;
      const tt = ease(clamp01(t * 1.4 - i * 0.12));
      text(`weight ${r.w.toFixed(1)}`, x + 24, y + 26, { size: 19, color: C.dim, font: "mono" });
      ctx.fillStyle = "#131d2e";
      ctx.beginPath(); ctx.roundRect(x + 24, y + 44, w - 48, 26, 6); ctx.fill();
      ctx.fillStyle = color;
      ctx.beginPath(); ctx.roundRect(x + 24, y + 44, (w - 48) * (get(r) / maxV) * tt, 26, 6); ctx.fill();
      text(label(r), x + w - 24, y + 30, { size: 24, color, align: "right", weight: 700, font: "mono" });
    });
  });
  if (t > 0.8) {
    const w1 = DEMO.weights.find((r) => r.w === 1);
    text(`weight=1 就是課本上的 A*，定理說保證最短解 — 實測 ${(w1.optimal_rate * 100).toFixed(1)}%。`
         + `差的那一點不是實作有錯，是前提沒滿足：h 有 ${(DEMO.heuristic.over_rate * 100).toFixed(1)}% 的局面高估了。`,
         960, 972, { size: 22, color: C.warn, align: "center" });
  }
}

// ── 場景：結果表 ────────────────────────────────────────────
function sceneTable(sc, local) {
  text("實測", 960, 176, { size: 34, color: C.fg, align: "center", weight: 700 });

  // 左：2x2x2（有正確答案）
  box(90, 220, 860, 700);
  text(`2×2×2 · ${fmt(DEMO.bench2.n)} 個局面，從全部 3,674,160 個裡均勻抽`, 126, 268, { size: 20, color: C.accent, weight: 600 });
  const head2 = ["", "解開率", "最短解率", "平均步數", "展開節點"];
  const colX = [126, 460, 590, 720, 860];
  head2.forEach((h, i) => text(h, colX[i] + (i ? 60 : 0), 316,
    { size: 17, color: C.dim, align: i ? "right" : "left" }));
  DEMO.bench2.rows.forEach((r, i) => {
    const y = 360 + i * 84;
    const tt = clamp01((local - 16 - i * 10) / 20);
    ctx.globalAlpha = tt;
    const hi = r.name.includes("A*");
    if (hi) { ctx.fillStyle = "rgba(56,189,248,.08)"; ctx.fillRect(108, y - 30, 824, 78); }
    text(r.name, colX[0], y, { size: 20, color: hi ? C.accent : C.fg, weight: hi ? 700 : 400 });
    const vals = [
      `${(r.solve_rate * 100).toFixed(1)}%`,
      r.optimal_rate == null ? "—" : `${(r.optimal_rate * 100).toFixed(1)}%`,
      r.mean_len == null ? "—" : r.mean_len.toFixed(2),
      fmt(Math.round(r.mean_nodes)),
    ];
    vals.forEach((v, k) => text(v, colX[k + 1] + 60, y, { size: 21, color: hi ? C.good : C.dim, align: "right", font: "mono" }));
    if (r.note) text(r.note, colX[0], y + 26, { size: 14.5, color: "#55647d" });
    ctx.globalAlpha = 1;
  });

  // 右：3x3x3（沒有正確答案）
  box(990, 220, 860, 700);
  text(`3×3×3 · 每個打亂深度 ${fmt(DEMO.bench3.n)} 個局面`, 1026, 268, { size: 20, color: C.accent, weight: 600 });
  text("沒有正確答案可比，所以只能問「解得開嗎、幾步」", 1026, 298, { size: 17, color: C.warn });
  const hx = [1026, 1420, 1610, 1820];
  ["打亂步數", "解開率", "平均步數", "展開節點"].forEach((h, i) =>
    text(h, hx[i] + (i ? 0 : 0), 342, { size: 17, color: C.dim, align: i ? "right" : "left" }));
  DEMO.bench3.by_depth.forEach((r, i) => {
    const y = 388 + i * 62;
    const tt = clamp01((local - 20 - i * 8) / 20);
    ctx.globalAlpha = tt;
    text(`${r.depth} 步`, hx[0], y, { size: 20, color: C.fg, font: "mono" });
    text(`${(r.solve_rate * 100).toFixed(0)}%`, hx[1], y,
         { size: 20, color: r.solve_rate > 0.9 ? C.good : r.solve_rate > 0.5 ? C.warn : C.bad, align: "right", font: "mono" });
    text(r.mean_len == null ? "—" : r.mean_len.toFixed(1), hx[2], y, { size: 20, color: C.dim, align: "right", font: "mono" });
    text(fmt(Math.round(r.mean_nodes)), hx[3], y, { size: 20, color: C.dim, align: "right", font: "mono" });
    ctx.globalAlpha = 1;
  });

  if (local > 150) {
    text("網路只會判斷「離終點多遠」，把方塊解開的是搜尋。", 960, 992,
         { size: 24, color: C.accent, align: "center", weight: 600 });
  }
}

// ── 主迴圈 ──────────────────────────────────────────────────
const DISPATCH = { cube: sceneCube, hist: sceneHist, bullets: sceneBullets, error: sceneError, weight: sceneWeight, table: sceneTable };

function renderFrame() {
  const sc = PLAN.scenes.find((s) => frame >= s.start && frame < s.start + s.frames)
    || PLAN.scenes[PLAN.scenes.length - 1];
  const local = frame - sc.start;
  chrome(sc, local);
  DISPATCH[sc.kind](sc, local);
  // 場景開頭淡入
  if (local < 10) {
    ctx.fillStyle = C.bg;
    ctx.globalAlpha = 1 - local / 10;
    ctx.fillRect(0, 0, W, H);
    ctx.globalAlpha = 1;
  }
  frame++;
}

window.__rec = {
  init() { PLAN = buildPlan(); frame = 0; return PLAN; },
  renderFrame,
  seek(f) { frame = f; },
};
