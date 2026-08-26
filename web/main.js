// 頁面的流程：載入、動畫、按鈕。規則在 cube.js，推論在 nn.js，搜尋在 search.js。

import { Cube } from "./cube.js";
import { ValueNet } from "./nn.js";
import { CubeView, drawNet } from "./render.js";
import { solve, greedyStep } from "./search.js";
import { exactSolve } from "./exact.js";

const $ = (id) => document.getElementById(id);
const stage = $("stage");
const ctx = stage.getContext("2d");

const CFG = await (await fetch("../shared/config.json", { cache: "no-store" })).json();
const COLORS = CFG.view.colors;

const app = {
  size: 2, cube: null, net: null, view: null,
  state: null, history: [],
  queue: [], anim: null, busy: false,
  h: null, exact: null, lastSeq: null,
};

// ── 載入一個尺寸 ─────────────────────────────────────────────
async function loadSize(size) {
  app.busy = true;
  app.size = size;
  app.cube = await Cube.load("../shared/moves.json", size);
  app.view = new CubeView(app.cube, COLORS);
  app.state = app.cube.solved();
  app.history = []; app.queue = []; app.anim = null; app.lastSeq = null;
  $("size2").classList.toggle("active", size === 2);
  $("size3").classList.toggle("active", size === 3);

  const max = CFG.sizes[String(size)].scrambleMax;
  $("depth").max = max;
  $("depth").value = Math.min(Number($("depth").value), max);
  onDepth();
  $("sizeNote").textContent = size === 2
    ? "固定 DBL 那顆角，只轉 U / R / F，共 3,674,160 個局面"
    : "六面都能轉，4.3 × 10¹⁹ 個局面";

  buildMoveButtons();
  app.net = null;
  $("netInfo").textContent = "載入權重…";
  try {
    app.net = await ValueNet.load(`policy-${size}x${size}.json`);
    const e = app.net.eval || {};
    const bits = [`${app.net.nParams.toLocaleString()} 參數`,
                  `練了 ${app.net.trainedIters.toLocaleString()} 輪`];
    if (e.mae !== undefined) bits.push(`平均差 ${Number(e.mae).toFixed(2)} 步`);
    $("netInfo").textContent = bits.join(" · ");
    $("netInfo").className = "note ok";
  } catch (err) {
    $("netInfo").textContent = `還沒有 policy-${size}x${size}.json — 先跑 python ml/export_policy.py --size ${size}`;
    $("netInfo").className = "note warn";
  }
  $("btnSolve").disabled = $("btnGreedy").disabled = !app.net;
  app.busy = false;
  refresh();
}

function buildMoveButtons() {
  const box = $("moveBtns");
  box.innerHTML = "";
  app.cube.moveNames.forEach((name, i) => {
    const b = document.createElement("button");
    b.className = "mv";
    b.textContent = name;
    b.onclick = () => { if (!app.busy) { push(i); } };
    box.appendChild(b);
  });
}

// ── 動畫佇列 ─────────────────────────────────────────────────
// 每個動作畫成 180ms 的轉動。動畫本身不改狀態，
// 轉完才把置換套上去——所以「畫面」跟「狀態」永遠是同一件事的兩個時間點，不會漂移。
const TURN_MS = 180;

function push(move, record = true) {
  if (record) app.history.push(move);
  app.queue.push(move);
}

function undo() {
  if (app.busy || !app.history.length) return;
  const m = app.history.pop();
  app.queue.push(app.cube.inverseMove[m]);
}

function stepAnim(now) {
  if (!app.anim && app.queue.length) {
    const m = app.queue.shift();
    const name = app.cube.moveNames[m];
    const face = name[0];
    const ccw = name.endsWith("'");
    app.anim = { move: m, face, dir: ccw ? 1 : -1, t0: now };
  }
  if (app.anim) {
    const p = Math.min(1, (now - app.anim.t0) / TURN_MS);
    // ease-in-out，看起來比等速自然
    const e = p < 0.5 ? 2 * p * p : 1 - 2 * (1 - p) * (1 - p);
    app.anim.angle = app.anim.dir * e * Math.PI / 2;
    if (p >= 1) {
      app.state = app.cube.apply(app.state, app.anim.move);
      app.anim = null;
      if (!app.queue.length) refresh();
    }
  }
}

// ── 畫面 ─────────────────────────────────────────────────────
function draw() {
  ctx.fillStyle = "#070b14";
  ctx.fillRect(0, 0, stage.width, stage.height);

  const scale = app.size === 2 ? 62 : 46;
  app.view.draw(ctx, app.state, {
    cx: stage.width / 2, cy: 248, scale,
    anim: app.anim ? { face: app.anim.face, angle: app.anim.angle } : null,
  });

  // 攤平圖：把背面三片也顯示出來，不然「解開了沒」用看的不準
  const cell = app.size === 2 ? 22 : 17;
  const fw = app.cube.n * cell + 2;
  drawNet(ctx, app.cube, app.state, COLORS, stage.width / 2 - 2 * fw, 452, cell);

  ctx.fillStyle = "#64748b";
  ctx.font = "12px ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.fillText("攤平圖（六面都看得到）", stage.width / 2, 444);

  if (app.cube.isSolved(app.state) && !app.anim) {
    ctx.fillStyle = "#4ade80";
    ctx.font = "600 15px system-ui";
    ctx.fillText("解開了", stage.width / 2, 636);
  }
  requestAnimationFrame((t) => { stepAnim(t); draw(); });
}

// ── 每次狀態穩定下來就重算兩個數字 ───────────────────────────
function refresh() {
  const solvedNow = app.cube.isSolved(app.state);
  app.h = app.net ? app.net.value(app.cube, app.state) : null;
  $("hudH").textContent = app.h === null ? "–" : (solvedNow ? "0.00" : app.h.toFixed(2));

  if (app.size === 2) {
    const r = exactSolve(app.cube, app.state);
    app.exact = r ? r.dist : null;
    $("hudExact").textContent = app.exact === null ? "?" : `${app.exact} 步`;
    $("hudExact").style.fontSize = "";
    if (app.h !== null && app.exact !== null) {
      const d = (solvedNow ? 0 : app.h) - app.exact;
      $("hudErr").textContent = (d >= 0 ? "+" : "") + d.toFixed(2);
      $("hudErr").style.color = Math.abs(d) <= 1 ? "var(--good)" : "var(--warn)";
    } else $("hudErr").textContent = "–";
  } else {
    app.exact = null;
    $("hudExact").textContent = "算不出來";
    $("hudExact").style.fontSize = "13px";
    $("hudErr").textContent = "–";
  }
}

// ── 按鈕 ─────────────────────────────────────────────────────
function onDepth() { $("depthLabel").textContent = `${$("depth").value} 步`; }
$("depth").oninput = onDepth;
$("weight").oninput = () => { $("weightLabel").textContent = (Number($("weight").value) / 10).toFixed(1); };

$("size2").onclick = () => loadSize(2);
$("size3").onclick = () => loadSize(3);
$("btnUndo").onclick = undo;

$("btnReset").onclick = () => {
  if (app.busy) return;
  app.state = app.cube.solved();
  app.history = []; app.queue = []; app.lastSeq = null;
  $("seq").textContent = "";
  $("searchInfo").style.display = "none";
  refresh();
};

$("btnScramble").onclick = () => {
  if (app.busy) return;
  const d = Number($("depth").value);
  app.state = app.cube.solved();
  app.history = []; app.queue = [];
  const { seq } = app.cube.scramble(d);
  seq.forEach((m) => push(m));
  $("seq").innerHTML = `打亂 ${d} 步：${app.cube.toStr(seq)}`;
  $("searchInfo").style.display = "none";
};

$("btnGreedy").onclick = () => {
  if (app.busy || !app.net) return;
  const { move, values } = greedyStep(app.cube, app.net, app.state);
  push(move);
  const parts = app.cube.moveNames.map((n, i) =>
    `${n} ${values[i] < 0 ? "解開" : values[i].toFixed(2)}`);
  $("seq").innerHTML = `每個轉法之後網路猜還要幾步 → ${parts.join("　")}<br>選了 <b>${app.cube.moveNames[move]}</b>`;
};

$("btnSolve").onclick = async () => {
  if (app.busy || !app.net) return;
  if (app.cube.isSolved(app.state)) { $("seq").textContent = "已經是解開的了"; return; }
  app.busy = true;
  $("btnSolve").disabled = true;
  const info = $("searchInfo");
  info.style.display = "block";
  info.className = "note ok";
  const w = Number($("weight").value) / 10;
  const cfg = CFG.sizes[String(app.size)].search;
  const r = await solve(app.cube, app.net, app.state, {
    weight: w,
    // 2x2x2 用 batch=1（就是課本上的 A*）：批次一大就會把 weight 的效果吃掉，
    // 而且節點數會多十幾倍。3x3x3 的樹夠大，批次才有意義。
    batch: app.size === 2 ? 1 : 20,
    // 3x3x3 的網路有 297 萬個參數，純 JS 跑一個局面約 5 毫秒。
    // 打亂 8 步左右（幾百個節點）在瀏覽器裡還等得起，30 步要三萬個節點——那要半小時。
    // 所以這裡給一個會在合理時間內放棄的預算，深的局面交給 GPU（ml/benchmark.py）。
    maxNodes: app.size === 2 ? 60000 : 600,
    onProgress: ({ expanded }) => { info.textContent = `搜尋中… 已展開 ${expanded.toLocaleString()} 個節點`; },
  });
  app.busy = false;
  $("btnSolve").disabled = false;

  if (!r.seq) {
    info.className = "note warn";
    info.textContent = `展開 ${r.expanded.toLocaleString()} 個節點還是沒找到（${(r.ms / 1000).toFixed(1)}s）。`
      + (app.size === 3
        ? "3×3×3 的網路有 297 萬個參數，純 JS 一個局面約 5 毫秒——打亂 8 步以內在這裡解得動，"
          + "30 步要展開三萬個節點，那是半小時。深的局面交給 GPU：python ml/benchmark.py --size 3。"
          + "不過上面那個「網路猜還要幾步」是即時的，你可以自己轉幾下看它怎麼變。"
        : "");
    return;
  }
  const line = [`A*（weight=${w}）解開：<b>${r.seq.length} 步</b>`,
                `展開 ${r.expanded.toLocaleString()} 個節點`,
                `${r.ms.toFixed(0)} ms`];
  if (app.size === 2 && app.exact !== null) {
    const extra = r.seq.length - app.exact;
    line.splice(1, 0, extra === 0 ? "<b>就是最短解</b>" : `比最短解多 ${extra} 步`);
  }
  info.textContent = "";
  info.style.display = "none";
  $("seq").innerHTML = `${line.filter(Boolean).join(" · ")}<br>${app.cube.toStr(r.seq)}`;
  r.seq.forEach((m) => push(m, false));
  app.history = [];
};

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "r" || e.key === "R") $("btnReset").click();
  if (e.key === "s" || e.key === "S") $("btnScramble").click();
  if (e.key === " ") { e.preventDefault(); $("btnSolve").click(); }
  if (e.key === "u" || e.key === "U") undo();
});

await loadSize(2);
draw();
