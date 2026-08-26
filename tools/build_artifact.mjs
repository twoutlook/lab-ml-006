/** 把引擎、權重、實測數字全部灌進 artifact 樣板，產出單一 HTML。
 *    node tools/build_artifact.mjs
 *  輸出 out/artifact.html
 *
 * artifact 是一個檔案，所以 web/ 底下那五個 ES module 要串成一段。
 * 串法是把 `import ...` 整行刪掉、把 `export ` 前綴拿掉，然後照相依順序接起來。
 * 這代表模組之間不能有同名的頂層宣告——render.js 的 NET_ORDER 和 exact.js 的 ekey
 * 就是為了這件事改名的（各自單獨跑的時候本來沒問題）。
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, "$1"), "..");
const rd = (p) => readFileSync(path.join(ROOT, p), "utf8");
const rj = (p) => JSON.parse(rd(p));

// 相依順序：cube -> nn -> render -> search -> exact
const MODULES = ["web/cube.js", "web/nn.js", "web/render.js", "web/search.js", "web/exact.js"];
const strip = (src) => src
  .split("\n")
  .filter((l) => !/^\s*import\s/.test(l))
  .join("\n")
  .replace(/^export\s+/gm, "");

const moves = rj("shared/moves.json");
const policy = rj("web/policy-2x2.json");
const demo = rj("web/demo.json");
const b2 = rj("ml/checkpoints/benchmark-2x2.json");

const engine = [
  `// ── 轉動表（shared/moves.json 的 2x2x2 那份，由 ml/gen_moves.py 從幾何算出來）──`,
  `const MOVES2 = ${JSON.stringify(moves["2"])};`,
  `const MOVES3 = ${JSON.stringify(moves["3"])};`,
  ...MODULES.map((m) => `\n// ──────── ${m} ────────\n${strip(rd(m))}`),
].join("\n");

const DATA = {
  hist: demo.hist,
  total2: demo.total2,
  heuristic: demo.heuristic,
  greedy2: demo.greedy2,
  weights: demo.weights,
  bench2: { n: demo.bench2.n, rows: demo.bench2.rows, truth_mean: b2.truth_mean,
            batch: b2.rows.find((r) => r.batch)?.batch ?? null },
  bench3: demo.bench3,
  randomStats: demo.randomStats,
  batchAblation: b2.batch_ablation,
};

// 3x3x3 那個沙盒用的資料。權重 21.88 MB 塞不進 artifact 的 16 MB，
// 所以頁面上不做 3x3x3 的推論——只放一顆先在 GPU 上解好的方塊給它重播，
// 連同沿路每一站網路猜幾步（也是先算好的）。幾 KB 而已。
{
  const c3 = demo.cases["3"][0];
  const run3 = c3.runs[demo.defaultWeight["3"]];
  DATA.demo3 = {
    scramble: c3.scramble,
    solution: run3.seq,
    h: c3.hScramble.concat(run3.h),
    nodes: run3.nodes,
    ms: run3.ms,
    weight: Number(demo.defaultWeight["3"]),
  };
}

// HTML 原始碼為了好讀而換的行，落在兩個全形字之間會被瀏覽器算成一個空格
// （「…網路猜的；\n      橘色帶子…」會變成「網路猜的； 橘色帶子」）。
// Chrome 只在特定條件下才會把這種段落換行吃掉，實測是吃不掉的。
// 與其把整段中文寫成一行讓原始碼難讀，不如在這裡處理掉——
// 只動 <script> 之前的部分，免得改到 JS 裡的字串。
const CJK = "\u2E80-\u9FFF\uF900-\uFAFF\uFF00-\uFF60\uFFE0-\uFFE6\u3000-\u303F";
function joinCJK(html) {
  // 用一般字串串接，不用樣板字面值——樣板字面值裡的 \n \s 會被當成跳脫序列處理掉，
  // 組出來的規則式會變成「換行後接零個以上的字母 s」，看起來對但完全不會命中。
  // 換行的兩側可能夾著行內標籤（…幾步</strong>\n    真正…），那些也要一起吃掉
  const TAG = "(?:<\\/?(?:strong|em|b|i|span|code|a)\\b[^>]*>)*";
  const re = new RegExp("([" + CJK + "]" + TAG + ")[\\r\\n]+[ \\t]*(" + TAG + "[" + CJK + "])", "g");
  let prev;
  do { prev = html; html = html.replace(re, "$1$2"); } while (html !== prev);
  return html;
}

let tpl = rd("tools/artifact_template.html");
{
  const cut = tpl.indexOf("<script>");
  tpl = joinCJK(tpl.slice(0, cut)) + tpl.slice(cut);
}
let yt = null;
try { yt = rj("out/youtube.json"); } catch { /* 還沒上傳 */ }
if (yt && yt.video_url) {
  tpl = tpl.replaceAll("__YT_URL__", yt.video_url);
} else {
  // 沒有影片就把那兩段整個拿掉，不要留一個指向 __YT_URL__ 的死連結
  tpl = tpl.replace(/<!--YT_START-->[\s\S]*?<!--YT_END-->/, "")
           .replace(/<!--YT_START2-->[\s\S]*?<!--YT_END2-->/, "");
}

const out = tpl
  .replace("/*__DATA__*/", JSON.stringify(DATA))
  .replace("/*__POLICY__*/", JSON.stringify(policy))
  .replace("/*__ENGINE__*/", engine);

mkdirSync(path.join(ROOT, "out"), { recursive: true });
const dst = path.join(ROOT, "out", "artifact.html");
writeFileSync(dst, out, "utf8");
const mb = Buffer.byteLength(out) / 1e6;
console.log(`${dst}  ${mb.toFixed(2)} MB${mb > 16 ? "  ← 超過 artifact 的 16 MB 上限！" : ""}`);
console.log(`  權重 ${policy.n_params.toLocaleString()} 參數 · 練了 ${policy.trained_iters.toLocaleString()} 輪`);
console.log(`  2x2x2 實測 ${DATA.bench2.n} 局（batch=${DATA.bench2.batch}）· 3x3x3 每深度 ${DATA.bench3.n} 局`);
console.log(`  影片連結 ${yt && yt.video_url ? yt.video_url : "（無，已移除該區塊）"}`);
