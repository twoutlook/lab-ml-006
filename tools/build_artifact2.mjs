/** GOAL002 的圖文版：把「先解角」灌成單一 HTML。
 *    node tools/build_artifact2.mjs
 *  輸出 out/artifact2.html
 *
 * 設計系統不重寫一份——直接把第一份 artifact 的 <style> 抽出來注入。
 * 兩頁是同一個系列，配色與版式本來就該一致；抄一份的話遲早會漂掉。
 */
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import path from "node:path";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, "$1"), "..");
const rd = (p) => readFileSync(path.join(ROOT, p), "utf8");
const rj = (p) => JSON.parse(rd(p));

const MODULES = ["web/cube.js", "web/render.js", "web/corners3.js"];
const strip = (src) => src.split("\n").filter((l) => !/^\s*import\s/.test(l)).join("\n")
  .replace(/^export\s+/gm, "");

const moves = rj("shared/moves.json");
const g2 = rj("ml/checkpoints/goal2.json");
const d2 = rj("web/demo2.json");
const b3 = rj("ml/checkpoints/benchmark-3x3.json");
const g3 = rj("out/goal3.json");          // 三張表的值分佈 + IDA* 尺度
const s2 = rj("out/stage2.json");         // 先解角繞了多遠

// ── 共用的設計系統：從第一份 artifact 抽 <style> ──
const first = rd("tools/artifact_template.html");
const style = first.slice(first.indexOf("<style>"), first.indexOf("</style>") + 8);
if (!style.startsWith("<style>")) throw new Error("抽不到第一份 artifact 的 <style>");

const engine = [
  `const MOVES3 = ${JSON.stringify(moves["3"])};`,
  // 角塊的兩張座標移動表（約 1.4 MB）。有了它們，瀏覽器就能用雙向 BFS
  // 現算出精確的角塊距離——不必搬那張 88 MB 的表。
  `const CORNER_TABLES = ${rd("web/corner_tables.json")};`,
  ...MODULES.map((m) => `\n// ──────── ${m} ────────\n${strip(rd(m))}`),
].join("\n");

// E2 的原始紀錄有上百筆，頁面只需要彙總後的剖析，別把明細灌進去
const DATA = {
  pdb: g2.corner_pdb,
  e1: g2.e1,
  e2: { n: g2.e2.n, mae: g2.e2.mae, bias: g2.e2.bias, over_rate: g2.e2.over_rate,
        within_1: g2.e2.within_1, by_true_net: g2.e2.by_true_net,
        by_true_pdb: g2.e2.by_true_pdb,
        by_depth: [...new Set(g2.e2.records.map((r) => r.depth))].sort((a, b) => a - b)
          .map((d) => {
            const g = g2.e2.records.filter((r) => r.depth === d);
            return { depth: d, n: g.length,
                     mean_true: +(g.reduce((s, r) => s + r.true, 0) / g.length).toFixed(2),
                     mean_nodes: Math.round(g.reduce((s, r) => s + r.nodes, 0) / g.length) };
          }) },
  e3: { n: g2.e3.n, mean_lower: g2.e3.mean_lower, mean_found: g2.e3.mean_found,
        mean_gap_upper: g2.e3.mean_gap_upper },
  e4: g2.e4,
  net: g2.net,
  bench3: { by_depth: b3.by_depth, n: b3.n },
  ida: d2.idaVsAstar,
  // 互動台要用的：示範方塊的兩種解法，以及沿路的估計值
  demo: { scramble: d2.scramble, oneShot: d2.oneShot, staged: d2.staged,
          cornerStickers: d2.cornerStickers },
  // 第二階段（邊）：三張表的值分佈、IDA* 節點數怎麼隨深度長、先解角的繞路
  edges: { bounds: g3.bounds, scale: g3.scale, build: rj("ml/checkpoints/edges-build.json") },
  stage2: s2,
};

const CJK = "⺀-鿿豈-﫿＀-｠￠-￦　-〿";
function joinCJK(html) {
  const TAG = "(?:<\\/?(?:strong|em|b|i|span|code|a)\\b[^>]*>)*";
  const re = new RegExp("([" + CJK + "]" + TAG + ")[\\r\\n]+[ \\t]*(" + TAG + "[" + CJK + "])", "g");
  let prev;
  do { prev = html; html = html.replace(re, "$1$2"); } while (html !== prev);
  return html;
}

let tpl = rd("tools/artifact2_template.html");
{
  const cut = tpl.indexOf("<script>");
  tpl = joinCJK(tpl.slice(0, cut)) + tpl.slice(cut);
}
let yt = null;
try { yt = rj("out/youtube2.json"); } catch { /* 還沒上傳 */ }
let first_url = null;
try { first_url = rj("out/artifact1.json").url; } catch { /* 還沒填 */ }

tpl = tpl.replace("/*__STYLE__*/", "").replace("<!--STYLE-->", style);
if (yt && yt.video_url) tpl = tpl.replaceAll("__YT_URL__", yt.video_url);
else tpl = tpl.replace(/<!--YT_START-->[\s\S]*?<!--YT_END-->/g, "");
if (first_url) tpl = tpl.replaceAll("__ARTIFACT1_URL__", first_url);

const out = tpl
  .replace("/*__DATA__*/", JSON.stringify(DATA))
  .replace("/*__ENGINE__*/", engine);

mkdirSync(path.join(ROOT, "out"), { recursive: true });
const dst = path.join(ROOT, "out", "artifact2.html");
writeFileSync(dst, out, "utf8");
const mb = Buffer.byteLength(out) / 1e6;
console.log(`${dst}  ${mb.toFixed(2)} MB${mb > 16 ? "  ← 超過 16 MB 上限！" : ""}`);
console.log(`  角塊表 ${DATA.pdb.states.toLocaleString()} 狀態 · E2 精確答案 ${DATA.e2.n} 筆`);
console.log(`  影片連結 ${yt && yt.video_url ? yt.video_url : "（無，已移除該區塊）"}`);
console.log(`  上一篇連結 ${first_url || "（無）"}`);
