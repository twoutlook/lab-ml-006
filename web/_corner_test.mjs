/** 驗證瀏覽器端的角塊求解器：距離要跟 Python 那張 88 MB 的表逐一相同，
 *  而且回傳的解轉回去之後角塊必須真的歸位。
 *    python ml/export_corner_tables.py && node web/_corner_test.mjs */
import { readFileSync } from "node:fs";
import path from "node:path";
const HERE = path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, "$1");
globalThis.atob = (s) => Buffer.from(s, "base64").toString("binary");
const { CornerSolver } = await import("./corners3.js");
const { Cube } = await import("./cube.js");

const spec = JSON.parse(readFileSync(path.join(HERE, "corner_tables.json"), "utf8"));
const moves = JSON.parse(readFileSync(path.join(HERE, "..", "shared", "moves.json"), "utf8"));
const cases = JSON.parse(readFileSync(path.join(HERE, "_corner_cases.json"), "utf8")).cases;
const cube = new Cube(moves["3"]);
const sv = new CornerSolver(spec);

let bad = 0, worst = 0, nodes = 0;
const byDist = new Map();   // 每個距離各自統計耗時，看最壞情況多久
for (const [i, c] of cases.entries()) {
  const st = Uint8Array.from(c.state);
  const t1 = performance.now();
  const r = sv.solve(sv.index(st));
  const dt = performance.now() - t1;
  const b = byDist.get(c.dist) || { n: 0, ms: 0, nodes: 0 };
  b.n++; b.ms += dt; b.nodes += r ? r.expanded : 0; byDist.set(c.dist, b);
  if (!r) { console.error(`#${i} 找不到解`); bad++; continue; }
  if (r.dist !== c.dist) { console.error(`#${i} 距離不符：JS ${r.dist} vs Python ${c.dist}`); bad++; }
  let cur = st;
  for (const m of r.seq) cur = cube.apply(cur, m);
  if (sv.index(cur) !== sv.solved) { console.error(`#${i} 回傳的解沒把角塊轉回去`); bad++; }
  worst = Math.max(worst, r.dist); nodes += r.expanded;
}
console.log(`角塊距離對帳：${cases.length - bad}/${cases.length} 與 Python 的表完全相同`);
console.log(`  最遠 ${worst} 步（角塊的上帝之數就是 14）`);
console.log(`
  距離   題數      平均展開狀態      平均耗時`);
for (const d of [...byDist.keys()].sort((a, b2) => a - b2)) {
  const b = byDist.get(d);
  console.log(`  ${String(d).padStart(4)} ${String(b.n).padStart(6)} ${Math.round(b.nodes / b.n).toLocaleString().padStart(16)} ${(b.ms / b.n).toFixed(0).padStart(10)} ms`);
}
process.exit(bad ? 1 : 0);
