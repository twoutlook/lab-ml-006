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

let bad = 0, t0 = Date.now(), worst = 0, nodes = 0;
for (const [i, c] of cases.entries()) {
  const st = Uint8Array.from(c.state);
  const r = sv.solve(sv.index(st));
  if (!r) { console.error(`#${i} 找不到解`); bad++; continue; }
  if (r.dist !== c.dist) { console.error(`#${i} 距離不符：JS ${r.dist} vs Python ${c.dist}`); bad++; }
  let cur = st;
  for (const m of r.seq) cur = cube.apply(cur, m);
  if (sv.index(cur) !== sv.solved) { console.error(`#${i} 回傳的解沒把角塊轉回去`); bad++; }
  worst = Math.max(worst, r.dist); nodes += r.expanded;
}
const ms = (Date.now() - t0) / cases.length;
console.log(`角塊距離對帳：${cases.length - bad}/${cases.length} 與 Python 的表完全相同`);
console.log(`  最遠 ${worst} 步 · 平均展開 ${Math.round(nodes / cases.length).toLocaleString()} 個狀態 · ${ms.toFixed(0)} ms/題`);
process.exit(bad ? 1 : 0);
