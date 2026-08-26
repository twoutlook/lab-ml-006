/**
 * 跨語言對帳：JS 這邊的引擎與推論，必須跟 Python 那邊一致。
 *
 *   python ml/_parity_dump.py --size 2
 *   node web/_parity_test.mjs 2
 *
 * 這個專案的置換表本來就只有一份（shared/moves.json），
 * 所以「規則不一致」的風險比前幾個專案低——但「用錯那張表」還是會發生：
 * 例如把 new[j] = old[perm[j]] 寫反，四步一輪的動作照樣是四步一輪，
 * 打亂再倒著轉回去也照樣會回到原狀，測不出來。只有逐格比對抓得到。
 *
 * 推論那半更重要：nn.js 只要層序或殘差接錯一個地方，頁面照樣跑，只是解得爛。
 * 基準是「匯出後那份 policy.json 算出來的值」，不是原始 PyTorch 網路——
 * 摺 BatchNorm 與四捨五入本來就有約 0.006 步的誤差，混進來就分不出是哪一種錯了。
 */
import { readFileSync } from "node:fs";
import path from "node:path";

const HERE = path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, "$1");
const size = Number(process.argv[2] || 2);

// node 沒有 fetch 本機檔案的能力，用同樣的資料手動組出 Cube / ValueNet
const { Cube } = await import("./cube.js");
const { ValueNet } = await import("./nn.js");

const moves = JSON.parse(readFileSync(path.join(HERE, "..", "shared", "moves.json"), "utf8"));
const cube = new Cube(moves[String(size)]);

const casesPath = path.join(HERE, `_parity_cases-${size}x${size}.json`);
const data = JSON.parse(readFileSync(casesPath, "utf8"));

let bad = 0;
for (let i = 0; i < data.cases.length; i++) {
  const { moves: seq, state } = data.cases[i];
  let s = cube.solved();
  for (const m of seq) s = cube.apply(s, m);
  for (let j = 0; j < s.length; j++) {
    if (s[j] !== state[j]) {
      console.error(`案例 ${i} 第 ${j} 片貼紙不一致：JS ${s[j]} vs Python ${state[j]}`);
      console.error(`  動作序列 ${cube.toStr(seq)}`);
      bad++;
      break;
    }
  }
}
console.log(`轉動對帳：${data.cases.length - bad}/${data.cases.length} 逐格相同`);

if (data.values) {
  const spec = JSON.parse(readFileSync(path.join(HERE, `policy-${size}x${size}.json`), "utf8"));
  const net = new ValueNet(spec);
  let maxErr = 0;
  for (let i = 0; i < data.cases.length; i++) {
    const s = Uint8Array.from(data.cases[i].state);
    const v = net.value(cube, s);
    maxErr = Math.max(maxErr, Math.abs(v - data.values[i]));
  }
  const ok = maxErr < 1e-5;
  console.log(`推論對帳：最大誤差 ${maxErr.toExponential(2)} 步  ${ok ? "✓" : "✗ 太大了"}`);
  if (!ok) bad++;
} else {
  console.log("推論對帳：跳過（_parity_cases 裡沒有 values）");
}

// 順便驗一次雙向 BFS：它算出來的最短解，轉回去必須真的解開
if (size === 2) {
  const { exactSolve } = await import("./exact.js");
  let worst = 0, checked = 0;
  for (const c of data.cases.slice(0, 12)) {
    const s = Uint8Array.from(c.state);
    const r = exactSolve(cube, s);
    let cur = s;
    for (const m of r.seq) cur = cube.apply(cur, m);
    if (!cube.isSolved(cur)) { console.error("雙向 BFS 給的解轉不回去！"); bad++; }
    worst = Math.max(worst, r.dist);
    checked++;
  }
  console.log(`雙向 BFS：${checked} 題都解得開，最遠 ${worst} 步（上帝之數 14）`);
}

process.exit(bad ? 1 : 0);
