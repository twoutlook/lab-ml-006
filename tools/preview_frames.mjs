/** 抽幾格關鍵畫面存成 PNG，用來檢查版面，不用等整支影片錄完。
 *    node tools/preview_frames.mjs 20 900 1800 3200
 *  加 --page 就改成截互動頁（web/index.html），順便把 console 錯誤印出來。
 *    node tools/preview_frames.mjs --page
 */
import puppeteer from "puppeteer-core";
import { mkdirSync, existsSync, writeFileSync } from "node:fs";
import path from "node:path";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, "$1"), "..");
const OUT = path.join(ROOT, "out", "preview");
const CHROME = ["C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"].find(existsSync);

const args = process.argv.slice(2);
const pageMode = args.includes("--page");
const targets = args.map(Number).filter((n) => Number.isFinite(n));
mkdirSync(OUT, { recursive: true });

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true,
  args: ["--no-sandbox", "--font-render-hinting=none", "--hide-scrollbars"] });
const page = await browser.newPage();
page.on("pageerror", (e) => console.error("PAGE ERROR:", e.message));
page.on("console", (m) => console.log(`console.${m.type()}:`, m.text()));

if (pageMode) {
  await page.setViewport({ width: 1280, height: 1400, deviceScaleFactor: 1 });
  await page.goto("http://localhost:8000/web/", { waitUntil: "networkidle0", timeout: 60000 });
  await new Promise((r) => setTimeout(r, 1200));
  const shots = [["idle", null]];
  for (const [name, fn] of shots) {
    if (fn) await fn();
    const f = path.join(OUT, `page_${name}.png`);
    writeFileSync(f, await page.screenshot({ type: "png" }));
    console.log(f);
  }
  // 打亂 -> 讓網路解，看整條流程會不會爆
  await page.click("#btnScramble");
  await new Promise((r) => setTimeout(r, 3000));
  writeFileSync(path.join(OUT, "page_scrambled.png"), await page.screenshot({ type: "png" }));
  const hud = await page.evaluate(() => ({
    h: document.getElementById("hudH").textContent,
    exact: document.getElementById("hudExact").textContent,
    err: document.getElementById("hudErr").textContent,
    seq: document.getElementById("seq").textContent,
  }));
  console.log("打亂後：", hud);
  await page.click("#btnSolve");
  await new Promise((r) => setTimeout(r, 6000));
  writeFileSync(path.join(OUT, "page_solving.png"), await page.screenshot({ type: "png" }));
  console.log("解完：", await page.evaluate(() => document.getElementById("seq").textContent));
  await new Promise((r) => setTimeout(r, 6000));
  writeFileSync(path.join(OUT, "page_solved.png"), await page.screenshot({ type: "png" }));
  console.log(OUT);
  await browser.close();
} else {
  await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
  await page.goto("http://localhost:8000/web/record.html", { waitUntil: "networkidle0" });
  await page.waitForFunction("window.__rec !== undefined");
  const plan = await page.evaluate(() => window.__rec.init());
  const max = Math.max(...targets);
  for (let i = 0; i <= max; i++) {
    const url = await page.evaluate((want) => {
      window.__rec.renderFrame();
      return want ? document.getElementById("stage").toDataURL("image/png") : null;
    }, targets.includes(i));
    if (url) {
      const sc = plan.scenes.find((s) => i >= s.start && i < s.start + s.frames);
      const f = path.join(OUT, `f${String(i).padStart(5, "0")}_${sc ? sc.id : "end"}.png`);
      writeFileSync(f, Buffer.from(url.slice(url.indexOf(",") + 1), "base64"));
      console.log(f);
    }
  }
  await browser.close();
}
