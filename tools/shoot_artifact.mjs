/** 把 out/artifact.html 開起來截圖，順便把 console 的錯誤印出來。
 *    node tools/shoot_artifact.mjs            # 中文全頁 + 幾張局部
 *    node tools/shoot_artifact.mjs --en       # 英文版
 *  輸出 out/preview/artifact*.png
 */
import puppeteer from "puppeteer-core";
import { mkdirSync, existsSync, writeFileSync } from "node:fs";
import path from "node:path";

const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, "$1"), "..");
const OUT = path.join(ROOT, "out", "preview");
const CHROME = ["C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"].find(existsSync);
const en = process.argv.includes("--en");
const dark = process.argv.includes("--dark");
mkdirSync(OUT, { recursive: true });

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true,
  args: ["--no-sandbox", "--font-render-hinting=none", "--hide-scrollbars"] });
const page = await browser.newPage();
const errs = [];
page.on("pageerror", (e) => errs.push("PAGE ERROR: " + e.message));
page.on("console", (m) => { if (m.type() === "error") errs.push("console: " + m.text()); });
await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: dark ? "dark" : "light" }]);
await page.setViewport({ width: 1280, height: 1000, deviceScaleFactor: 1 });
// artifact 是被包在 <body> 裡送出去的，本機預覽要自己補外殼
await page.goto("http://127.0.0.1:8031/out/artifact.html", { waitUntil: "networkidle0", timeout: 90000 });
if (en) await page.click("#le");
await new Promise((r) => setTimeout(r, 1500));

const tag = `${en ? "en" : "zh"}${dark ? "-dark" : ""}`;
const full = path.join(OUT, `artifact-${tag}.png`);
writeFileSync(full, await page.screenshot({ type: "png", fullPage: true }));
console.log(full);

// 互動台：打亂 -> 解
await page.click("#bScr");
await new Promise((r) => setTimeout(r, 3500));
const hud = await page.evaluate(() => ({
  h: document.getElementById("rH").textContent,
  e: document.getElementById("rE").textContent,
  s: document.getElementById("status").textContent,
}));
console.log("打亂後：", hud);
await page.click("#bSolve");
await new Promise((r) => setTimeout(r, 8000));
console.log("解完：", await page.evaluate(() => document.getElementById("status").textContent));
const bench = await page.$("#cubewrap");
if (bench) writeFileSync(path.join(OUT, `artifact-bench-${tag}.png`),
  await (await page.$(".bench")).screenshot({ type: "png" }));

if (errs.length) { console.log("\n--- 錯誤 ---"); errs.forEach((e) => console.log(e)); }
else console.log("\n沒有 console 錯誤");
await browser.close();
