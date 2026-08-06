// Gate harness: boot the compiled artifact headless, assert the
// computable properties, and screenshot every character in idle AND walk
// — the renders are the review surface (webkit: six real defects were
// invisible in markup and found only by looking).
//
// Hard gates (non-compensable):
//   BOOTS            zero console errors, WebGL context up, frames advance
//   ON_THE_FLOOR     ground clearance ~0 (nothing floating, nothing buried)
//   ANIMATION_PLAYS  pose samples differ across frames (idle AND walk)
//   BUDGET           triangle count within the chunky-primitive budget
//   RENDERS_CONTENT  screenshot is neither blank nor background-only
import { chromium } from 'playwright-core';
import { mkdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const ARTIFACT = resolve(process.env.OUT_HTML ?? 'dist/forge.html');
const specs = JSON.parse(readFileSync(process.env.SPECS_FILE ?? 'specs/reference.json', 'utf8'));
mkdirSync('dist/shots', { recursive: true });

const browser = await chromium.launch({ channel: 'msedge', args: ['--use-angle=swiftshader'] });
const page = await browser.newPage({ viewport: { width: 900, height: 900 } });
const consoleErrors = [];
page.on('console', (m) => { if (m.type() === 'error') consoleErrors.push(m.text()); });
page.on('pageerror', (e) => consoleErrors.push(String(e)));

await page.goto(`file://${ARTIFACT}`);
await page.waitForTimeout(1200);

let failures = 0;
const fail = (gate, detail) => { failures += 1; console.log(`  FAIL ${gate}: ${detail}`); };
const pass = (gate, detail = '') => console.log(`  pass ${gate} ${detail}`);

for (let i = 0; i < specs.length; i++) {
  console.log(`\n== ${specs[i].name} (${specs[i].archetype}) ==`);
  await page.evaluate((idx) => {
    document.querySelectorAll('#picker button')[idx].click();
  }, i);
  await page.waitForTimeout(400);

  const m = await page.evaluate(() => window.__forge.measure);

  // BOOTS: frames advancing.
  const f0 = await page.evaluate(() => window.__forge.frames());
  await page.waitForTimeout(500);
  const f1 = await page.evaluate(() => window.__forge.frames());
  if (f1 <= f0) fail('BOOTS', `frames stuck at ${f1}`);
  else pass('BOOTS', `${f1 - f0} frames/500ms`);

  // ON_THE_FLOOR: assembled bounding box touches y=0 within tolerance.
  if (Math.abs(m.groundClearance) > 0.06 * m.height) {
    fail('ON_THE_FLOOR', `clearance ${m.groundClearance.toFixed(3)} vs height ${m.height.toFixed(2)}`);
  } else pass('ON_THE_FLOOR', `clearance ${m.groundClearance.toFixed(3)}`);

  // BUDGET: chunky primitives, not accidental mesh soup.
  if (m.triangles > 60000) fail('BUDGET', `${m.triangles} triangles`);
  else pass('BUDGET', `${m.meshes} meshes / ${m.triangles} tris`);

  // ANIMATION_PLAYS in both modes: sampled joint pose must vary.
  for (const mode of ['idle', 'walk']) {
    await page.evaluate((mm) => window.__forge.setMode(mm), mode);
    const poses = [];
    for (let k = 0; k < 5; k++) {
      poses.push(await page.evaluate(() => window.__forge.pose()));
      await page.waitForTimeout(120);
    }
    const spread = Math.max(...poses) - Math.min(...poses);
    if (spread < 1e-3) fail('ANIMATION_PLAYS', `${mode} pose spread ${spread}`);
    else pass('ANIMATION_PLAYS', `${mode} spread ${spread.toFixed(3)}`);
    const shot = `dist/shots/${specs[i].name}-${mode}.png`;
    await page.screenshot({ path: shot });
  }

  // RENDERS_CONTENT: center crop of the idle shot must not be one color.
  const px = await page.evaluate(() => {
    const c = document.querySelector('canvas');
    const g = document.createElement('canvas');
    g.width = 200; g.height = 200;
    const ctx = g.getContext('2d');
    ctx.drawImage(c, c.width / 2 - 220, c.height / 2 - 220, 440, 440, 0, 0, 200, 200);
    const d = ctx.getImageData(0, 0, 200, 200).data;
    const seen = new Set();
    for (let j = 0; j < d.length; j += 40) seen.add(`${d[j] >> 4},${d[j + 1] >> 4},${d[j + 2] >> 4}`);
    return seen.size;
  });
  if (px < 8) fail('RENDERS_CONTENT', `center crop has ${px} distinct colors`);
  else pass('RENDERS_CONTENT', `${px} distinct colors in center crop`);
}

if (consoleErrors.length) fail('BOOTS', `console errors: ${consoleErrors.slice(0, 3).join(' | ')}`);
await browser.close();
console.log(failures === 0 ? '\nALL_GATES_GREEN' : `\n${failures} GATE FAILURES`);
process.exit(failures === 0 ? 0 : 1);
