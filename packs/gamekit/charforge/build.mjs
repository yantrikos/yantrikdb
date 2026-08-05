// Build: bundle app+compiler+three into ONE self-contained forge.html.
// A build step that cannot fail is a build step that lies (webkit lesson)
// — esbuild errors and missing output are both fatal here.
import { build } from 'esbuild';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';

const specs = JSON.parse(readFileSync('specs/reference.json', 'utf8'));

const result = await build({
  entryPoints: ['src/app.js'],
  bundle: true,
  minify: true,
  format: 'iife',
  write: false,
  logLevel: 'error',
});
if (result.errors.length) {
  console.error(result.errors);
  process.exit(1);
}
const js = result.outputFiles[0].text;

const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>charforge</title>
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; background: #14101e; }
  canvas { display: block; }
  #hud { position: fixed; top: 14px; left: 16px; display: flex; gap: 8px; align-items: center;
         font: 14px/1 system-ui, sans-serif; }
  #name { color: #f5edE0; opacity: .85; letter-spacing: .04em; margin-right: 6px;
          text-transform: lowercase; }
  #hud button { background: #ffffff14; color: #f5ede0; border: 1px solid #ffffff2e;
                border-radius: 8px; padding: 7px 14px; cursor: pointer; font: inherit; }
  #hud button:hover { background: #ffffff26; }
</style>
</head>
<body>
<div id="hud"><span id="name"></span><span id="picker"></span><button id="walk">walk</button></div>
<script>window.__CHARFORGE_SPECS__ = ${JSON.stringify(specs)};</script>
<script>${js}</script>
</body>
</html>`;

mkdirSync('dist', { recursive: true });
writeFileSync('dist/forge.html', html);
if (readFileSync('dist/forge.html', 'utf8').length < 100_000) {
  console.error('dist/forge.html suspiciously small — three.js missing from bundle?');
  process.exit(1);
}
console.log(`BUILD_OK dist/forge.html (${(html.length / 1024).toFixed(0)} KB, ${specs.length} specs)`);
