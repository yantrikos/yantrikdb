#!/usr/bin/env node
// The prompt-to-character command — the agentic front door.
//
//   node forge.mjs "a brave little turtle-hearted knight kid"
//
// Brief → local model emits a spec (bounded grammar, strictly
// validated) → compiler builds a single self-contained HTML artifact →
// path printed. No code is written or changed anywhere in the loop;
// "recoding" happens only when the LIBRARY grows a new body plan or
// part type, which is a versioned release, not a per-character event.
//
// Agent usage: call this as a tool. Exit 0 = artifact ready; the JSON
// report on stdout carries spec, violations, and artifact path.
import { execFileSync } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { GRAMMAR, validate } from './src/protocol.mjs';

const brief = process.argv[2];
if (!brief) {
  console.error('usage: node forge.mjs "<character brief>" [name]');
  process.exit(2);
}
const OLLAMA = process.env.CHARFORGE_OLLAMA ?? 'http://192.168.4.35:11434';
const MODEL = process.env.CHARFORGE_MODEL ?? 'qwen3.6:27b';

const res = await fetch(`${OLLAMA}/api/chat`, {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify({
    model: MODEL,
    stream: false,
    think: false,
    options: { temperature: 0.7, num_predict: 600 },
    messages: [
      { role: 'system', content: GRAMMAR },
      { role: 'user', content: `Brief: ${brief}` },
    ],
  }),
});
if (!res.ok) {
  console.error(`model call failed: ${res.status} ${await res.text()}`);
  process.exit(1);
}
const raw = (await res.json()).message?.content ?? '';
let spec;
try {
  spec = JSON.parse(raw.trim());
} catch {
  const m = raw.match(/\{[\s\S]*\}/);
  if (!m) { console.error(`model emitted no JSON:\n${raw}`); process.exit(1); }
  spec = JSON.parse(m[0]);
}
const violations = validate(spec);
const name = (process.argv[3] ?? spec.name ?? 'character').replace(/[^a-z0-9-]/gi, '').toLowerCase() || 'character';
spec.name = name;

mkdirSync('specs/generated', { recursive: true });
const specPath = `specs/generated/${name}.json`;
writeFileSync(specPath, JSON.stringify([spec], null, 2));
const outHtml = `dist/${name}.html`;
execFileSync(process.execPath, ['build.mjs'], {
  env: { ...process.env, SPECS_FILE: specPath, OUT_HTML: outHtml },
  stdio: ['ignore', 'ignore', 'inherit'],
});

console.log(JSON.stringify({
  brief,
  name,
  spec,
  protocol_violations: violations,
  artifact: resolve(outHtml),
}, null, 2));
if (violations.length) {
  console.error(`NOTE: ${violations.length} protocol violation(s) — spec was clamped by the compiler: ${violations.join(', ')}`);
}
