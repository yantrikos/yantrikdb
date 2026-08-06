// Model-in-the-loop run: qwen3.6:27b drives the charforge spec grammar
// against SEALED briefs (written before any output was seen, never
// tuned after). Protocol-conformance is measured STRICTLY before the
// compiler's forgiving clamps ever run — count the model, not the
// harness (the sol no-fallback law from the webkit experiment).
import { writeFileSync, readFileSync } from 'node:fs';

const OLLAMA = process.env.CHARFORGE_OLLAMA ?? 'http://192.168.4.35:11434';
const MODEL = process.env.CHARFORGE_MODEL ?? 'qwen3.6:27b';

const BRIEFS = [
  { id: 'honeybear', text: 'A grumpy old bear who runs a mountain honey shop. He disapproves of you specifically, but sells you honey anyway.' },
  { id: 'zoomfox', text: 'A tiny hyperactive fox kit, the star of an endless-runner game. Built for speed, vibrating with energy.' },
  { id: 'moonbun', text: 'A sleepy moon rabbit mascot for a bedtime puzzle game. Soft, dozy, faintly luminous colors.' },
  { id: 'scrapdog', text: 'A scrappy junkyard guard dog with a heart of gold. Tough posture, friendly eyes.' },
  { id: 'skategirl', text: 'A fearless nine-year-old skater girl who just landed her first kickflip. Triumphant, a little scuffed up.' },
  { id: 'libraryboy', text: 'A shy bookish boy who spends recess in the library. Gentle, tidy, quietly pleased with his book fort.' },
];

import { GRAMMAR, validate } from './src/protocol.mjs';

async function ask(brief) {
  const res = await fetch(`${OLLAMA}/api/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: MODEL,
      stream: false,
      think: false, // measured trap: thinking eats the whole budget
      options: { temperature: 0.7, num_predict: 600 },
      messages: [
        { role: 'system', content: GRAMMAR },
        { role: 'user', content: `Brief: ${brief}` },
      ],
    }),
  });
  if (!res.ok) throw new Error(`ollama ${res.status}: ${await res.text()}`);
  const data = await res.json();
  return data.message?.content ?? '';
}

const specs = [];
const report = [];
for (const b of BRIEFS) {
  process.stdout.write(`asking ${MODEL} for "${b.id}"... `);
  const t0 = Date.now();
  const raw = await ask(b.text);
  const secs = ((Date.now() - t0) / 1000).toFixed(0);
  let spec = null;
  let parseNote = 'ok';
  try {
    spec = JSON.parse(raw.trim());
  } catch {
    // One tolerated extraction: a fenced or prefixed JSON object. This
    // is RECORDED as a violation — the instruction said no fences.
    const m = raw.match(/\{[\s\S]*\}/);
    if (m) { spec = JSON.parse(m[0]); parseNote = 'VIOLATION: needed extraction from prose/fences'; }
    else parseNote = 'FATAL: no JSON object found';
  }
  const violations = spec ? validate(spec) : ['unparseable'];
  report.push({ brief: b.id, seconds: +secs, parse: parseNote, violations, spec });
  if (spec) { spec.name = spec.name || b.id; specs.push(spec); }
  console.log(`${secs}s, parse=${parseNote}, violations=${violations.length ? violations.join(',') : 'none'}`);
}

writeFileSync('specs/qwen27b.json', JSON.stringify(specs, null, 2));
writeFileSync('dist/qwen27b-report.json', JSON.stringify(report, null, 2));
const clean = report.filter((r) => r.parse === 'ok' && r.violations.length === 0).length;
console.log(`\nPROTOCOL: ${clean}/${BRIEFS.length} specs fully conformant (strict, pre-clamp)`);
console.log('specs written to specs/qwen27b.json');
