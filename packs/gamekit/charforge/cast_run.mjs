// Agentic cast generation: qwen3.6:27b is handed ONE game concept and
// acts as the game's character director — stage 1 it DECIDES the cast
// (roles + briefs, its own judgment of what the game needs), stage 2 it
// generates every character spec through the forge grammar. The harness
// moves messages and compiles; every creative decision is the model's.
//
//   node cast_run.mjs            (uses the sealed default concept)
//   node cast_run.mjs "<your game concept>" <slug>
import { writeFileSync, mkdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { resolve } from 'node:path';
import { GRAMMAR, validate } from './src/protocol.mjs';

const OLLAMA = process.env.CHARFORGE_OLLAMA ?? 'http://192.168.4.35:11434';
const MODEL = process.env.CHARFORGE_MODEL ?? 'qwen3.6:27b';

const CONCEPT = process.argv[2] ?? (
  'Moonlight Bakery: a cozy night-time adventure game. A young baker ' +
  'delivers magical pastries through a sleeping forest town, dodging a ' +
  'grumpy rival and befriending the townsfolk. Tone: warm, gentle, a ' +
  'little mischievous.'
);
const SLUG = (process.argv[3] ?? 'moonlight-bakery').replace(/[^a-z0-9-]/gi, '-').toLowerCase();

const PLAN_PROMPT = `You are the character director for a small 3D game built from
chunky toy-like characters. The engine can build: creatures (dog, cat, rabbit,
bear, fox) and human kids. Nothing else — no adults, no monsters, no objects.

Given the game concept, decide the CAST this game actually needs: 5 to 7
characters covering the player character, at least one antagonist/rival, and
supporting roles that make the world feel alive. For each, write a vivid
one-sentence design brief a character artist could work from (personality,
age/build, palette mood — not geometry).

Reply with ONLY a JSON array, no prose:
[{ "role": "player|rival|companion|npc", "name_hint": "short lowercase name", "brief": "..." }]`;

async function chat(system, user, maxTokens) {
  const res = await fetch(`${OLLAMA}/api/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      model: MODEL,
      stream: false,
      think: false,
      options: { temperature: 0.8, num_predict: maxTokens },
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: user },
      ],
    }),
  });
  if (!res.ok) throw new Error(`ollama ${res.status}: ${await res.text()}`);
  return (await res.json()).message?.content ?? '';
}

function extractJson(raw, opener, closer) {
  try { return JSON.parse(raw.trim()); } catch { /* fall through */ }
  const a = raw.indexOf(opener);
  const b = raw.lastIndexOf(closer);
  if (a < 0 || b <= a) throw new Error(`no ${opener}...${closer} found in model output:\n${raw}`);
  return JSON.parse(raw.slice(a, b + 1));
}

// ── stage 1: the model decides the cast ─────────────────────────────
console.log(`concept: ${CONCEPT}\n`);
process.stdout.write(`stage 1 — ${MODEL} plans the cast... `);
const t0 = Date.now();
const castRaw = await chat(PLAN_PROMPT, `Game concept: ${CONCEPT}`, 1200);
const cast = extractJson(castRaw, '[', ']');
console.log(`${((Date.now() - t0) / 1000).toFixed(0)}s, ${cast.length} characters planned:`);
for (const c of cast) console.log(`  [${c.role}] ${c.name_hint}: ${c.brief}`);

// ── stage 2: the model generates each character through the grammar ─
const specs = [];
const report = { concept: CONCEPT, model: MODEL, cast, characters: [] };
for (const c of cast) {
  process.stdout.write(`\nstage 2 — spec for "${c.name_hint}" (${c.role})... `);
  const t = Date.now();
  const raw = await chat(GRAMMAR, `Brief: ${c.brief}`, 600);
  let spec;
  let parse = 'ok';
  try {
    spec = extractJson(raw, '{', '}');
    if (raw.trim()[0] !== '{') parse = 'VIOLATION: extraction needed';
  } catch (e) {
    console.log(`FATAL: ${e.message}`);
    report.characters.push({ ...c, parse: 'unparseable' });
    continue;
  }
  const violations = validate(spec);
  spec.name = (c.name_hint ?? spec.name ?? c.role).replace(/[^a-z0-9-]/gi, '').toLowerCase();
  specs.push(spec);
  report.characters.push({ ...c, parse, violations, spec, seconds: +((Date.now() - t) / 1000).toFixed(0) });
  console.log(`${((Date.now() - t) / 1000).toFixed(0)}s, ${spec.archetype}, violations=${violations.length ? violations.join(',') : 'none'}`);
}

mkdirSync('specs/generated', { recursive: true });
const specPath = `specs/generated/cast-${SLUG}.json`;
writeFileSync(specPath, JSON.stringify(specs, null, 2));
const outHtml = `dist/cast-${SLUG}.html`;
execFileSync(process.execPath, ['build.mjs'], {
  env: { ...process.env, SPECS_FILE: specPath, OUT_HTML: outHtml },
  stdio: ['ignore', 'inherit', 'inherit'],
});
writeFileSync(`dist/cast-${SLUG}-report.json`, JSON.stringify(report, null, 2));

const clean = report.characters.filter((r) => r.parse === 'ok' && r.violations?.length === 0).length;
console.log(`\nCAST: ${specs.length}/${cast.length} built, ${clean}/${cast.length} strictly conformant`);
console.log(`artifact: ${resolve(outHtml)}`);
console.log('CAST_DONE');
