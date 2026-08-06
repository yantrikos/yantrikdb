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

const GRAMMAR = `You design characters for a 3D toy-like game engine by emitting a JSON spec.
The engine assembles chunky primitive characters (spheres, capsules, cones) —
charm comes entirely from PROPORTIONS and PALETTE. You cannot emit geometry,
only the spec. Schema (every field required, exactly these fields):

{
  "name": string (short, lowercase),
  "archetype": "dog" | "cat" | "rabbit" | "bear" | "fox" | "kid",
  "proportions": {
    "head":  0-100 (bigger = bigger head = cuter/younger),
    "snout": 0-100 (bigger = longer muzzle),
    "ears":  0-100 (bigger = larger ears),
    "body":  0-100 (bigger = longer torso),
    "limbs": 0-100 (bigger = longer legs),
    "tail":  0-100 (bigger = larger tail),
    "chunk": 0-100 (bigger = rounder, heavier build)
  },
  "palette": {
    "seed": "#rrggbb" (the ONE free color — the character's base fur),
    "belly": "light" | "none",
    "nose": "dark" | "pink"
  },
  "face": {
    "expression": "happy" | "alert" | "sleepy" | "determined",
    "brows": boolean,
    "tongue": boolean
  },
  "motion": { "energy": 0-100 (animation tempo/amplitude) },
  "human": ONLY when archetype is "kid": {
    "hair": "crop" | "bob" | "pigtails" | "buns" | "spikes" | "swoop",
    "hairColor": "#rrggbb",
    "skin": "porcelain" | "fair" | "tan" | "brown" | "deep",
    "outfit": "tee-shorts" | "dress" | "overalls"
  }
}
For "kid": the palette seed colors the OUTFIT (skin comes only from the
named skin field); snout/ears/tail proportions are ignored; head 70-85
reads as a young child.

Worked example — brief: "a loyal chunky puppy, tongue out, delighted to see you":
{"name":"biscuit","archetype":"dog","proportions":{"head":78,"snout":60,"ears":85,"body":55,"limbs":45,"tail":45,"chunk":72},"palette":{"seed":"#e8b06d","belly":"light","nose":"dark"},"face":{"expression":"happy","brows":true,"tongue":true},"motion":{"energy":65}}
(big head + high chunk = puppy charm; warm biscuit seed; happy + tongue sells the delight)

Reply with ONLY the JSON object for the brief. No prose, no markdown fences.`;

// Strict validation — the protocol gate. Violations are recorded, not
// silently repaired.
const ARCH = ['dog', 'cat', 'rabbit', 'bear', 'fox', 'kid'];
const EXPR = ['happy', 'alert', 'sleepy', 'determined'];
const PROPS = ['head', 'snout', 'ears', 'body', 'limbs', 'tail', 'chunk'];
function validate(spec) {
  const v = [];
  if (typeof spec.name !== 'string' || !spec.name) v.push('name');
  if (!ARCH.includes(spec.archetype)) v.push(`archetype=${spec.archetype}`);
  for (const k of PROPS) {
    const x = spec.proportions?.[k];
    if (!Number.isFinite(x) || x < 0 || x > 100) v.push(`proportions.${k}=${x}`);
  }
  if (!/^#[0-9a-fA-F]{6}$/.test(spec.palette?.seed ?? '')) v.push(`seed=${spec.palette?.seed}`);
  if (!['light', 'none'].includes(spec.palette?.belly)) v.push(`belly=${spec.palette?.belly}`);
  if (!['dark', 'pink'].includes(spec.palette?.nose)) v.push(`nose=${spec.palette?.nose}`);
  if (!EXPR.includes(spec.face?.expression)) v.push(`expression=${spec.face?.expression}`);
  if (typeof spec.face?.brows !== 'boolean') v.push('brows');
  if (typeof spec.face?.tongue !== 'boolean') v.push('tongue');
  const e = spec.motion?.energy;
  if (!Number.isFinite(e) || e < 0 || e > 100) v.push(`energy=${e}`);
  if (spec.archetype === 'kid') {
    if (!['crop', 'bob', 'pigtails', 'buns', 'spikes', 'swoop'].includes(spec.human?.hair)) v.push(`hair=${spec.human?.hair}`);
    if (!/^#[0-9a-fA-F]{6}$/.test(spec.human?.hairColor ?? '')) v.push(`hairColor=${spec.human?.hairColor}`);
    if (!['porcelain', 'fair', 'tan', 'brown', 'deep'].includes(spec.human?.skin)) v.push(`skin=${spec.human?.skin}`);
    if (!['tee-shorts', 'dress', 'overalls'].includes(spec.human?.outfit)) v.push(`outfit=${spec.human?.outfit}`);
  }
  return v;
}

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
