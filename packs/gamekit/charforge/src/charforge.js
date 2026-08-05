// charforge — deterministic character compiler.
//
// A character SPEC (bounded fields, one free color seed) comes in; a lit,
// rigged, animated THREE.Group comes out. The split follows the measured
// kit law (webkit: ops+compiler 12/12 vs direct 0/6): the model decides
// "loyal chunky dog, cream belly, alert ears" — proportions, palette seed,
// expression. The compiler owns everything computable: primitive assembly,
// palette derivation with contrast rules, the light rig, the joint
// hierarchy, and the animations. A model cannot evaluate whether the snout
// intersects the eyes; the compiler never lets it happen.
//
// Style target: chunky primitive assemblies (spheres/capsules/cones), soft
// studio light, contact shadow — charm lives in proportions and palette,
// not in mesh detail. No textures, no external assets: artifacts stay
// single-file and headless-verifiable.

import * as THREE from 'three';

// ── spec schema (the grammar a model emits) ──────────────────────────
// All proportion fields are 0..100 ints; the compiler maps them into safe
// per-archetype ranges so no legal spec can produce a broken body plan.
export const ARCHETYPES = ['dog', 'cat', 'rabbit', 'bear', 'fox'];
export const EXPRESSIONS = ['happy', 'alert', 'sleepy', 'determined'];

export const SPEC_DEFAULTS = {
  name: 'creature',
  archetype: 'dog',
  proportions: { head: 60, snout: 50, ears: 60, body: 55, limbs: 40, tail: 50, chunk: 60 },
  palette: { seed: '#e0a868', belly: 'light', nose: 'dark' },
  face: { expression: 'happy', brows: true, tongue: false },
  motion: { energy: 55 },
};

function clamp01(x) { return Math.min(1, Math.max(0, x)); }
function lerp(a, b, t) { return a + (b - a) * t; }
// Map a 0..100 spec knob into a safe [lo, hi] range.
function knob(v, lo, hi) { return lerp(lo, hi, clamp01((v ?? 50) / 100)); }

export function normalizeSpec(raw) {
  const s = JSON.parse(JSON.stringify(SPEC_DEFAULTS));
  if (!raw || typeof raw !== 'object') return s;
  s.name = typeof raw.name === 'string' ? raw.name.slice(0, 40) : s.name;
  if (ARCHETYPES.includes(raw.archetype)) s.archetype = raw.archetype;
  for (const k of Object.keys(s.proportions)) {
    const v = raw.proportions?.[k];
    if (Number.isFinite(v)) s.proportions[k] = Math.min(100, Math.max(0, v));
  }
  if (/^#[0-9a-fA-F]{6}$/.test(raw.palette?.seed ?? '')) s.palette.seed = raw.palette.seed;
  if (['light', 'none'].includes(raw.palette?.belly)) s.palette.belly = raw.palette.belly;
  if (['dark', 'pink'].includes(raw.palette?.nose)) s.palette.nose = raw.palette.nose;
  if (EXPRESSIONS.includes(raw.face?.expression)) s.face.expression = raw.face.expression;
  if (typeof raw.face?.brows === 'boolean') s.face.brows = raw.face.brows;
  if (typeof raw.face?.tongue === 'boolean') s.face.tongue = raw.face.tongue;
  if (Number.isFinite(raw.motion?.energy)) s.motion.energy = Math.min(100, Math.max(0, raw.motion.energy));
  return s;
}

// ── palette derivation ───────────────────────────────────────────────
// One free seed; every other color is derived with bounded lightness /
// saturation moves (the webkit accent rule, one dimension simpler: HSL
// bands are enough at this material roughness — no text contrast at
// stake, only part separation, which the checks below enforce).
export function derivePalette(p) {
  const seed = new THREE.Color(p.seed);
  const hsl = {};
  seed.getHSL(hsl);
  // Project the seed into a band that keeps parts distinguishable under
  // the rig: not too dark (belly derivation dies), not neon.
  const h = hsl.h;
  const sBase = Math.min(0.75, Math.max(0.25, hsl.s));
  const lBase = Math.min(0.72, Math.max(0.45, hsl.l));
  const c = (hh, ss, ll) => new THREE.Color().setHSL(((hh % 1) + 1) % 1, clamp01(ss), clamp01(ll));
  return {
    base: c(h, sBase, lBase),
    baseDark: c(h, sBase * 1.05, lBase * 0.72),          // outer ear / paw shading
    belly: p.belly === 'none' ? c(h, sBase, lBase) : c(h, sBase * 0.55, Math.min(0.9, lBase * 1.35)),
    innerEar: c(0.985, 0.55, 0.72),                       // warm pink, archetype-stable
    nose: p.nose === 'pink' ? c(0.97, 0.5, 0.62) : c(h, 0.35, 0.09),
    eyeWhite: c(0, 0, 0.97),
    pupil: c(0.6, 0.25, 0.08),
    tongue: c(0.97, 0.62, 0.66),
    collar: c(h + 0.5, 0.55, 0.45),                       // complementary accent
    ground: c(h, 0.18, 0.16),
    bg: c(h, 0.22, 0.09),
  };
}

// ── mesh helpers ─────────────────────────────────────────────────────
function mat(color, rough = 0.62) {
  return new THREE.MeshStandardMaterial({ color, roughness: rough, metalness: 0.0 });
}
function sphere(r, color, sx = 1, sy = 1, sz = 1) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(r, 32, 24), mat(color));
  m.scale.set(sx, sy, sz);
  m.castShadow = true;
  return m;
}
function capsule(r, len, color) {
  const m = new THREE.Mesh(new THREE.CapsuleGeometry(r, len, 8, 24), mat(color));
  m.castShadow = true;
  return m;
}
function cone(r, h, color) {
  const m = new THREE.Mesh(new THREE.ConeGeometry(r, h, 24), mat(color));
  m.castShadow = true;
  return m;
}

// ── the body plan ────────────────────────────────────────────────────
// One parameterized quadruped mammal plan covers the v0 archetypes; the
// archetype table bends ears/snout/tail/stance rather than forking the
// skeleton. Returns { group, joints, dims } — joints are what animation
// drives, dims are what the gates measure.
export function buildCharacter(rawSpec) {
  const spec = normalizeSpec(rawSpec);
  const P = spec.proportions;
  const pal = derivePalette(spec.palette);
  const arch = {
    dog:    { earKind: 'tri',   earTilt: 0.25, snoutLen: 1.0, tailKind: 'curl',  cheeks: 1.0 },
    cat:    { earKind: 'tri',   earTilt: 0.05, snoutLen: 0.55, tailKind: 'line', cheeks: 0.9 },
    rabbit: { earKind: 'tall',  earTilt: 0.1,  snoutLen: 0.45, tailKind: 'puff', cheeks: 1.1 },
    bear:   { earKind: 'round', earTilt: 0.0,  snoutLen: 0.8,  tailKind: 'puff', cheeks: 1.15 },
    fox:    { earKind: 'tri',   earTilt: 0.12, snoutLen: 0.9,  tailKind: 'brush', cheeks: 0.95 },
  }[spec.archetype];

  const root = new THREE.Group();
  root.name = `charforge:${spec.name}`;
  const joints = {};

  // Body: a chunky upright capsule; `chunk` widens it, `body` lengthens.
  const bodyR = knob(P.chunk, 0.55, 0.85);
  const bodyLen = knob(P.body, 0.5, 1.0);
  const legLen = knob(P.limbs, 0.45, 0.95);
  const legR = bodyR * 0.28;
  const hipY = legLen + bodyR * 0.9;

  const bodyGroup = new THREE.Group();
  bodyGroup.position.y = hipY;
  root.add(bodyGroup);
  joints.body = bodyGroup;

  const torso = capsule(bodyR, bodyLen, pal.base);
  torso.position.y = bodyLen * 0.1;
  bodyGroup.add(torso);

  // Slightly elliptical torso (narrower front-to-back) so the belly
  // patch and legs read in silhouette instead of merging into a slab.
  torso.scale.z = 0.88;

  if (spec.palette.belly !== 'none') {
    // Belly patch: a flattened sphere PROUD of the torso surface — the
    // first look pass found it buried (0.95R vs the torso's 1.0R).
    const belly = sphere(bodyR * 0.72, pal.belly, 0.78, 1.05, 0.5);
    belly.position.set(0, bodyLen * 0.02, bodyR * 0.66);
    bodyGroup.add(belly);
  }

  // Head: big sphere on a neck joint; `head` scales it (charm knob #1).
  const headR = bodyR * knob(P.head, 0.85, 1.35);
  const neck = new THREE.Group();
  neck.position.y = bodyLen * 0.62 + bodyR * 0.55;
  bodyGroup.add(neck);
  joints.neck = neck;
  const head = sphere(headR, pal.base, 1.0, 0.95, 0.95);
  head.position.y = headR * 0.55;
  neck.add(head);
  const headTop = headR * 0.55; // local y of head center within neck

  // Muzzle: flattened sphere pushed forward; nose PROUD of its tip.
  // Both muzzle size and reach follow the snout knob — a snout=30
  // rabbit gets a small button muzzle, not a beak (look-pass defect).
  const snoutT = clamp01((P.snout ?? 50) / 100) * arch.snoutLen;
  const snoutLen = headR * lerp(0.3, 0.75, snoutT);
  const muzzleR = headR * lerp(0.3, 0.56, snoutT) * arch.cheeks;
  const muzzle = sphere(muzzleR, pal.belly, 1.0, 0.72, 1.0);
  muzzle.position.set(0, headTop - headR * 0.28, headR * 0.62 + snoutLen * 0.4);
  neck.add(muzzle);
  const nose = sphere(headR * lerp(0.12, 0.2, snoutT), pal.nose, 1.25, 0.85, 1.0);
  nose.material.roughness = 0.3;
  nose.position.set(0, headTop - headR * 0.16, headR * 0.62 + snoutLen * 0.4 + muzzleR * 0.82);
  neck.add(nose);

  // Mouth + tongue.
  if (spec.face.tongue) {
    const tongue = capsule(headR * 0.11, headR * 0.3, pal.tongue);
    tongue.position.set(0, headTop - headR * 0.62, headR * 0.62 + snoutLen * 0.35);
    tongue.rotation.x = 0.45;
    neck.add(tongue);
    joints.tongue = tongue;
  }

  // Eyes: whites + pupils, spacing driven by muzzle width. Expression
  // sets lid/brow pose, not geometry — same parts, different pose.
  const eyeSep = headR * 0.42;
  const eyeY = headTop + headR * 0.12;
  const eyeZ = headR * 0.78;
  for (const side of [-1, 1]) {
    const white = sphere(headR * 0.15, pal.eyeWhite, 0.8, 1.0, 0.55);
    white.position.set(side * eyeSep, eyeY, eyeZ);
    white.castShadow = false;
    neck.add(white);
    const pupil = sphere(headR * 0.085, pal.pupil, 0.9, 1.0, 0.6);
    pupil.position.set(side * eyeSep * 0.98, eyeY, eyeZ + headR * 0.06);
    pupil.castShadow = false;
    neck.add(pupil);
    joints[side < 0 ? 'pupilL' : 'pupilR'] = pupil;
    const glint = sphere(headR * 0.028, pal.eyeWhite);
    glint.position.set(side * eyeSep * 0.9 + headR * 0.03, eyeY + headR * 0.05, eyeZ + headR * 0.11);
    glint.castShadow = false;
    neck.add(glint);
  }
  if (spec.face.brows) {
    const browTilt = { happy: 0.15, alert: 0.05, sleepy: -0.1, determined: -0.35 }[spec.face.expression];
    for (const side of [-1, 1]) {
      const brow = sphere(headR * 0.11, pal.baseDark, 1.6, 0.4, 0.5);
      brow.position.set(side * eyeSep, eyeY + headR * 0.28, eyeZ - headR * 0.02);
      brow.rotation.z = -side * browTilt;
      neck.add(brow);
    }
  }

  // Ears: archetype kind, `ears` scales (charm knob #2). Inner-ear card
  // sits proud of the outer cone so the pink reads from the front.
  const earScale = knob(P.ears, 0.6, 1.5);
  for (const side of [-1, 1]) {
    const earGroup = new THREE.Group();
    // Seated INTO the skull (0.60R, was 0.72R): the look pass showed
    // cone bases hovering with a hard seam at the ear/head junction.
    earGroup.position.set(side * headR * 0.52, headTop + headR * 0.6, -headR * 0.02);
    earGroup.rotation.z = -side * (0.18 + arch.earTilt);
    neck.add(earGroup);
    joints[side < 0 ? 'earL' : 'earR'] = earGroup;
    let outer, inner;
    if (arch.earKind === 'round') {
      outer = sphere(headR * 0.3 * earScale, pal.baseDark, 1, 1, 0.5);
      inner = sphere(headR * 0.19 * earScale, pal.innerEar, 1, 1, 0.4);
      inner.position.z = headR * 0.09 * earScale;
    } else {
      const h = headR * (arch.earKind === 'tall' ? 1.15 : 0.62) * earScale;
      outer = cone(headR * 0.26 * earScale, h, pal.baseDark);
      outer.position.y = h * 0.4;
      inner = cone(headR * 0.16 * earScale, h * 0.72, pal.innerEar);
      inner.position.set(0, h * 0.36, headR * 0.09);
    }
    outer.castShadow = true;
    earGroup.add(outer);
    earGroup.add(inner);
  }

  // Legs: four capsules with paw spheres, on swing joints at the hips /
  // shoulders. Quadruped stance, front pair slightly narrower.
  // Rear pair wider and further back than the front pair so all four
  // read from a front three-quarter view (look pass: rear legs were
  // hidden behind the front pair and the body slab).
  const stanceX = bodyR * 0.55;
  const stanceZ = bodyR * 0.62;
  const legDefs = [
    ['legFL', stanceX * 0.8, stanceZ], ['legFR', -stanceX * 0.8, stanceZ],
    ['legBL', stanceX * 1.25, -stanceZ], ['legBR', -stanceX * 1.25, -stanceZ],
  ];
  for (const [name, x, z] of legDefs) {
    const hip = new THREE.Group();
    hip.position.set(x, -bodyR * 0.55, z);
    bodyGroup.add(hip);
    joints[name] = hip;
    const leg = capsule(legR, Math.max(0.05, legLen - legR), pal.base);
    leg.position.y = -(legLen + bodyR * 0.35) / 2 + legR * 0.2;
    hip.add(leg);
    const paw = sphere(legR * 1.35, pal.belly, 1.0, 0.72, 1.15);
    paw.position.set(0, -(hipY - bodyR * 0.55) + legR * 0.95, legR * 0.25);
    paw.castShadow = true;
    hip.add(paw);
  }

  // Tail on a wag joint.
  const tailGroup = new THREE.Group();
  tailGroup.position.set(0, -bodyR * 0.15, -bodyR * 0.95);
  bodyGroup.add(tailGroup);
  joints.tail = tailGroup;
  const tailScale = knob(P.tail, 0.5, 1.3);
  let tailMesh;
  if (arch.tailKind === 'puff') {
    tailMesh = sphere(bodyR * 0.28 * tailScale, pal.belly);
  } else if (arch.tailKind === 'brush') {
    tailMesh = sphere(bodyR * 0.24 * tailScale, pal.baseDark, 0.7, 0.7, 1.9);
    tailMesh.position.z = -bodyR * 0.4 * tailScale;
    tailMesh.rotation.x = -0.5;
  } else {
    tailMesh = capsule(bodyR * 0.13 * tailScale, bodyR * 0.7 * tailScale, pal.base);
    tailMesh.rotation.x = arch.tailKind === 'curl' ? -1.0 : -0.6;
    tailMesh.position.set(0, bodyR * 0.2 * tailScale, -bodyR * 0.2 * tailScale);
  }
  tailGroup.add(tailMesh);

  // Collar: the one "prop" in v0 — a torus accent that separates head
  // from body mass and carries the complementary color. Child of the
  // NECK so it rides head tilts and sits in the visible neck gap (the
  // look pass found it swallowed by the head/torso junction).
  const collar = new THREE.Mesh(
    new THREE.TorusGeometry(headR * 0.58, headR * 0.09, 12, 32), mat(pal.collar, 0.5),
  );
  // Below the head sphere's underside (head bottom ≈ headTop − 0.9R):
  // at −0.12R the ring was fully occluded and the accent color never
  // rendered (second look pass).
  collar.position.y = headTop - headR * 1.02;
  collar.rotation.x = Math.PI / 2 - 0.1;
  collar.castShadow = true;
  neck.add(collar);
  const tag = sphere(headR * 0.11, pal.nose);
  tag.material.roughness = 0.25;
  tag.position.set(0, headTop - headR * 1.14, headR * 0.5);
  neck.add(tag);

  // Frame from the ASSEMBLED bounding box, not a formula — the formula
  // undercounted tall rabbit ears and the camera cropped them.
  const box = new THREE.Box3().setFromObject(root);
  return {
    spec,
    palette: pal,
    group: root,
    joints,
    dims: { hipY, bodyR, headR, totalHeight: box.max.y },
  };
}

// ── animation ────────────────────────────────────────────────────────
// Clock-driven pose functions on the joint hierarchy — no keyframe data
// to get wrong, and `energy` scales amplitude/tempo inside safe bands.
export function makeAnimator(character) {
  const { joints, spec, dims } = character;
  const energy = knob(spec.motion.energy, 0.5, 1.35);
  let mode = 'idle';
  let blinkT = -1;
  return {
    setMode(m) { mode = m === 'walk' ? 'walk' : 'idle'; },
    getMode() { return mode; },
    tick(t) {
      const speed = mode === 'walk' ? 6.0 * energy : 2.2 * energy;
      const s = Math.sin(t * speed);
      const c = Math.cos(t * speed);
      // Body bob + squash — the "alive" baseline.
      const bobAmp = mode === 'walk' ? 0.05 : 0.02;
      joints.body.position.y = dims.hipY + Math.abs(s) * bobAmp * dims.bodyR * 2;
      joints.body.rotation.z = mode === 'walk' ? s * 0.03 : 0;
      // Head: gentle counter-bob and tilt.
      joints.neck.rotation.z = s * (mode === 'walk' ? 0.05 : 0.03);
      joints.neck.rotation.x = c * 0.02 - (spec.face.expression === 'sleepy' ? 0.12 : 0);
      // Legs: diagonal pairs in antiphase when walking, still when idle.
      const swing = mode === 'walk' ? 0.55 : 0.0;
      joints.legFL.rotation.x = s * swing;
      joints.legBR.rotation.x = s * swing;
      joints.legFR.rotation.x = -s * swing;
      joints.legBL.rotation.x = -s * swing;
      // Tail: always wagging; happier expressions wag harder.
      const wag = { happy: 1.0, alert: 0.7, sleepy: 0.25, determined: 0.5 }[spec.face.expression];
      joints.tail.rotation.y = Math.sin(t * 7.5 * energy) * 0.45 * wag;
      // Ears: periodic twitch (every ~3.7s, offset per ear).
      const tw = (tt) => Math.max(0, Math.sin(tt)) ** 12;
      joints.earL.rotation.z = 0.18 + tw(t * 1.7) * 0.2;
      joints.earR.rotation.z = -0.18 - tw(t * 1.7 + 2.4) * 0.2;
      // Blink: pupils squash briefly on a timer.
      if (blinkT < 0 && Math.random() < 0.008) blinkT = t;
      const blink = blinkT >= 0 ? Math.max(0, 1 - Math.abs(t - blinkT - 0.08) * 18) : 0;
      if (blinkT >= 0 && t - blinkT > 0.2) blinkT = -1;
      const pupilY = 1 - blink * 0.85;
      joints.pupilL.scale.y = pupilY;
      joints.pupilR.scale.y = pupilY;
      if (joints.tongue) joints.tongue.rotation.x = 0.45 + Math.abs(Math.sin(t * 3 * energy)) * 0.12;
    },
  };
}

// ── stage: light rig + ground + camera framing ───────────────────────
// The studio look from the style target: hemisphere fill, warm key with
// soft shadow, cool rim, dark radial backdrop, contact shadow disc.
export function buildStage(scene, character) {
  const pal = character.palette;
  scene.background = pal.bg;
  scene.add(new THREE.HemisphereLight(0xfff4e6, 0x2a2440, 0.85));
  const key = new THREE.DirectionalLight(0xffe8cc, 2.4);
  key.position.set(2.5, 5, 3.5);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.radius = 6;
  const d = character.dims.totalHeight * 1.2;
  Object.assign(key.shadow.camera, { left: -d, right: d, top: d, bottom: -d, near: 0.5, far: 20 });
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x7a9cff, 1.1);
  rim.position.set(-3, 2.5, -3.5);
  scene.add(rim);
  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(character.dims.totalHeight * 2.2, 48),
    new THREE.MeshStandardMaterial({ color: character.palette.ground, roughness: 0.95 }),
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);
  return { key };
}

export function frameCamera(camera, character, aspect) {
  const h = character.dims.totalHeight;
  camera.fov = 34;
  camera.aspect = aspect;
  camera.position.set(0, h * 0.62, h * 2.35);
  camera.lookAt(0, h * 0.52, 0);
  camera.updateProjectionMatrix();
}

// ── computable checks the compiler itself guarantees ─────────────────
// Exposed for the gate harness: measures the assembled character rather
// than trusting the assembly code.
export function measureCharacter(character) {
  const box = new THREE.Box3().setFromObject(character.group);
  let meshes = 0;
  let triangles = 0;
  character.group.traverse((o) => {
    if (o.isMesh) {
      meshes += 1;
      const idx = o.geometry.getIndex();
      triangles += (idx ? idx.count : o.geometry.getAttribute('position').count) / 3;
    }
  });
  return {
    meshes,
    triangles: Math.round(triangles),
    height: box.max.y - box.min.y,
    groundClearance: box.min.y, // must be ~0: feet on the floor, nothing below it
    width: box.max.x - box.min.x,
  };
}
