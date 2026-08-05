# charforge — game-kit asset lane, v0

The character half of the game construction kit: a bounded SPEC goes in,
a lit, rigged, animated character comes out as one self-contained HTML
artifact. The split follows the measured kit law (webkit: ops+compiler
12/12 vs direct generation 0/6): the model decides archetype,
proportions, palette seed, and expression; the compiler owns everything
computable — primitive assembly, palette derivation, the light rig, the
joint hierarchy, animations, and the gates.

Style target: chunky primitive assemblies (spheres/capsules/cones), soft
studio light, contact shadow. Charm lives in proportions and palette,
not mesh detail. No textures, no network, no image models.

## Layout

- `src/charforge.js` — the compiler: spec → THREE.Group + joints + dims.
- `src/app.js` — turntable viewer (drag to turn, walk/idle toggle).
- `specs/reference.json` — the ceiling character (`biscuit`, hand-tuned
  against the style-target screenshot) plus two archetype probes.
- `build.mjs` — bundles app + compiler + three into `dist/forge.html`
  (single file, ~500 KB, CSP-clean).
- `gates.mjs` — headless (Edge/SwiftShader) hard gates + screenshots:
  BOOTS, ON_THE_FLOOR, BUDGET, ANIMATION_PLAYS (idle+walk),
  RENDERS_CONTENT. Screenshots land in `dist/shots/` — they are the
  review surface; the defects that matter are only found by looking.

## Run

```
npm install
node build.mjs     # -> dist/forge.html (open in any browser)
node gates.mjs     # headless gates + screenshots
```

## Spec grammar (what a model will emit)

```json
{
  "name": "biscuit",
  "archetype": "dog",              // dog|cat|rabbit|bear|fox
  "proportions": {                  // 0..100, mapped into safe ranges
    "head": 78, "snout": 60, "ears": 85,
    "body": 55, "limbs": 45, "tail": 45, "chunk": 72
  },
  "palette": { "seed": "#e8b06d", "belly": "light", "nose": "dark" },
  "face": { "expression": "happy", "brows": true, "tongue": true },
  "motion": { "energy": 65 }
}
```

One free value (the palette seed); everything else is bounded. No legal
spec can produce a broken body plan — that guarantee is the compiler's,
not the model's.

## Known v0 properties

- Collar/tag accent is occluded frontally on big-head characters (a
  head wider than the neck ring hides it); it reads on the turntable.
- One quadruped body plan; biped stance, props, and per-archetype
  silhouette work (rabbit ears rounder + more vertical) are next.
- The kill experiment this exists for: kit-arm vs direct-arm on sealed
  character briefs, hard gates + blinded charm preference — run it
  before building more archetypes (see the game-kit arc memory).
