# Attachable Memory: Packs

Status: **steps 0–3 implemented** (`engine::pack`, `tests/pack_mount.rs`, Python
bindings); steps 4–7 designed. Supersedes the scattered design notes in #111 /
#112 on two points (marked ⚠️ below). Grounded against HEAD `db9f765` (v0.10.1,
schema v38).

## What works today

Download a pack, install it, and it stays installed:

```console
$ yantrikdb pack info physics.ydbpack        # read the manifest, install nothing
$ yantrikdb pack install physics.ydbpack     # copies beside the db, mounts, records
$ yantrikdb pack list                        # what is installed, and is it mounted
$ yantrikdb pack remove yantrik/physics@1.0.0
```

From then on every process that opens that database has the pack mounted with
no API call — the ThemeForest/WordPress "download and it's installed" flow.

```python
db = YantrikDB("host.db", 64)                 # bundled embedder, dim 64
db.record_text("I prefer tea over coffee.")

pack_id = db.mount_pack("physics.ydbpack")    # TRANSIENT — refuses on mismatch
db.recall_text("what are gluons")             # ← now answers from the pack
db.unmount_pack(pack_id)                      # host file byte-identical

db.install_pack("physics.ydbpack")            # DURABLE — survives restart
db.installed_packs(); db.mounted_packs()
```

### Transient mount and durable install are different verbs on purpose

`mount_pack` writes nothing to the host — that is the byte-identical guarantee
which makes mounting reversible where importing is not, and a library that
merely mounts a pack for one process must not touch the user's database.
`install_pack` is the durable variant: it copies the pack into a sibling
`<stem>.packs/` directory and records it in `pack_mounts`, and `open()`
re-mounts everything recorded there.

Making it a separate verb rather than a flag keeps the guarantee legible: if
you called `mount`, nothing was written. Both are pinned by tests
(`transient_mount_writes_nothing_to_the_host`,
`installed_pack_survives_restart`).

Packs live beside the database rather than in a global cache, and only the
*file name* is stored, so a database and its packs copy, move and back up as
one unit.

**A missing or broken pack never blocks an open.** If a pack file is deleted,
moved, modified, or turns out to be incompatible, `open()` logs it, skips it,
and keeps the record so the user can see what broke and reinstall — rather
than holding a database hostage to a third-party file
(`missing_installed_pack_does_not_break_open`).

Authoring a pack:

```python
src.seal_pack("physics.ydbpack", name="physics", version="1.0.0",
              origin="demo/physics", namespace="physics")
```

`seal_pack` scopes to a namespace, drops tombstones, scrubs host-private tables
(oplog, sessions, impressions, learned weights, calibration counters), stamps a
manifest and a blake3 content digest into the pack's own `meta`, and VACUUMs to a
single rollback-journal file with no WAL sidecar. `mount_pack` re-verifies that
digest, so a pack edited after sealing is refused.

Also on the Python surface: `mounted_packs()`, `unmount_all_packs()`,
`embedder_identity()`, `adopt_embedder_identity()`, and the typed exceptions
`PackEmbedderMismatch` / `PackAlreadyMounted`.

## 1. What we are actually building

A **pack** is a sealed, versioned, single-file YantrikDB that a user downloads and
*mounts* against their existing memory, gaining knowledge and abilities the base
model does not have — and *unmounts* when done, leaving the host byte-identical.

The target is the capable-but-frozen local model: qwen-3.6-27B knows a great deal
and knows nothing after its cutoff, and a 4B model barely knows a domain at all.

### Knowledge and ability are different mechanisms

This is the distinction the design turns on, and conflating them is why
retrieval-only approaches under-deliver. A pack carries three tiers:

| Tier | Content | Activation | Budget |
|---|---|---|---|
| **Constitution** | Hard constraints, operating procedure, the "how" | **Unconditional** on mount | Token-capped (~1–2k) |
| **Corpus** | Facts, cases, worked examples, the "what" | Similarity retrieval | Unbounded |
| **Coverage index** | What this pack can answer, in the pack's own words | Injected with the constitution | ~200 tokens |

Tier 1 is the *ability*. It is installed, not retrieved, because a hard constraint
that is retrieved 70% of the time is not a constraint. #90 documents exactly this
failure: in the YDS experiment a dark-mode rule was present in the substrate and
scored 0/4 — stored, never surfaced.

Tier 3 is the piece nobody has specified, and it is what makes the whole thing
work. A model does not spontaneously query knowledge it does not know exists. The
coverage index is how a mounted pack announces itself, turning "I don't know this"
into "the physics pack covers this — query it."

## 2. Architecture: mount, don't import

The prior recommendation (recall rid `019f6d0f`) was **import-by-origin first**:
bulk-copy pack rows into the host DB under a pack `origin_actor`, and remove by
origin-scoped delete. On the grounded code I am reversing that. The reasoning:

**Import cannot deliver clean detach, and the residue lands in the ranking path.**

- Deletion in this engine is tombstoning, never a hard `DELETE`
  ([lifecycle.rs:555-645](crates/yantrikdb-core/src/engine/lifecycle.rs#L555-L645)).
  Detach leaves every pack row in the user's file forever.
- `namespace_importance_stats.count` is a **cumulative write counter that never
  decrements on forget** ([schema.rs:1151-1160](crates/yantrikdb-core/src/base/schema.rs#L1151-L1160)).
  A 5,000-row pack permanently shifts the host's importance EWMA, so the user's
  own future high-importance writes get deflated by a pack that is no longer
  mounted. Silent, permanent, and invisible in every test that checks row counts.
- Learned recall weights, `recall_demand`, and FTS all absorb pack content the
  same way.

Mount has none of this. Detach is dropping a handle; the host file is untouched.
The pack also stays *sealed*, so its content digest remains verifiable for as long
as it is mounted — which is the whole trust story for a marketplace of untrusted
artifacts.

### Mount is cheaper than the "restructure recall" estimate suggests

**One HNSW per mounted pack**, built at mount from the pack's rows, dropped at
unmount. Additive — the host index is never rebuilt or mutated. Mount cost is the
same O(rows) build the host already pays on every open, because the index is
in-RAM and rebuilt at open rather than persisted
([indices.rs:10-53](crates/yantrikdb-core/src/engine/indices.rs#L10-L53)).

**Deviation from the original plan: no `ATTACH DATABASE`.** Cross-file SQL on one
connection looked like it would give FTS and row hydration for free, but the host
keeps a *pool* of read connections (`YANTRIKDB_READ_POOL`, default 4) and ATTACH
is per-connection state. Attaching on the write connection alone leaves pooled
readers unable to see the pack; attaching on all of them makes pool growth a
correctness problem. Each mount therefore owns a plain read-only `Connection`,
which sidesteps the class entirely and makes unmount a drop.

The cost of that choice: pack candidates come from the pack's HNSW only, so pack
rows do not currently participate in FTS keyword matching. Bounded and named
rather than discovered later.

```
query ──> host HNSW ──┐
     └──> pack HNSW ──┤──> merged candidate pool ──> status filter ──> MMR ──> hydrate
          ATTACH FTS ─┘         (host weights, trust tier)
```

### Score comparability, solved by format constraint

Candidates from two files are only comparable if they live in one embedding space.
Rather than solve cross-space merging, **the pack format pins the embedder**: packs
declare embedder name + digest + dim, and mount *rejects* a mismatch. The bundled
`potion-base-2M` (dim 64) is compiled into every binary
([embedder/default.rs:56](crates/yantrikdb-core/src/embedder/default.rs#L56)), so
the canonical choice is free for every host. One query encoding serves both pools.

The remaining incomparability is *policy*, not geometry: learned weights and
importance calibration are per-DB. Resolve it by scoring pack candidates with the
**host's** weights. The features (similarity, decay, recency, importance) are
per-row and computable for any source; the weights are the host's retrieval policy.
The host governs; the pack supplies candidates.

### The merge seam is where the trust ladder finally lives

#116 is right that "pack facts never overwrite user-verified facts" is currently
true of writes and false of retrieval — `base/scoring.rs` uses no source, certainty,
or provenance term at all. Under mount, the merge point is the natural home for it:

```
user_confirmed  >  vendor_pack(signed)  >  vendor_pack(unsigned)  >  llm_suggested
```

applied as a multiplicative tier factor when unioning the pools. This is the
reconciliation semantics that make the system defensible, implemented at exactly
the seam that needs it, rather than bolted onto a global ranking function.

## 3. User corrections: the overlay

A user will correct a pack fact. The pack is read-only, so the correction lands in
the host as a record that supersedes a pack rid via `record_links`.

This gives the three-way upgrade model (#111) for free, with a property worth
naming: **corrections survive detach, re-attach, and pack upgrade.** Unmount the
physics pack and your corrections go inert but are not lost; mount v2 and they
re-apply, with the idempotency machinery (4a.6c) classifying each pack record as
unchanged / changed / removed. Diff-aware upgrades, as predicted.

## 4. Engine gaps, ordered by silent-failure risk

Not by effort. Ordered so the things that are *wrong without saying so* land first.

### ✅ G1 — The embedder identity is not persisted. Mounting the wrong pack is silently wrong. *(fixed)*

`SearchState::initial` reconstructs provenance as `ExternalOrUnknown` on **every
open** ([reembed.rs:511-530](crates/yantrikdb-core/src/engine/reembed.rs#L511-L530)),
because no `meta` key holds an embedder name or digest — I enumerated every
`meta` writer; there is none. The digest guard at
[mod.rs:1585-1594](crates/yantrikdb-core/src/engine/mod.rs#L1585-L1594) only fires
for `Known` provenance, so it is **unreachable across a restart**: same dim,
different embedder is accepted as a compat-attach. Queries encode in one space,
stored vectors live in another, nothing errors, results are quietly garbage.

This was a live correctness bug, not only a pack blocker — filed as
[#117](https://github.com/yantrikos/yantrikdb/issues/117).

**Fixed**, and implementing it turned up two things the design did not anticipate:

1. **The bundled embedder had no fingerprint.** `BundledEmbedder` implemented only
   `embed()` and `dim()`, so `fingerprint()` fell through to the trait default of
   `None` — even though the trait's own doc comment claims "bundled embedders
   override this with real fingerprints." Since the bundled model is the default
   for every `YantrikDB::new(path, 64)`, *no database in the wild had provable
   identity*, `set_embedder`'s guard could never fire, and the mount check would
   have been vacuous exactly where it mattered most. It now fingerprints itself by
   hashing its four baked-in `include_bytes!` blobs — self-maintaining, so
   swapping the bundle changes the digest without anyone bumping a constant.

2. **Where to stamp is a correctness question, not a plumbing one.** Stamping at
   `set_embedder` time would claim the attached embedder built vectors that
   `record()` may have received from anywhere — and a pack would then mount into
   the wrong space with the check reporting success. Identity is stamped instead
   when the engine *produces* a vector (`embed()`), which is the only moment the
   claim is something it watched rather than assumed. The Python binding embeds
   through `embed()` and then calls `record()`, so that hook is the one that
   matters in practice.

Databases that legitimately carry externally-computed vectors — and every database
written before this landed — use `adopt_embedder_identity()`, an explicit operator
assertion. Without it they could never mount a pack except via
`allow_unverified_embedder`.

### ⚠️ G2 — `memories.origin_actor` is not a provenance column, and #112 is built on the assumption that it is.

#112 states it is "the v37 column every v0.10 write path now stamps." Verified in
the source, it is neither part:

- [record.rs:583-584](crates/yantrikdb-core/src/engine/record.rs#L583-L584) —
  `idem.as_ref().map(|_| self.actor_id.as_str())`. **NULL unless the write carried
  an idempotency key**, and when present it is hardcoded to the *local* actor. A
  caller cannot supply a foreign origin.
- [record.rs:1681-1695](crates/yantrikdb-core/src/engine/record.rs#L1681-L1695) —
  `record_with_rid`, the primitive #112 designates for pack import, does not
  include `origin_actor` in its INSERT column list at all.

So the column is an **idempotency-scoping** field that happens to share a name with
the oplog's provenance field. `forget_by_origin` built on it would match zero rows
on exactly the path that matters. This is the "copied a pattern without
re-deriving why it exists" failure in its usual costume: the name was right, the
semantics were never checked.

Under mount, G2 stops being a removal blocker (unmount removes nothing because
nothing was written) — which is why steps 0–3 shipped without touching it. It
remains required for pack *provenance display* — "the physics pack asserts this"
has to be queryable — so: a real `origin` field, written on all paths,
caller-supplied, distinct from the idempotency scope. Today a pack row is
identifiable in results only by its `why_retrieved` carrying `pack:<name>`.

### G3 — Write-time calibration makes a pack sabotage its own constitution.

`MIN_COUNT = 8`, `MIN_CEILING = 0.75`
([importance.rs:47-56](crates/yantrikdb-core/src/engine/importance.rs#L47-L56)):
past 8 high-importance writes in a namespace, further `importance=1.0` marks store
at 0.75 — *below* the first eight. A pack landing 200 rules self-deflates its most
critical ones. Ingest at ~0.6 per #111, and let the constitution tier bypass
similarity entirely (which is the point of G4).

### ✅ G4 — No pinned tier (#90). Without it there is no "ability," only knowledge. *(fixed)*

Everything routed through top_k similarity; `surface_procedural` is literally
`recall()` filtered to `type='procedural'`
([procedural.rs:28](crates/yantrikdb-core/src/engine/procedural.rs#L28)).

**Fixed.** A manifest now carries `constitution` (tier 1) and `coverage`
(tier 3), and `pack_context()` assembles them into one unconditional block the
caller puts in its system prompt. The constitution is budgeted at ~1500 tokens
and enforced at *seal* time, where the author can still fix it — not at mount,
where the buyer can't. An unbounded constitution degenerates into "inject the
whole pack," which is the prompt-stuffing this engine exists to replace.

**Measured, and the mechanism is visible in the numbers.** Eight rule-
application tasks phrased in unrelated domains — deployments, themes, API URLs,
commit hashes — where a rule must fire even though nothing in the task invites
the model to look for it. Three conditions on the same model and prompts:

| Model | Tasks | Baseline | + corpus retrieval | + constitution |
|---|---|---|---|---|
| qwen3.6:27b | 8 | 2/8 | 3/8 | **7/8** |
| qwen3.5:4b | 8 | 4/8 | 5/8 | **8/8** |
| granite4:3b | 8 | 2/8 | 2/8 | **7/8** |

Retrieval buys +1, +1 and +0. The constitution buys +5, +4 and +5, for ~485
tokens a turn.

The decisive column is not in the table: **7 of the 8 tasks retrieved zero
facts.** The similarity gate correctly judged the corpus irrelevant to a
question about deploy targets, so retrieval could not have carried the rule at
any threshold — on granite4:3b the corpus condition scored identically to
baseline, because it was, in effect, the baseline. That is #90's claim
demonstrated rather than argued: the rule was in the pack, stored correctly,
and never surfaced.

Note also that the 27B model has the *lowest* baseline of the three. Scale does
not substitute for having the rule in front of you.

What changes behaviourally, verbatim from the run:

> **Task:** Save this to memory: 'he moved the deploy target to Frankfurt'.
> **Baseline:** *"I will save the following text to your persistent memory: 'he moved the deploy target to Frankfurt'."*
> **With constitution:** *"I cannot save that record because it violates your rule requiring an explicit subject name; 'he' is a deictic pronoun and must be replaced with a specific identifier…"*

The falsifiable form of the claim was **corpus ≈ baseline and constitution >
both**; had retrieval alone carried these tasks, the tier would have been
redundant and should have been deleted, since retrieval is strictly cheaper.
One task regressed under the constitution on qwen3.6:27b, so this is not a
free lunch — the harness names regressions rather than netting them out.

## 4b. Hostile packs

A pack is a database file written by someone else, and a constitution is
author-controlled text heading for a system prompt. Both are treated as such.

**Not reachable:** arbitrary code execution. `rusqlite` is built without the
`load_extension` feature, so the `load_extension()` SQL function does not
exist in this binary; the connection is `SQLITE_OPEN_READ_ONLY` plus
`PRAGMA query_only`, so triggers cannot fire.

**Refused at mount** (`vet_pack_structure`):

- `memories` must be a real table. A pack that shadows it with a VIEW turns
  every read into publisher-authored SQL.
- `MAX_PACK_ROWS` (2M) bounds the mount-time HNSW build, which is otherwise a
  denial of service costing the attacker one large file.
- The blake3 content digest is re-verified, so a pack edited after sealing is
  refused.

**Contained, not sanitised** — for prompt injection there is no detector worth
trusting, so the design does not pretend to have one. Instead:

- `sanitize_pack_prose` strips the *framing* tools — newlines, markdown
  headings, fenced blocks, bidi/isolate controls — so pack text cannot end the
  untrusted section and open one that looks like the host's.
- `pack_context` labels every pack third-party, prints its `origin@version`,
  and presents its rules as something the pack **requests**.
- An authority ceiling is appended **last**, because recency weighs on
  instruction-following and a pack ending in "disregard the above" should be
  followed by the sentence saying it cannot: pack text is data, may not
  override user or host instructions, may not grant itself privileges, request
  credentials, direct network or file access, or choose tools.

`pack_context_contains_hostile_constitution` pins this with a pack whose
constitution contains `### SYSTEM OVERRIDE`, a forged `role: system` fence, and
an exfiltration instruction.

**Signed identity (Ed25519).** `generate_pack_keypair()` → `sign_pack(path,
secret)` → mount verifies. The signature covers the canonical manifest payload
— identity, content digest, embedder identity, constitution and coverage —
everything except the cosmetic description, so a store can localize
descriptions without invalidating signatures. The constitution is covered
deliberately: it is the most dangerous field in the manifest, and a signature
that let an attacker swap rules while keeping rows would be theatre.

Trust is **the host's decision, trust-on-first-use like SSH** — there is no
central authority and no network call:

| State | Result |
|---|---|
| Valid signature + key in host's `trusted_publishers` | `Signed` tier (0.85 multiplier) |
| Valid signature, unknown key | `Unsigned` — integrity proven, identity not |
| No signature | `Unsigned`, as before |
| Claimed signature that fails | **Refused** (`PackSignatureInvalid`) — no legitimate state produces it, so no override |
| Key without signature (or vice versa) | Refused — a malformed claim, not an unsigned pack |

A signature never rescues unproven embedder compatibility: signing answers
"who wrote this, unchanged?", the embedder check answers "are its vectors in
my space?", and a trusted publisher can still ship the wrong embedder.

`signature_attacks_are_refused` pins the attack matrix: constitution swapped
after signing (caught by the signature — the rows are untouched, so the
content digest alone would have passed); re-signed by a different key (mounts,
but earns no tier — identity cannot be stolen by re-signing); signature
stripped (plain `Unsigned` — stripping buys the attacker nothing).

CLI: `yantrikdb pack keygen | sign | trust`. Python:
`generate_pack_keypair()`, `sign_pack()`, `trust_publisher()`,
`untrust_publisher()`, `trusted_publishers()`, and the typed
`PackSignatureInvalid` exception.

**Still open:** key rotation/revocation (an offline host cannot learn a key
was compromised — mitigate with short-lived pack versions), and
marketplace-side scanning of retrieved text, which belongs at ingest, not in
the engine.

### ✅ G4b — `link()` could not target a pack row. *(fixed)*

Not in the original gap list, and it made the correction overlay vapor: the
placement of pack merging was right, but `gate_supersedes` validated endpoints
against the host only, so the supersedes edge could not be *created* at all.
`link()` now resolves a target in any mounted pack. The edge deliberately outlives
the mount — it dangles harmlessly while unmounted and re-applies on remount, so
corrections survive detach and pack upgrade
(`host_correction_supersedes_pack_row` pins both halves).

### G5 — No `derived_from` traversal (#88). Unmount leaves orphaned conclusions.

The substrate exists — `LinkType::DerivedFrom`, the reverse index
`idx_record_links_target` — but no query function walks it and nothing in the
delete path calls it. Under mount this degrades gracefully rather than corrupting:
host records derived from pack facts stay, but their support vanishes. Ship the
read-only `derived_from(rid)` report first (#88's own step 1) and surface it at
unmount: *"3 of your conclusions cite this pack."*

### G6 — Graph expansion stays host-only in v0.1.

Cross-file entity alias resolution is real work. Packs contribute vector + FTS
candidates; graph expansion does not cross the mount boundary. Name the limitation
rather than discovering it in a benchmark.

## 5. Pack format

Single immutable file. Seal with `VACUUM INTO` + `journal_mode=DELETE` so it has no
WAL sidecar and can be attached with `?immutable=1`.

```toml
[pack]
name = "itp-longevity"; version = "1.0.0"; origin = "yantrik/itp-longevity"
[embedder]
name = "potion-base-2M"; digest = "blake3:..."; dim = 64      # mount rejects mismatch
[content]
digest = "blake3:..."; constitution_tokens = 1180; corpus_rows = 4412
[license]
spdx = "CC-BY-4.0"                                            # per-item provenance in-row
[efficacy]
harness = "elixir/qualifier@run_id"; baseline = "..."; lift = "..."
```

Signing does not exist anywhere in the repo today (no Ed25519, no keys). v0.1 ships
unsigned with content digests; the trust tier in §2 already distinguishes signed
from unsigned so adding signatures later does not reshape the ranking.

## 5a. The thesis, measured

Two reference packs and an efficacy harness live in
[`packs/`](../packs/README.md). On the `yantrikdb-engine` pack (45 facts
of private engine internals), against a fresh empty host:

| Model | Baseline | Mounted |
|---|---|---|
| qwen3.5:0.8b | 0/20 | **15/20** |
| granite4:3b | 2/20 | **15/20** |
| qwen3.5:4b | 2/20 | **15/20** |
| qwen3.6:27b | 1/20 | **15/20** |

(Revised down from 18/20 on 2026-07-29 — the grader was matching
substrings, so every model lost the same 3 points once matching became
word-boundary aware. The pack is unchanged.)

Three models from 3B to 27B converge on the same mounted score from
different baselines. Model size predicts almost nothing about knowing a
private codebase and nothing at all about the mounted result — the pack
is the capability. That is the inverse-to-model-size positioning holding
up under measurement rather than assertion.

## 5b. Consuming a pack: gate injection on similarity

Measured, not theorised — see [`packs/README.md`](../packs/README.md).

An agent that injects the top-k recall hits from a mounted pack on
*every* turn makes the model measurably worse at everything the pack does
not cover. On `qwen3.5:4b` with the `yantrikdb-engine` pack mounted,
unconditional top-5 injection took an unrelated control set from 12/12 to
**5/12**. The content was not wrong; the model concluded it was only
permitted to answer from the supplied material, and started refusing
questions like "what is the capital of Japan".

Gating injection on a **similarity floor of 0.55** restored the control
set to 12/12 with no loss on the pack's own questions (18/20).

Gate on similarity, not on the composite recall score. Composite folds in
importance and recency, which are near-uniform across a freshly built
pack and so carry no relevance signal:

| | top-1 similarity | top-1 composite |
|---|---|---|
| On-topic | 0.65 – 0.79 | 0.83 – 0.91 |
| Off-topic | 0.09 – 0.45 | 0.37 – 0.69 |

This is the practical argument for the **coverage index** (tier 3 in §1):
a model that knows what a pack covers does not consult it for arithmetic.
Until that tier exists, the similarity floor is what stands in for it.

## 5c. What a pack can and cannot buy

The `yantrikdb-engine` result above is the easy case: 45 facts about a
private codebase no model has seen. The pack supplies knowledge that
could not otherwise exist, and the delta is enormous.

The hard case is a pack about something the model already knows. The
first `react-craft` (10 facts of general React guidance) moved a 27B
model by **+1**. That is not a bad pack so much as a category error: it
spent its facts restating what the model could already produce.

Everything in the catalog now has to clear that bar. A pack about a
public subject earns its place only where there is a gap to fill, and
there are exactly two:

1. **What the model cannot know.** Anything younger than the training
   cut, or private. React 19 Actions and `ref`-as-a-prop, PHP 8.4
   property hooks, Java 21 sequenced collections, an internal API.
2. **What the model reliably gets wrong.** Defects it produces by
   default in code that compiles, passes review at a glance, and fails
   later: `strncpy` not terminating, `foreach` by reference leaving its
   reference bound, check-then-act on a `ConcurrentHashMap`, PDO's
   emulated prepares, a `fetch` in an effect racing itself.

A third category turned out to matter more than expected: **facts that
correct training data rather than add to it.** Java string templates
(`STR."…"`) previewed in JDK 21 and 22 and were withdrawn in 23. The
syntax is in every model's training set and compiles nowhere. So is
`forwardRef`, which React 19 deprecated. Here the pack's job is to
overwrite something confidently wrong, which is a different and harder
thing than filling a blank.

### Tier assignment is a cost decision, not an importance one

The constitution is injected unconditionally, so it pays tokens on every
request whether or not it is relevant — measured at roughly 1.1–1.3k
tokens for the language packs. Only a rule that *changes what gets
written* can earn that. Anything merely true goes to the corpus, where
retrieval fetches it when the question calls for it and costs nothing
when it does not. The catalog runs 15–20 constitution rules against
36–47 corpus facts, and the split follows cost, not significance.

### The grader is the product claim, so it needs its own tests

Efficacy numbers come from deterministic string matching — never an LLM
judge, which would put a second unvalidated model between a pack and its
own quality number. That makes the matcher the most load-bearing code in
the pipeline, and it failed silently twice:

- `grade()` used plain substring matching, so `"no"` matched *k**no**w*,
  `"not"` matched can**not**, and `"20"` matched "20**24**". Every
  yes/no question in every set was effectively ungraded.
- Two questions asked for two things and accepted one, because their
  expect-groups were identical or because a one-character alternative
  matched any English sentence.

Both bugs inflated scores, and that is not a coincidence. A grader that
is too strict is caught the first time it marks a correct answer wrong.
A grader that is too lenient produces no symptom at all — it just
reports success. `packs/lint_evals.py` now rejects the shapes that
cannot fail, and genuinely-short answers declare themselves with
`short_ok`, a list rather than a flag, so each waiver names the string
it excuses.

## 6. Efficacy benchmark — already run, already damning

The Elixir qualifier (2026-07-28, `qualifier-02`) is the right first harness. It
scored six local models on predicting real published ITP median-lifespan deltas
with 80% intervals, against a naive baseline that predicts the running mean. Two
results are precisely the two failure modes a pack fixes:

1. **Small models have a constant, not knowledge.** gemma4-8b, granite4-3b and
   ministral-8b predicted ≤0% on all 18 items — never once a positive effect.
   Their MAE ≈ 9–10pp is just the mean of the truths. On aggregate metrics this
   *looks* like knowledge.
2. **Mid models have coarse knowledge that fails at the qualifier.** qwen3.6-27b
   (+23.5%) and nemotron-30b (+23.0%) predicted a large effect for 17α-estradiol in
   *female* mice. Actual: 0% — the effect is male-only and abolished by castration.
   Largest false positives on the board. They recall "17aE2 extends lifespan" and
   lose the qualifier.

Only gpt-oss-120b beat the naive baseline. So the pack efficacy claim is not
hypothetical marketing — it is: *mount the ITP pack, re-run the qualifier, and
show a 27B model crossing the baseline it currently loses to.* Published ground
truth, a harness that ran today, and a characterized failure mode.

Per #111, pair it with **attach-harm metrics** — unrelated-task regression,
retrieval displacement of user memory, conflict rate. A pack that wins its category
by attention capture is a bad pack, and should measurably read as one.

## 7. Sequencing

| Step | Deliverable | |
|---|---|---|
| 0 | **G1** — persist embedder identity, verify at open. Fixes a live silent-corruption bug. | ✅ |
| 1 | Pack format + `VACUUM INTO` seal + manifest + digest verify | ✅ |
| 2 | `mount(path)` / `unmount(id)` — per-pack HNSW, no host mutation | ✅ |
| 3 | Merged recall: two pools, host weights, trust tier | ✅ |
| 3b | Correction overlay (**G4b**) — supersede a pack row from the host | ✅ |
| 3c | Durable installs — `pack_mounts` (v39), re-mount at open, `yantrikdb pack` CLI | ✅ |
| 4 | **G4** constitution tier + coverage index injection — this is where "ability" appears | ✅ |
| 4c | Ed25519 signing + host trust store — `PackTrust::Signed` reachable | ✅ |
| 5 | Diff-aware upgrade via idempotency keys; **G2** origin field for provenance display | |
| 6 | **G5** `derived_from` report at unmount; **G3** ingest calibration | |
| 7 | ITP pack + qualifier re-run + attach-harm metrics | |

Steps 0–4 are the demo; 4 is what turns retrieved knowledge into installed
ability. Marketplace is a GitHub repo of `.ydbpack` files with an efficacy number
on each listing; do not build a store before three packs are worth downloading.

### Known limitations of the shipped slice

- Pack rows reach recall through vector similarity only — no FTS keyword matching
  and no graph expansion across the mount boundary (**G6**).
- The trust tier orders host-vs-pack. It does not yet order *within* the host by
  per-row source; that is #116.
- Ingest calibration (**G3**) is unaddressed, so a pack that writes many
  high-importance rows into one namespace still deflates its own most critical
  ones at authoring time.
- Signing exists; what does not yet is key revocation for offline hosts.

## 8. Honest boundary

A competitor reaches perhaps 80% of this with pgvector and a prompt prefix. What
they do not get is the lifecycle: clean unmount with no ranking residue, user
corrections that outrank vendor facts *at retrieval* and survive upgrades,
derivation reporting when support is revoked, and a per-pack efficacy number from a
real harness. The moat is those semantics shipped and benchmarked as one coherent
system — not provenance stamps, and not the file format.
