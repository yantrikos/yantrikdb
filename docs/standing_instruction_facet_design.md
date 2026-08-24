# Standing-Instruction Facet Design

Status: v1 opt-in implementation merged. Product replay passed exactly; the
full-400 evaluation preserved the target-category lift but rejected global
default-on composition under the frozen no-regression gates.

## Evidence

A query-independent transform extracted only user-authored turns beginning with
`Always`, persisted their source turn identity, and placed the complete
conversation-level set before ordinary memories under a lower total token
budget than control. Canonical instruction retention moved from `0.9284`
(`36/40`) to `1.0000` (`40/40`).

The frozen `instruction_following` arm passed in two fresh model sessions:

| run | control | treatment | delta | wins/ties/losses | bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| discovery | 0.80000 | 0.90000 | +0.10000 | 7/32/1 | [+0.01875, +0.20000] |
| replication | 0.75000 | 0.89375 | +0.14375 | 9/30/1 | [+0.05625, +0.24375] |

This promotes the mechanism, not the benchmark fixture. The product must retain
authoritative provenance and explicitness; it must not turn every inferred
preference or assistant acknowledgement into an instruction.

## Record Contract

A standing instruction is an answer-sized, source-backed facet with this
logical shape:

```text
facet_type       = standing_instruction
text             = exact user-authored directive
evidence_ids     = one or more engine-resolved source RIDs
source_actor     = user
detector_version = explicit_always_v1
first_mention_at = earliest source occurrence time
first_mention_turn = earliest source turn when available
created_at       = newest evidence availability time
namespace        = namespace shared by every source
```

The minimal implementation may use the existing synthesis descriptor with
`synthesis_axis = "standing_instruction"` and
`synthesis_granularity = "atomic"`. If a generic typed `facet_type` column is
introduced for later facet kinds, it must remain nullable and NULL must preserve
ordinary-record behavior bit for bit. Recall-critical type information cannot
exist only in encrypted JSON metadata.

`evidence_ids` are resolved through `synthesis_dependencies`, never trusted
from caller metadata. Correction or forget of any evidence invalidates the
facet through the existing lifecycle. A facet with missing, cross-namespace,
inactive, assistant-authored, or revision-mismatched evidence is unverified and
recall-ineligible.

### Time Semantics

Standing instructions normally describe no real-world event date. Their
ordering clock is therefore the source record's historical occurrence time:

- `first_mention_at` is the earliest source occurrence time;
- `created_at` is the newest source availability time, as for every synthesis;
- `first_mention_turn` is used for conversation ordering when present;
- ingestion time is used only when the source genuinely has no historical
  timestamp.

The detector must never invent a date from the directive text. For a
single-source instruction with no explicit historical timestamp,
`first_mention_at` and `created_at` both fall back to that source record's
creation time. This is the same fallback contract used by organized concerns.

## Write-Time Detection

Version 1 is deliberately narrow:

1. Inspect only a complete direct user turn. Speaker provenance must be
   verified by the ingestion boundary, not guessed from text.
2. Ignore leading whitespace and require the first lexical token to be
   `Always`, case-insensitively.
3. Store the complete directive without paraphrasing. Transport annotations
   are excluded before detection; ordinary punctuation is preserved.
4. Reject assistant/system/tool turns, quoted or embedded occurrences, empty
   directives, titles such as `Always Sunny`, and prose such as
   `I do not always ...`.
5. Emit at most one facet per normalized directive and source revision. The
   idempotency key includes the source RID, normalized directive, and detector
   version.

The detector is an admission candidate generator, not an authority oracle. A
record is eligible only after the provenance and dependency checks above pass.
The API must also permit an application to submit an explicit typed standing
instruction without relying on English detection, subject to the same
user-provenance rules.

Version 1 does not detect `Never`, `Stop`, preference-like statements, or
assistant summaries. Those need separate evidence and contradiction semantics.

## False-Fire Audit

The implementation must expose a dry-run audit with these counts, scoped by
namespace and detector version:

```text
user_turns_scanned
candidates
accepted
rejected_unverified_provenance
rejected_non_directive
duplicate_candidates
would_write
```

Dry-run never writes records, dependencies, idempotency claims, or cursor
progress. The test corpus must include positive directives and adversarial
negatives for assistant acknowledgements, quoted instructions, titles,
negation, mid-sentence `always`, empty suffixes, and cross-namespace evidence.
False-fire examples become permanent regression tests before default-on.

## Recall Salience

Standing instructions use a dedicated facet lane before ordinary context
assembly. They do not compete invisibly as ordinary vector hits.

- Only active, verified, same-namespace facets are eligible.
- User-authored directives outrank assistant acknowledgements categorically.
- When the eligible set fits the configured facet budget, return the complete
  set in first-mention order. This preserves the tested 3-5 item mechanism.
- When it exceeds the budget, relevance may select a bounded subset, with a
  deterministic recency/diversity fallback. The response must expose omitted
  facet count; silent truncation is not acceptable.
- Facets consume a separately reported context budget. If the caller requests
  a fixed total budget, ordinary low-ranked blocks are removed only after the
  facet set is frozen, matching the evaluated transform.
- Existing `recall` behavior remains unchanged unless facet inclusion is
  explicitly enabled. Default-on requires the acceptance gates below.

The returned record remains an ordinary auditable result: RID, text, score,
`why_retrieved`, clocks, source actor, and evidence links are visible. The lane
does not inject hidden prompt text.

## Supersession And Conflicts

An `Always` directive does not automatically expire. A later explicit directive
may coexist until the application or a future contradiction policy links or
supersedes it. Version 1 must not silently deactivate an older instruction from
embedding similarity alone. `correct` and `forget` retain their existing
revision and invalidation semantics.

This conservative rule can surface conflicting instructions, which is safer
than permanently choosing the wrong one at write time. Conflict resolution is
a separate typed-facet capability and requires its own benchmark gate.

## Persistence And Compatibility

- New descriptor fields are additive and nullable/defaulted.
- Old databases migrate without rewriting ordinary records or embeddings.
- Oplog payloads carry the typed facet descriptor and complete dependency
  closure; followers fail closed as `unverified` when evidence is absent.
- Packs and replication preserve facet type, detector version, provenance,
  clocks, lifecycle state, and evidence links.
- Namespace filters apply at admission, replication, audit, and recall.
- Idempotent replay returns the same RID for the same source revision and
  detector output; changed output for the same generation conflicts.

## Product API

Names are illustrative; behavior is normative:

```python
db.extract_facets(
    source_rids,
    facet_types=["standing_instruction"],
    dry_run=False,
)

db.recall(
    query,
    include_facets=["standing_instruction"],
    facet_limit=8,
    facet_token_budget=2048,
)
```

The extraction call is bounded, resumable, and safe to retry. The recall call
reports selected and omitted facet counts. Language-specific automatic
detection is replaceable; the persistence and recall contracts are not.

## Acceptance Gates

1. Unit and integration tests prove detector false-fire behavior, provenance
   rejection, namespace isolation, idempotency, correction/forget invalidation,
   replication, encrypted-metadata operation, and date fallback.
2. A product replay over the frozen 40 contexts reconstructs the evaluated
   panels from persisted facets with exact ordered query IDs and no gold/query
   access during extraction.
3. The frozen instruction category retains at least `+0.05` mean lift with more
   wins than losses in a fresh session.
4. Before default-on, a preregistered full-400 paired run must show no overall
   regression and no material category regression outside instruction
   following. Exact thresholds and artifact hashes are frozen before calls.
5. Automatic background extraction remains opt-in until real, consented corpus
   false-fire rates and write amplification are measured. Real companion
   memories are never sent to an external benchmark model.

Failure at a gate leaves the feature opt-in and records the result. It does not
authorize broadening the detector.

## Implementation Status — v1 Scope (amended at implementation review)

The v1 implementation (feat/standing-instruction-facet) was cold-reviewed
against this contract; the following scope decisions were made jointly by
implementer and reviewer and are normative until amended:

**Narrowed for v1, by explicit decision:**

- **The recall lane reads the host store only.** Facet DATA is fully
  preserved through pack seal/mount (rows, dependencies, provenance,
  clocks — verified by review), but a mounted pack's facets are not
  lane-visible. Enabling that requires deliberate design for trust-tier
  interaction with facet salience and cross-namespace semantics under
  mount. The ignored test
  `mounted_pack_preserves_standing_instruction_facet_lane` in
  `tests/pack_mount.rs` is the named v1.1 acceptance.

**Deferred, named so absence is never mistaken for completion:**

- **Explicit typed submission API** — the contract requires an
  application-submitted path that bypasses English detection under the
  same provenance rules; v1 is detector-admission only.
- **Ordinary auditable hit fields** — lane results carry rid, text,
  clocks, and evidence links, but not yet `score`, `why_retrieved`, or
  an explicit `source_actor` field.

**Detector narrowings (documented false-negatives, by design):**

- A directive whose first body word is capitalized ("Always CC Priya…")
  does not fire.
- The function-word lead stoplist includes the copula family, so
  "Always be X" directives do not fire — lowercase "always be
  closing"-class titles are otherwise indistinguishable from prose.
