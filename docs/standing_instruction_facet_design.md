
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
- **`first_mention_turn`** — v1 captures `first_mention_at` only; turn
  identity from source metadata is not yet extracted.
- **Ordinary auditable hit fields** — lane results carry rid, text,
  clocks, and evidence links, but not yet `score`, `why_retrieved`, or
  an explicit `source_actor` field.

**Detector narrowings (documented false-negatives, by design):**

- A directive whose first body word is capitalized ("Always CC Priya…")
  does not fire.
- The function-word lead stoplist includes the copula family, so
  "Always be X" directives do not fire — lowercase "always be
  closing"-class titles are otherwise indistinguishable from prose.
