# What a claim-graph sees that a vector DB doesn't

*A Wirecard case study against the RFC 008 substrate.*

## The problem, in one sentence

If your agent thinks 20 news outlets citing the same FT story are 20 independent sources, your agent is wrong — and the standard tools (vector databases, knowledge graphs, LLM context windows) can't tell the difference. This is one case where that mistake had six-year consequences, and what a different kind of database does about it.

## The case

In 2020, Wirecard AG — a DAX-30 German payment processor — collapsed after its auditor (EY) and its own filings had attested for six years to €1.9 billion held in escrow at two Philippine banks. On 2020-06-18 the Financial Times obtained internal documents showing the cash wasn't there. On 2020-06-19 and 2020-06-21, BPI, BDO, and Bangko Sentral ng Pilipinas formally denied ever holding the accounts. Wirecard filed insolvency on 2020-06-25.

The public record about that €1.9B carries six claims from five sources:

| # | Source | Polarity | Valid from → to | Lineage |
|---|---|---|---|---|
| 1 | wirecard.filing | **+1 (exists)** | 2018-12-31 → 2020-06-18 | `[wirecard]` |
| 2 | ey.audit | **+1 (exists)** | 2014-01-01 → 2020-06-18 | `[wirecard, ey]` |
| 3 | ft.investigation | −1 (denies) | 2019-01-30 → ∞ | `[ft]` |
| 4 | bpi.statement | −1 (denies) | 2020-06-18 → ∞ | `[bpi]` |
| 5 | bdo.statement | −1 (denies) | 2020-06-19 → ∞ | `[bdo]` |
| 6 | bsp.circular | −1 (denies) | 2020-06-21 → ∞ | `[bsp, bpi, bdo]` |

The naive tally: 2 yes, 4 no. Attack-dominant by 2.

The problem with that tally is buried in the lineage column.

## What the naive tally misses

**EY was not an independent second source for Wirecard's position.** EY audited the documents Wirecard provided; those documents were forged trustee confirmation letters. EY's claim that the €1.9B existed shares its epistemic chain with Wirecard's claim — not with any independent bank verification.

**BSP was not an independent fourth source for the denial.** The Bangko Sentral ng Pilipinas circular restated the two commercial banks' own denials. It is correct and authoritative, but it is not a fourth witness.

A system that counts claims rather than modeling epistemic dependency calls this 2-to-4. A system that tracks source lineage per claim calls it ~1.6 to ~3.6.

## What the substrate does

RFC 008 defines `⊕` (the accumulation operator) as:

> σ(c) = Σ_k ω_k · mass_k
> ω_k = 1 / (1 + 0.5·D_k + 0.3·P_k + 0.7·S_k)

D_k is the Jaccard similarity of claim k's `source_lineage` set against the union of every *other* live claim's lineage (leave-one-out). If claim k shares sources with the rest of the set, its ω_k drops below 1 and its mass is discounted.

For Wirecard:

```
  naive σ (support count) = 2.000
  naive α (attack count)  = 4.000

  substrate σ (dependence-discounted) = 1.600
  substrate α (dependence-discounted) = 3.578
```

The 1.6 reflects the fact that EY's `[wirecard, ey]` overlaps `[wirecard]` by 50% — EY's ω ≈ 0.6, Wirecard's ω = 1.0. Two claims become effectively 1.6 independent supports.

The 3.578 reflects BSP's `[bsp, bpi, bdo]` overlap with `[bpi]` and `[bdo]` individually — BSP's ω drops. Four denials become effectively 3.6 independent witnesses.

This is not a tuning knob. It is grounded in set-level lineage, computed deterministically, order-invariant.

## Ablation — which layer does what

| Model | σ (supports) | α (attacks) | Says |
|---|---|---|---|
| **Naive polarity count** | 2.000 | 4.000 | 2-to-4, attack-dominant |
| **+ lineage discount (⊕ only)** | 1.600 | 3.578 | Closer to 1.6-to-3.6 — EY and BSP partially redundant |
| **+ temporal split (⊕ + ⋈)** | 1.600 / 3.578 | (as above) + PRESENT_TENSE_CONFLICT flag on 4 of 8 opposite pairs; 4 pairs marked as historical state changes, not active contradictions | Distinguishes "Wirecard still claims this right now, FT denies" from "Wirecard pulled its claim, then BDO denied" |

The rows matter for different decisions. If all you need is "which side has more witnesses," row 1 is enough. If you need "how many independent sources actually agree," you need row 2. If you need to answer "is this an ongoing contradiction I have to resolve *now*, or a superseded claim from last month," you need row 3.

Each row adds structure, not a knob. The outputs are deterministic functions of the input lineage + temporal fields; there are no hyperparameters tuned on Wirecard.

## Why this is not just polarity scoring

The obvious critique: "you made a scoring function and called it a database primitive."

Partially fair. ω_k does produce numbers. What makes this a substrate choice and not a score:

**Inspectability.** Every component of the output is a deterministic function of schema fields you can query: `support_mass`, `support_effective_independence`, `same_source_opposite_polarity_count`, `temporal_overlap_conflict_count`, `heuristic_flags`. If ⊕'s output surprises you, you read the lineage columns and see exactly why. A neural ranker or hand-tuned confidence score doesn't give you that.

**Determinism.** Same inputs always produce same outputs. The recompute is order-invariant (proved by property tests), content-hashed (so repeat computation is a no-op), and reproducible across machines. A scoring function with tunable weights produces different outputs as you tune; this substrate has fixed definitional weights (0.5 / 0.3 / 0.7) and explicitly rejects "learn them" as a v1 path.

**Structural differentiation.** The contest operator doesn't collapse into a scalar. It produces five separate counters and a bitset because "same-source contradiction" (BPI saying yes to BPI-sourced claim) is a different failure mode than "extraction-pathology contradiction" (two extractors on the same document disagreeing) and a different mode from "historical state change" (Wirecard pulled its claim before BDO denied). Flattening these into one confidence number discards the information that tells you what to *do* about the disagreement.

**Non-goals.** This is not a trust score. It does not tell you who to believe. It does not handle "unknown unknowns" — if your lineage labels are wrong, ⊕ is wrong the same way. It does not replace source vetting. What it does is make the disagreement pattern *structurally visible* in a way a polarity scalar does not.

If after reading this you still think "that's just scoring with extra steps," that's a legitimate read — and one of the things I'm trying to find out from posting this.

## What else fires

The contest operator `⋈` produces a structured Γ(c) — not a scalar confidence, a shape of contest.

```
  same_source_opposite_polarity_count            = 0
  same_artifact_extractor_polarity_conflict_count = 0
  temporal_overlap_conflict_count                 = 4
  temporal_separable_opposition_count             = 4
  referent_schema_heterogeneity_count             = 0
  heuristic_flags bitmap:  PRESENT_TENSE_CONFLICT set
```

The temporal split is the interesting one. Of the 2×4=8 opposite-polarity pairs, four have *overlapping* validity intervals (real present-tense contradictions) and four are *temporally separable* — meaning one side's assertion had already expired before the other's was made.

Concretely: Wirecard kept asserting the €1.9B through 2020-06-18. BDO denied starting 2020-06-19. BSP denied starting 2020-06-21. By the time BDO and BSP spoke, Wirecard's own filings no longer claimed the balance (the company had pulled it). Those pairs count as *separable opposition* — states changing over time, not contradictions requiring reconciliation.

A system that conflates the two produces a noisier contradiction signal than it should. A system that separates them tells you exactly which disputes are active *right now* and which are historical state changes.

## What temporal coherence says about the whole story

```
  τ (temporal_coherence) = 0.8
```

τ counts polarity flips across the ordered claim sequence — one flip out of five transitions. Wirecard's balance existed (per its own and EY's filings) for six years, then was flipped to "doesn't exist" by the banks' denials. τ = 0.8 reflects that most of the sequence was coherent, with one decisive inversion.

If τ were 1.0, nothing ever changed. If τ were 0.0, the polarity flipped on every claim. 0.8 is the signature of a long-held assertion that failed.

## What the cognitive moves trace

The RFC 008 substrate records not just what was claimed but what reasoning transformed the record. For Wirecard:

- **FT contradiction_triage**: consumes Wirecard + EY claims, produces the FT denial
- **BPI source_audit**: consumes Wirecard's claim, produces BPI's denial
- **BDO source_audit**: same, produces BDO's denial
- **BSP contradiction_triage**: consumes BPI + BDO, produces the central bank determination

Each move is append-only. Each specifies its inputs, outputs, and version. Each can be corrected via a `move_correction_event` without mutating the original.

The `source_audit` move carries a precondition axiom: it requires `ψ_ancestral < 1.0` (there must be at least some external ancestry to audit). Applied to a fully self-generated claim, the substrate refuses — incoherent by definition. For Wirecard the audit was applicable because the claim had external ancestry (the forged documents still came from somewhere real).

## What this is not

This is not a scandal report. The fraud is a matter of public record; the German courts and the Munich Prosecutor handled that in 2023. What the substrate demonstrates is the epistemic accounting that would have made the disagreement between sources legible at query time.

Specifically, it is not:
- a scoring function
- a confidence estimate
- a trust weight tuned to flatter the "right" answer

It is a deterministic, order-invariant, content-hashed materialization of what the source record actually supports, given the observable dependencies between sources.

## Honest acknowledgments of what didn't fire

- **`same_source_opposite_polarity_count = 0`** — the gate requires *identical* normalized source_lineage sets with opposite polarity. EY's `[wirecard, ey]` is not identical to Wirecard's `[wirecard]`, even though they overlap. The strict gate is by design (precision over recall). The lineage overlap gets caught by `⊕`'s ω_k, not by the contest counter. Different layers, different granularities.

- **`DUPLICATION_RISK` unset** — σ = 1.6 is below the threshold of 2.0 with `support_effective_independence < 2.0`. For the flag to fire we'd need more supports sharing lineage. This matches the intuition: two claims collapsing to 1.6 independence is not rumor amplification, it's minor dependence.

- **`ψ_a = 0.0`** — the ancestry BFS found no self-generated claims in the two-hop upstream of the proposition's claims. Correct: no part of this chain is self-generated by a reasoning agent.

Each non-firing is the substrate being honest about what it can and can't see from the data given.

## The 120-line example

[`examples/showcase_wirecard.rs`](../../crates/yantrikdb-core/examples/showcase_wirecard.rs) runs this end-to-end:

```bash
cargo run --example showcase_wirecard
```

No HTTP server, no configuration. Opens an in-memory database, seeds the six claims, records the moves, prints the substrate state. The code is the proof.

## What this doesn't prove

One adversarial case is not a benchmark. What Wirecard tells us:

1. **The substrate math is tractable and produces meaningful numbers** — not just axiomatic in tests, but survives a real sourced case.
2. **The discounts reflect a real epistemic story**, not a curated one — EY *was* dependent on Wirecard's documents, BSP *was* dependent on the commercial banks. The substrate didn't have to be told; the lineage did the work.
3. **The temporal split catches something a scalar confidence would miss** — "states changing over time" is not the same failure mode as "sources disagree right now," and the substrate calls it out.

What it doesn't tell us:

- Whether `⊕`'s weights (0.5, 0.3, 0.7) are calibrated for the generic case.
- Whether the substrate helps on domains where source_lineage is weak or unknown.
- Whether automated extraction of lineage from natural-language sources is feasible.
- Whether outsiders see this as a meaningful abstraction or as a formalized scoring function in a trenchcoat.

Those last one is the point of posting this walkthrough. The math is what it is; whether the abstraction is *useful* depends on whether someone besides me finds it legible enough to build on. If the answer is "I'd just use Postgres + application logic," that's a genuine answer.

## Known weaknesses (surfaced by adversarial review)

Before publishing this, I ran the essay past a panel of eight AI models in red-team mode — different families, different priors — asking each to find the sharpest technical attacks. The same four objections kept coming back, plus two that caught real issues:

**Lineage extraction is the unsolved problem.** The substrate works if `source_lineage` is correctly populated. Extracting lineage from natural-language sources (who cites whom, which audit depends on which document) is not a solved problem. In Wirecard, I hand-curated the lineage column — that's the entire load-bearing assumption. Every attacker flagged this as the essay's central unearned claim. Defensible if the substrate is pitched at domains with already-structured provenance (citations, code dependencies, workflow audits) rather than as a drop-in for "any corpus of text."

**The weights (0.5, 0.3, 0.7) are arbitrary.** Yes. The essay says "not a tuning knob" because the weights are fixed in v1, not because they're calibrated. Swap them to 0.3 / 0.5 / 0.9 and Wirecard's σ changes. "Learn them from labeled data" is deferred to a v2 that doesn't exist yet. Until then, treat ω_k as a structural *shape* — monotone in overlap, bounded, deterministic — not as a calibrated probability.

**Jaccard measures label overlap, not information dependence — and can invert evidence theory.** An FT investigation based on a Wirecard-insider leak has zero Jaccard overlap with Wirecard's own claim but is 100% informationally dependent on it. The mirror case matters just as much: two independent auditors citing the same SEC filing are *corroborating*, not redundant, but the substrate discounts them for sharing `[sec]` in lineage. The substrate catches declared-source redundancy; it doesn't catch hidden causal dependence, and it can under-weight genuine corroboration. In fraud detection the multi-source consensus signal is the strongest thing you have — a discount that treats it as partial duplication can invert the decision in exactly the cases you wanted to get right.

**Monotonicity violation (genuinely surprising).** Adding a claim with entirely unrelated lineage changes existing claims' ω values — because `D_k` is measured against the *rest-of-set* union, which grows. Concretely: A=[w], B=[w,e] → D_A = 0.5. Add orthogonal C=[x] → D_A drops to 0.333, ω_A rises. A weighs *more* after an unrelated claim joins the set. Two readings: (a) this is what "independence relative to live set" is supposed to mean, and the rise reflects A's larger relative diversity; (b) it violates independence of irrelevant alternatives and lets attackers inflate ω by padding with spurious claims. The substrate as it stands doesn't distinguish these readings. Watch for gaming if lineage is ever user-supplied.

**Adversarial lineage padding.** Closely related to the above. If an attacker controls `source_lineage` on a claim they're trying to boost, they can add unique nonsense tokens (`["real_source_A", "unique_decoy_1", "unique_decoy_2", ...]`) to reduce their Jaccard against the rest of the live set. The substrate's answer to this is "lineage is supposed to be objective provenance, not attacker-controlled metadata" — but that assumption needs enforcing at the ingestion layer, not at the math layer.

**Comparison to Bayesian belief networks.** Five of the eight red-team models independently suggested this. A BN with Dirichlet priors over source reliability and explicit conditional independence edges gets you principled uncertainty propagation — it *does* reason under uncertainty in a way ⊕ does not. The substrate's pitch vs. a BN is not "better uncertainty math" — it's "first-class schema fields, inspectable counters, content-hashed determinism." If what you need is P(claim true | evidence), use pgmpy or Pyro. If what you need is "which claims in my live set share provenance, what's the temporal contest shape, which moves produced each output," that's the substrate's territory.

## Phase 2 negative result (correction, added later)

This essay's original framing — "substrate produces better decisions on contested evidence" — did not survive contact with a real agent. I'm leaving the rest of the essay intact but adding this correction because the finding is important.

### What I tested

Three adversarial cases, each designed to make a different substrate mechanism earn its place: (A) rumor amplification where ⊕ should discount dependent sources; (B) temporal state change where the overlap/separable split should distinguish "contradiction" from "state change"; (C) same-source retraction where `SAME_SOURCE_CONFLICT` should fire. Local Qwen 3.6 (36B MoE) with tool access to the five RFC 008 HTTP endpoints, 3 cases × 3 conditions × 2 runs = 18 runs.

### What happened

The infrastructure worked — Qwen reliably called the tools (7-9 calls per run). But the substrate's intended mechanisms **did not fire correctly** in any of the three cases:

| Case | What happened | Why |
|---|---|---|
| A rumor amp | σ=4.545 vs α=2.000. **Substrate agreed with the wrong majority.** | Qwen populated lineage correctly, but ⊕'s weights cap per-claim discount at ω=0.667 — mathematically incapable of flipping 5-vs-2 dependence. |
| B temporal | `temporal_separable_opposition_count = 0`. Substrate missed the state-change pattern entirely. | Qwen didn't populate `valid_from`/`valid_to` when ingesting claims, so the temporal gate couldn't fire. |
| C same-source | `same_source_opposite_polarity_count = 0`. SAME_SOURCE_CONFLICT didn't fire. | Qwen used `[reuters, reuters_original]` vs `[reuters, reuters_correction]` — related but not character-identical. The gate requires exact normalized-set equality. |

Accuracy: **tool-use conditions averaged 5/12 correct; bare (no substrate) averaged 2/6 correct.** No meaningful improvement from tool access. Several runs came back as "unclear" — Qwen was *hedging*, reconciling its narrative intuition (often correct) with substrate output (often contradictory). The substrate may have made the model less decisive without making it more correct.

### The gold-state ablation

To distinguish "operator broken" from "extraction broken," I then hand-populated the *ideal* structured state for each case and re-ran:

| Case | Gold-state result | Interpretation |
|---|---|---|
| A | σ=3.333, α=2.000. **Still wrong** even with all 5 supports sharing identical `[nova_pharma]` lineage (max Jaccard). | **Operator is structurally incapable.** ω_min = 0.4 under the current functional form; σ_min at N=5 = 2.0 = α. No coefficient tuning can fix it. |
| B | `temporal_separable_opposition_count = 9`, no PRESENT_TENSE_CONFLICT. **Correct.** | Extraction/ergonomics problem. Gate works when fields are populated. |
| C | `same_source_opposite_polarity_count = 1`, SAME_SOURCE_CONFLICT flag set. **Correct.** | Canonicalization problem. Gate works when lineage is normalized. |

### What this means

1. **Phase 1's "success" was a demo harness.** The Wirecard showcase above worked because I hand-populated the `source_lineage` fields to satisfy the gates. A real agent populating from narrative text does not produce that structured state.

2. **⊕ in its current form is mathematically wrong for rumor amplification** — the use case it was most directly designed to handle. Any linear combination of bounded [0,1] discount terms with weights summing to 1.5 has a per-claim floor of ω = 0.4, so σ at N=5 cannot drop below 2.0. Replacing ⊕ with a cluster-collapse operator (treat claims sharing upstream provenance as effectively one source) would fix Case A; retuning the existing coefficients cannot.

3. **Cases B and C are solvable with an extractor/canonicalizer** that normalizes source names, extracts temporal intervals from narrative, and ensures claims about the same proposition use the same `(src, rel_type, dst)` triple. That preprocessor was never built; I did it by hand in Wirecard.

4. **"AGI-capable substrate" language was overstated.** What was built is a *structured provenance/temporal/conflict annotation schema*, with a partially-working operator layer, usable for audit and inspection. Not a decision-making mechanism for autonomous agents.

### Where this leaves the claims

- **Source-lineage discounting demonstrably helps on hand-populated cases** (the Wirecard example above still stands as a concrete demonstration of the *concept*).
- **The substrate does NOT reliably improve decisions when a real agent populates the state.**
- **The extractor/canonicalizer is the load-bearing problem**, not the substrate arithmetic.
- **⊕ needs replacement, not retuning**, for the rumor-amplification use case.

### What comes next

RFC 009 (design-only, no code until red-teamed) will focus on:

- **Layer 1** — extractor/canonicalizer: source alias resolution, citation-chain normalization, temporal interval extraction, canonical proposition triples. This is the real research problem.
- **Layer 2** — cluster-collapse aggregation replacing ⊕. Identify shared-origin claim clusters; compute effective support at cluster level, not per claim.
- **Layer 3** — reposition current substrate as structured audit scaffold. Stop claiming decision improvement. Surface the provenance/temporal/conflict signatures for inspection, not as authoritative weights.
- **Eval** — 30-50 adversarial fixtures across rumor, corroboration, retraction, aliasing, temporal succession, missing fields, noisy extraction. Gold-state vs extracted-state vs model-native ablations.

### Methodological note

The 8-model red-team before Phase 2 predicted every one of these failure modes. The mistake was not listening harder and running the ablation sooner. RFC 009 will run gold-state eval on day 1 of design, not day 90 of implementation.

Phase 2 raw data: [`yantrikdb-server/docs/phase2/results.json`](https://github.com/yantrikos/yantrikdb-server/blob/server/docs/phase2/results.json). Gold-state ablation: [`yantrikdb-server/docs/phase2/gold_state_results.json`](https://github.com/yantrikos/yantrikdb-server/blob/server/docs/phase2/gold_state_results.json). Both preserved so the negative result is reproducible.

## Invitation

If you read this and thought "yes, I've hit that problem," or "no, that's not a problem I have," or "this is clever but where's the product," — I'd like to hear it. Open an issue at [github.com/yantrikos/yantrikdb](https://github.com/yantrikos/yantrikdb) or email developer@pranab.co.in. One adversarial case is necessary but not sufficient; figuring out which adversarial case to test next is a conversation, not a solo decision.

---

*Part of [RFC 008](../../crates/yantrikdb-core/src/engine/warrant.rs) — Warrant Flow & Reflexive Epistemic Control.*
