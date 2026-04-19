# What a claim-graph sees that a vector DB doesn't

A Wirecard case study against the RFC 008 substrate.

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

Those are the next adversarial cases. One showcase is a necessary condition, not a sufficient one.

---

*Part of [RFC 008](../../crates/yantrikdb-core/src/engine/warrant.rs) — Warrant Flow & Reflexive Epistemic Control.*
