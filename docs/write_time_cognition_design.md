# Write-time cognition: conflicts, succession, and temporal adjacency at ingestion

**Status:** agreed design, unimplemented. **Date:** 2026-08-17.
**Prerequisites shipped:** claim-keyed conflict detection (0.15.1),
maintenance-debt surfacing channel (0.15.2 + mcp 0.19.1), event-time
extraction at write (0.15.0).

## The idea in one sentence

The write path already extracts a memory's claims and event time; checking
*only that delta* against the indexed store makes contradiction detection and
succession linking an ingestion-time property — surfaced in the write's own
response, at the exact moment the caller has the context to resolve it.

## Why now and not before

1. **Precision.** Eager detection with the pre-0.15.1 detector (0/16 on its
   own open set) would have nagged callers with garbage on every write. The
   claim-keyed detector with phantom-subject admission makes write-time
   surfacing safe.
2. **A channel exists.** The maintenance-debt work made the tool response a
   legitimate carrier for substrate signals, with the flat-data-plus-suggest
   convention already established (and the measured warning against urgency
   prose — the rank-flip experiment — already encoded).
3. **Temporal coordinates exist.** Event time is stamped at write since
   0.15.0; adjacency is an index question, not an extraction question.

## Components

### A. Incremental conflict check (the delta pattern, third instance)

After a write's claims land, check **only the new claims** against existing
claims sharing a `(subject, functional-attribute)` key — a handful of indexed
lookups, never a store scan. Same shape as delta-vs-cold in the vector tier
and debt-vs-think in the cognition tier.

- Fires the existing claim-keyed classification: overlapping windows →
  conflict; ordered values → `possible_temporal_succession` with the
  supersedes hint. Nothing destructive, ever — findings are rows plus a
  response payload.
- Batch writes: per-item checks inside the existing inline-extraction path.
- Single writes: extraction is currently ASYNC (~1–3s) — see prerequisites.

### B. Temporal adjacency at write

Link each memory to its event-time neighbors (prev/next within namespace) as
it lands. Makes "what happened before/after X" an index walk; gives
successions found by (A) a place to offer the `supersedes` link so update
chains build themselves at ingestion. Schema: two nullable columns or a tiny
adjacency table — decide against migration cost; must be backfillable lazily.

### C. Surfacing (zero new machinery)

- The write's own response carries its own findings:
  `{"conflict": {"with_rid": ..., "kind": "succession", "their_claim": ...,
  "suggest": "correct() or supersede"}}` — flat data + suggest, rate-limit
  n/a (a write's own conflict is always relevant to that write).
- Everything else rides the existing on-change `open_conflicts` field.

## Prerequisites now load-bearing (both already cataloged)

1. **§3 async-extraction gap (read-your-writes).** Single-record extraction
   is async, so either (a) the conflict check rides the async worker and the
   finding surfaces on the *next* tool response (still in-band, one call
   late), or (b) extraction gains a barrier/inline mode. Decide by measuring
   the inline cost on the write path — the 1000/s budget rules.
2. **The two-thread harness + F1/F2.** This adds work to exactly the
   background-worker-vs-foreground class where the cataloged races live. The
   harness lands in this arc, not after it.

## Interaction with relation-extraction quality (symbiotic, same arc)

The write-time check sees only what extraction mints. Today's `heuristic_v1`
relations are mostly junk (`leads` co-occurrence); the functional set rarely
fires. The arc interleaves: better relation extraction (verb-frame or
pattern-based minting of version-of / located-at / state-of claims) directly
raises what (A) can catch. Measure both on the same labeled substrate.

## Gates (the protocol, pre-registered)

1. **First-hand:** on the production store copy, write "CT128 runs engine
   0.15.2" → the write response must carry the succession finding against the
   existing 0.15.1 memory, with the supersede suggestion. The store owner can
   verify every fixture personally.
2. **Precision before recall, again:** replay the 16-labeled-shapes suite at
   write time — zero may fire. New true-positive fixtures from the store's
   real succession history (engine versions, deploy states, pin ranges).
3. **Write-path cost:** p50/p99 write latency before/after on the bench
   (wedge_repro / server bench) — the check must be lost in the noise, or it
   gates behind a config flag until it is.
4. **No BEAM claim without a run.** Prediction registered now: BEAM
   contradiction_resolution may gain (fresh stores, extraction permitting);
   everything else flat. Paired run only after the cheap gates pass.

## Explicit non-goals

- No auto-resolution at write time. Suggest, never act.
- No LLM anywhere in the ingest path (the zero-LLM-at-ingest claim stands).
- No global rescans on write — delta only, indexed only.
