# AMB Complete Facet Composition V3 Replication Preregistration

Status: protocol frozen before the v2 event-ordering loss rows were inspected.

## Purpose

V2 produced a positive overall paired interval and passed four of five gates,
but event ordering moved `-0.030486`, below the frozen `-0.025` harm floor.
Its category interval included zero and wins were nearly symmetric, so one
fresh replication may distinguish answer variance from a repeatable product
cost. V3 is replication, not mechanism tuning.

## Byte-Identical Arm

V3 reuses the exact v2 contexts and manifest:

- Manifest SHA-256:
  `b1b3b009b4f5d531d58edb80f157d46f02d80169535f190e1d603806cb673989`
- Control SHA-256:
  `918f572927b75ab1bb2ae3edf5656eada132cf9a644953f31bf693c695d46863`
- Treatment SHA-256:
  `43a222af04195cc2d18d3a2ad15124a73208afeb0df352c956e69e86c5c5d9fb`
- Model: `deepseek-v4-flash:0731-cloud`
- One answer and one judge per arm per query
- Seed: `20260825`
  This intentionally differs from the v2 run seed and pooled-bootstrap seed so
  the replication preserves identical contexts while sampling independent
  answer and judge randomness.
- Synthetic BEAM data only; no real companion memories

The run uses a new output and initially absent checkpoint. Checkpoint/resume is
allowed only for interruption recovery within that run. No v2 answer, judge,
or score is reused as a v3 observation.

## Zero-Call Funding Gate

Before spending on v3, inspect only the ten v2 event-ordering loss rows. A row
has a plausible facet collision when a composed directive governs dates,
timelines, scheduling, deadlines, event chronology, or ordered output and the
control-to-treatment answer change could reasonably be affected by that rule.

- If at least 8/10 loss rows have no plausible collision, fund v3.
- If three or more have a plausible collision, do not run an identical v3.
  Investigate the interaction and preregister a changed mechanism instead.

This inspection cannot change the arm, score gates, pooling method, or
thresholds below.

## V3 Gates

For v3 alone, all original gates except the v2-failed event floor must pass:

1. Instruction-following delta is at least `+0.05`, with more wins than
   losses.
2. Overall delta is non-negative and its paired 95% interval lower bound is at
   least `-0.01`.
3. The pooled other-nine-category delta is at least `-0.005`.
4. Summarization delta is at least `-0.01`.
5. No category other than event ordering has delta below `-0.025`.
6. Event-ordering delta is non-negative.

The non-negative event condition is intentionally asymmetric. If the true
effect is exactly zero, a 40-row point estimate has roughly a 50% chance of
being non-negative. A false opt-in decision leaves the capability available;
a false default promotion exposes users to harm, so promotion carries the
burden of proof.

## Pooled Gates

Pool v2 and v3 by query ID, preserving the two paired deltas for each of the
400 queries. The pooled point estimate is the mean of all 800 deltas. The
pooled 95% interval uses 20,000 cluster-bootstrap resamples with seed
`20260824`: sample 400 query IDs with replacement and include both run deltas
for each sampled query. This avoids treating two observations of the same
query as unrelated rows.

Both pooled gates must pass:

1. Event-ordering delta is at least `-0.025`.
2. Overall paired interval lower bound is at least `-0.01`.

## Decision

Promote complete-lane composition to default only if the zero-call funding
gate authorizes v3, every v3 gate passes, and both pooled gates pass.

Any failure leaves this exact mechanism opt-in and triggers interaction
investigation. There will be no third identical run. A future promotion attempt
must change the mechanism based on evidence and receive a fresh preregistration
before scoring.
