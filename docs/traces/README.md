# Trace contracts (v0.10 reliability gate)

Thirteen labeled, versioned trace contracts (`T01.toml`–`T13.toml`) — the
release-blocking consumer-behavior gate designed in the three-seat v0.10 review
(engine: Claude/yantrikdb-core; consumer: nuron; validity: gpt-5.6-sol). Each is a
documented real failure, not a hypothetical; T01's fixture is nuron's actual
false-accusation incident, T08's is their false-retry loop.

## The assertion-style law (anti-flake)

Traces assert **filtering / typed-status / typed-field invariants** wherever
possible. Rank-equality assertions are allowed ONLY where the Phase-0 determinism
seams make them reproducible (seeded HNSW via `YANTRIKDB_HNSW_SEED`, `ORDER BY`
rebuild scans, injectable clock, failpoint handshakes — all behind the non-default
`testing` cargo feature). Nothing in this gate may depend on wall-clock timing,
scheduler luck, or approximate-rank ties. A flaky gate gets weakened and then
deleted; an invariant gate survives.

Prose-first stamps are how "the verdict ranked 5th" survived; measurement-first is
how it dies.

## Status lifecycle

`pending → implemented` (must set `implemented_since` + `test_path`).
`implemented → pending` is structurally illegal — enforced by
`tests/trace_registry.rs` on the current checkout and by CI base-diff against the
merge target for historical non-regression.

## Thresholds

- **T01-class (superseded outranks head at k=1): hard zero from day one.**
- T03-class (aged at k=1): monitored one release, then ratio-gated with a
  sample-size floor (`rate ≤ X%` over `≥ N` labeled queries; X and N chosen from
  the monitoring release's data).
- T12 computes the stale-memory action rate from the same fixtures in CI.

## Attribution

Trace wishlist: nuron (consumer seat), 2026-07-14, from documented incidents.
T07's certainty-unchanged assertion and T11's censored-feedback rule: nuron.
T13's sealed protocol: sol (validity seat). Determinism requirements per trace:
sol's trace audit. Assembled by the engine seat.
