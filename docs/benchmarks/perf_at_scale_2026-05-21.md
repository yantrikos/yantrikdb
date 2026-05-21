# YantrikDB perf-at-scale baseline — 2026-05-21

First committed perf baseline. Source: [`examples/perf_at_scale.rs`](../../crates/yantrikdb-core/examples/perf_at_scale.rs).

## Methodology

For each N ∈ {100, 1000, 2000, 5000}:
1. Open a fresh `:memory:` `YantrikDB` at `dim=384` and spawn the
   materializer + compactor workers via `spawn_all_workers`.
2. Seed N memories with **hash-based pseudo-random unit-norm embeddings**
   (SplitMix64-style per-dimension advance — components uncorrelated
   across dimensions and across seeds, no sin-aliasing at moderate N).
   Time the full seed; report `ingest = N / seed_seconds`.
3. Run 200 recall queries at `top_k=10`. Each query embedding is a
   deterministically-perturbed copy of one of the seeded embeddings,
   constructed via Gram-Schmidt orthogonalisation to give an **exact
   cosine similarity of 0.95** to the target. This is the principled
   way to control query-target distance — naive per-dim Gaussian noise
   on a high-dim unit vector destroys angular alignment after
   renormalisation. With cos_sim=0.95 against random embeddings, a
   healthy ANN index returns the matching rid in the top-10 with very
   high probability; therefore `recall@10` directly measures whether
   the HNSW index actually finds it.
4. Report ingest throughput, recall@10 accuracy, and recall latency
   p50/p95/p99 for each scale.

The `record_or_wait` helper retries on `Backpressure` after sleeping
`retry_after_ms`, so the reported ingest throughput is the **sustained
rate under saturation** (not pure cold-path latency, which is bounded
by `delta_max=256` writes per cycle on a fresh DB).

## Host

| Field        | Value                                                        |
|--------------|--------------------------------------------------------------|
| OS           | Windows 11 Pro 10.0.26200, x64                              |
| CPU          | AMD Ryzen 9 5950X (16C / 32T, 3.4 GHz base)                 |
| RAM          | 128 GB                                                      |
| Rust         | 1.91.1 (ed61e7d7e 2025-11-07)                               |
| Engine       | yantrikdb v0.7.19                                           |
| Git SHA      | `ab1b91d` (v0.7.19 release commit)                          |
| Profile      | `--release` (LTO off, default release flags)                |

## Results

| N | seed total | ingest | recall@10 | p50 | p95 | p99 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 8ms | 12345 rec/s | 1.000 | 3633.3µs | 6232.2µs | 12725.4µs |
| 1000 | 1714ms | 583 rec/s | 1.000 | 4992.6µs | 6362.0µs | 15801.4µs |
| 2000 | 5048ms | 396 rec/s | 1.000 | 6324.6µs | 8521.1µs | 16608.3µs |
| 5000 | 18158ms | 275 rec/s | 1.000 | 9616.6µs | 17323.9µs | 28070.5µs |

## Interpretation

- **recall@10 = 1.000 across all scales** at `cos_sim_target=0.95`.
  The HNSW index reliably returns the correct match in the top-10 when
  the query is at 0.95 cosine similarity. This is the "easy" accuracy
  regime; harder thresholds (cos_sim=0.8, 0.7) are follow-up bench
  shapes worth adding later to probe approximate-NN tail behaviour.

- **Ingest throughput drops from 12,345 → 275 rec/s** as N grows.
  This is the v0.7.18+ `Backpressure` retry path doing its job: the
  first 256 records land in the delta tier at ~12k rec/s (no contention),
  but as the cold HNSW grows, each compactor cycle costs more wall
  time, and writes block behind `retry_after_ms` sleeps. At N=5000 the
  steady-state ingest is dominated by the compact-vs-write race. This
  is the production-honest sustained throughput — callers experience
  the same backpressure waits.

- **Recall latency scales sub-linearly with N** (HNSW behaving correctly):
  p50 grows from 3.6ms at N=100 to 9.6ms at N=5000 — ~3× latency for
  50× corpus, consistent with HNSW's log-factor search cost.

- **p99 / p50 ratio** is roughly 2-4× across all scales. p99 tail is
  driven by background compaction cycles colliding with recall path
  (both touch the `Arc<DeltaIndex>`); see
  [`docs/decoupled_write_path_rfc.md`](../decoupled_write_path_rfc.md)
  for the Phase-2 design that reduces this further.

## Caveats and follow-ups

- **Not a CT 132 LXC apples-to-apples baseline.** This run is on a
  16-core 5950X dev host with 128 GB RAM and a Windows scheduler.
  The production-comparable baseline is yantrikdb-server's CT 132
  throughput harness (running on a 2-core LXC). Compare to that
  separately. This baseline is for **engine perf-vs-scale** as seen
  by a single-process consumer.

- **All scales fit in RAM trivially.** N=5000 × dim=384 × f32 ≈ 7.5MB
  of embeddings + SQLite WAL. No I/O-bound effects measured here.
  Future runs should include 100k and 1M scale points to probe
  HNSW cold-tier rebuild latency under realistic memory pressure.

- **Single-thread reads, no concurrent writer.** Recall p99 may grow
  significantly under concurrent write load. The decoupled-write-path
  RFC's Phase 2 is what hardens that scenario; see
  [`docs/wedge_concurrency_sweep_2026-05-07.md`](../wedge_concurrency_sweep_2026-05-07.md)
  for the analysis that motivated the current architecture.

- **cos_sim=0.95 is the "easy" recall regime.** A useful follow-up
  bench shape: sweep `cos_sim_target ∈ {0.95, 0.85, 0.75, 0.65}` at
  fixed N to characterise the recall@K vs query-distance curve. That's
  the actual ANN-quality measurement; this baseline only validates
  that the index is functional, not that it's well-tuned.

- **No graph-expanded recall.** This baseline isolates pure
  vector recall. The `recall_with_graph` path is a separate measurement;
  see `docs/benchmarks/` (future) for that variant.

## Raw data

JSON sidecar with full per-row payload (machine-diffable for regression
tracking): [`perf_at_scale_2026-05-21.json`](perf_at_scale_2026-05-21.json).

## Reproduction

```bash
cargo run --release --example perf_at_scale -p yantrikdb
```

Output: markdown table to stdout, JSON sidecar + per-scale progress to
stderr.
