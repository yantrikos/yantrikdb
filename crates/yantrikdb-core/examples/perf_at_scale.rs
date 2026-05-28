//! User-facing perf-at-scale baseline.
//!
//! For each N in `SCALES`, this example:
//!   1. Opens a fresh `:memory:` `YantrikDB` at `dim=384` and spawns the
//!      materializer + compactor workers.
//!   2. Seeds N memories with hash-based pseudo-random unit-norm embeddings
//!      (essentially orthogonal at high dim), recording total seed time
//!      and computing the corresponding ingest throughput (records / sec).
//!   3. Runs `N_QUERIES` recall queries at `top_k=10`, where each query
//!      embedding is a deterministically-perturbed copy of one of the
//!      seeded embeddings with a fixed **cosine similarity** to the target
//!      (`COS_SIM_TARGET`, default 0.95). This is the principled way to
//!      control query-target distance — naive per-dim Gaussian noise on
//!      a high-dim unit vector destroys angular alignment after
//!      renormalisation. With cos_sim=0.95 + random embeddings the
//!      target should land in top-10 trivially for a healthy engine,
//!      so recall@10 measures whether the index actually finds it under
//!      approximate-NN.
//!   4. Prints a markdown table to stdout summarising the run.
//!
//! Run with:
//! ```bash
//! cargo run --release --example perf_at_scale > docs/benchmarks/perf_at_scale_<date>.md
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};
use yantrikdb::engine::materializer::{recommended_worker_count, spawn_all_workers};
use yantrikdb::error::YantrikDbError;
use yantrikdb::YantrikDB;

const DIM: usize = 384;
const SCALES: [usize; 4] = [100, 1000, 2000, 5000];
const N_QUERIES: usize = 200;
const TOP_K: usize = 10;
/// Target cosine similarity between the query embedding and its matching
/// memory embedding. 0.95 is a reasonable "near-neighbor" target — close
/// enough that a healthy ANN index should return the matching rid in the
/// top-10 of a random N-memory corpus, far enough that exact-match is
/// not being measured.
const COS_SIM_TARGET: f32 = 0.95;

/// Hash-based deterministic pseudo-random embedding. SplitMix64-style
/// state advance per dimension gives uncorrelated components across
/// dimensions AND across seeds — no sin-aliasing at moderate N.
fn deterministic_embedding(seed: usize) -> Vec<f32> {
    let mut state: u64 = (seed as u64)
        .wrapping_add(0x9E37_79B9_7F4A_7C15)
        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let raw: Vec<f32> = (0..DIM)
        .map(|_| {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // Map to roughly [-1, 1)
            let signed = z as i64;
            (signed as f32) / (i64::MAX as f32)
        })
        .collect();
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return vec![1.0 / (DIM as f32).sqrt(); DIM];
    }
    raw.iter().map(|x| x / norm).collect()
}

/// Generate a query embedding with exact cosine similarity ≈ COS_SIM_TARGET
/// to the target embedding. Uses Gram-Schmidt to construct an orthogonal
/// noise component, then combines: query = α·target + β·orthog where
/// α = COS_SIM_TARGET, β = sqrt(1 − α²).
fn perturbed_query(target_embedding: &[f32], query_iter: usize) -> Vec<f32> {
    // Generate a random noise vector seeded by query_iter (so each query
    // has a different perturbation direction).
    let mut state: u64 = (query_iter as u64)
        .wrapping_mul(0xD135_3727_2374_4AF7)
        .wrapping_add(0xC2B2_AE3D_27D4_EB4F);
    let noise: Vec<f32> = (0..DIM)
        .map(|_| {
            state = state.wrapping_add(0xD135_3727_2374_4AF7);
            let mut z = state;
            z = (z ^ (z >> 33)).wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            z = (z ^ (z >> 33)).wrapping_mul(0xC4CE_B9FE_1A85_EC53);
            z ^= z >> 33;
            let signed = z as i64;
            (signed as f32) / (i64::MAX as f32)
        })
        .collect();
    // Gram-Schmidt: noise_orth = noise − dot(noise, target) · target
    let dot: f32 = noise
        .iter()
        .zip(target_embedding.iter())
        .map(|(n, t)| n * t)
        .sum();
    let orth: Vec<f32> = noise
        .iter()
        .zip(target_embedding.iter())
        .map(|(n, t)| n - dot * t)
        .collect();
    let orth_norm: f32 = orth.iter().map(|x| x * x).sum::<f32>().sqrt();
    let orth_unit: Vec<f32> = if orth_norm > 0.0 {
        orth.iter().map(|x| x / orth_norm).collect()
    } else {
        // Pathological: pick an arbitrary orthogonal direction
        let mut e = vec![0.0_f32; DIM];
        e[0] = 1.0;
        e
    };
    let alpha = COS_SIM_TARGET;
    let beta = (1.0 - alpha * alpha).sqrt();
    let combined: Vec<f32> = target_embedding
        .iter()
        .zip(orth_unit.iter())
        .map(|(t, o)| alpha * t + beta * o)
        .collect();
    let norm: f32 = combined.iter().map(|x| x * x).sum::<f32>().sqrt();
    combined.iter().map(|x| x / norm).collect()
}

#[allow(clippy::too_many_arguments)]
fn record_or_wait(
    db: &YantrikDB,
    text: &str,
    embedding: &[f32],
    metadata: &serde_json::Value,
) -> String {
    loop {
        match db.record(
            text, "episodic", 0.5, 0.0, 604_800.0, metadata, embedding, "default", 0.8, "general",
            "user", None,
        ) {
            Ok(rid) => return rid,
            Err(YantrikDbError::Backpressure { retry_after_ms, .. }) => {
                std::thread::sleep(Duration::from_millis(retry_after_ms.max(1)));
            }
            Err(e) => panic!("perf_at_scale record failed: {e}"),
        }
    }
}

fn percentile(samples: &[f64], p: f64) -> f64 {
    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (((sorted.len() as f64 - 1.0) * p / 100.0).round() as usize).min(sorted.len() - 1);
    sorted[idx]
}

struct Row {
    n: usize,
    seed_ms: f64,
    ingest_per_sec: f64,
    recall_at_10: f64,
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
}

fn run_one(n: usize) -> Row {
    eprintln!("=== N = {n} ===");
    let db = Arc::new(YantrikDB::new(":memory:", DIM).unwrap());
    let _workers = spawn_all_workers(&db, recommended_worker_count());
    let meta = serde_json::json!({});

    eprintln!("  seeding {n} memories at dim={DIM}...");
    let seed_start = Instant::now();
    let mut rids: Vec<String> = Vec::with_capacity(n);
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n {
        let emb = deterministic_embedding(i);
        let rid = record_or_wait(&db, &format!("memory {i}"), &emb, &meta);
        rids.push(rid);
        embeddings.push(emb);
    }
    let seed_ms = seed_start.elapsed().as_secs_f64() * 1000.0;
    let ingest_per_sec = (n as f64) / (seed_ms / 1000.0);
    eprintln!(
        "  seeded {} in {:.0}ms ({:.0} records/sec)",
        n, seed_ms, ingest_per_sec
    );

    eprintln!(
        "  running {N_QUERIES} recall queries (top_k={TOP_K}, cos_sim_target={COS_SIM_TARGET})..."
    );
    let mut latencies_us: Vec<f64> = Vec::with_capacity(N_QUERIES);
    let mut hits: usize = 0;
    for q in 0..N_QUERIES {
        let target = q % n;
        let query = perturbed_query(&embeddings[target], q);
        let target_rid = &rids[target];

        let q_start = Instant::now();
        let results = db
            .recall(
                &query, TOP_K, None, None, false, false, None, true, None, None, None, None, None,
            )
            .expect("recall failed");
        let lat_us = q_start.elapsed().as_secs_f64() * 1_000_000.0;
        latencies_us.push(lat_us);

        if results.iter().any(|r| r.rid == *target_rid) {
            hits += 1;
        }
    }
    let recall_at_10 = hits as f64 / N_QUERIES as f64;
    let p50_us = percentile(&latencies_us, 50.0);
    let p95_us = percentile(&latencies_us, 95.0);
    let p99_us = percentile(&latencies_us, 99.0);
    eprintln!(
        "  recall@10={:.3}  p50={:.1}µs  p95={:.1}µs  p99={:.1}µs",
        recall_at_10, p50_us, p95_us, p99_us
    );

    Row {
        n,
        seed_ms,
        ingest_per_sec,
        recall_at_10,
        p50_us,
        p95_us,
        p99_us,
    }
}

fn main() {
    eprintln!(
        "YantrikDB perf-at-scale baseline \
         (dim={DIM}, queries/scale={N_QUERIES}, cos_sim_target={COS_SIM_TARGET})\n"
    );

    let mut rows: Vec<Row> = Vec::new();
    for &n in &SCALES {
        rows.push(run_one(n));
    }

    println!("| N | seed total | ingest | recall@10 | p50 | p95 | p99 |");
    println!("|---:|---:|---:|---:|---:|---:|---:|");
    for r in &rows {
        println!(
            "| {} | {:.0}ms | {:.0} rec/s | {:.3} | {:.1}µs | {:.1}µs | {:.1}µs |",
            r.n, r.seed_ms, r.ingest_per_sec, r.recall_at_10, r.p50_us, r.p95_us, r.p99_us
        );
    }

    eprintln!("\n--- JSON sidecar ---");
    let json = serde_json::json!({
        "host": "see baseline markdown header",
        "dim": DIM,
        "queries_per_scale": N_QUERIES,
        "top_k": TOP_K,
        "cos_sim_target": COS_SIM_TARGET,
        "rows": rows.iter().map(|r| serde_json::json!({
            "n": r.n,
            "seed_ms": r.seed_ms,
            "ingest_per_sec": r.ingest_per_sec,
            "recall_at_10": r.recall_at_10,
            "p50_us": r.p50_us,
            "p95_us": r.p95_us,
            "p99_us": r.p99_us,
        })).collect::<Vec<_>>(),
    });
    eprintln!("{}", serde_json::to_string_pretty(&json).unwrap());
}
