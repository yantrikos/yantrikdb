//! Reproducible benchmark for the substitution-index change (0010).
//!
//! `check_redundancy_in_namespace` used to issue one JOIN per token pair of
//! every near-duplicate memory pair. This harness builds a store of that shape
//! and times `check_redundancy` end to end, so the same binary can be built
//! from the parent commit and from the change and the two numbers compared.
//!
//!     cargo run --release --example substitution_index_bench -- [namespaces] [records]
//!
//! Defaults: 8 namespaces, 30 records each — the per-namespace scan is capped
//! at 30 rows (`LIMIT 30`), so more records only widen the table, not the scan.
//!
//! The corpus is synthetic and generated here; no stored memory content is used.

use std::time::Instant;

use yantrikdb::{triggers, YantrikDB};

/// Unit vector with `seed` spread over the first dimensions, normalised.
fn embedding(seed: f32, dim: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..dim)
        .map(|i| ((seed * 13.0) + i as f32 * 0.7).sin())
        .collect();
    let norm = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    raw.iter().map(|x| x / norm).collect()
}

/// A long, realistic-length record with `marker` woven in.
///
/// `reps` controls length: each filler sentence is ~150 characters, so the
/// default (14) yields ~2,100-character records — the shape of a real session
/// note. Length is what drives the cost here: the number of token pairs in the
/// symmetric difference grows with it.
fn text(idx: usize, marker: &str, reps: usize) -> String {
    let filler = "Der Abschnitt beschreibt den Aufbau der Ablage, die Reihenfolge \
                  der Schritte und die Bedingungen, unter denen der Lauf wiederholt \
                  wird; die Entscheidung bleibt beim Menschen.";
    let body = std::iter::repeat(filler)
        .take(reps)
        .collect::<Vec<_>>()
        .join(" ");
    // Distinct tokens matter more than length: the cost is the product of the
    // two symmetric differences. A repeated filler has ~25 unique tokens no
    // matter how long it gets, which understates the effect by orders of
    // magnitude. Real session notes differ in almost every noun.
    let distinct: Vec<String> = (0..120).map(|j| format!("w{idx}_{j}")).collect();
    format!("{marker} Variante {idx}. {body} {}", distinct.join(" "))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let namespaces: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(8);
    let per_ns: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(30);
    let reps: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(14);
    let dim = 1024;
    let dir = std::env::temp_dir().join(format!("ydb-subbench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("bench.db");
    let _ = std::fs::remove_file(&path);

    let open = Instant::now();
    let db = YantrikDB::new(path.to_str().unwrap(), dim).expect("open store");
    println!("PERF_OPEN seconds={:.3}", open.elapsed().as_secs_f64());

    // Two near-duplicate families per namespace: they differ by one token in
    // the middle, so the symmetric token difference is large — exactly the
    // shape that made the per-pair JOIN expensive.
    let seed_start = Instant::now();
    for ns in 0..namespaces {
        let namespace = format!("bench-{ns}");
        for i in 0..per_ns {
            let marker = if i % 2 == 0 { "postgresql" } else { "mysql" };
            let emb = embedding(1.0 + i as f32 * 0.00002, dim);
            db.record(
                &text(i, marker, reps),
                "episodic",
                0.5,
                0.0,
                604_800.0,
                &serde_json::json!({}),
                &emb,
                &namespace,
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        }
    }
    println!(
        "PERF_SEED seconds={:.3}",
        seed_start.elapsed().as_secs_f64()
    );

    let run = Instant::now();
    let triggers_found = triggers::check_redundancy(&db, 0.85).unwrap();
    let seconds = run.elapsed().as_secs_f64();
    println!(
        "PERF_CHECK name=redundancy seconds={seconds:.3} triggers={} namespaces={namespaces} records_per_ns={per_ns} filler_reps={reps} approx_chars={}",
        triggers_found.len(),
        reps * 150
    );

    drop(db);
    std::fs::remove_dir_all(&dir).ok();
}
