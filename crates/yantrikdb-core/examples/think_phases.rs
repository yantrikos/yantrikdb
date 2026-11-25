//! Development-only harness: which trigger check costs the time?
//!
//! `think()` runs the checks in `cognition.rs` back to back, so a single
//! wall-clock number cannot say which one is responsible. This calls each one
//! in isolation on a scratch copy and prints seconds + RSS delta.
//!
//! cargo run --release --example think_phases -- <db-path>

use std::time::Instant;

use yantrikdb::triggers as t;
use yantrikdb::YantrikDB;

fn rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmRSS:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
        })
        .unwrap_or(0)
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: think_phases <db-path>");
    let t0 = Instant::now();
    let db = YantrikDB::new(&path, 1024).expect("open corpus copy");
    println!("PERF_OPEN seconds={:.3}", t0.elapsed().as_secs_f64());

    macro_rules! timed {
        ($name:literal, $call:expr) => {{
            let rss0 = rss_kb();
            let t = Instant::now();
            let r = $call;
            let secs = t.elapsed().as_secs_f64();
            let n = match &r {
                Ok(v) => v.len(),
                Err(_) => 0,
            };
            println!(
                "PERF_CHECK name={} seconds={:.3} triggers={} rss_delta_kb={} ok={}",
                $name,
                secs,
                n,
                rss_kb().saturating_sub(rss0),
                r.is_ok()
            );
        }};
    }

    timed!("decay", t::check_decay_triggers(&db, 0.7, 0.5, 20));
    timed!("consolidation", t::check_consolidation_triggers(&db, 50));
    timed!("conflict_escalation", t::check_conflict_escalation(&db));
    timed!("temporal_drift", t::check_temporal_drift(&db));
    timed!("redundancy", t::check_redundancy(&db, 0.85));
    timed!("relationship_insight", t::check_relationship_insight(&db));
    timed!("valence_trend", t::check_valence_trend(&db));
    timed!("entity_anomaly", t::check_entity_anomaly(&db));
}
