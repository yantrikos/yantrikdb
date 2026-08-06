//! Open-time capture — the per-lane candidate-stream dump the eleventh
//! determinism source demanded (2026-08-06, co-iteration wheel).
//!
//! Five convicted nondeterminism sources were fixed from outside
//! inference; the sixth survived every fix and both sides agreed no
//! further fix ships without a per-stage capture naming the mechanism
//! (rule 3: no fix without conviction). The signature being chased:
//! N fresh opens of the SAME file, queried inside a drift-free burst
//! (<0.5s, ~1e-8 decay drift), disagree on top-5 ordering — arm B of
//! hermes benchmarks/determinism_burst.py — while one instance queried
//! repeatedly is stable. The mechanism is therefore minted at OPEN, not
//! per query, and this module exists to show WHICH stage first diverges
//! across instances.
//!
//! Enabled by setting YANTRIKDB_CAPTURE to a directory; every recall
//! appends JSONL to capture_<pid>.jsonl there. Each line carries the
//! engine instance address (distinguishes instances within one
//! process — HashMap RandomState is per-instance, so in-process opens
//! are the exact regime under test) and f64 score BITS, because the
//! question "bit-identical scores with diverging picks?" decides
//! between summation jitter and an unnamed structural order.
//!
//! Disabled (the default) this is one OnceLock read per hook site.
//!
//! **Status: SUPPORTED, not debug** (hermes objection sustained,
//! 2026-08-06): this instrument convicted the eleventh AND twelfth
//! determinism sources when every user-side gate — hash-pinned,
//! error-barred, determinism-checked — could not see them. It is
//! load-bearing for anyone gating this engine. The shallow version of
//! the same fact (the candidate pool) ships in the public API as
//! `recall_explained` / `explain=True`; capture remains the deep
//! per-stage, per-MMR-step, bit-level view, and its env-var interface
//! is a stable contract: `YANTRIKDB_CAPTURE=<dir>` → JSONL per pid,
//! stage names hnsw_pool / lex_by_rid / scored_pre_reserve /
//! pool_post_reserve_sorted / mmr_step / final.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::OnceLock;

fn dir() -> Option<&'static PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| std::env::var_os("YANTRIKDB_CAPTURE").map(PathBuf::from))
        .as_ref()
}

pub(crate) fn enabled() -> bool {
    dir().is_some()
}

/// Exact f64 identity as hex bits — two scores that print the same at
/// any decimal precision still differ here if they differ at all.
pub(crate) fn bits(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

pub(crate) fn emit(inst: usize, stage: &str, data: serde_json::Value) {
    let Some(d) = dir() else { return };
    let path = d.join(format!("capture_{}.jsonl", std::process::id()));
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
        let line = serde_json::json!({"inst": inst, "stage": stage, "data": data});
        let _ = writeln!(f, "{line}");
    }
}
