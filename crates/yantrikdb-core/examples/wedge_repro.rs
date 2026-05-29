//! Wedge reproducer — empirical baseline for the lock-scope audit.
//!
//! Spawns N writer threads + M reader threads against a single YantrikDB
//! instance and prints per-second latency snapshots so we can see the wedge
//! develop (or fail to develop) at varying concurrency levels.
//!
//! Usage:
//!   cargo run --release --example wedge_repro -- \
//!       --writers 32 --readers 8 --duration-secs 60 \
//!       --dim 384 --warmup-records 5000 --db-path ./wedge_repro.db
//!
//! Default values:
//!   writers          = 32
//!   readers          = 8
//!   duration-secs    = 60
//!   dim              = 384  (matches bge-base-en-v1.5)
//!   warmup-records   = 5000
//!   db-path          = ./wedge_repro.db (deleted at start)
//!
//! What to look for:
//!   - write p99 climbing monotonically over time → lock contention building
//!   - read p99 spiking when writer concurrency increases → reader-writer starvation
//!   - throughput plateau under added writer threads → serialization on vec_index lock
//!
//! Output is per-second + final summary. No external crates beyond what the
//! engine already pulls in.

use std::env;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use yantrikdb::engine::materializer::spawn_compactor;
use yantrikdb::YantrikDB;

/// **Phase 7 soak prep — RSS instrumentation.**
///
/// Read this process's resident set size in bytes. Linux reads
/// `/proc/self/status`'s `VmRSS:` line; other platforms return `None`
/// (the harness still runs, just prints `rss=  N/A` in those rows).
///
/// The soak validation acceptance gate from
/// `docs/decoupled_write_path_rfc.md` includes "RSS bounded over 1h
/// sustained load". Without this hook the operator has to read RSS
/// out-of-band (htop / Get-Process); inline per-second capture means
/// the wedge_repro output is the single source of truth for the run.
#[cfg(target_os = "linux")]
fn read_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            // Format: "VmRSS:    123456 kB"
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(not(target_os = "linux"))]
fn read_rss_bytes() -> Option<u64> {
    // Windows + macOS not wired here. The Phase 7 soak runs on the
    // Linux homelab (CT 168 reference), and adding a sysinfo or
    // windows-sys dep is heavier than the value warrants for dev
    // runs. If you need RSS on Windows interactively, watch
    // `Get-Process` in another pane.
    None
}

fn fmt_mib(bytes: Option<u64>) -> String {
    match bytes {
        Some(b) => format!("{}", b / (1024 * 1024)),
        None => "N/A".to_string(),
    }
}

#[derive(Clone, Copy)]
struct Config {
    writers: usize,
    readers: usize,
    duration_secs: u64,
    dim: usize,
    warmup_records: usize,
}

fn parse_args() -> (Config, String) {
    let mut cfg = Config {
        writers: 32,
        readers: 8,
        duration_secs: 60,
        dim: 384,
        warmup_records: 5000,
    };
    let mut db_path = String::from("./wedge_repro.db");
    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--writers" => {
                cfg.writers = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--readers" => {
                cfg.readers = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--duration-secs" => {
                cfg.duration_secs = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--dim" => {
                cfg.dim = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--warmup-records" => {
                cfg.warmup_records = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--db-path" => {
                db_path = args[i + 1].clone();
                i += 2;
            }
            "--help" | "-h" => {
                println!(
                    "{}",
                    include_str!("wedge_repro.rs")
                        .lines()
                        .take(28)
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown arg: {}", other);
                std::process::exit(2);
            }
        }
    }
    (cfg, db_path)
}

/// Deterministic embedding from a seed — same shape `bench_utils::vec_seed_dim` uses.
fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    raw.iter().map(|x| x / norm).collect()
}

/// Each thread captures (second_bucket, duration_ns) tuples into its local Vec
/// to avoid contention on a shared aggregator. Aggregation happens at the end.
type LatencySample = (u64, u64);

fn run() {
    let (cfg, db_path) = parse_args();
    println!("=== wedge_repro ===");
    println!(
        "config: writers={} readers={} duration={}s dim={} warmup={} db={}",
        cfg.writers, cfg.readers, cfg.duration_secs, cfg.dim, cfg.warmup_records, db_path
    );

    // Clean slate
    let _ = std::fs::remove_file(&db_path);

    let db = Arc::new(YantrikDB::new(&db_path, cfg.dim).expect("YantrikDB::new"));
    // Phase 5 — spawn compactor so the delta drains into cold periodically.
    // Without this, delta scan grows unbounded and reads degrade linearly.
    let _compactor_guard = spawn_compactor(&db);
    println!("[setup] DB opened at {}", db_path);

    // Warmup — seed the HNSW graph so subsequent inserts cost realistic O(M·log N)
    let t0 = Instant::now();
    let meta = serde_json::json!({});
    for i in 0..cfg.warmup_records {
        let emb = vec_seed(i as f32 * 0.37, cfg.dim);
        // v0.6.6: bounded ingest queue surfaces Backpressure when the
        // materializer thread can't keep up with single-threaded warmup
        // throughput. Honor retry_after_ms — that's the contract any
        // real client would follow.
        loop {
            let res = db.record(
                &format!("warmup memory {}", i),
                if i % 2 == 0 { "episodic" } else { "semantic" },
                0.5,
                0.0,
                604800.0,
                &meta,
                &emb,
                "default",
                0.8,
                "general",
                "user",
                None,
            );
            match res {
                Ok(_) => break,
                Err(e) => {
                    let msg = format!("{e}");
                    // v0.6.6 ingest queue surfaces backpressure as
                    // either `Backpressure { ... }` or a free-text
                    // "ingest queue full ... retry after Nms" depending
                    // on the path. Match either.
                    let is_backpressure =
                        msg.contains("Backpressure") || msg.contains("ingest queue full");
                    if is_backpressure {
                        let ms: u64 = msg
                            .split("retry after ")
                            .nth(1)
                            .and_then(|s| s.split("ms").next())
                            .and_then(|s| s.trim().parse().ok())
                            .or_else(|| {
                                msg.split("retry_after_ms:")
                                    .nth(1)
                                    .and_then(|s| s.split([' ', ',', '}']).next())
                                    .and_then(|s| s.trim().parse().ok())
                            })
                            .unwrap_or(50);
                        thread::sleep(std::time::Duration::from_millis(ms));
                        continue;
                    } else {
                        panic!("warmup record (non-backpressure): {e}");
                    }
                }
            }
        }
        if i > 0 && i % 1000 == 0 {
            println!(
                "[setup] warmup {}/{} ({:.1}s elapsed)",
                i,
                cfg.warmup_records,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    println!(
        "[setup] warmup done in {:.1}s — {} records seeded",
        t0.elapsed().as_secs_f64(),
        cfg.warmup_records
    );

    let stop = Arc::new(AtomicBool::new(false));
    let started_at = Instant::now();

    // Per-thread sample storage
    let writer_samples: Vec<Arc<Mutex<Vec<LatencySample>>>> = (0..cfg.writers)
        .map(|_| Arc::new(Mutex::new(Vec::with_capacity(100_000))))
        .collect();
    let reader_samples: Vec<Arc<Mutex<Vec<LatencySample>>>> = (0..cfg.readers)
        .map(|_| Arc::new(Mutex::new(Vec::with_capacity(100_000))))
        .collect();

    let writes_total = Arc::new(AtomicU64::new(0));
    let reads_total = Arc::new(AtomicU64::new(0));

    // ── Spawn writer threads ──
    let mut writer_handles = Vec::new();
    for w_id in 0..cfg.writers {
        let db_c = Arc::clone(&db);
        let stop_c = Arc::clone(&stop);
        let samples_c = Arc::clone(&writer_samples[w_id]);
        let writes_c = Arc::clone(&writes_total);
        let dim = cfg.dim;
        let started = started_at;
        let h = thread::spawn(move || {
            let meta = serde_json::json!({});
            let mut local_seq: u64 = 0;
            while !stop_c.load(Ordering::Relaxed) {
                let seed = (w_id as u64 * 1_000_000 + local_seq) as f32 * 0.137 + 100_000.0;
                let emb = vec_seed(seed, dim);
                let text = format!("writer {} seq {}", w_id, local_seq);
                let t = Instant::now();
                let _ = db_c.record(
                    &text, "episodic", 0.5, 0.0, 604800.0, &meta, &emb, "default", 0.8, "general",
                    "user", None,
                );
                let dur_ns = t.elapsed().as_nanos() as u64;
                let bucket = started.elapsed().as_secs();
                samples_c.lock().unwrap().push((bucket, dur_ns));
                writes_c.fetch_add(1, Ordering::Relaxed);
                local_seq += 1;
            }
        });
        writer_handles.push(h);
    }

    // ── Spawn reader threads ──
    let mut reader_handles = Vec::new();
    for r_id in 0..cfg.readers {
        let db_c = Arc::clone(&db);
        let stop_c = Arc::clone(&stop);
        let samples_c = Arc::clone(&reader_samples[r_id]);
        let reads_c = Arc::clone(&reads_total);
        let dim = cfg.dim;
        let started = started_at;
        let h = thread::spawn(move || {
            let mut q_seed: u64 = (r_id as u64 + 1) * 31;
            while !stop_c.load(Ordering::Relaxed) {
                let query = vec_seed((q_seed as f32) * 0.71 + 7.0, dim);
                let t = Instant::now();
                let _ = db_c.recall(
                    &query, 10, None, None, false, false, None, false, None, None, None, None, None,
                );
                let dur_ns = t.elapsed().as_nanos() as u64;
                let bucket = started.elapsed().as_secs();
                samples_c.lock().unwrap().push((bucket, dur_ns));
                reads_c.fetch_add(1, Ordering::Relaxed);
                q_seed = q_seed.wrapping_add(1);
            }
        });
        reader_handles.push(h);
    }

    // ── Per-second progress printer ──
    // RSS column is right-aligned MiB (or N/A on non-Linux). Tracks
    // the soak acceptance gate "RSS bounded over 1h sustained load".
    println!(
        "\n {:>4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
        "sec", "w/s", "r/s", "w_p50us", "w_p99us", "r_p50us", "r_p99us", "delta", "rss_MiB"
    );

    let rss_baseline = read_rss_bytes();
    let mut rss_peak = rss_baseline.unwrap_or(0);
    let mut prev_w = 0u64;
    let mut prev_r = 0u64;
    for sec in 0..cfg.duration_secs {
        thread::sleep(Duration::from_secs(1));
        let w = writes_total.load(Ordering::Relaxed);
        let r = reads_total.load(Ordering::Relaxed);
        // Snapshot per-thread durations for this second bucket and aggregate
        let mut w_durs: Vec<u64> = Vec::new();
        for s in &writer_samples {
            for (b, d) in s.lock().unwrap().iter() {
                if *b == sec {
                    w_durs.push(*d);
                }
            }
        }
        let mut r_durs: Vec<u64> = Vec::new();
        for s in &reader_samples {
            for (b, d) in s.lock().unwrap().iter() {
                if *b == sec {
                    r_durs.push(*d);
                }
            }
        }
        let (w_p50, w_p99) = pcts_us(&mut w_durs);
        let (r_p50, r_p99) = pcts_us(&mut r_durs);
        let delta_now = db.delta_len();
        let rss_now = read_rss_bytes();
        if let Some(b) = rss_now {
            if b > rss_peak {
                rss_peak = b;
            }
        }
        println!(
            " {:>4} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8}",
            sec + 1,
            w - prev_w,
            r - prev_r,
            w_p50,
            w_p99,
            r_p50,
            r_p99,
            delta_now,
            fmt_mib(rss_now)
        );
        prev_w = w;
        prev_r = r;
    }

    // ── Stop and join ──
    stop.store(true, Ordering::Relaxed);
    for h in writer_handles {
        let _ = h.join();
    }
    for h in reader_handles {
        let _ = h.join();
    }

    // ── Summary ──
    let mut all_w: Vec<u64> = Vec::new();
    for s in &writer_samples {
        all_w.extend(s.lock().unwrap().iter().map(|(_, d)| *d));
    }
    let mut all_r: Vec<u64> = Vec::new();
    for s in &reader_samples {
        all_r.extend(s.lock().unwrap().iter().map(|(_, d)| *d));
    }

    println!("\n=== summary ===");
    println!("total writes: {}", all_w.len());
    println!("total reads:  {}", all_r.len());
    if !all_w.is_empty() {
        let (p50, p95, p99, p999) = quad_pcts_us(&mut all_w);
        println!(
            "writes p50={}us p95={}us p99={}us p99.9={}us",
            p50, p95, p99, p999
        );
    }
    if !all_r.is_empty() {
        let (p50, p95, p99, p999) = quad_pcts_us(&mut all_r);
        println!(
            "reads  p50={}us p95={}us p99={}us p99.9={}us",
            p50, p95, p99, p999
        );
    }
    let elapsed = started_at.elapsed().as_secs_f64();
    println!(
        "wall: {:.1}s | write tput: {:.0}/s | read tput: {:.0}/s",
        elapsed,
        all_w.len() as f64 / elapsed,
        all_r.len() as f64 / elapsed
    );

    // RSS summary — soak acceptance gate "RSS bounded".
    let rss_final = read_rss_bytes();
    println!(
        "rss: baseline={} MiB peak={} MiB final={} MiB | delta={} cold={} (engine pressure: {:.1}%)",
        fmt_mib(rss_baseline),
        fmt_mib(Some(rss_peak)),
        fmt_mib(rss_final),
        db.delta_len(),
        db.cold_len(),
        100.0 * db.delta_len() as f64 / db.delta_max() as f64,
    );
}

fn pcts_us(durs: &mut [u64]) -> (u64, u64) {
    if durs.is_empty() {
        return (0, 0);
    }
    durs.sort_unstable();
    let p = |q: f64| -> u64 {
        let idx = ((durs.len() as f64 * q) as usize).min(durs.len() - 1);
        durs[idx] / 1000
    };
    (p(0.50), p(0.99))
}

fn quad_pcts_us(durs: &mut [u64]) -> (u64, u64, u64, u64) {
    if durs.is_empty() {
        return (0, 0, 0, 0);
    }
    durs.sort_unstable();
    let p = |q: f64| -> u64 {
        let idx = ((durs.len() as f64 * q) as usize).min(durs.len() - 1);
        durs[idx] / 1000
    };
    (p(0.50), p(0.95), p(0.99), p(0.999))
}

fn main() {
    run();
}
