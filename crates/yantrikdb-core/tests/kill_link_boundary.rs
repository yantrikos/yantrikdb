//! v0.10 Phase 0 — the link-boundary kill proof (sol Q3 converged choice).
//!
//! Proves two things at once:
//! 1. The hand-rolled failpoint handshake works (park + `FAILPOINT:<name>`
//!    line + parent kill — deterministic, not sleep roulette).
//! 2. Commit A's atomicity claim is REAL: a process killed between the
//!    record_links insert and the oplog insert (inside their shared
//!    SAVEPOINT) leaves NEITHER row — no local edge without a replication
//!    event, no replication event without an edge.
//!
//! Mechanics: this test re-invokes its own test binary to run the
//! `child_parks_at_link_boundary` helper (ignored by default) in a child
//! process with the failpoint armed via `YANTRIKDB_FAILPOINTS`. The parent
//! waits for the handshake line with a bounded timeout, kills the child,
//! reopens the database, and asserts `PRAGMA quick_check` plus
//! neither-row-committed.
#![cfg(feature = "testing")]

use std::io::BufRead;
use std::process::{Command, Stdio};
use std::time::Duration;

use yantrikdb::YantrikDB;

fn test_vec(seed: f32, dim: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..dim).map(|i| ((seed + i as f32) * 0.7).sin()).collect();
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    raw.iter().map(|x| x / norm).collect()
}

fn record(db: &YantrikDB, text: &str, seed: f32) -> String {
    db.record(
        text,
        "semantic",
        0.5,
        0.0,
        604800.0,
        &serde_json::json!({}),
        &test_vec(seed, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap()
}

/// CHILD half — ignored in normal runs; the parent invokes it explicitly.
/// Opens the DB at `YANTRIKDB_KILL_DB`, writes two records, then calls
/// `link()`, which parks at `link.between_row_and_oplog` (armed via env
/// by the parent). Parked forever; the parent kills us.
#[test]
#[ignore]
fn child_parks_at_link_boundary() {
    let db_path = std::env::var("YANTRIKDB_KILL_DB").expect("child needs YANTRIKDB_KILL_DB");
    let db = YantrikDB::new(&db_path, 8).unwrap();
    let old = record(&db, "predecessor fact", 1.0);
    let new = record(&db, "successor fact", 2.0);
    // Handshake so the parent can distinguish "records committed" from
    // "engine still opening" before it starts waiting on the failpoint.
    println!("CHILD_READY:{old}:{new}");
    use std::io::Write;
    let _ = std::io::stdout().flush();
    // Parks inside the link savepoint (between row insert and oplog insert).
    let _ = db.link(
        &new,
        &yantrikdb::RecordLink {
            target_rid: old,
            link_type: yantrikdb::LinkType::Supersedes,
        },
    );
    unreachable!("parent must kill us at the failpoint");
}

/// PARENT half — the actual proof.
#[test]
fn kill_between_link_row_and_oplog_leaves_neither_row() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("kill.db");
    let db_path_str = db_path.to_str().unwrap().to_string();

    let exe = std::env::current_exe().unwrap();
    let mut child = Command::new(exe)
        .args([
            "child_parks_at_link_boundary",
            "--exact",
            "--ignored",
            "--nocapture",
        ])
        .env("YANTRIKDB_KILL_DB", &db_path_str)
        .env("YANTRIKDB_FAILPOINTS", "link.between_row_and_oplog")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn child");

    // Bounded handshake wait (no sleeps): reader thread + channel timeout.
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in std::io::BufReader::new(stdout)
            .lines()
            .map_while(|l| l.ok())
        {
            let _ = tx.send(line);
        }
    });

    let deadline = Duration::from_secs(60);
    let mut saw_ready = false;
    let mut saw_failpoint = false;
    let start = std::time::Instant::now();
    while start.elapsed() < deadline {
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(line) if line.starts_with("CHILD_READY:") => saw_ready = true,
            Ok(line) if line == "FAILPOINT:link.between_row_and_oplog" => {
                saw_failpoint = true;
                break;
            }
            Ok(_) => {}
            Err(_) => {
                if let Ok(Some(_)) = child.try_wait() {
                    break; // child died early — assertions below will tell us
                }
            }
        }
    }
    assert!(saw_ready, "child never reached the link call");
    assert!(saw_failpoint, "child never hit the armed failpoint");

    // Kill mid-transaction — the deterministic version of a crash.
    child.kill().expect("kill child");
    let _ = child.wait();

    // Reopen and prove: structurally sound, and NEITHER half of the link
    // write survived (SAVEPOINT rolled back on process death).
    let db = YantrikDB::new(&db_path_str, 8).unwrap();
    let conn = db.conn();
    let qc: String = conn
        .query_row("PRAGMA quick_check", [], |r| r.get(0))
        .unwrap();
    assert_eq!(qc, "ok", "quick_check after kill");
    let links: i64 = conn
        .query_row("SELECT COUNT(*) FROM record_links", [], |r| r.get(0))
        .unwrap();
    assert_eq!(links, 0, "no record_links row survived the kill");
    let link_ops: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM oplog WHERE op_type = 'link'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(link_ops, 0, "no link oplog op survived the kill");
    // The two records committed before the link are intact (wholly old).
    let memories: i64 = conn
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(memories, 2, "pre-link records intact");
}
