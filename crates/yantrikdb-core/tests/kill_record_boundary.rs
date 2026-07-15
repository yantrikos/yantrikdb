//! v0.10 Item 4a.6a — the record-boundary kill proof.
//!
//! This is the test that actually discriminates. Every other assertion about
//! 4a.6a passes on the OLD code too, because the old design reached the same
//! FINAL state by different means: it wrote the memories row, and if the vector
//! append then failed it reclaimed the row with a compensating DELETE. Observing
//! the end state cannot tell "never written" from "written then deleted".
//!
//! The difference is only visible when the process dies BETWEEN the row and its
//! oplog provenance — which is exactly the failure the old comment in record.rs
//! recorded as "over 39 days on trader's `default` DB, this leaked 23k rows".
//! The compensating DELETE never covered that case: a crash beats it.
//!
//! Before 4a.6a: the memories INSERT autocommitted on its own, so a kill here
//! left a durable row with NO oplog op — an orphan, invisible to replication and
//! to any oplog inspector, and no compensation could run because the process was
//! gone.
//!
//! After 4a.6a: the row, the session updates, the `record` op and the
//! `materialize_record_post` enqueue share one transaction, so a kill at this
//! boundary rolls back all of it. Neither half survives.
//!
//! Mechanics mirror `kill_link_boundary.rs` (v0.10 Phase 0): re-invoke this test
//! binary to run the ignored child helper with the failpoint armed, wait for the
//! handshake line with a bounded timeout (no sleep roulette), kill the child,
//! reopen, and assert.
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

/// CHILD half — ignored in normal runs; the parent invokes it explicitly.
#[test]
#[ignore]
fn child_parks_at_record_boundary() {
    let db_path = std::env::var("YANTRIKDB_KILL_DB").expect("child needs YANTRIKDB_KILL_DB");
    let db = YantrikDB::new(&db_path, 8).unwrap();

    // Handshake BEFORE any write. The failpoint is armed process-wide via
    // YANTRIKDB_FAILPOINTS and it lives inside record(), so a "baseline" record
    // here would park at the failpoint before the parent ever saw us — unlike
    // kill_link_boundary, whose failpoint is in link() and whose setup records
    // freely. The engine is open at this point, which is all the parent needs.
    println!("CHILD_READY:opened");
    use std::io::Write;
    let _ = std::io::stdout().flush();

    // Parks inside the transaction, after the memories INSERT, before the oplog
    // op. The parent kills us here.
    let _ = db.record(
        "the record that must not survive",
        "semantic",
        0.5,
        0.0,
        604800.0,
        &serde_json::json!({}),
        &test_vec(2.0, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    );
    unreachable!("parent must kill us at the failpoint");
}

/// PARENT half — the actual proof.
#[test]
fn kill_between_record_row_and_oplog_leaves_neither() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("kill_record.db");
    let db_path_str = db_path.to_str().unwrap().to_string();

    let exe = std::env::current_exe().unwrap();
    let mut child = Command::new(exe)
        .args([
            "child_parks_at_record_boundary",
            "--exact",
            "--ignored",
            "--nocapture",
        ])
        .env("YANTRIKDB_KILL_DB", &db_path_str)
        .env("YANTRIKDB_FAILPOINTS", "record.between_row_and_oplog")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn child");

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
            Ok(line) if line == "FAILPOINT:record.between_row_and_oplog" => {
                saw_failpoint = true;
                break;
            }
            Ok(_) => {}
            Err(_) => {
                if let Ok(Some(_)) = child.try_wait() {
                    break;
                }
            }
        }
    }
    assert!(saw_ready, "child never opened the engine");
    assert!(saw_failpoint, "child never hit the armed failpoint");

    // Kill mid-transaction — the deterministic version of a crash.
    child.kill().expect("kill child");
    let _ = child.wait();

    let db = YantrikDB::new(&db_path_str, 8).unwrap();
    let conn = db.conn();

    let qc: String = conn
        .query_row("PRAGMA quick_check", [], |r| r.get(0))
        .unwrap();
    assert_eq!(qc, "ok", "quick_check after kill");

    // THE assertion. On the pre-4a.6a code this row exists (its INSERT
    // autocommitted) with no oplog op to account for it — a 23k-leak orphan.
    let orphan: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories WHERE text = ?1",
            rusqlite::params!["the record that must not survive"],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        orphan, 0,
        "a memories row survived the kill with no oplog provenance — the orphan leak"
    );

    // The killed write was the only write, so the database must be empty. On the
    // pre-4a.6a code this is 1 — the autocommitted orphan.
    let memories: i64 = conn
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(memories, 0, "the killed record's row survived");

    let record_ops: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM oplog WHERE op_type = 'record'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(record_ops, 0, "the killed write logged an op");

    // Every surviving row has provenance: no row without an op, no op without a
    // row. That is the invariant the old three-autocommit shape could not hold.
    let rows_without_ops: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories m WHERE NOT EXISTS \
             (SELECT 1 FROM oplog o WHERE o.op_type = 'record' AND o.target_rid = m.rid)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        rows_without_ops, 0,
        "a memories row exists with no record op"
    );
}
