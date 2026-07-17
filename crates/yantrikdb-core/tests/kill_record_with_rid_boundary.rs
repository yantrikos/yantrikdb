//! v0.10 Item 4a.6d-3 — the record_with_rid-boundary kill proof.
//!
//! `record_with_rid` is the cluster apply primitive AND a public origin API,
//! and until 4a.6d-3 it had the exact pre-4a.6a shape `record()` was cured of:
//! the memories row committed in its own savepoint, and the
//! `"record_with_rid"` oplog op was written AFTERWARDS on a separate
//! autocommit. A process death between the two left a durable row with no
//! oplog provenance — and on THIS path the orphan is unrepairable by retry:
//! the row exists, so the retry takes the `was_new_row = false` arm, which
//! deliberately skips `log_op`. The op is not late; it is lost FOREVER, and
//! with it the write's replication to every peer.
//!
//! After 4a.6d-3: the row, the session updates, the `record_with_rid` op and
//! the entity-materialization enqueue share one savepoint, so a kill at this
//! boundary rolls back all of it. Neither half survives; the retry then takes
//! the fresh-row path and writes everything.
//!
//! Mechanics mirror `kill_record_boundary.rs`: re-invoke this test binary to
//! run the ignored child helper with the failpoint armed, wait for the
//! handshake line with a bounded timeout, kill the child, reopen, assert.
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
fn child_parks_at_record_with_rid_boundary() {
    let db_path = std::env::var("YANTRIKDB_KILL_DB").expect("child needs YANTRIKDB_KILL_DB");
    let db = YantrikDB::new(&db_path, 8).unwrap();

    println!("CHILD_READY:opened");
    use std::io::Write;
    let _ = std::io::stdout().flush();

    // Parks at the row/oplog boundary. The parent kills us here.
    let _ = db.record_with_rid(
        "0198c1c2-0000-7000-8000-00000000feed",
        "the replicated write that must not survive",
        "semantic",
        0.5,
        0.0,
        604800.0,
        &serde_json::json!({}),
        &test_vec(3.0, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
        1_750_000_000_000_000,
        &["KillProofEntity"],
        "test-embedder",
        None,
        yantrikdb::provenance::WriteAdmission::Origin,
    );
    unreachable!("parent must kill us at the failpoint");
}

/// PARENT half — the actual proof.
#[test]
fn kill_between_record_with_rid_row_and_oplog_leaves_neither() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("kill_rwr.db");
    let db_path_str = db_path.to_str().unwrap().to_string();

    let exe = std::env::current_exe().unwrap();
    let mut child = Command::new(exe)
        .args([
            "child_parks_at_record_with_rid_boundary",
            "--exact",
            "--ignored",
            "--nocapture",
        ])
        .env("YANTRIKDB_KILL_DB", &db_path_str)
        .env(
            "YANTRIKDB_FAILPOINTS",
            "record_with_rid.between_row_and_oplog",
        )
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
            Ok(line) if line == "FAILPOINT:record_with_rid.between_row_and_oplog" => {
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

    child.kill().expect("kill child");
    let _ = child.wait();

    let db = YantrikDB::new(&db_path_str, 8).unwrap();
    let conn = db.conn();

    let qc: String = conn
        .query_row("PRAGMA quick_check", [], |r| r.get(0))
        .unwrap();
    assert_eq!(qc, "ok", "quick_check after kill");

    // THE assertion. On the pre-4a.6d-3 code the row exists (its savepoint
    // RELEASEd before the kill) with no record_with_rid op — an orphan no
    // retry can repair, because was_new_row=false skips log_op forever.
    let memories: i64 = conn
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        memories, 0,
        "the killed record_with_rid's row survived without oplog provenance — \
         unrepairable by retry (was_new_row=false skips log_op)"
    );

    let ops: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM oplog WHERE op_type = 'record_with_rid'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(ops, 0, "the killed write logged an op");

    // Both-or-neither, stated as the standing invariant.
    let rows_without_ops: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM memories m WHERE NOT EXISTS \
             (SELECT 1 FROM oplog o WHERE o.op_type = 'record_with_rid' \
              AND o.target_rid = m.rid)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        rows_without_ops, 0,
        "a memories row exists with no record_with_rid op"
    );
}
