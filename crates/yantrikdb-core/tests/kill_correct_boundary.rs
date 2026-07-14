//! v0.10 Item 3 — the correction kill proof.
//!
//! Complements the Phase-0 link kill proof, but asserts the OPPOSITE
//! property because the failpoint sits at a different place in the
//! protocol. `correct.between_commit_and_delta` fires AFTER the single
//! SQL transaction (revision + memories UPDATE + `correct` oplog intent)
//! has committed and BEFORE the in-memory delta tombstone+append. So a
//! process killed here must leave:
//!
//! - SQL fully consistent: new text + new embedding + revision row +
//!   `correct` oplog op ALL durable (the correction is not lost); and
//! - the vector index self-healing on reopen: it rebuilds from the
//!   memories table, so the record is retrievable under its NEW meaning
//!   despite the crash landing before the in-memory index update.
//!
//! This is the kill-safety guarantee for a text-changing correction:
//! boundary #6 (durable intent) + rebuild-from-SQL recovery, proven by
//! an actual mid-protocol process kill rather than reasoning.
#![cfg(all(feature = "testing", feature = "bundled-embedder"))]

use std::io::BufRead;
use std::process::{Command, Stdio};
use std::time::Duration;

use yantrikdb::YantrikDB;

const OLD_TEXT: &str = "I love hiking in the mountains at dawn";
const NEW_TEXT: &str = "the quarterly financial revenue grew twenty percent";

/// CHILD half — ignored in normal runs; the parent invokes it explicitly.
/// Opens a bundled-embedder DB, records one memory, then calls `correct()`
/// with a text change, which re-embeds, commits the SQL transaction, and
/// parks at `correct.between_commit_and_delta`. The parent kills us there.
#[test]
#[ignore]
fn child_parks_at_correct_boundary() {
    let db_path = std::env::var("YANTRIKDB_KILL_DB").expect("child needs YANTRIKDB_KILL_DB");
    let db = YantrikDB::with_default(&db_path).unwrap();
    let rid = db
        .record_text(
            OLD_TEXT,
            "semantic",
            0.6,
            0.0,
            604800.0,
            &serde_json::json!({}),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    println!("CHILD_READY:{rid}");
    use std::io::Write;
    let _ = std::io::stdout().flush();
    // Parks AFTER the SQL commit, BEFORE the delta tombstone+append.
    let _ = db.correct(&rid, Some(NEW_TEXT), None, None, None, "topic corrected");
    unreachable!("parent must kill us at the failpoint");
}

/// PARENT half — the actual proof.
#[test]
fn kill_after_correct_commit_keeps_sql_consistent_and_index_rebuilds() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("kill_correct.db");
    let db_path_str = db_path.to_str().unwrap().to_string();

    let exe = std::env::current_exe().unwrap();
    let mut child = Command::new(exe)
        .args([
            "child_parks_at_correct_boundary",
            "--exact",
            "--ignored",
            "--nocapture",
        ])
        .env("YANTRIKDB_KILL_DB", &db_path_str)
        .env("YANTRIKDB_FAILPOINTS", "correct.between_commit_and_delta")
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

    let deadline = Duration::from_secs(90); // bundled embedder load + embed
    let mut rid = String::new();
    let mut saw_failpoint = false;
    let start = std::time::Instant::now();
    while start.elapsed() < deadline {
        match rx.recv_timeout(Duration::from_secs(10)) {
            Ok(line) if line.starts_with("CHILD_READY:") => {
                rid = line["CHILD_READY:".len()..].to_string();
            }
            Ok(line) if line == "FAILPOINT:correct.between_commit_and_delta" => {
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
    assert!(!rid.is_empty(), "child never recorded the memory");
    assert!(saw_failpoint, "child never hit the armed failpoint");

    // Kill mid-protocol — the deterministic crash, AFTER SQL commit.
    child.kill().expect("kill child");
    let _ = child.wait();

    // Reopen: the index rebuilds from SQL on open().
    let db = YantrikDB::with_default(&db_path_str).unwrap();
    {
        let conn = db.conn();
        let qc: String = conn
            .query_row("PRAGMA quick_check", [], |r| r.get(0))
            .unwrap();
        assert_eq!(qc, "ok", "quick_check after kill");

        // The correction is DURABLE: new text committed, not lost.
        let text: String = conn
            .query_row("SELECT text FROM memories WHERE rid = ?1", [&rid], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(text, NEW_TEXT, "corrected text is durable after kill");

        // Revision row + correct oplog intent both committed (boundary #6:
        // the replication event cannot be lost separately from the mutation).
        let revs: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM record_revisions WHERE rid = ?1",
                [&rid],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(revs, 1, "revision row durable");
        let ops: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM oplog WHERE op_type = 'correct' AND target_rid = ?1",
                [&rid],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(ops, 1, "correct oplog intent durable (boundary #6)");
    }

    // The vector index self-healed from SQL: the record is retrievable
    // under its NEW meaning even though the crash landed before the
    // in-memory delta update.
    let q = db.embed("financial revenue report").unwrap();
    let hits = db
        .recall(
            &q, 5, None, None, false, false, None, true, None, None, None, None, None, false,
        )
        .unwrap();
    assert!(
        hits.iter().any(|h| h.rid == rid),
        "record retrievable under new meaning after index rebuild"
    );
}
