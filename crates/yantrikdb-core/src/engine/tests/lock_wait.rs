//! #298: writes wait busy_timeout behind another process's write lock.
//!
//! The engine's write transactions used to be DEFERRED: they read first and
//! then upgraded to a write lock. SQLite never runs the busy handler for that
//! upgrade (sqlite3BtreeBeginTrans retries only while the connection holds no
//! transaction), so while another connection held the write lock `record()`
//! failed at once with "database is locked" instead of waiting up to
//! busy_timeout (5 s). Each test holds the write lock from a second
//! connection for HOLD, then checks the engine call both succeeded and
//! actually waited for the lock to come free.

use super::*;
use std::sync::mpsc;
use std::time::{Duration, Instant};

const HOLD: Duration = Duration::from_millis(400);

/// Hold the write lock on `path` from a second connection for HOLD. Returns
/// once the lock is held; the thread commits and exits after HOLD.
fn hold_write_lock(path: &std::path::Path) -> std::thread::JoinHandle<()> {
    let path = path.to_path_buf();
    let (tx, rx) = mpsc::channel();
    let handle = std::thread::spawn(move || {
        let other = rusqlite::Connection::open(&path).unwrap();
        other.execute_batch("BEGIN IMMEDIATE").unwrap();
        tx.send(()).unwrap();
        std::thread::sleep(HOLD);
        other.execute_batch("COMMIT").unwrap();
    });
    rx.recv().unwrap();
    handle
}

fn file_db(dir: &tempfile::TempDir) -> (YantrikDB, std::path::PathBuf) {
    let path = dir.path().join("lock_wait.db");
    let db = YantrikDB::new(path.to_str().unwrap(), 8).unwrap();
    // One committed row, so later writes have something to read first.
    record_one(&db, "seed memory about the harbour", 0.5);
    (db, path)
}

fn record_one(db: &YantrikDB, text: &str, seed: f32) -> String {
    db.record(
        text,
        "semantic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(seed, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap()
}

/// Waited for the lock: not instant, and not past busy_timeout.
fn assert_waited(started: Instant, what: &str) {
    let waited = started.elapsed();
    assert!(
        waited >= HOLD - Duration::from_millis(100),
        "{what} returned after {waited:?}, before the other connection let \
         go of the write lock; it cannot have waited for it"
    );
    assert!(
        waited < Duration::from_secs(4),
        "{what} took {waited:?}, close to busy_timeout"
    );
}

#[test]
fn record_waits_for_another_connections_write_lock() {
    let dir = tempfile::tempdir().unwrap();
    let (db, path) = file_db(&dir);

    let holder = hold_write_lock(&path);
    let started = Instant::now();
    let rid = db.record(
        "the ferry leaves at nine",
        "semantic",
        0.5,
        0.0,
        604800.0,
        &empty_meta(),
        &vec_seed(1.5, 8),
        "default",
        0.8,
        "general",
        "user",
        None,
    );
    assert_waited(started, "record()");
    holder.join().unwrap();

    let rid = rid.expect("record() must wait busy_timeout, not fail with database is locked");
    assert!(
        db.get(&rid).unwrap().is_some(),
        "the waited write must land"
    );
}

#[test]
fn record_batch_waits_for_another_connections_write_lock() {
    let dir = tempfile::tempdir().unwrap();
    let (db, path) = file_db(&dir);

    let holder = hold_write_lock(&path);
    let started = Instant::now();
    let rids = db.record_batch(&[RecordInput {
        created_at: None,
        idempotency_key: None,
        text: "the ferry leaves at ten".into(),
        memory_type: "semantic".into(),
        importance: 0.5,
        valence: 0.0,
        half_life: 604800.0,
        metadata: serde_json::json!({}),
        embedding: vec_seed(2.5, 8),
        namespace: "default".into(),
        certainty: 0.8,
        domain: "general".into(),
        source: "user".into(),
        emotional_state: None,
    }]);
    assert_waited(started, "record_batch()");
    holder.join().unwrap();

    let rids = rids.expect("record_batch() must wait busy_timeout");
    assert_eq!(rids.len(), 1);
}

/// The category, not just the two instances above: any transaction the
/// engine opens on its write connection, by either route, must take the
/// write lock before its first read.
#[test]
fn engine_transactions_take_the_write_lock_up_front() {
    let dir = tempfile::tempdir().unwrap();
    let (db, path) = file_db(&dir);

    // unchecked_transaction(): the connection's default behaviour.
    {
        let holder = hold_write_lock(&path);
        let conn = db.conn();
        let started = Instant::now();
        let tx = conn.unchecked_transaction().unwrap();
        assert_waited(started, "unchecked_transaction()");
        let n: i64 = tx
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        tx.execute("DELETE FROM memories WHERE rid = 'no-such-rid'", [])
            .unwrap();
        tx.commit().unwrap();
        assert!(n >= 1);
        holder.join().unwrap();
    }

    // An outermost SavepointGuard.
    {
        let holder = hold_write_lock(&path);
        let conn = db.conn();
        let started = Instant::now();
        let sp = crate::engine::savepoint::SavepointGuard::new(&conn, "lock_wait_probe").unwrap();
        assert_waited(started, "SavepointGuard::new()");
        let _: i64 = conn
            .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
            .unwrap();
        conn.execute("DELETE FROM memories WHERE rid = 'no-such-rid'", [])
            .unwrap();
        sp.release().unwrap();
        assert!(conn.is_autocommit(), "release must end the transaction");
        holder.join().unwrap();
    }
}

/// A savepoint nested inside an open transaction stays a plain savepoint:
/// releasing it must not commit the outer transaction.
#[test]
fn nested_savepoint_does_not_commit_the_outer_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let (db, _path) = file_db(&dir);
    let conn = db.conn();

    let outer = crate::engine::savepoint::SavepointGuard::new(&conn, "outer_probe").unwrap();
    conn.execute_batch("CREATE TABLE nested_probe (x INTEGER)")
        .unwrap();
    let inner = crate::engine::savepoint::SavepointGuard::new(&conn, "inner_probe").unwrap();
    conn.execute("INSERT INTO nested_probe VALUES (1)", [])
        .unwrap();
    inner.release().unwrap();
    assert!(
        !conn.is_autocommit(),
        "releasing the inner savepoint must leave the outer transaction open"
    );
    drop(outer);
    assert!(conn.is_autocommit(), "dropping the outer guard must end it");
    let gone = conn
        .query_row("SELECT COUNT(*) FROM nested_probe", [], |r| {
            r.get::<_, i64>(0)
        })
        .is_err();
    assert!(
        gone,
        "dropping the outer guard must roll back the inner work"
    );
}
