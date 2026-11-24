//! #298: writes wait busy_timeout behind another connection's write lock.
//!
//! The engine's write transactions used to be DEFERRED: they read first and
//! then upgraded to a write lock. SQLite never runs the busy handler for that
//! upgrade (sqlite3BtreeBeginTrans retries only while the connection holds no
//! transaction), so while another connection held the write lock `record()`
//! failed at once with "database is locked" instead of waiting up to
//! busy_timeout (5 s). If another writer committed between the read and the
//! write, it failed with SQLITE_BUSY_SNAPSHOT instead.
//!
//! The tests avoid wall-clock thresholds where they can. A holder thread
//! sets a flag just before it commits; a call that returns Ok with the flag
//! still clear cannot have waited for the lock.

use super::*;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

const HOLD: Duration = Duration::from_millis(400);

/// The write lock, held from a second connection for HOLD.
struct Holder {
    handle: std::thread::JoinHandle<()>,
    releasing: Arc<AtomicBool>,
}

impl Holder {
    /// Returns once the lock is held.
    fn take(path: &std::path::Path) -> Self {
        let path = path.to_path_buf();
        let releasing = Arc::new(AtomicBool::new(false));
        let flag = releasing.clone();
        let (tx, rx) = mpsc::channel();
        let handle = std::thread::spawn(move || {
            let other = rusqlite::Connection::open(&path).unwrap();
            other.execute_batch("BEGIN IMMEDIATE").unwrap();
            tx.send(()).unwrap();
            std::thread::sleep(HOLD);
            flag.store(true, Ordering::SeqCst);
            other.execute_batch("COMMIT").unwrap();
        });
        rx.recv().unwrap();
        Self { handle, releasing }
    }

    /// True once the holder has started to let go of the lock.
    fn released(&self) -> bool {
        self.releasing.load(Ordering::SeqCst)
    }

    fn join(self) {
        self.handle.join().unwrap();
    }
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

fn batch_item(text: &str, seed: f32) -> RecordInput {
    RecordInput {
        created_at: None,
        idempotency_key: None,
        text: text.into(),
        memory_type: "semantic".into(),
        importance: 0.5,
        valence: 0.0,
        half_life: 604800.0,
        metadata: serde_json::json!({}),
        embedding: vec_seed(seed, 8),
        namespace: "default".into(),
        certainty: 0.8,
        domain: "general".into(),
        source: "user".into(),
        emotional_state: None,
    }
}

/// A second connection, opened with busy_timeout 0, that can tell whether
/// the write lock is free right now.
fn write_lock_is_free(path: &std::path::Path) -> bool {
    let probe = rusqlite::Connection::open(path).unwrap();
    probe.busy_timeout(Duration::ZERO).unwrap();
    match probe.execute_batch("BEGIN IMMEDIATE") {
        Ok(()) => {
            probe.execute_batch("ROLLBACK").unwrap();
            true
        }
        Err(rusqlite::Error::SqliteFailure(e, _))
            if e.code == rusqlite::ErrorCode::DatabaseBusy =>
        {
            false
        }
        Err(e) => panic!("probing the write lock failed for another reason: {e}"),
    }
}

#[test]
fn record_waits_for_another_connections_write_lock() {
    let dir = tempfile::tempdir().unwrap();
    let (db, path) = file_db(&dir);

    let holder = Holder::take(&path);
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
    let waited = holder.released();
    holder.join();

    let rid = rid.expect("record() must wait busy_timeout, not fail with database is locked");
    assert!(waited, "record() returned before the lock was released");
    assert!(
        db.get(&rid).unwrap().is_some(),
        "the waited write must land"
    );
}

#[test]
fn record_batch_waits_for_another_connections_write_lock() {
    let dir = tempfile::tempdir().unwrap();
    let (db, path) = file_db(&dir);

    let holder = Holder::take(&path);
    let rids = db.record_batch(&[batch_item("the ferry leaves at ten", 2.5)]);
    let waited = holder.released();
    holder.join();

    let rids = rids.expect("record_batch() must wait busy_timeout");
    assert!(
        waited,
        "record_batch() returned before the lock was released"
    );
    assert_eq!(rids.len(), 1);
}

/// The category, not just the instances above: a transaction the engine
/// opens on its write connection, by either route, holds the write lock
/// from its first statement, before it has written anything.
#[test]
fn engine_transactions_take_the_write_lock_up_front() {
    let dir = tempfile::tempdir().unwrap();
    let (db, path) = file_db(&dir);
    let conn = db.conn();

    // unchecked_transaction(): the connection's default behaviour.
    let tx = conn.unchecked_transaction().unwrap();
    assert!(
        !write_lock_is_free(&path),
        "unchecked_transaction() must take the write lock at BEGIN"
    );
    let n: i64 = tx
        .query_row("SELECT COUNT(*) FROM memories", [], |r| r.get(0))
        .unwrap();
    assert!(n >= 1);
    tx.commit().unwrap();
    assert!(write_lock_is_free(&path), "commit must release it");

    // An outermost SavepointGuard.
    let sp = crate::engine::savepoint::SavepointGuard::new(&conn, "lock_wait_probe").unwrap();
    assert!(
        !write_lock_is_free(&path),
        "an outermost SavepointGuard must take the write lock at BEGIN"
    );
    sp.release().unwrap();
    assert!(conn.is_autocommit(), "release must end the transaction");
    assert!(write_lock_is_free(&path), "release must release it");
}

/// The SQLITE_BUSY_SNAPSHOT shape: another connection commits writes
/// continually while the engine records. A deferred read-then-write
/// transaction fails at once whenever a commit lands between its read and
/// its write; one that takes the write lock first waits instead.
///
/// Only a FAST failure is the bug. A contender that retakes the lock this
/// often can still starve a write past busy_timeout (SQLite's busy handler
/// polls with no fairness; slow commits on a Windows runner make it likely),
/// and that write then fails after ~5 s — the wait the fix exists to give.
#[test]
fn writes_survive_a_continually_committing_second_connection() {
    let dir = tempfile::tempdir().unwrap();
    let (db, path) = file_db(&dir);

    // The contender's commit count is shared. The writes start only once it
    // is committing, and after each write the test waits (briefly) for its
    // next commit, so commits land between writes however loaded the runner
    // is. WRITES stays well under the delta tier's 256-entry backpressure
    // limit: these tests run no compactor.
    const WRITES: usize = 120;
    const MIN_INTERLEAVED_COMMITS: u64 = 60;
    let stop = Arc::new(AtomicBool::new(false));
    let commits = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let contender = {
        let path = path.clone();
        let stop = stop.clone();
        let commits = commits.clone();
        std::thread::spawn(move || {
            let other = rusqlite::Connection::open(&path).unwrap();
            other.busy_timeout(Duration::from_secs(5)).unwrap();
            // Short lock holds: with the default synchronous=FULL each commit
            // flushes while holding the lock, which on a slow runner keeps
            // it held almost all the time and turns the test into a
            // starvation test.
            other
                .execute_batch(
                    "PRAGMA synchronous=OFF;                      CREATE TABLE IF NOT EXISTS contender (x INTEGER)",
                )
                .unwrap();
            while !stop.load(Ordering::SeqCst) {
                other
                    .execute_batch("BEGIN IMMEDIATE; INSERT INTO contender VALUES (1); COMMIT")
                    .unwrap();
                commits.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(Duration::from_micros(200));
            }
        })
    };
    while commits.load(Ordering::SeqCst) == 0 {
        std::thread::sleep(Duration::from_millis(1));
    }

    let first = commits.load(Ordering::SeqCst);
    let mut failures = Vec::new();
    let mut starved = 0usize;
    for i in 0..WRITES {
        let seed = 3.0 + i as f32 * 0.01;
        let text = format!("contended write number {i} about the harbour");
        let before = commits.load(Ordering::SeqCst);
        let started = std::time::Instant::now();
        let result = if i % 2 == 0 {
            db.record(
                &text,
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
            .map(|_| ())
        } else {
            db.record_batch(&[batch_item(&text, seed)]).map(|_| ())
        };
        if let Err(e) = result {
            let took = started.elapsed();
            if took < Duration::from_secs(1) {
                failures.push(format!("{e} after {took:?}"));
            } else {
                starved += 1;
            }
        }
        let deadline = std::time::Instant::now() + Duration::from_millis(200);
        while commits.load(Ordering::SeqCst) == before && std::time::Instant::now() < deadline {
            std::thread::yield_now();
        }
    }
    let interleaved = commits.load(Ordering::SeqCst) - first;
    stop.store(true, Ordering::SeqCst);
    contender.join().unwrap();

    assert!(
        starved * 4 < WRITES,
        "{starved} of {WRITES} writes timed out after busy_timeout; the test          no longer exercises the race it is for"
    );
    assert!(
        interleaved >= MIN_INTERLEAVED_COMMITS,
        "only {interleaved} contender commits landed during {WRITES} writes"
    );
    assert!(
        failures.is_empty(),
        "{} of {WRITES} writes failed fast beside a committing connection          (the #298 shape; {starved} more were starved past busy_timeout): {:?}",
        failures.len(),
        &failures[..failures.len().min(3)]
    );
}

/// A correction that waits for the write lock must not leave the correction
/// epoch odd while it waits: a concurrent recall spins on an odd epoch for
/// only ~1M iterations before failing with RecallContended.
#[test]
fn correct_waiting_for_the_lock_does_not_starve_recall() {
    let dir = tempfile::tempdir().unwrap();
    let (db, path) = file_db(&dir);
    let rid = record_one(&db, "the harbour opens at six", 4.0);
    let generation = db.search_state.load_full().generation;

    let run = |correction: &(dyn Fn() -> crate::error::Result<CorrectionResult> + Sync)| {
        let holder = Holder::take(&path);
        let mut recall_errors = Vec::new();
        let corrected = std::thread::scope(|s| {
            let worker = s.spawn(correction);
            while !worker.is_finished() {
                if let Err(e) = db.recall(
                    &vec_seed(4.0, 8),
                    5,
                    None,
                    None,
                    false,
                    false,
                    None,
                    true,
                    None,
                    None,
                    None,
                    None,
                    None,
                    false,
                    None,
                    None,
                ) {
                    recall_errors.push(e.to_string());
                }
            }
            worker.join().unwrap()
        });
        holder.join();
        (corrected, recall_errors)
    };

    // Scalar path (no text change).
    let (corrected, errors) =
        run(&|| db.correct(&rid, None, None, Some(0.9), None, "importance under lock"));
    corrected.expect("correct() must wait for the lock");
    assert!(
        errors.is_empty(),
        "recall failed while correct() waited for the lock: {:?}",
        &errors[..errors.len().min(3)]
    );

    // Text-changing path (re-embeds, publishes a vector).
    let (corrected, errors) = run(&|| {
        db.correct_with_embedding(
            &rid,
            Some("the harbour opens at seven"),
            &vec_seed(4.5, 8),
            generation,
            None,
            None,
            None,
            "text under lock",
        )
    });
    corrected.expect("a text correction must wait for the lock");
    assert!(
        errors.is_empty(),
        "recall failed while a text correction waited for the lock: {:?}",
        &errors[..errors.len().min(3)]
    );
}

/// A read-only snapshot must not take the write lock, even when it runs on
/// the write connection: with YANTRIKDB_READ_POOL=0, read_conn() falls back
/// to it, and that connection's default is now IMMEDIATE.
#[test]
fn thread_read_snapshot_does_not_wait_for_the_write_lock() {
    let dir = tempfile::tempdir().unwrap();
    let (mut db, path) = file_db(&dir);
    // The pool the env var would leave empty, without touching the env
    // (other tests run in parallel in this process).
    db.read_conns.clear();
    let query = crate::engine::thread::ThreadQuery {
        entities: vec!["harbour".to_string()],
        phrases: Vec::new(),
        topic_rids: Vec::new(),
    };

    let holder = Holder::take(&path);
    let out = db.recall_thread_v2("default", &query, 10);
    let waited = holder.released();
    holder.join();

    out.expect("a read snapshot must succeed under another writer's lock");
    assert!(
        !waited,
        "recall_thread_v2 waited for the write lock: its read snapshot took it"
    );
}

/// The rebuild runs its DELETE inside its own transaction, so a rebuild that
/// fails part-way keeps the previous counts instead of an empty table.
#[test]
fn failed_token_case_rebuild_keeps_the_old_counts() {
    let dir = tempfile::tempdir().unwrap();
    let (db, _path) = file_db(&dir);
    {
        let conn = db.conn();
        conn.execute(
            "INSERT OR REPLACE INTO token_case_stats (token, lower_n, cap_mid_n, cap_start_n) \
             VALUES ('sentinel', 7, 0, 0)",
            [],
        )
        .unwrap();
        conn.execute_batch(
            "CREATE TEMP TRIGGER refuse_token_case BEFORE INSERT ON token_case_stats \
             BEGIN SELECT RAISE(ABORT, 'refused by test'); END",
        )
        .unwrap();
    }

    let err = db
        .rebuild_token_case_stats()
        .expect_err("the trigger must make the rebuild fail");
    assert!(err.to_string().contains("refused by test"), "{err}");

    let conn = db.conn();
    assert!(
        conn.is_autocommit(),
        "a failed rebuild must end its transaction"
    );
    let sentinel: i64 = conn
        .query_row(
            "SELECT lower_n FROM token_case_stats WHERE token = 'sentinel'",
            [],
            |r| r.get(0),
        )
        .expect("the old counts must survive a failed rebuild");
    assert_eq!(sentinel, 7);
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

/// Rule 10 (CONCURRENCY.md), enforced on the source: every engine
/// transaction goes through `unchecked_transaction()` (IMMEDIATE by default
/// on the write connection) or `SavepointGuard`. A raw `SAVEPOINT`/`BEGIN`/
/// `RELEASE` string would open a DEFERRED transaction and bring #298 back,
/// and an explicit `Deferred` is allowed only for the one read-only snapshot
/// that may run on the write connection. A tripwire, not a proof: it reads
/// string literals and spellings, not what the code executes.
#[test]
fn no_bare_savepoint_or_begin_outside_the_guard() {
    fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let p = entry.unwrap().path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|e| e == "rs") {
                out.push(p);
            }
        }
    }
    /// A string literal whose text starts (after whitespace) with a
    /// transaction keyword. Upper case anywhere (the codebase's SQL
    /// convention); any case on a line that executes SQL, so word lists
    /// that happen to contain "begin" or "release" stay quiet.
    fn opens_transaction_literal(code: &str) -> bool {
        let runs_sql = ["execute", "prepare", "query"]
            .iter()
            .any(|m| code.contains(m));
        let lower = code.to_ascii_lowercase();
        lower.match_indices('"').any(|(i, _)| {
            let rest = lower[i + 1..].trim_start();
            let original = code[i + 1..].trim_start();
            ["savepoint", "begin", "release"].iter().any(|kw| {
                let word = rest.starts_with(kw)
                    && rest[kw.len()..]
                        .chars()
                        .next()
                        .is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_');
                word && (runs_sql || original.starts_with(&kw.to_ascii_uppercase()))
            })
        })
    }

    let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let mut files = Vec::new();
    walk(&crates.join("yantrikdb-core").join("src"), &mut files);
    walk(&crates.join("yantrikdb-python").join("src"), &mut files);
    assert!(files.len() > 50, "walked too few files: {}", files.len());

    let norm = |p: &std::path::Path| p.to_string_lossy().replace('\\', "/");
    let mut offenders = Vec::new();
    for f in &files {
        let path = norm(f);
        // The guard itself, and tests that drive SQLite directly on their
        // own connections (this file's holders, the guard's own tests).
        if path.ends_with("engine/savepoint.rs") || path.contains("engine/tests/") {
            continue;
        }
        let text = std::fs::read_to_string(f).unwrap();
        for (i, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            let deferred = code.contains("TransactionBehavior::Deferred")
                && !path.ends_with("engine/thread.rs");
            if opens_transaction_literal(code) || deferred {
                offenders.push(format!("{path}:{}: {}", i + 1, line.trim()));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "open transactions with SavepointGuard or unchecked_transaction(), \
         not raw SQL or an explicit Deferred (Rule 10, #298):\n{}",
        offenders.join("\n")
    );
}
