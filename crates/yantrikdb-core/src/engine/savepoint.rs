//! RAII savepoint guard (v0.10 Item 4a.6d-2a, issue #91).
//!
//! `record_batch`'s error arms used to unwind the savepoint by hand — and got
//! it wrong in every way that class of code gets it wrong. The INSERT-failure
//! arm ran `ROLLBACK TO batch_record` and returned WITHOUT `RELEASE`, and the
//! fallible calls before the INSERT (`serde_json::to_string`, the encrypt
//! wrappers) `?`-returned without even the `ROLLBACK TO`. Both leave the
//! outermost savepoint OPEN on the engine's single shared connection:
//! `SAVEPOINT` on an autocommit conn starts a transaction that only ends when
//! that savepoint is RELEASEd or the tx rolls back, so every later write on
//! the conn silently nests inside the abandoned savepoint — durable only if
//! some unrelated future write happens to close it, all lost together if
//! anything rolls back. Two error arms, two different manual unwind attempts,
//! both wrong: the rule wants ONE owner.
//!
//! The guard is that owner. Construct it right after `SAVEPOINT`; every early
//! `?`-return and every unwinding panic hits `Drop`, which runs
//! `ROLLBACK TO name; RELEASE name` — SQLite's idiom for "undo the work AND
//! close the frame". The happy path calls [`SavepointGuard::release`], which
//! consumes the guard so a released savepoint cannot also be rolled back.

use rusqlite::Connection;

/// RAII for `SAVEPOINT name … RELEASE name` on a shared connection.
///
/// The name is `&'static str` by design: savepoint names cannot be bound as
/// SQL parameters, so accepting runtime strings here would invite injection
/// through the back door. Every caller names its savepoint in source.
pub(crate) struct SavepointGuard<'a> {
    conn: &'a Connection,
    name: &'static str,
    released: bool,
}

impl<'a> SavepointGuard<'a> {
    /// Open the savepoint. On an autocommit connection this BEGINS a
    /// transaction that stays open until this guard releases or drops.
    pub(crate) fn new(conn: &'a Connection, name: &'static str) -> rusqlite::Result<Self> {
        conn.execute_batch(&format!("SAVEPOINT {name}"))?;
        Ok(Self {
            conn,
            name,
            released: false,
        })
    }

    /// Commit the savepoint's work. Consumes the guard: released and rolled
    /// back are mutually exclusive by construction.
    ///
    /// If the RELEASE itself fails (e.g. `SQLITE_FULL` at the outermost
    /// commit), the guard drops un-released and `Drop` rolls the work back —
    /// an Err from this method means NOT COMMITTED, never "maybe".
    pub(crate) fn release(mut self) -> rusqlite::Result<()> {
        self.conn.execute_batch(&format!("RELEASE {}", self.name))?;
        self.released = true;
        Ok(())
    }
}

impl Drop for SavepointGuard<'_> {
    fn drop(&mut self) {
        if !self.released {
            // ROLLBACK TO undoes the work but leaves the savepoint frame on
            // the stack; the RELEASE closes it (and, for the outermost
            // savepoint, ends the transaction). Best-effort: an unwind path
            // has no way to surface a second error, and the conn poisons no
            // locks — the next writer's own savepoint would otherwise nest.
            let _ = self
                .conn
                .execute_batch(&format!("ROLLBACK TO {n}; RELEASE {n}", n = self.name));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::YantrikDB;

    /// These pin the #91 property directly and deterministically — no failpoint
    /// needed: `Connection::is_autocommit()` is false exactly while a
    /// transaction (here: the outermost savepoint) is open.
    fn db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    #[test]
    fn release_commits_and_returns_conn_to_autocommit() {
        let db = db();
        let conn = db.conn();
        assert!(conn.is_autocommit(), "precondition");

        let sp = SavepointGuard::new(&conn, "sp_release").unwrap();
        assert!(
            !conn.is_autocommit(),
            "outermost savepoint must open a transaction"
        );
        conn.execute_batch("CREATE TABLE sp_probe (x INTEGER)")
            .unwrap();
        sp.release().unwrap();

        assert!(conn.is_autocommit(), "release must close the transaction");
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM sp_probe", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 0, "released work must be durable (table exists)");
    }

    /// THE #91 bug shape: an error path that abandons the savepoint. The old
    /// manual arms left `is_autocommit() == false` here — every later write on
    /// the shared conn then silently joined the abandoned transaction.
    #[test]
    fn drop_without_release_rolls_back_and_closes_the_frame() {
        let db = db();
        let conn = db.conn();

        {
            let _sp = SavepointGuard::new(&conn, "sp_drop").unwrap();
            conn.execute_batch("CREATE TABLE sp_gone (x INTEGER)")
                .unwrap();
            // falls out of scope un-released — the early-`?`-return shape
        }

        assert!(
            conn.is_autocommit(),
            "drop must ROLLBACK TO *and* RELEASE — a bare ROLLBACK TO leaves \
             the transaction open (#91)"
        );
        let missing = conn
            .query_row("SELECT COUNT(*) FROM sp_gone", [], |r| r.get::<_, i64>(0))
            .is_err();
        assert!(missing, "dropped savepoint's work must be rolled back");
    }

    #[test]
    fn unwind_rolls_back_and_closes_the_frame() {
        let db = db();
        let conn = db.conn();

        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _sp = SavepointGuard::new(&conn, "sp_panic").unwrap();
            conn.execute_batch("CREATE TABLE sp_panic_probe (x INTEGER)")
                .unwrap();
            panic!("simulated panic inside the savepoint");
        }));
        assert!(res.is_err(), "panic must propagate");

        assert!(
            conn.is_autocommit(),
            "an unwinding panic must not leave the savepoint open"
        );
        let missing = conn
            .query_row("SELECT COUNT(*) FROM sp_panic_probe", [], |r| {
                r.get::<_, i64>(0)
            })
            .is_err();
        assert!(missing, "unwound savepoint's work must be rolled back");
    }
}
