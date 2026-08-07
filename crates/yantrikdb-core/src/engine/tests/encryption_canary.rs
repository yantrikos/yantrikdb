//! "Encrypted means encrypted" — enforced by scanning the raw file.
//!
//! 0.13.2. Every previous encryption test asked the ENGINE whether it
//! had encrypted something, which is the same instrument answering for
//! its own claim. This suite asks the DISK. It exists because a canary
//! byte-scan of a released 0.13.1 database found every record's full
//! text sitting in plaintext in `oplog.payload`: `memories.text` was
//! properly sealed, the write-ahead projection was not, and
//! `is_encrypted()` reported true the whole time.
//!
//! The rule these tests pin: on an encrypted database, no byte of
//! recorded content appears anywhere in the file — no table exempt, no
//! sidecar exempt, no vintage exempt.

use super::*;
use helpers::empty_meta;
use std::path::Path;

const CANARY: &str = "ZANZIBAR-FALCON-7731 launch codes in the blue vault";
const CANARY_ENTITY: &str = "Priyamvada reports to Deepankar on skunkworks";

fn key() -> [u8; 32] {
    let mut k = [0u8; 32];
    for (i, b) in k.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(7).wrapping_add(11);
    }
    k
}

/// Every byte of the database and its sidecars (-wal, -shm).
fn raw_bytes(base: &Path) -> Vec<u8> {
    let mut blob = Vec::new();
    let dir = base.parent().unwrap();
    let name = base.file_name().unwrap().to_string_lossy().to_string();
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let fname = entry.file_name().to_string_lossy().to_string();
        if fname.starts_with(&name) {
            if let Ok(b) = std::fs::read(entry.path()) {
                blob.extend_from_slice(&b);
            }
        }
    }
    blob
}

fn contains(blob: &[u8], needle: &str) -> bool {
    blob.windows(needle.len()).any(|w| w == needle.as_bytes())
}

fn write_canaries(db: &YantrikDB) {
    for t in [CANARY, CANARY_ENTITY] {
        db.record_text(
            t,
            "semantic",
            0.6,
            0.0,
            604800.0,
            &empty_meta(),
            "secret",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    }
}

#[test]
fn encrypted_database_leaks_no_plaintext_to_disk() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("enc.db");
    {
        let db = YantrikDB::new_encrypted(path.to_str().unwrap(), 64, &key()).unwrap();
        assert!(db.is_encrypted());
        write_canaries(&db);
        // The oplog seal must be complete, not merely applied to new rows.
        assert_eq!(
            db.oplog_plaintext_rows().unwrap(),
            0,
            "every oplog payload must be sealed on an encrypted database"
        );
    }

    let blob = raw_bytes(&path);
    assert!(!blob.is_empty(), "the scan must actually read the file");
    for needle in [
        CANARY,
        CANARY_ENTITY,
        "ZANZIBAR-FALCON-7731",
        "Priyamvada",
        "Deepankar",
        "skunkworks",
        "blue vault",
    ] {
        assert!(
            !contains(&blob, needle),
            "PLAINTEXT LEAK: {needle:?} found in the raw database file"
        );
    }
}

#[test]
fn the_canary_scan_can_actually_fail() {
    // The control. A scan that cannot detect plaintext proves nothing
    // about the encrypted case — an instrument must be shown capable of
    // producing the negative result before its positive one counts.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("plain.db");
    {
        let db = YantrikDB::with_default(path.to_str().unwrap()).unwrap();
        write_canaries(&db);
    }
    let blob = raw_bytes(&path);
    assert!(
        contains(&blob, "ZANZIBAR-FALCON-7731"),
        "control: an UNencrypted database must show the canary, or the \
         scan is measuring nothing"
    );
}

#[test]
fn sealed_oplog_still_replicates_and_materializes() {
    // The seal is an at-rest property. Reading back through the engine's
    // own paths must be unchanged — a fix that quietly broke replication
    // or the materializer would trade one silent defect for another.
    use crate::replication::{extract_ops_since, extract_ops_since_enc};

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("repl.db");
    let db = YantrikDB::new_encrypted(path.to_str().unwrap(), 64, &key()).unwrap();
    write_canaries(&db);

    let ops = extract_ops_since_enc(&db.conn(), db.encryption(), None, None, None, 100).unwrap();
    assert!(!ops.is_empty(), "records must produce replicable ops");
    let carries_text = ops.iter().any(|o| {
        o.payload
            .get("text")
            .and_then(|v| v.as_str())
            .map(|t| t.contains("ZANZIBAR"))
            .unwrap_or(false)
    });
    assert!(
        carries_text,
        "a keyed reader must get the real payload back, not an empty one"
    );

    // And the keyless reader must REFUSE rather than hand back a payload
    // that silently parsed to {} — the failure mode that would have
    // turned this security fix into replication data loss.
    let keyless = extract_ops_since(&db.conn(), None, None, None, 100);
    match keyless {
        Err(_) => {}
        Ok(entries) => assert!(
            entries.iter().all(|o| o.payload.get("text").is_none()),
            "a keyless read must not surface record text"
        ),
    }
}

#[test]
fn migration_erases_freed_pages_not_merely_live_rows() {
    // 0.13.4. The 0.13.2 migration sealed every row and left the
    // plaintext in the pages those rows used to occupy:
    // `oplog_plaintext_rows()` read 0 while a raw byte scan still found
    // the canary. Rows are not bytes, and a count that reports the
    // former while claiming the latter is the same defect the migration
    // was written to fix.
    //
    // Enough rows that the seal cannot fit in the original pages, so the
    // old ones are genuinely freed rather than overwritten in place.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("freed.db");
    let k = key();
    const LEAK: &str = "ZANZIBAR-FALCON-7731-FREED-PAGE-CANARY";
    {
        let db = YantrikDB::new_encrypted(path.to_str().unwrap(), 64, &k).unwrap();
        write_canaries(&db);
        let conn = db.conn();
        for i in 0..2000 {
            conn.execute(
                "INSERT INTO oplog (op_id, op_type, timestamp, target_rid, payload, applied) \
                 VALUES (?1, 'record', 0.0, ?2, ?3, 1)",
                rusqlite::params![
                    format!("legacy-{i}"),
                    format!("rid-{i}"),
                    format!(r#"{{"rid":"rid-{i}","text":"{LEAK}"}}"#)
                ],
            )
            .unwrap();
        }
    }

    let db = YantrikDB::new_encrypted(path.to_str().unwrap(), 64, &k).unwrap();
    assert_eq!(db.oplog_plaintext_rows().unwrap(), 0, "rows must be sealed");
    drop(db);

    let blob = raw_bytes(&path);
    assert!(
        !contains(&blob, LEAK),
        "PLAINTEXT SURVIVES IN FREED PAGES: every oplog row was sealed and \
         the count read 0, but the bytes are still in the file. Sealing is \
         not erasing — the migration must rewrite the file."
    );
}

#[test]
fn pre_fix_plaintext_rows_are_healed_on_open() {
    // The migration. Simulates a database written by 0.13.1 or earlier
    // by writing a bare-JSON payload directly, then proves that opening
    // the database seals it and that the row still reads correctly.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("legacy.db");
    let k = key();
    {
        let db = YantrikDB::new_encrypted(path.to_str().unwrap(), 64, &k).unwrap();
        write_canaries(&db);
        db.conn()
            .execute(
                "INSERT INTO oplog (op_id, op_type, timestamp, target_rid, payload, applied) \
                 VALUES ('legacy-op-1', 'record', 0.0, 'rid-legacy', ?1, 1)",
                rusqlite::params![r#"{"text":"ZANZIBAR-FALCON-7731 legacy row"}"#],
            )
            .unwrap();
        assert_eq!(
            db.oplog_plaintext_rows().unwrap(),
            1,
            "the simulated legacy row must register as unsealed"
        );
    }

    let db = YantrikDB::new_encrypted(path.to_str().unwrap(), 64, &k).unwrap();
    assert_eq!(
        db.oplog_plaintext_rows().unwrap(),
        0,
        "opening an encrypted database must heal pre-0.13.2 plaintext rows"
    );
    let stored: String = db
        .conn()
        .query_row(
            "SELECT payload FROM oplog WHERE op_id = 'legacy-op-1'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(stored.starts_with("ENCv1:"), "healed row must be marked");
    let back = db.decode_oplog_payload(&stored).unwrap();
    assert!(
        back.contains("legacy row"),
        "healed row must still decode to its original payload"
    );
    drop(db);

    let blob = raw_bytes(&path);
    assert!(
        !contains(&blob, "ZANZIBAR-FALCON-7731 legacy row"),
        "the healed legacy payload must be gone from the raw file"
    );
}
