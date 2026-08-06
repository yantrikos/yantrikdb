use super::*;

pub(super) fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
    let raw: Vec<f32> = (0..dim).map(|i| (seed + i as f32) * 0.1).collect();
    let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
    raw.iter().map(|x| x / norm).collect()
}

pub(super) fn empty_meta() -> serde_json::Value {
    serde_json::json!({})
}

pub(super) fn seed_proposition(db: &YantrikDB, pid: &str) {
    // Derive per-proposition (src, rel, dst) so multiple seed calls in the
    // same DB don't collide on UNIQUE(src, rel_type, dst, namespace).
    let src = format!("src_{}", pid);
    let rel = format!("rel_{}", pid);
    let dst = format!("dst_{}", pid);
    db.conn()
        .execute(
            "INSERT OR IGNORE INTO propositions (proposition_id, src, rel_type, dst, namespace, created_at) \
             VALUES (?1, ?2, ?3, ?4, 'default', 0.0)",
            rusqlite::params![pid, src, rel, dst],
        )
        .unwrap();
}

pub(super) fn seed_contest_claim(
    db: &YantrikDB,
    proposition_id: &str,
    claim_id: &str,
    extractor: &str,
    polarity: i32,
    source_lineage_json: &str,
    source_memory_rid: Option<&str>,
    namespace: &str,
    valid_from: Option<f64>,
    valid_to: Option<f64>,
) {
    // Derive per-proposition entity triples so tests that seed multiple
    // propositions in one DB don't collide on the claim UNIQUE constraint
    // (src, dst, rel_type, extractor, polarity, namespace). The proposition
    // row uses 'X'/'rel'/'Y' by default (seed_proposition), but claim rows
    // can use any triple — the FK is to proposition_id, not to the triple.
    let src = format!("src_{}", proposition_id);
    let dst = format!("dst_{}", proposition_id);
    let rel = format!("rel_{}", proposition_id);
    db.conn()
        .execute(
            "INSERT INTO claims (claim_id, src, dst, rel_type, created_at, \
             extractor, polarity, namespace, proposition_id, regime_tag, \
             self_generated, source_lineage, modality_signal, weight, \
             source_memory_rid, valid_from, valid_to) \
             VALUES (?1, ?2, ?3, ?4, 0.0, ?5, ?6, ?7, ?8, 'default', 0, \
             ?9, 'text', 1.0, ?10, ?11, ?12)",
            rusqlite::params![
                claim_id,
                src,
                dst,
                rel,
                extractor,
                polarity,
                namespace,
                proposition_id,
                source_lineage_json,
                source_memory_rid,
                valid_from,
                valid_to,
            ],
        )
        .unwrap();
}

/// Helper: list columns of a SQLite table via PRAGMA.
pub(super) fn table_columns(conn: &rusqlite::Connection, table: &str) -> Vec<String> {
    let mut stmt = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .unwrap();
    stmt.query_map([], |row| row.get::<_, String>(1))
        .unwrap()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap()
}

/// Helper: assert an index exists in sqlite_master.
pub(super) fn index_exists(conn: &rusqlite::Connection, index: &str) -> bool {
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='index' AND name=?1",
            params![index],
            |row| row.get(0),
        )
        .unwrap();
    count == 1
}
