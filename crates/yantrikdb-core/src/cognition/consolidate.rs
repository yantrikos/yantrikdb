use std::collections::{BTreeMap, HashMap, HashSet};

use rusqlite::OptionalExtension;

use crate::engine::YantrikDB;
use crate::error::Result;
use crate::serde_helpers::{deserialize_f32, serialize_f32};
use crate::types::*;

const CONSOLIDATION_SCAN_OFFSET_META: &str = "consolidation_scan_offset";
const CONSOLIDATION_SCAN_MULTIPLIER: usize = 20;
const CONSOLIDATION_SCAN_FLOOR: usize = 100;
const CONSOLIDATION_SCAN_CEILING: usize = 500;
const CONSOLIDATION_SCAN_OVERLAP: usize = 25;

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum();
    let norm_a: f64 = a
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    let norm_b: f64 = b
        .iter()
        .map(|&x| (x as f64) * (x as f64))
        .sum::<f64>()
        .sqrt();
    // Issue #62 (defense-in-depth port of the #60 hnsw fix): `== 0.0`
    // misses NaN (`NaN == 0.0` is false), so a NaN norm sailed through and
    // this function returned NaN — poisoning recall scores AND the
    // response-level confidence. `!(n > 0.0)` catches NaN, zero, and
    // negatives; a degenerate vector now degrades to 0.0 similarity
    // instead of propagating NaN.
    if !(norm_a > 0.0) || !(norm_b > 0.0) {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Find clusters of related memories using greedy agglomerative approach.
///
/// Two memories can cluster together if:
///   - Embedding similarity >= sim_threshold
///   - Created within time_window_days of each other
///   - If `entities_by_rid` is Some AND both memories have non-empty entity sets,
///     they must share at least one entity. This guards against false merges
///     where two sentences are cosine-similar but refer to different entities
///     (e.g., "Alice is CEO" vs "Sarah is CTO" — same shape, different people).
///     When either memory has no entities the guard falls back to cosine-only
///     (no regression on memories predating entity extraction).
pub fn find_clusters(
    memories: &[MemoryWithEmbedding],
    entities_by_rid: Option<&HashMap<String, HashSet<String>>>,
    sim_threshold: f64,
    time_window_days: f64,
    min_cluster_size: usize,
    max_cluster_size: usize,
) -> Vec<Vec<usize>> {
    if memories.len() < min_cluster_size {
        return vec![];
    }

    // Sort by creation time (return indices)
    let mut indices: Vec<usize> = (0..memories.len()).collect();
    indices.sort_by(|&a, &b| memories[a].created_at.total_cmp(&memories[b].created_at));

    let mut used = HashSet::new();
    let mut clusters: Vec<Vec<usize>> = Vec::new();

    for (seed_pos, &i) in indices.iter().enumerate() {
        if used.contains(&i) {
            continue;
        }

        let mut cluster = vec![i];
        used.insert(i);

        // Candidates are the records LATER in the time order — indexed by
        // POSITION, not by raw index value. The old `j <= i` compared the
        // memory-index values, so clustering depended on storage order:
        // on any corpus where created_at != insertion order (event-time,
        // historical import, replication) it silently under-clustered, and
        // mine_topic_clusters (rows fed DESC) returned ZERO clusters for
        // every distinct-timestamp input. `used` already prevents
        // re-seeding and cosine is symmetric, so position-after-seed is the
        // correct and complete candidate set.
        for &j in &indices[seed_pos + 1..] {
            if used.contains(&j) {
                continue;
            }

            // Time proximity check
            let time_diff = (memories[j].created_at - memories[i].created_at).abs();
            if time_diff > time_window_days * 86400.0 {
                continue;
            }

            // Entity-overlap guard: if both memories have entities, require at
            // least one shared one. Skips the guard if either side is empty
            // (covers memories written before extraction existed).
            if let Some(emap) = entities_by_rid {
                let ei = emap.get(&memories[i].rid);
                let ej = emap.get(&memories[j].rid);
                if let (Some(a), Some(b)) = (ei, ej) {
                    if !a.is_empty() && !b.is_empty() && a.is_disjoint(b) {
                        continue;
                    }
                }
            }

            // Similarity check
            let sim = cosine_similarity(&memories[i].embedding, &memories[j].embedding);
            if sim >= sim_threshold {
                cluster.push(j);
                used.insert(j);

                if cluster.len() >= max_cluster_size {
                    break;
                }
            }
        }

        if cluster.len() >= min_cluster_size {
            clusters.push(cluster);
        }
    }

    clusters
}

/// Generate an extractive summary by selecting the most important memory
/// and combining key facts from the cluster.
pub fn extractive_summary(memories: &[MemoryWithEmbedding]) -> String {
    let mut ranked: Vec<&MemoryWithEmbedding> = memories.iter().collect();
    ranked.sort_by(|a, b| b.importance.total_cmp(&a.importance));

    let lead = &ranked[0].text;
    let additional: Vec<&str> = ranked[1..]
        .iter()
        .filter_map(|m| {
            let text = m.text.trim();
            if !text.is_empty() && text != lead.as_str() {
                Some(text)
            } else {
                None
            }
        })
        .collect();

    if additional.is_empty() {
        lead.clone()
    } else {
        let mut parts = vec![lead.as_str()];
        parts.extend(additional);
        parts.join(" | ")
    }
}

/// Consolidate an explicit cluster using CALLER-SUPPLIED text.
///
/// The default [`consolidate`] path writes [`extractive_summary`] (every
/// cluster text joined with `" | "`) carrying [`mean_embedding`] — the mean
/// of the members' vectors. Measured on BEAM (2026-08-11): that costs
/// accuracy, and the mechanism is EMBEDDING DILUTION rather than lost
/// content. Nothing is dropped — all N texts survive in the join — but N
/// precise vectors collapse into one average that matches no specific query
/// well, and when it does hit it drags N chunks of text into the answerer's
/// context budget.
///
/// This path exists so a caller can supply a real synthesis (e.g. from a
/// small local model) instead. The crucial difference is not the prose: it
/// is that the new memory is embedded FROM ITS OWN TEXT via the engine's
/// embedder, so the vector describes what the record actually says.
///
/// Bookkeeping is identical to [`consolidate`] — the same entity transfer,
/// `consolidation_members` CRDT rows, source marking, and oplog entry — so
/// the two paths cannot drift. The caller owns only the text.
///
/// `embedding`: `None` embeds `text` with the ENGINE's embedder. Callers
/// whose embedder lives outside the engine (the Python binding's
/// `set_embedder` path, where the engine itself has none) pass the vector
/// they computed for `text`. Either way the vector describes THIS TEXT —
/// that is the contract, and it is what distinguishes this path from the
/// cluster-mean default.
///
/// Refuses an empty cluster, unknown rids, and members already consolidated
/// (double-consolidating would orphan the first consolidation's members).
pub fn consolidate_cluster(
    db: &YantrikDB,
    source_rids: &[String],
    text: &str,
    embedding: Option<&[f32]>,
) -> Result<serde_json::Value> {
    if source_rids.is_empty() {
        return Err(crate::error::YantrikDbError::InvalidInput(
            "consolidate_cluster: source_rids is empty".into(),
        ));
    }
    if text.trim().is_empty() {
        return Err(crate::error::YantrikDbError::InvalidInput(
            "consolidate_cluster: text is empty — the caller owns the synthesis".into(),
        ));
    }
    if let Some(v) = embedding {
        crate::validate::validate_embedding("consolidate_cluster", v, db.embedding_dim())?;
    }
    let members = load_cluster_members(db, source_rids)?;
    commit_consolidation(
        db,
        &members,
        text,
        embedding.map(|v| v.to_vec()),
        false,
        None,
    )
}

/// Store a synthesis of `source_rids` BESIDE them, leaving every source live.
///
/// # Why this exists
///
/// `consolidate_cluster` retires its sources: it sets
/// `consolidation_status = 'consolidated'` (which the default recall filter
/// excludes) and multiplies their importance by 0.3. That is correct when the
/// synthesis is meant to REPLACE detail, and destructive when it is not.
///
/// Measured on BEAM 100k, the replacing form bought +18.8pp on abstention and
/// +16.8pp on temporal_reasoning — the two categories where noise hurts — and
/// cost -21.6 on summarization and -17.5 on preference_following, because the
/// verbatim detail those need had been hidden. Net -6.2pp, which is why the
/// whole mechanism was written off as churn rather than as mis-scoped.
///
/// # Why an ADDITIVE synthesis is worth storing at all
///
/// Reading BEAM's per-nugget judgments shows 26% of all lost points sit on
/// abstract topic labels — 'Initial project setup', 'Integration test
/// coverage' — and NONE of those strings occurs anywhere in 12.4M characters
/// of the stored conversations. They are abstractions over a span of turns,
/// not quotes. No retrieval policy can ever surface a phrase that was never
/// written; the abstraction has to be SYNTHESISED and stored. This is the
/// write-side half of that, and keeping the sources live is what separates it
/// from the version that already failed.
pub fn summarize_cluster(
    db: &YantrikDB,
    source_rids: &[String],
    text: &str,
    embedding: Option<&[f32]>,
) -> Result<serde_json::Value> {
    if source_rids.is_empty() {
        return Err(crate::error::YantrikDbError::InvalidInput(
            "summarize_cluster: source_rids is empty".into(),
        ));
    }
    if text.trim().is_empty() {
        return Err(crate::error::YantrikDbError::InvalidInput(
            "summarize_cluster: text is empty — the caller owns the synthesis".into(),
        ));
    }
    if let Some(v) = embedding {
        crate::validate::validate_embedding("summarize_cluster", v, db.embedding_dim())?;
    }
    let members = load_cluster_members(db, source_rids)?;
    commit_consolidation(
        db,
        &members,
        text,
        embedding.map(|v| v.to_vec()),
        true,
        None,
    )
}

struct SynthesisWrite<'a> {
    axis: &'a str,
    granularity: &'a str,
    metadata: &'a serde_json::Value,
    idempotency_key: &'a str,
}

fn build_synthesis_admission(
    db: &YantrikDB,
    cluster: &[MemoryWithEmbedding],
    write: &SynthesisWrite<'_>,
) -> Result<SynthesisAdmission> {
    let namespace = cluster
        .first()
        .map(|memory| memory.namespace.as_str())
        .unwrap_or("default");
    let conn = db.conn();
    let current_revision = |rid: &str| -> Result<i64> {
        Ok(conn.query_row(
            "SELECT COALESCE(MAX(revision_num), 0) FROM record_revisions WHERE rid = ?1",
            rusqlite::params![rid],
            |row| row.get(0),
        )?)
    };
    let mut closure: BTreeMap<String, SynthesisDependency> = BTreeMap::new();

    for direct in cluster {
        let state: Option<String> = conn.query_row(
            "SELECT synthesis_state FROM memories WHERE rid = ?1",
            rusqlite::params![&direct.rid],
            |row| row.get(0),
        )?;
        if state.as_deref().is_some_and(|state| state != "verified") {
            return Err(crate::error::YantrikDbError::InvalidInput(format!(
                "record_synthesis: direct source {} has synthesis_state {:?}",
                direct.rid, state
            )));
        }
        closure.insert(
            direct.rid.clone(),
            SynthesisDependency {
                source_rid: direct.rid.clone(),
                source_revision_num: current_revision(&direct.rid)?,
                is_direct: true,
            },
        );

        let inherited: Vec<SynthesisDependency> = {
            let mut stmt = conn.prepare(
                "SELECT source_rid, source_revision_num, namespace \
                 FROM synthesis_dependencies WHERE synthesis_rid = ?1 \
                 ORDER BY source_rid",
            )?;
            let rows = stmt.query_map(rusqlite::params![&direct.rid], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?;
            let mut dependencies = Vec::new();
            for row in rows {
                let (source_rid, source_revision_num, dependency_namespace) = row?;
                if dependency_namespace != namespace {
                    return Err(crate::error::YantrikDbError::InvalidInput(format!(
                        "record_synthesis: inherited source {source_rid} crosses namespace boundary"
                    )));
                }
                let (status, actual_namespace): (String, String) = conn.query_row(
                    "SELECT consolidation_status, namespace FROM memories WHERE rid = ?1",
                    rusqlite::params![&source_rid],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )?;
                let observed_revision = current_revision(&source_rid)?;
                if status != "active"
                    || actual_namespace != namespace
                    || observed_revision != source_revision_num
                {
                    return Err(crate::error::YantrikDbError::InvalidInput(format!(
                        "record_synthesis: inherited source {source_rid} is no longer valid"
                    )));
                }
                dependencies.push(SynthesisDependency {
                    source_rid,
                    source_revision_num,
                    is_direct: false,
                });
            }
            dependencies
        };
        if state.is_some() && inherited.is_empty() {
            return Err(crate::error::YantrikDbError::InvalidInput(format!(
                "record_synthesis: synthesized source {} has no authoritative dependencies",
                direct.rid
            )));
        }
        for dependency in inherited {
            closure
                .entry(dependency.source_rid.clone())
                .and_modify(|existing| {
                    existing.is_direct |= dependency.is_direct;
                })
                .or_insert(dependency);
        }
    }

    let dependencies: Vec<SynthesisDependency> = closure.into_values().collect();
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"yantrikdb:synthesis-evidence:v1\0");
    for dependency in &dependencies {
        hasher.update(dependency.source_rid.as_bytes());
        hasher.update(b"\0");
        hasher.update(&dependency.source_revision_num.to_le_bytes());
    }
    Ok(SynthesisAdmission {
        axis: write.axis.to_string(),
        granularity: write.granularity.to_string(),
        logical_key: write.idempotency_key.to_string(),
        evidence_version: hasher.finalize().to_hex().to_string(),
        dependencies,
    })
}

/// Persist one query-independent synthesized item beside its evidence.
///
/// `axis` names the representation this item belongs to (for example
/// `asked`, `contributed`, `decided`, or `who_said`). Multiple axes may point
/// at the same evidence; recall chooses among them later. `granularity` is
/// either `atomic` or `rollup`, so fine children remain first-class even when
/// a coarser node is also stored.
///
/// The caller must provide a stable `idempotency_key` for the logical item.
/// A retry over identical evidence returns the original record and rejects a
/// changed model output. When the authoritative evidence set changes, the
/// engine writes a new generation and atomically supersedes the older logical
/// generation; syntheses depending on the retired generation are invalidated.
#[allow(clippy::too_many_arguments)]
pub fn record_synthesis(
    db: &YantrikDB,
    source_rids: &[String],
    text: &str,
    embedding: Option<&[f32]>,
    axis: &str,
    granularity: &str,
    metadata: &serde_json::Value,
    idempotency_key: &str,
) -> Result<serde_json::Value> {
    if source_rids.is_empty() {
        return Err(crate::error::YantrikDbError::InvalidInput(
            "record_synthesis: source_rids is empty".into(),
        ));
    }
    if text.trim().is_empty() {
        return Err(crate::error::YantrikDbError::InvalidInput(
            "record_synthesis: text is empty".into(),
        ));
    }
    let valid_axis = !axis.is_empty()
        && axis.len() <= 64
        && axis
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-');
    if !valid_axis {
        return Err(crate::error::YantrikDbError::InvalidInput(
            "record_synthesis: axis must be 1-64 lowercase ASCII letters, digits, '_' or '-'"
                .into(),
        ));
    }
    if !matches!(granularity, "atomic" | "rollup") {
        return Err(crate::error::YantrikDbError::InvalidInput(
            "record_synthesis: granularity must be 'atomic' or 'rollup'".into(),
        ));
    }
    if !metadata.is_object() && !metadata.is_null() {
        return Err(crate::error::YantrikDbError::InvalidInput(
            "record_synthesis: metadata must be an object or null".into(),
        ));
    }
    if let Some(v) = embedding {
        crate::validate::validate_embedding("record_synthesis", v, db.embedding_dim())?;
    }
    let mut canonical_rids = source_rids.to_vec();
    canonical_rids.sort();
    canonical_rids.dedup();
    let members = load_cluster_members(db, &canonical_rids)?;
    let write = SynthesisWrite {
        axis,
        granularity,
        metadata,
        idempotency_key,
    };
    commit_consolidation(
        db,
        &members,
        text,
        embedding.map(|v| v.to_vec()),
        true,
        Some(&write),
    )
}

/// Load the named memories, refusing anything that would make the
/// consolidation incoherent. Shared by [`consolidate_cluster`].
fn load_cluster_members(
    db: &YantrikDB,
    source_rids: &[String],
) -> Result<Vec<MemoryWithEmbedding>> {
    let mut out = Vec::with_capacity(source_rids.len());
    for rid in source_rids {
        let row: Option<(String, String, f64, f64, f64, f64, String, String, String)> = {
            let conn = db.conn();
            conn.query_row(
                "SELECT type, text, created_at, importance, valence, half_life, \
                 metadata, namespace, consolidation_status \
                 FROM memories WHERE rid = ?1",
                rusqlite::params![rid],
                |r| {
                    Ok((
                        r.get(0)?,
                        r.get(1)?,
                        r.get(2)?,
                        r.get(3)?,
                        r.get(4)?,
                        r.get(5)?,
                        r.get(6)?,
                        r.get(7)?,
                        r.get(8)?,
                    ))
                },
            )
            .optional()?
        };
        let (
            memory_type,
            stored_text,
            created_at,
            importance,
            valence,
            half_life,
            meta,
            ns,
            status,
        ) = row.ok_or_else(|| {
            crate::error::YantrikDbError::NotFound(format!(
                "consolidate_cluster: memory {rid} not found"
            ))
        })?;
        if status != "active" {
            return Err(crate::error::YantrikDbError::InvalidInput(format!(
                "consolidate_cluster: memory {rid} is '{status}', not active — \
                 consolidating it again would orphan the first consolidation's members"
            )));
        }
        // Namespaces are tenant boundaries. A cluster spanning two of them
        // would store its synthesis in the FIRST member's namespace — text
        // synthesized from tenant B's memories, readable by tenant A — and
        // (in the destructive form) retire records in a namespace the
        // caller never named. Refuse the incoherent input instead.
        if let Some(first) = out.first() {
            let first: &MemoryWithEmbedding = first;
            if first.namespace != ns {
                return Err(crate::error::YantrikDbError::InvalidInput(format!(
                    "consolidate_cluster: members span namespaces \
                     ('{}' vs '{}' at {rid}) — a cluster must be \
                     consolidated within one namespace",
                    first.namespace, ns
                )));
            }
        }
        out.push(MemoryWithEmbedding {
            rid: rid.clone(),
            memory_type,
            text: db.decrypt_text(&stored_text)?,
            embedding: Vec::new(), // unused on this path: the text is re-embedded
            created_at,
            importance,
            valence,
            half_life,
            last_access: created_at,
            metadata: serde_json::from_str(&db.decrypt_text(&meta)?)
                .unwrap_or_else(|_| serde_json::json!({})),
            namespace: ns,
        });
    }
    Ok(out)
}

/// Compute the mean embedding of a set of memories.
pub fn mean_embedding(memories: &[MemoryWithEmbedding]) -> Vec<f32> {
    let n = memories.len() as f32;
    let dim = memories[0].embedding.len();
    let mut result = vec![0.0f32; dim];
    for mem in memories {
        for (i, &v) in mem.embedding.iter().enumerate() {
            result[i] += v;
        }
    }
    result.iter_mut().for_each(|v| *v /= n);
    result
}

/// Find clusters of memories that are candidates for consolidation.
///
/// When `require_entity_overlap` is true, candidate pairs must share at least
/// one entity in `memory_entities` (or have no entities on either side). This
/// prevents cosine-only false merges across distinct named subjects.
pub fn find_consolidation_candidates(
    db: &YantrikDB,
    sim_threshold: f64,
    time_window_days: f64,
    min_cluster_size: usize,
    limit: usize,
    require_entity_overlap: bool,
) -> Result<Vec<Vec<MemoryWithEmbedding>>> {
    find_consolidation_candidates_page(
        db,
        sim_threshold,
        time_window_days,
        min_cluster_size,
        limit,
        0,
        require_entity_overlap,
    )
    .map(|(clusters, _)| clusters)
}

/// Scan one stable, time-ordered page of consolidation inputs.
///
/// `offset` belongs to the maintenance cursor, while `scan_limit` is the
/// number of new rows examined by this page. A small look-behind keeps
/// clusters that straddle adjacent pages discoverable without preventing the
/// cursor from moving through a large corpus.
fn find_consolidation_candidates_page(
    db: &YantrikDB,
    sim_threshold: f64,
    time_window_days: f64,
    min_cluster_size: usize,
    scan_limit: usize,
    offset: usize,
    require_entity_overlap: bool,
) -> Result<(Vec<Vec<MemoryWithEmbedding>>, usize)> {
    if scan_limit == 0 {
        return Ok((Vec::new(), 0));
    }

    let overlap = offset.min(CONSOLIDATION_SCAN_OVERLAP);
    let page_offset = offset - overlap;
    let page_limit = scan_limit.saturating_add(overlap);

    // Phase 1: query rows while holding the conn lock, then drop it.
    // Scope is explicit so the guard CANNOT live across the subsequent
    // calls to db.decrypt_text / db.decrypt_embedding in Phase 2. See
    // CONCURRENCY.md Rule 4: never hold db.conn() across a call taking
    // `&YantrikDB`. decrypt_text/decrypt_embedding don't currently take
    // db.conn(), but a future refactor could, and that silent deadlock
    // would be expensive to find.
    type RawRow = (
        String,
        String,
        String,
        Vec<u8>,
        f64,
        f64,
        f64,
        f64,
        f64,
        String,
        String,
    );
    let raw_rows: Vec<RawRow> = {
        let conn = db.conn();
        let sql = "SELECT rid, type, text, embedding, created_at, importance, valence, \
             half_life, last_access, metadata, namespace \
             FROM memories \
             WHERE consolidation_status = 'active' \
             AND storage_tier = 'hot' \
             AND type IN ('episodic', 'semantic') \
             ORDER BY namespace ASC, created_at ASC, rid ASC \
             LIMIT ?1 OFFSET ?2";
        let mut stmt = conn.prepare(&sql)?;
        let mapped = stmt.query_map(
            rusqlite::params![page_limit as i64, page_offset as i64],
            |row| {
                Ok((
                    row.get::<_, String>("rid")?,
                    row.get::<_, String>("type")?,
                    row.get::<_, String>("text")?,
                    row.get::<_, Vec<u8>>("embedding")?,
                    row.get::<_, f64>("created_at")?,
                    row.get::<_, f64>("importance")?,
                    row.get::<_, f64>("valence")?,
                    row.get::<_, f64>("half_life")?,
                    row.get::<_, f64>("last_access")?,
                    row.get::<_, String>("metadata")?,
                    row.get::<_, String>("namespace")?,
                ))
            },
        )?;
        let collected: std::result::Result<Vec<RawRow>, _> = mapped.collect();
        collected?
    }; // conn, stmt, mapped all dropped here before Phase 2
    let scanned_count = raw_rows.len();

    // Phase 2: decrypt. Safe to call `db.decrypt_*` now because no conn
    // guard is held.
    let memories: Vec<MemoryWithEmbedding> = raw_rows
        .into_iter()
        .map(
            |(
                rid,
                memory_type,
                stored_text,
                stored_emb,
                created_at,
                importance,
                valence,
                half_life,
                last_access,
                stored_meta,
                namespace,
            )| {
                let text = db.decrypt_text(&stored_text)?;
                let meta_str = db.decrypt_text(&stored_meta)?;
                let emb_blob = db.decrypt_embedding(&stored_emb)?;
                Ok(MemoryWithEmbedding {
                    rid,
                    memory_type,
                    text,
                    embedding: deserialize_f32(&emb_blob),
                    created_at,
                    importance,
                    valence,
                    half_life,
                    last_access,
                    metadata: serde_json::from_str(&meta_str)
                        .unwrap_or(serde_json::Value::Object(Default::default())),
                    namespace,
                })
            },
        )
        .collect::<Result<Vec<_>>>()?;

    // Derived summaries are OUTPUTS of consolidation, never inputs. A
    // summary is by construction similar to its own still-live sources, so
    // without this filter the next automatic (destructive) pass clusters
    // them together and retires summary AND sources — silently undoing the
    // additive `summarize_cluster` guarantee — and the additive pass
    // re-summarizes its own summaries into churn. Sources alone may still
    // cluster; that is ordinary consolidation.
    let memories: Vec<MemoryWithEmbedding> = memories
        .into_iter()
        .filter(|m| m.metadata.get("consolidated_from").is_none())
        .collect();

    // Load memory→entities map (single query) so the entity-overlap guard
    // can prune false-positive pairs before cosine clustering.
    let entities_by_rid: Option<HashMap<String, HashSet<String>>> = if require_entity_overlap {
        let rids: Vec<String> = memories.iter().map(|m| m.rid.clone()).collect();
        if rids.is_empty() {
            None
        } else {
            let conn = db.conn();
            let placeholders = vec!["?"; rids.len()].join(",");
            let sql = format!(
                "SELECT memory_rid, entity_name FROM memory_entities WHERE memory_rid IN ({})",
                placeholders
            );
            let mut stmt = conn.prepare(&sql)?;
            let params_vec: Vec<&dyn rusqlite::ToSql> =
                rids.iter().map(|s| s as &dyn rusqlite::ToSql).collect();
            let rows = stmt
                .query_map(params_vec.as_slice(), |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let mut map: HashMap<String, HashSet<String>> = HashMap::new();
            for (rid, name) in rows {
                map.entry(rid).or_default().insert(name);
            }
            Some(map)
        }
    } else {
        None
    };

    // Group memories by namespace to prevent cross-namespace consolidation
    let mut by_namespace: HashMap<String, Vec<MemoryWithEmbedding>> = HashMap::new();
    for mem in memories {
        by_namespace
            .entry(mem.namespace.clone())
            .or_default()
            .push(mem);
    }

    let mut result: Vec<Vec<MemoryWithEmbedding>> = Vec::new();
    for (_ns, ns_memories) in by_namespace {
        let cluster_indices = find_clusters(
            &ns_memories,
            entities_by_rid.as_ref(),
            sim_threshold,
            time_window_days,
            min_cluster_size,
            10,
        );
        for indices in cluster_indices {
            result.push(
                indices
                    .into_iter()
                    .map(|i| ns_memories[i].clone())
                    .collect(),
            );
        }
    }

    result.sort_by(|a, b| {
        let a = &a[0];
        let b = &b[0];
        a.namespace
            .cmp(&b.namespace)
            .then_with(|| a.created_at.total_cmp(&b.created_at))
            .then_with(|| a.rid.cmp(&b.rid))
    });

    Ok((result, scanned_count))
}

/// Run the full consolidation pipeline.
pub fn consolidate(
    db: &YantrikDB,
    sim_threshold: f64,
    time_window_days: f64,
    min_cluster_size: usize,
    limit: usize,
    require_entity_overlap: bool,
    dry_run: bool,
) -> Result<Vec<serde_json::Value>> {
    if limit == 0 {
        return Ok(Vec::new());
    }

    // `limit` remains the maximum number of consolidations this call may
    // commit. Candidate discovery gets a wider bounded window so the default
    // limit of five can actually find pairs, and a durable offset prevents
    // every think() call from examining the same first five rows forever.
    let scan_limit = limit
        .saturating_mul(CONSOLIDATION_SCAN_MULTIPLIER)
        .clamp(CONSOLIDATION_SCAN_FLOOR, CONSOLIDATION_SCAN_CEILING);
    let (offset, eligible_count): (usize, usize) = {
        let conn = db.conn();
        let stored_offset = conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                [CONSOLIDATION_SCAN_OFFSET_META],
                |row| row.get::<_, String>(0),
            )
            .optional()?
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let count = conn.query_row(
            "SELECT COUNT(*) FROM memories \
             WHERE consolidation_status = 'active' \
             AND storage_tier = 'hot' \
             AND type IN ('episodic', 'semantic')",
            [],
            |row| row.get::<_, i64>(0),
        )? as usize;
        (stored_offset.min(count), count)
    };

    let (mut clusters, _scanned_count) = find_consolidation_candidates_page(
        db,
        sim_threshold,
        time_window_days,
        min_cluster_size,
        scan_limit,
        offset,
        require_entity_overlap,
    )?;
    clusters.truncate(limit);

    if dry_run {
        return Ok(clusters
            .iter()
            .map(|cluster| {
                serde_json::json!({
                    "cluster_size": cluster.len(),
                    "texts": cluster.iter().map(|m| m.text.clone()).collect::<Vec<_>>(),
                    "preview_summary": extractive_summary(cluster),
                    "source_rids": cluster.iter().map(|m| m.rid.clone()).collect::<Vec<_>>(),
                })
            })
            .collect());
    }

    let mut results = Vec::new();
    for cluster in &clusters {
        // The default path keeps its historical behaviour: extractive join
        // carrying the cluster's MEAN embedding. `consolidate_cluster` is the
        // opt-in alternative for callers supplying their own synthesis.
        let summary_text = extractive_summary(cluster);
        let mean_emb = mean_embedding(cluster);
        results.push(commit_consolidation(
            db,
            cluster,
            &summary_text,
            Some(mean_emb),
            false,
            None,
        )?);
    }

    let next_offset = if offset.saturating_add(scan_limit) >= eligible_count {
        0
    } else {
        offset + scan_limit
    };
    db.conn().execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)",
        rusqlite::params![CONSOLIDATION_SCAN_OFFSET_META, next_offset.to_string()],
    )?;

    Ok(results)
}

/// Write one consolidated memory for `cluster` and retire its members.
///
/// THE single place consolidation bookkeeping lives, so the default
/// (extractive + mean-embedding) and caller-supplied paths cannot drift:
/// entity transfer, `consolidation_members` CRDT rows, source marking with
/// importance decay, scoring-cache invalidation, and the oplog entry.
///
/// `embedding`: `Some(v)` uses the caller's vector (the default path passes
/// the cluster mean, preserving historical behaviour). `None` embeds
/// `summary_text` with the engine's embedder — which is the point of the
/// caller-supplied path, since a vector derived from the actual text
/// retrieves for what the record says rather than for a cluster average.
fn commit_consolidation(
    db: &YantrikDB,
    cluster: &[MemoryWithEmbedding],
    summary_text: &str,
    embedding: Option<Vec<f32>>,
    // ADDITIVE mode keeps the sources fully live: the synthesis is stored
    // beside them rather than over them. See `summarize_cluster`.
    additive: bool,
    synthesis: Option<&SynthesisWrite<'_>>,
) -> Result<serde_json::Value> {
    let ts = crate::time::now_secs();
    {
        let source_rids: Vec<String> = cluster.iter().map(|m| m.rid.clone()).collect();

        // 3. Aggregate importance
        let max_importance = cluster.iter().map(|m| m.importance).fold(0.0f64, f64::max);
        let consolidated_importance = (max_importance * 1.1).min(1.0);

        // Mean valence
        let mean_valence: f64 =
            cluster.iter().map(|m| m.valence).sum::<f64>() / cluster.len() as f64;

        // Longer half-life for consolidated memories
        let max_half_life = cluster.iter().map(|m| m.half_life).fold(0.0f64, f64::max);
        let consolidated_half_life = max_half_life * 1.5;

        let cluster_namespace = cluster
            .first()
            .map(|m| m.namespace.as_str())
            .unwrap_or("default");

        let admission = synthesis
            .map(|write| build_synthesis_admission(db, cluster, write))
            .transpose()?;
        let effective_idempotency_key = admission.as_ref().map(|admission| {
            format!(
                "{}:evidence-v1:{}",
                admission.logical_key, admission.evidence_version
            )
        });

        // Availability time is the newest evidence member. The synthesis
        // must not appear in recall_as_of before all evidence it depends on
        // existed. First mention is a separate clock in metadata; collapsing
        // the two either leaks future evidence or misorders the item.
        let cluster_span_end = cluster
            .iter()
            .map(|m| m.created_at)
            .fold(f64::NEG_INFINITY, f64::max);
        let (first_mention_at, evidence_span_end_at, evidence_ids) =
            if let Some(admission) = &admission {
                let conn = db.conn();
                let mut first = f64::INFINITY;
                let mut last = f64::NEG_INFINITY;
                let mut ids = Vec::with_capacity(admission.dependencies.len());
                for dependency in &admission.dependencies {
                    let (created_at, synthesis_state): (f64, Option<String>) = conn.query_row(
                        "SELECT created_at, synthesis_state FROM memories WHERE rid = ?1",
                        rusqlite::params![&dependency.source_rid],
                        |row| Ok((row.get(0)?, row.get(1)?)),
                    )?;
                    first = first.min(created_at);
                    last = last.max(created_at);
                    if synthesis_state.is_none() {
                        ids.push(dependency.source_rid.clone());
                    }
                }
                (first, last, ids)
            } else {
                let first = cluster
                    .iter()
                    .map(|m| {
                        m.metadata
                            .get("first_mention_at")
                            .and_then(serde_json::Value::as_f64)
                            .filter(|v| v.is_finite())
                            .unwrap_or(m.created_at)
                    })
                    .fold(f64::INFINITY, f64::min);
                let last = cluster
                    .iter()
                    .map(|m| {
                        m.metadata
                            .get("evidence_span_end_at")
                            .and_then(serde_json::Value::as_f64)
                            .filter(|v| v.is_finite())
                            .unwrap_or(m.created_at)
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                let mut ids: Vec<String> = cluster.iter().map(|m| m.rid.clone()).collect();
                ids.sort();
                ids.dedup();
                (first, last, ids)
            };
        let span_end = if admission.is_some() {
            evidence_span_end_at
        } else {
            cluster_span_end
        };
        let summary_created_at = span_end.is_finite().then_some(span_end);

        // Caller metadata is additive. Reserved provenance and clock fields
        // are engine-owned so a synthesis cannot claim evidence it was not
        // actually linked to or move its first-mention/availability clocks.
        let mut meta = synthesis
            .map(|s| s.metadata.clone())
            .filter(serde_json::Value::is_object)
            .unwrap_or_else(|| serde_json::json!({}));
        let meta_obj = meta
            .as_object_mut()
            .expect("record_synthesis normalized metadata to an object");
        meta_obj.insert("consolidated_from".into(), serde_json::json!(source_rids));
        meta_obj.insert("evidence_ids".into(), serde_json::json!(evidence_ids));
        meta_obj.insert("cluster_size".into(), serde_json::json!(cluster.len()));
        meta_obj.insert(
            "first_mention_at".into(),
            serde_json::json!(first_mention_at),
        );
        meta_obj.insert(
            "evidence_span_end_at".into(),
            serde_json::json!(evidence_span_end_at),
        );
        if let Some(s) = synthesis {
            meta_obj.insert(
                "synthesis_kind".into(),
                serde_json::json!("multi_axis_item"),
            );
            meta_obj.insert("synthesis_axis".into(), serde_json::json!(s.axis));
            meta_obj.insert("granularity".into(), serde_json::json!(s.granularity));
            meta_obj.insert("synthesis_version".into(), serde_json::json!(1));
            meta_obj.insert("synthesis_available_at".into(), serde_json::json!(span_end));
            let admission = admission
                .as_ref()
                .expect("synthesis write always builds an admission descriptor");
            meta_obj.insert(
                "synthesis_logical_key".into(),
                serde_json::json!(admission.logical_key),
            );
            meta_obj.insert(
                "synthesis_evidence_version".into(),
                serde_json::json!(admission.evidence_version),
            );
        } else {
            // Legacy unkeyed consolidation keeps its maintenance timestamp.
            // Keyed synthesis omits wall time because metadata participates in
            // the idempotency digest and must be byte-stable across retries.
            meta_obj.insert("consolidation_time".into(), serde_json::json!(ts));
        }

        let idempotency_key = effective_idempotency_key.as_deref();
        let record_source = if synthesis.is_some() {
            "inference"
        } else {
            "user"
        };
        let preexisting_rid: Option<String> = if let Some(key) = idempotency_key {
            db.conn()
                .query_row(
                    "SELECT rid FROM memories \
                     WHERE origin_actor = ?1 AND namespace = ?2 \
                       AND idempotency_key = ?3 LIMIT 1",
                    rusqlite::params![db.actor_id(), cluster_namespace, key],
                    |row| row.get(0),
                )
                .optional()?
        } else {
            None
        };

        let consolidated_rid = match &embedding {
            Some(emb) => db.record_with_idempotency_sync_only(
                summary_text,
                "semantic",
                consolidated_importance,
                mean_valence,
                consolidated_half_life,
                &meta,
                emb,
                cluster_namespace,
                0.8,
                "general",
                record_source,
                None,
                idempotency_key,
                summary_created_at,
                admission.as_ref(),
            )?,
            // Engine-embedded: the vector comes from the synthesis itself.
            None => db.record_text_with_idempotency_sync_only(
                summary_text,
                "semantic",
                consolidated_importance,
                mean_valence,
                consolidated_half_life,
                &meta,
                cluster_namespace,
                0.8,
                "general",
                record_source,
                None,
                idempotency_key,
                summary_created_at,
                admission.as_ref(),
            )?,
        };

        // The record call above is still required on a retry: it verifies the
        // canonical payload digest and rejects a changed model output under
        // the same logical key. Once that succeeds, every remaining side
        // effect was already committed by the first attempt.
        if let Some(existing) = preexisting_rid {
            debug_assert_eq!(existing, consolidated_rid);
            return Ok(serde_json::json!({
                "consolidated_rid": consolidated_rid,
                "source_rids": source_rids,
                "evidence_ids": evidence_ids,
                "cluster_size": cluster.len(),
                "summary": summary_text,
                "importance": consolidated_importance,
                "embedded_from_text": embedding.is_none(),
                "first_mention_at": first_mention_at,
                "evidence_span_end_at": evidence_span_end_at,
                "synthesis_logical_key": admission.as_ref().map(|a| &a.logical_key),
                "synthesis_evidence_version": admission.as_ref().map(|a| &a.evidence_version),
                "idempotent_replay": true,
            }));
        }

        // 5. Transfer entity relationships
        let mut all_entities = std::collections::HashSet::new();
        for mem in cluster {
            let edges = db.get_edges(&mem.rid)?;
            for edge in &edges {
                all_entities.insert(edge.src.clone());
                all_entities.insert(edge.dst.clone());
                if edge.src == mem.rid {
                    db.relate(&consolidated_rid, &edge.dst, &edge.rel_type, edge.weight)?;
                } else if edge.dst == mem.rid {
                    db.relate(&edge.src, &consolidated_rid, &edge.rel_type, edge.weight)?;
                }
            }
        }

        // 6. Insert consolidation_members (set-union CRDT) and mark sources
        {
            let conn = db.conn();
            let hlc_ts = db.tick_hlc();
            let hlc_bytes = hlc_ts.to_bytes().to_vec();
            let actor_id = db.actor_id().to_string();

            for source_rid in &source_rids {
                conn.execute(
                    "INSERT OR IGNORE INTO consolidation_members \
                     (consolidation_rid, source_rid, hlc, actor_id) \
                     VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![consolidated_rid, source_rid, hlc_bytes, actor_id],
                )?;

                // ADDITIVE mode stops here: membership is recorded for
                // provenance, but the source keeps its status and its
                // importance. Marking it 'consolidated' would remove it from
                // every default recall (the filter admits only 'active'
                // unless include_consolidated), and the 0.3x importance
                // demotes it even when it is included. That pair is what made
                // consolidation destructive: measured on BEAM it bought
                // +18.8pp on abstention and +16.8pp on temporal_reasoning
                // while costing -21.6 on summarization and -17.5 on
                // preference_following, because the verbatim detail those
                // categories need had been hidden.
                if additive {
                    continue;
                }
                conn.execute(
                    "UPDATE memories \
                     SET consolidation_status = 'consolidated', \
                         consolidated_into = ?1, \
                         updated_at = ?2, \
                         importance = importance * 0.3 \
                     WHERE rid = ?3",
                    rusqlite::params![consolidated_rid, ts, source_rid],
                )?;
                // Update scoring cache: mark as consolidated, reduce importance
                db.cache_mark_consolidated(source_rid, 0.3);
            }
        } // conn lock released before log_op

        // 7. Log the operation
        let logged_emb = match &embedding {
            Some(v) => v.clone(),
            // Engine-embedded path: hash the stored vector so the oplog entry
            // still identifies what was written.
            None => db.embed(summary_text).unwrap_or_default(),
        };
        let emb_hash = blake3::hash(&serialize_f32(&logged_emb))
            .as_bytes()
            .to_vec();
        db.log_op(
            "consolidate",
            Some(&consolidated_rid),
            &serde_json::json!({
                "consolidated_rid": consolidated_rid,
                "source_rids": source_rids,
                "cluster_size": cluster.len(),
                "text": summary_text,
                "importance": consolidated_importance,
                "valence": mean_valence,
                "half_life": consolidated_half_life,
                "metadata": meta,
                "idempotency_key": idempotency_key,
                "synthesis": admission,
                "additive": additive,
                "record_source": record_source,
                "summary_preview": &summary_text[..summary_text.floor_char_boundary(200)],
            }),
            Some(&emb_hash),
        )?;

        Ok(serde_json::json!({
            "consolidated_rid": consolidated_rid,
            "source_rids": source_rids,
            "evidence_ids": evidence_ids,
            "cluster_size": cluster.len(),
            "summary": summary_text,
            "importance": consolidated_importance,
            "entities_linked": all_entities.len(),
            "embedded_from_text": embedding.is_none(),
            "first_mention_at": first_mention_at,
            "evidence_span_end_at": evidence_span_end_at,
            "idempotent_replay": false,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// find_clusters must cluster by POSITION in the time order, not by raw
    /// index value. Before 2026-08-15 the `j <= i` pair-filter compared
    /// index VALUES, so on a corpus where created_at != insertion order
    /// (here: reverse) every cluster collapsed to a singleton — the bug
    /// that made mine_topic_clusters return zero topics for real data.
    #[test]
    fn clusters_by_time_position_not_index_value() {
        // Five identical embeddings (all similar), created_at DESCENDING by
        // index — the reverse permutation that fired the bug.
        let emb = vec![1.0f32, 0.0, 0.0, 0.0];
        let mems: Vec<MemoryWithEmbedding> = (0..5)
            .map(|i| MemoryWithEmbedding {
                rid: format!("m{i}"),
                memory_type: "episodic".to_string(),
                text: format!("note {i}"),
                embedding: emb.clone(),
                created_at: 500.0 - i as f64 * 100.0,
                importance: 0.5,
                valence: 0.0,
                half_life: 604800.0,
                last_access: 0.0,
                metadata: serde_json::Value::Null,
                namespace: "default".to_string(),
            })
            .collect();
        let clusters = find_clusters(&mems, None, 0.9, f64::MAX, 2, 100);
        // All five are mutually similar and within the window → ONE cluster
        // of all five. The bug produced five singletons (returned as none,
        // since min_cluster_size=2).
        assert_eq!(clusters.len(), 1, "expected one cluster, got {clusters:?}");
        assert_eq!(clusters[0].len(), 5, "all five records must cluster");
    }

    use super::*;

    #[test]
    fn cosine_similarity_guards_nan_and_zero_norms() {
        // Issue #62 defect B: the pre-#60 `== 0.0` guard let NaN norms
        // through and returned NaN. Degenerate vectors must degrade to a
        // finite 0.0 similarity instead.
        let nan_vec = vec![f32::NAN, 1.0, 0.0];
        let finite = vec![1.0f32, 0.0, 0.0];
        let zero = vec![0.0f32, 0.0, 0.0];

        let s = cosine_similarity(&nan_vec, &finite);
        assert!(s.is_finite(), "NaN vector must not yield NaN similarity");
        assert_eq!(s, 0.0);
        assert_eq!(cosine_similarity(&finite, &nan_vec), 0.0);
        assert_eq!(cosine_similarity(&zero, &finite), 0.0);
        assert!((cosine_similarity(&finite, &finite) - 1.0).abs() < 1e-9);
    }

    fn make_mem(
        rid: &str,
        text: &str,
        embedding: Vec<f32>,
        created_at: f64,
        importance: f64,
    ) -> MemoryWithEmbedding {
        MemoryWithEmbedding {
            rid: rid.to_string(),
            memory_type: "episodic".to_string(),
            text: text.to_string(),
            embedding,
            created_at,
            importance,
            valence: 0.0,
            half_life: 604800.0,
            last_access: created_at,
            metadata: serde_json::json!({}),
            namespace: "default".to_string(),
        }
    }

    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim)
            .map(|i| (seed * (i as f32 + 1.0) * 1.7).sin() + (seed * (i as f32 + 2.0) * 0.3).cos())
            .collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec_seed(1.0, 8);
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_find_clusters_basic() {
        let now = 1000000.0;
        let mems = vec![
            make_mem("a", "t1", vec_seed(1.0, 8), now, 0.5),
            make_mem("b", "t2", vec_seed(1.05, 8), now + 100.0, 0.5),
            make_mem("c", "t3", vec_seed(10.0, 8), now + 200.0, 0.5),
        ];

        let clusters = find_clusters(&mems, None, 0.9, 7.0, 2, 10);
        assert_eq!(clusters.len(), 1);
        assert!(clusters[0].contains(&0)); // "a"
        assert!(clusters[0].contains(&1)); // "b"
    }

    #[test]
    fn test_find_clusters_entity_overlap_blocks_false_merge() {
        // Regression: cosine-similar memories referring to different entities
        // should NOT cluster when entity-overlap guard is active.
        let now = 1000000.0;
        let mems = vec![
            make_mem("a", "Alice is CEO", vec_seed(1.0, 8), now, 0.5),
            make_mem("b", "Sarah is CTO", vec_seed(1.02, 8), now + 100.0, 0.5),
        ];
        let mut entities: HashMap<String, HashSet<String>> = HashMap::new();
        entities.insert(
            "a".to_string(),
            ["Alice"].iter().map(|s| s.to_string()).collect(),
        );
        entities.insert(
            "b".to_string(),
            ["Sarah"].iter().map(|s| s.to_string()).collect(),
        );

        // Without guard: would cluster (high cosine).
        let unguarded = find_clusters(&mems, None, 0.9, 7.0, 2, 10);
        assert_eq!(unguarded.len(), 1, "cosine-only should merge");

        // With guard: should NOT cluster (disjoint entities).
        let guarded = find_clusters(&mems, Some(&entities), 0.9, 7.0, 2, 10);
        assert_eq!(guarded.len(), 0, "entity-overlap guard should block merge");
    }

    #[test]
    fn test_find_clusters_entity_overlap_allows_shared_entity() {
        let now = 1000000.0;
        let mems = vec![
            make_mem("a", "Alice is CEO", vec_seed(1.0, 8), now, 0.5),
            make_mem(
                "b",
                "Acme's CEO is Alice",
                vec_seed(1.02, 8),
                now + 100.0,
                0.5,
            ),
        ];
        let mut entities: HashMap<String, HashSet<String>> = HashMap::new();
        entities.insert(
            "a".to_string(),
            ["Alice"].iter().map(|s| s.to_string()).collect(),
        );
        entities.insert(
            "b".to_string(),
            ["Alice", "Acme"].iter().map(|s| s.to_string()).collect(),
        );

        let guarded = find_clusters(&mems, Some(&entities), 0.9, 7.0, 2, 10);
        assert_eq!(guarded.len(), 1, "shared entity should allow merge");
    }

    #[test]
    fn test_find_clusters_entity_overlap_falls_back_when_empty() {
        // Memories without extracted entities should still cluster by cosine
        // alone (no regression on memories predating entity extraction).
        let now = 1000000.0;
        let mems = vec![
            make_mem("a", "t1", vec_seed(1.0, 8), now, 0.5),
            make_mem("b", "t2", vec_seed(1.05, 8), now + 100.0, 0.5),
        ];
        let entities: HashMap<String, HashSet<String>> = HashMap::new();
        let clusters = find_clusters(&mems, Some(&entities), 0.9, 7.0, 2, 10);
        assert_eq!(
            clusters.len(),
            1,
            "empty entity map falls back to cosine-only"
        );
    }

    #[test]
    fn test_extractive_summary_single() {
        let mems = vec![make_mem("a", "The cat sat", vec_seed(1.0, 8), 0.0, 0.5)];
        assert_eq!(extractive_summary(&mems), "The cat sat");
    }

    #[test]
    fn test_extractive_summary_multi() {
        let mems = vec![
            make_mem("a", "Low importance", vec_seed(1.0, 8), 0.0, 0.1),
            make_mem("b", "High importance lead", vec_seed(2.0, 8), 0.0, 0.9),
        ];
        let summary = extractive_summary(&mems);
        assert!(summary.starts_with("High importance lead"));
    }

    #[test]
    fn test_mean_embedding() {
        let mems = vec![
            make_mem("a", "t1", vec![1.0, 2.0, 3.0], 0.0, 0.5),
            make_mem("b", "t2", vec![3.0, 4.0, 5.0], 0.0, 0.5),
        ];
        let mean = mean_embedding(&mems);
        assert_eq!(mean, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn bounded_consolidation_advances_past_an_unproductive_first_page() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let embedding = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let ten_days = 10.0 * 86400.0;

        // The first 100 records are too far apart to cluster. The only valid
        // pair is beyond that page, reproducing the starvation caused by the
        // old unordered `LIMIT 5` query.
        for i in 0..102 {
            let created_at = if i == 101 {
                100.0 * ten_days + 1.0
            } else {
                i as f64 * ten_days
            };
            db.record_with_idempotency(
                &format!("memory {i}"),
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &embedding,
                "default",
                0.8,
                "general",
                "user",
                None,
                None,
                Some(created_at),
            )
            .unwrap();
        }

        let first = consolidate(&db, 0.99, 7.0, 2, 1, false, false).unwrap();
        assert!(
            first.is_empty(),
            "the first page intentionally has no cluster"
        );
        let cursor: String = db
            .conn()
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                [CONSOLIDATION_SCAN_OFFSET_META],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(cursor, "100", "a real pass advances the scan cursor");

        let preview = consolidate(&db, 0.99, 7.0, 2, 1, false, true).unwrap();
        assert_eq!(preview.len(), 1, "the second page exposes the distant pair");
        let cursor_after_preview: String = db
            .conn()
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                [CONSOLIDATION_SCAN_OFFSET_META],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            cursor_after_preview, "100",
            "dry-run preview must not consume the page"
        );

        let second = consolidate(&db, 0.99, 7.0, 2, 1, false, false).unwrap();
        assert_eq!(second.len(), 1);
        let wrapped: String = db
            .conn()
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                [CONSOLIDATION_SCAN_OFFSET_META],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(wrapped, "0", "the cursor wraps after reaching the tail");
    }
}

#[cfg(test)]
mod additive_summary_tests {
    use crate::YantrikDB;

    fn seed(db: &YantrikDB, texts: &[&str]) -> Vec<String> {
        texts
            .iter()
            .enumerate()
            .map(|(i, t)| {
                let mut v = vec![0.0f32; 8];
                v[i % 8] = 1.0;
                db.record_with_idempotency(
                    t,
                    "episodic",
                    0.8,
                    0.0,
                    604800.0,
                    &serde_json::json!({}),
                    &v,
                    "default",
                    0.8,
                    "work",
                    "user",
                    None,
                    None,
                    None,
                )
                .unwrap()
            })
            .collect()
    }

    /// THE POINT OF THE WHOLE THING. `consolidate_cluster` retires its
    /// sources; `summarize_cluster` must not. If the sources stop being
    /// 'active' the default recall filter drops them, which is exactly what
    /// cost -21.6pp on summarization and -17.5pp on preference_following.
    #[test]
    fn additive_summary_leaves_every_source_live() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rids = seed(
            &db,
            &["set up the database schema", "configured the local server"],
        );
        super::summarize_cluster(&db, &rids, "Initial project setup", Some(&vec![0.5f32; 8]))
            .unwrap();
        for r in &rids {
            let m = db.get_memory(r).unwrap().unwrap();
            assert_eq!(m.consolidation_status, "active", "source {r} was retired");
            assert!((m.importance - 0.8).abs() < 1e-6, "source {r} was demoted");
        }
    }

    /// A derived summary must never become a clustering INPUT: it is
    /// similar to its own live sources by construction, so the automatic
    /// (destructive) pass would cluster them together and retire summary
    /// AND sources — undoing the additive guarantee silently. Sources
    /// alone staying clusterable is ordinary consolidation and fine.
    #[test]
    fn derived_summaries_are_never_clustering_inputs() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        // Identical embeddings so the pair is a guaranteed cluster.
        let rids = seed(
            &db,
            &["set up the database schema", "configured the local server"],
        );
        super::summarize_cluster(&db, &rids, "Initial project setup", Some(&vec![0.5f32; 8]))
            .unwrap();
        let clusters =
            super::find_consolidation_candidates(&db, 0.0, f64::MAX, 2, 100, false).unwrap();
        for cluster in &clusters {
            for m in cluster {
                assert!(
                    m.metadata.get("consolidated_from").is_none(),
                    "summary {} offered as a clustering input",
                    m.rid
                );
            }
        }
    }

    /// The contrast case: the replacing form still retires sources, so the
    /// two behaviours cannot silently converge.
    #[test]
    fn replacing_consolidation_still_retires_sources() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rids = seed(
            &db,
            &["set up the database schema", "configured the local server"],
        );
        super::consolidate_cluster(&db, &rids, "Initial project setup", Some(&vec![0.5f32; 8]))
            .unwrap();
        let m = db.get_memory(&rids[0]).unwrap().unwrap();
        assert_eq!(m.consolidation_status, "consolidated");
        assert!(m.importance < 0.8);
    }

    /// The synthesis must be a first-class, retrievable memory — an
    /// abstraction nobody can retrieve is the problem this exists to fix.
    #[test]
    fn the_abstraction_itself_is_stored_and_findable() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rids = seed(
            &db,
            &["set up the database schema", "configured the local server"],
        );
        let out =
            super::summarize_cluster(&db, &rids, "Initial project setup", Some(&vec![0.5f32; 8]))
                .unwrap();
        let new_rid = out["consolidated_rid"]
            .as_str()
            .or_else(|| out["rid"].as_str())
            .expect("summary rid in result");
        let m = db.get_memory(new_rid).unwrap().unwrap();
        assert_eq!(m.text, "Initial project setup");
        assert_eq!(m.consolidation_status, "active");
    }
}
