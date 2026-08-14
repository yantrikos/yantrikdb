use std::collections::{HashMap, HashSet};

use rusqlite::OptionalExtension;

use crate::engine::YantrikDB;
use crate::error::Result;
use crate::serde_helpers::{deserialize_f32, serialize_f32};
use crate::types::*;

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

    for &i in &indices {
        if used.contains(&i) {
            continue;
        }

        let mut cluster = vec![i];
        used.insert(i);

        for &j in &indices {
            if j <= i || used.contains(&j) {
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
    commit_consolidation(db, &members, text, embedding.map(|v| v.to_vec()), false)
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
    commit_consolidation(db, &members, text, embedding.map(|v| v.to_vec()), true)
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
        let sql = format!(
            "SELECT rid, type, text, embedding, created_at, importance, valence, \
             half_life, last_access, metadata, namespace \
             FROM memories \
             WHERE consolidation_status = 'active' \
             AND storage_tier = 'hot' \
             AND type IN ('episodic', 'semantic') \
             LIMIT {}",
            limit
        );
        let mut stmt = conn.prepare(&sql)?;
        let mapped = stmt.query_map([], |row| {
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
        })?;
        let collected: std::result::Result<Vec<RawRow>, _> = mapped.collect();
        collected?
    }; // conn, stmt, mapped all dropped here before Phase 2

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

    Ok(result)
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
    let clusters = find_consolidation_candidates(
        db,
        sim_threshold,
        time_window_days,
        min_cluster_size,
        limit,
        require_entity_overlap,
    )?;

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
        )?);
    }

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

        // 4. Record the new consolidated memory
        let meta = serde_json::json!({
            "consolidated_from": source_rids,
            "cluster_size": cluster.len(),
            "consolidation_time": ts,
        });

        let cluster_namespace = cluster
            .first()
            .map(|m| m.namespace.as_str())
            .unwrap_or("default");
        let consolidated_rid = match &embedding {
            Some(emb) => db.record(
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
                "user",
                None,
            )?,
            // Engine-embedded: the vector comes from the synthesis itself.
            None => db.record_text(
                summary_text,
                "semantic",
                consolidated_importance,
                mean_valence,
                consolidated_half_life,
                &meta,
                cluster_namespace,
                0.8,
                "general",
                "user",
                None,
            )?,
        };

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
                "summary_preview": &summary_text[..summary_text.floor_char_boundary(200)],
            }),
            Some(&emb_hash),
        )?;

        Ok(serde_json::json!({
            "consolidated_rid": consolidated_rid,
            "source_rids": source_rids,
            "cluster_size": cluster.len(),
            "summary": summary_text,
            "importance": consolidated_importance,
            "entities_linked": all_entities.len(),
            "embedded_from_text": embedding.is_none(),
        }))
    }
}

#[cfg(test)]
mod tests {
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
