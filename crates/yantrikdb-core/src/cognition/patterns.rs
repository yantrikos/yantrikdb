//! Pattern mining across memories.
//!
//! Rule-based pattern detection using existing data (embeddings, timestamps,
//! entities, valence). No ML dependencies. Five mining algorithms:
//! co-occurrence, temporal clusters, valence trends, topic clusters, entity hubs.

use rusqlite::params;

use crate::consolidate::{compression_gist_cannot_seed, find_clusters};
use crate::engine::YantrikDB;
use crate::error::Result;
use crate::types::{MemoryWithEmbedding, Pattern, PatternConfig, PatternMiningResult};

fn now() -> f64 {
    crate::time::now_secs()
}

/// Internal raw pattern before persistence.
struct RawPattern {
    pattern_type: String,
    confidence: f64,
    description: String,
    evidence_rids: Vec<String>,
    entity_names: Vec<String>,
    context: serde_json::Value,
    dedup_key: String,
}

// ── Mining algorithms ──

/// Find entity pairs that co-occur in memories via shared edge targets.
fn mine_co_occurrence(db: &YantrikDB, config: &PatternConfig) -> Result<Vec<RawPattern>> {
    let conn = db.conn();
    // Find entity pairs connected to the same memory rids
    let mut stmt = conn.prepare(
        "SELECT e1.src, e2.src, COUNT(DISTINCT e1.dst) as shared \
         FROM edges e1 \
         JOIN edges e2 ON e1.dst = e2.dst AND e1.src < e2.src \
         WHERE e1.tombstoned = 0 AND e2.tombstoned = 0 \
         GROUP BY e1.src, e2.src \
         HAVING shared >= ?1 \
         ORDER BY shared DESC \
         LIMIT 20",
    )?;

    let mut patterns = Vec::new();
    let rows = stmt.query_map(params![config.co_occurrence_min_count as i64], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
        ))
    })?;

    for row in rows {
        let (entity_a, entity_b, shared_count) = row?;
        let confidence = (shared_count as f64 / 10.0).min(1.0);
        let mut key_parts = vec![entity_a.clone(), entity_b.clone()];
        key_parts.sort();

        patterns.push(RawPattern {
            pattern_type: "co_occurrence".to_string(),
            confidence,
            description: format!(
                "'{}' and '{}' frequently appear together ({shared_count} shared connections)",
                entity_label(db, &conn, &entity_a),
                entity_label(db, &conn, &entity_b),
            ),
            evidence_rids: vec![],
            entity_names: vec![entity_a, entity_b],
            context: serde_json::json!({"shared_count": shared_count}),
            dedup_key: format!("co_occurrence:{}", key_parts.join(",")),
        });
    }

    Ok(patterns)
}

/// A one-line label for a pattern description.
///
/// `entity_names` stores names, but `relate()` puts record ids there too. When
/// the name resolves to a memory, the description shows a short text snippet
/// (what a human reads); otherwise the name stands as written. The rid stays in
/// `entity_names` and `context` either way — this changes the prose only.
fn entity_label(db: &YantrikDB, conn: &rusqlite::Connection, name: &str) -> String {
    if !crate::knowledge::graph::is_uuid_shaped(name) {
        return name.to_string();
    }
    // Only ACTIVE rows. `forget()` does not scrub the text — it sets
    // `consolidation_status = 'tombstoned'` and leaves the content in place —
    // so an unfiltered read would print a forgotten record's text into a
    // pattern description, which defeats the forget.
    let stored: Option<String> = conn
        .query_row(
            "SELECT text FROM memories WHERE rid = ?1 AND consolidation_status = 'active'",
            rusqlite::params![name],
            |row| row.get(0),
        )
        .ok();
    let Some(stored) = stored else {
        return name.to_string();
    };
    // `memories.text` is ciphertext at rest when a key is set. Never print the
    // stored bytes; a failed decryption is not worth aborting a mining pass.
    let Ok(text) = db.decrypt_text(&stored) else {
        return name.to_string();
    };
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let snippet: String = flat.chars().take(80).collect();
    if flat.chars().count() > 80 {
        format!("{snippet}…")
    } else {
        snippet
    }
}

/// Detect memories clustering around specific times of week.
fn mine_temporal_clusters(db: &YantrikDB, config: &PatternConfig) -> Result<Vec<RawPattern>> {
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "SELECT rid, created_at, valence FROM memories \
         WHERE type = 'episodic' AND consolidation_status = 'active'",
    )?;

    let rows: Vec<(String, f64, f64)> = stmt
        .query_map([], |row| {
            Ok((
                row.get::<_, String>("rid")?,
                row.get::<_, f64>("created_at")?,
                row.get::<_, f64>("valence")?,
            ))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    if rows.len() < config.temporal_cluster_min_events * 3 {
        return Ok(vec![]);
    }

    // Build hour-of-week histogram (168 buckets)
    let mut buckets: Vec<Vec<(String, f64)>> = (0..168).map(|_| Vec::new()).collect();
    for (rid, created_at, valence) in &rows {
        let secs = *created_at as u64;
        // Convert to hour-of-week (0 = Monday 00:00, 167 = Sunday 23:00)
        let hour_of_week = ((secs / 3600) % 168) as usize;
        buckets[hour_of_week].push((rid.clone(), *valence));
    }

    let counts: Vec<f64> = buckets.iter().map(|b| b.len() as f64).collect();
    let mean = counts.iter().sum::<f64>() / 168.0;
    if mean == 0.0 {
        return Ok(vec![]);
    }
    let variance = counts.iter().map(|c| (c - mean).powi(2)).sum::<f64>() / 168.0;
    let stddev = variance.sqrt();
    let threshold = mean + 2.0 * stddev;

    let day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ];
    let mut patterns = Vec::new();

    for (hour, bucket) in buckets.iter().enumerate() {
        if (bucket.len() as f64) > threshold && bucket.len() >= config.temporal_cluster_min_events {
            let day = day_names[hour / 24];
            let hour_of_day = hour % 24;
            let label = format!("{day} {hour_of_day}:00");

            let mean_valence: f64 =
                bucket.iter().map(|(_, v)| v).sum::<f64>() / bucket.len() as f64;
            let rids: Vec<String> = bucket.iter().map(|(r, _)| r.clone()).collect();

            let mut description =
                format!("Memories cluster around {label} ({} events)", bucket.len());
            // Was a hardcoded 0.3 mirroring (and ignoring) the config knob
            // of the same value — a sweep of it measured nothing.
            if mean_valence.abs() > config.valence_trend_delta_threshold {
                let tone = if mean_valence > 0.0 {
                    "positive"
                } else {
                    "negative"
                };
                description.push_str(&format!(" with {tone} emotional tone"));
            }

            patterns.push(RawPattern {
                pattern_type: "temporal_cluster".to_string(),
                confidence: ((bucket.len() as f64 - mean) / (bucket.len() as f64)).min(1.0),
                description,
                evidence_rids: rids,
                entity_names: vec![],
                context: serde_json::json!({
                    "hour_of_week": hour,
                    "label": label,
                    "event_count": bucket.len(),
                    "mean_valence": mean_valence,
                }),
                dedup_key: format!("temporal_cluster:{hour}"),
            });
        }
    }

    Ok(patterns)
}

/// Detect shifts in emotional valence over time.
fn mine_valence_trends(db: &YantrikDB, _config: &PatternConfig) -> Result<Vec<RawPattern>> {
    let ts = now();
    let conn = db.conn();

    // Weekly buckets over the last 5 weeks
    let mut weeks: Vec<Vec<f64>> = vec![Vec::new(); 5];
    let mut stmt = conn.prepare(
        "SELECT valence, created_at FROM memories \
         WHERE consolidation_status = 'active' \
         AND created_at > ?1 \
         ORDER BY created_at",
    )?;

    let cutoff = ts - 86400.0 * 35.0; // 5 weeks
    let rows = stmt.query_map(params![cutoff], |row| {
        Ok((row.get::<_, f64>(0)?, row.get::<_, f64>(1)?))
    })?;

    for row in rows {
        let (valence, created_at) = row?;
        let age_days = (ts - created_at) / 86400.0;
        let week_idx = (age_days / 7.0) as usize;
        if week_idx < 5 {
            weeks[week_idx].push(valence);
        }
    }

    let mut patterns = Vec::new();

    // Need at least 3 data points in recent and baseline
    let recent: Vec<f64> = weeks[0].iter().chain(weeks[1].iter()).copied().collect();
    let baseline: Vec<f64> = weeks[2]
        .iter()
        .chain(weeks[3].iter())
        .chain(weeks[4].iter())
        .copied()
        .collect();

    if recent.len() >= 3 && baseline.len() >= 3 {
        let recent_avg = recent.iter().sum::<f64>() / recent.len() as f64;
        let baseline_avg = baseline.iter().sum::<f64>() / baseline.len() as f64;
        let delta = recent_avg - baseline_avg;

        if delta.abs() > 0.3 {
            let direction = if delta > 0.0 { "positive" } else { "negative" };

            patterns.push(RawPattern {
                pattern_type: "valence_trend".to_string(),
                confidence: delta.abs().min(1.0),
                description: format!(
                    "Emotional tone has shifted {direction} by {:.2} over recent weeks",
                    delta.abs()
                ),
                evidence_rids: vec![],
                entity_names: vec![],
                context: serde_json::json!({
                    "recent_avg": recent_avg,
                    "baseline_avg": baseline_avg,
                    "delta": delta,
                    "direction": direction,
                }),
                dedup_key: format!("valence_trend:{direction}"),
            });
        }
    }

    Ok(patterns)
}

/// Detect recurring topic groups using existing clustering infrastructure.
fn mine_topic_clusters(db: &YantrikDB, config: &PatternConfig) -> Result<Vec<RawPattern>> {
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "SELECT rid, type, text, embedding, created_at, importance, valence, \
         half_life, last_access, metadata, namespace \
         FROM memories \
         WHERE consolidation_status = 'active' \
         AND embedding IS NOT NULL \
         ORDER BY created_at DESC \
         LIMIT 200",
    )?;

    let raw_rows: Vec<(
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
    )> = stmt
        .query_map([], |row| {
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
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    // Decrypt text, embedding, and metadata if encrypted
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
                // A record whose ciphertext cannot be read must not abort
                // the whole mining pass — the same policy `entity_label`
                // uses. Skip this row and keep mining the rest.
                let (Ok(text), Ok(meta_str), Ok(emb_blob)) = (
                    db.decrypt_text(&stored_text),
                    db.decrypt_text(&stored_meta),
                    db.decrypt_embedding(&stored_emb),
                ) else {
                    return Ok(None);
                };
                Ok(Some(MemoryWithEmbedding {
                    rid,
                    memory_type,
                    text,
                    embedding: crate::compression::decode_embedding_blob(&emb_blob),
                    created_at,
                    importance,
                    valence,
                    half_life,
                    last_access,
                    metadata: serde_json::from_str(&meta_str)
                        .unwrap_or(serde_json::Value::Object(Default::default())),
                    namespace,
                }))
            },
        )
        // `Ok(None)` marks a row whose ciphertext we could not read; drop it
        // and mine the rest instead of failing the pass.
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();

    if memories.len() < 5 {
        return Ok(vec![]);
    }

    let cluster_indices = find_clusters(
        &memories,
        None, // pattern mining is topic-level, not entity-gated
        config.topic_cluster_sim_threshold,
        config.topic_cluster_time_window_days,
        5,  // min cluster size for a "recurring topic"
        20, // max cluster size
        // Same bug class as consolidation (issue #253): a compression gist
        // seeding a topic cluster would mint a "recurring topic" whose text is
        // a summary of everything nearby.
        Some(&compression_gist_cannot_seed),
    );

    let mut patterns = Vec::new();
    let max_size = cluster_indices.iter().map(|c| c.len()).max().unwrap_or(1);

    for indices in cluster_indices {
        if indices.len() < 5 {
            continue;
        }

        // Check that cluster spans more than 7 days
        let min_ts = indices
            .iter()
            .map(|&i| memories[i].created_at)
            .fold(f64::INFINITY, f64::min);
        let max_ts = indices
            .iter()
            .map(|&i| memories[i].created_at)
            .fold(f64::NEG_INFINITY, f64::max);
        let span_days = (max_ts - min_ts) / 86400.0;
        if span_days < 7.0 {
            continue;
        }

        // Simple extractive summary: pick the highest-importance text
        let mut best_idx = indices[0];
        for &idx in &indices[1..] {
            if memories[idx].importance > memories[best_idx].importance {
                best_idx = idx;
            }
        }
        let summary = memories[best_idx].text.clone();
        let rids: Vec<String> = indices.iter().map(|&i| memories[i].rid.clone()).collect();

        let confidence = indices.len() as f64 / max_size as f64;

        let mut sorted_rids = rids.clone();
        sorted_rids.sort();

        patterns.push(RawPattern {
            pattern_type: "topic_cluster".to_string(),
            confidence,
            description: format!(
                "Recurring topic ({} memories over {span_days:.0} days): {summary}",
                indices.len()
            ),
            evidence_rids: rids,
            entity_names: vec![],
            context: serde_json::json!({
                "cluster_size": indices.len(),
                "span_days": span_days,
                "summary": summary,
            }),
            dedup_key: format!("topic_cluster:{}", sorted_rids.join(",")),
        });
    }

    Ok(patterns)
}

/// Detect high-degree entity hubs in the knowledge graph.
fn mine_entity_hubs(db: &YantrikDB, config: &PatternConfig) -> Result<Vec<RawPattern>> {
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "SELECT src, COUNT(*) as degree \
         FROM edges WHERE tombstoned = 0 \
         GROUP BY src HAVING degree >= ?1 \
         ORDER BY degree DESC \
         LIMIT 10",
    )?;

    let rows: Vec<(String, i64)> = stmt
        .query_map(params![config.entity_hub_min_degree as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let max_degree = rows.first().map(|(_, d)| *d).unwrap_or(1);
    let mut patterns = Vec::new();

    for (entity, degree) in rows {
        // A record id in the entity table is an artefact of `relate()`, not a
        // hub. Skipping it here keeps the pattern honest without touching the
        // entity store (that heal belongs to the admission rule).
        if crate::knowledge::graph::is_uuid_shaped(&entity) {
            continue;
        }
        let confidence = degree as f64 / max_degree as f64;

        patterns.push(RawPattern {
            pattern_type: "entity_hub".to_string(),
            confidence,
            description: format!("'{entity}' is a central entity with {degree} connections"),
            evidence_rids: vec![],
            entity_names: vec![entity.clone()],
            context: serde_json::json!({"entity": entity, "degree": degree}),
            dedup_key: format!("entity_hub:{entity}"),
        });
    }

    Ok(patterns)
}

// ── Cross-domain mining (V13) ──

/// Find similar memories across different domains using the HNSW index.
fn mine_cross_domain_patterns(db: &YantrikDB, config: &PatternConfig) -> Result<Vec<RawPattern>> {
    let conn = db.conn();

    // Get distinct active domains
    let mut stmt = conn.prepare(
        "SELECT DISTINCT domain FROM memories \
         WHERE consolidation_status = 'active' AND domain != 'general'",
    )?;
    let domains: Vec<String> = stmt
        .query_map([], |row| row.get(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    if domains.len() < 2 {
        return Ok(vec![]);
    }

    // Compute co-occurrence rates between domain pairs for surprise scoring
    let total_memories: f64 = conn.query_row(
        "SELECT COUNT(*) FROM memories WHERE consolidation_status = 'active'",
        [],
        |row| row.get::<_, i64>(0),
    )? as f64;

    let mut domain_counts: std::collections::HashMap<String, f64> =
        std::collections::HashMap::new();
    for d in &domains {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories WHERE consolidation_status = 'active' AND domain = ?1",
            params![d],
            |row| row.get(0),
        )?;
        domain_counts.insert(d.clone(), count as f64);
    }

    // Per domain, select candidates: 50% by importance, 30% by recency, 20% random
    let m = config.cross_domain_candidates_per_domain;
    let mut candidates: Vec<(String, String, Vec<f32>)> = Vec::new(); // (rid, domain, embedding)

    for domain in &domains {
        let imp_limit = (m * 50 / 100).max(1);
        let rec_limit = (m * 30 / 100).max(1);
        let rand_limit = m.saturating_sub(imp_limit + rec_limit).max(1);

        // By importance
        let mut stmt = conn.prepare(
            "SELECT rid, embedding FROM memories \
             WHERE consolidation_status = 'active' AND domain = ?1 AND embedding IS NOT NULL \
             ORDER BY importance DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![domain, imp_limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        for row in rows {
            let (rid, emb_blob) = row?;
            let emb_blob = db.decrypt_embedding(&emb_blob)?;
            candidates.push((
                rid,
                domain.clone(),
                crate::compression::decode_embedding_blob(&emb_blob),
            ));
        }

        // By recency
        let mut stmt = conn.prepare(
            "SELECT rid, embedding FROM memories \
             WHERE consolidation_status = 'active' AND domain = ?1 AND embedding IS NOT NULL \
             ORDER BY created_at DESC LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![domain, rec_limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        for row in rows {
            let (rid, emb_blob) = row?;
            let emb_blob = db.decrypt_embedding(&emb_blob)?;
            candidates.push((
                rid,
                domain.clone(),
                crate::compression::decode_embedding_blob(&emb_blob),
            ));
        }

        // Random sample
        let mut stmt = conn.prepare(
            "SELECT rid, embedding FROM memories \
             WHERE consolidation_status = 'active' AND domain = ?1 AND embedding IS NOT NULL \
             ORDER BY RANDOM() LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![domain, rand_limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Vec<u8>>(1)?))
        })?;
        for row in rows {
            let (rid, emb_blob) = row?;
            let emb_blob = db.decrypt_embedding(&emb_blob)?;
            candidates.push((
                rid,
                domain.clone(),
                crate::compression::decode_embedding_blob(&emb_blob),
            ));
        }
    }

    // Deduplicate candidates by rid
    let mut seen = std::collections::HashSet::new();
    candidates.retain(|(rid, _, _)| seen.insert(rid.clone()));

    // For each candidate, query HNSW for K=10 global neighbors (DeltaIndex merges cold + delta)
    let mut patterns = Vec::new();
    let mut pair_counts: std::collections::HashMap<(String, String), usize> =
        std::collections::HashMap::new();

    // Build rid→domain lookup
    let mut rid_domain: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    for (rid, domain, _) in &candidates {
        rid_domain.insert(rid.clone(), domain.clone());
    }

    // Also need domain for neighbors not in candidates
    // We'll query from DB as needed

    // **Issue #41 brainstorm-4 §1.** Snapshot SearchState once for the
    // pattern-detection scan so every neighbor query within this scan
    // hits the same generation-anchored index.
    let state = db.search_state.load_full();
    for (rid_a, domain_a, embedding) in &candidates {
        let neighbors = match state.vec_index.search(embedding, 10) {
            Ok(n) => n,
            Err(_) => continue,
        };

        for (rid_b, similarity) in neighbors {
            if &rid_b == rid_a || similarity < config.cross_domain_sim_threshold {
                continue;
            }

            // Get domain of neighbor
            let domain_b = if let Some(d) = rid_domain.get(&rid_b) {
                d.clone()
            } else {
                match conn.query_row(
                    "SELECT domain FROM memories WHERE rid = ?1",
                    params![rid_b],
                    |row| row.get::<_, String>(0),
                ) {
                    Ok(d) => d,
                    Err(_) => continue,
                }
            };

            // Must be different domain
            if &domain_b == domain_a || domain_b == "general" {
                continue;
            }

            // Domain pair key (sorted for symmetry)
            let pair = if domain_a < &domain_b {
                (domain_a.clone(), domain_b.clone())
            } else {
                (domain_b.clone(), domain_a.clone())
            };

            // Cap per domain pair
            let count = pair_counts.entry(pair.clone()).or_insert(0);
            if *count >= config.cross_domain_max_per_pair {
                continue;
            }

            // Compute domain surprise
            let count_a = domain_counts.get(domain_a).copied().unwrap_or(1.0);
            let count_b = domain_counts.get(&domain_b).copied().unwrap_or(1.0);
            let co_occurrence_rate = (count_a * count_b) / (total_memories * total_memories);
            let domain_surprise = 1.0 - co_occurrence_rate.min(1.0);

            // Check shared entities for entity_support
            let shared_entities: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM memory_entities me1 \
                 JOIN memory_entities me2 ON me1.entity_name = me2.entity_name \
                 WHERE me1.memory_rid = ?1 AND me2.memory_rid = ?2",
                    params![rid_a, rid_b],
                    |row| row.get(0),
                )
                .unwrap_or(0);
            let entity_support = 1.0 + 0.5 * shared_entities as f64;

            let score = similarity as f64 * domain_surprise * entity_support;

            // Dedup key (sorted rids)
            let dedup = if rid_a < &rid_b {
                format!("cross_domain:{rid_a}:{rid_b}")
            } else {
                format!("cross_domain:{rid_b}:{rid_a}")
            };

            *count += 1;

            patterns.push(RawPattern {
                pattern_type: "cross_domain".to_string(),
                confidence: score.min(1.0),
                description: format!(
                    "Cross-domain connection between {domain_a} and {domain_b} (sim={similarity:.2}, surprise={domain_surprise:.2})"
                ),
                evidence_rids: vec![rid_a.clone(), rid_b.clone()],
                entity_names: vec![],
                context: serde_json::json!({
                    "domain_a": domain_a,
                    "domain_b": domain_b,
                    "similarity": similarity,
                    "domain_surprise": domain_surprise,
                    "entity_support": entity_support,
                    "score": score,
                }),
                dedup_key: dedup,
            });
        }
    }

    // Sort by score descending, keep top results
    patterns.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    patterns.truncate(config.max_patterns);

    Ok(patterns)
}

/// Detect entities that bridge multiple domains.
fn mine_entity_bridges(db: &YantrikDB, config: &PatternConfig) -> Result<Vec<RawPattern>> {
    let conn = db.conn();

    // Get entity-domain counts
    let mut stmt = conn.prepare(
        "SELECT me.entity_name, m.domain, COUNT(*) as cnt \
         FROM memory_entities me \
         JOIN memories m ON m.rid = me.memory_rid \
         WHERE m.consolidation_status = 'active' AND m.domain != 'general' \
         GROUP BY me.entity_name, m.domain \
         HAVING cnt >= ?1",
    )?;

    let min_mentions = config.entity_bridge_min_mentions_per_domain as i64;
    let rows: Vec<(String, String, i64)> = stmt
        .query_map(params![min_mentions], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    // Group by entity
    let mut entity_domains: std::collections::HashMap<String, Vec<(String, i64)>> =
        std::collections::HashMap::new();
    for (entity, domain, count) in rows {
        entity_domains
            .entry(entity)
            .or_default()
            .push((domain, count));
    }

    // Filter: must span min_domains
    let min_domains = config.entity_bridge_min_domains;
    let mut patterns = Vec::new();

    // Total entity count for IDF
    let total_entities: f64 = conn.query_row("SELECT COUNT(*) FROM entities", [], |row| {
        row.get::<_, i64>(0)
    })? as f64;

    for (entity, domain_counts) in &entity_domains {
        if crate::knowledge::graph::is_uuid_shaped(entity) {
            continue;
        }
        if domain_counts.len() < min_domains {
            continue;
        }

        let domain_count = domain_counts.len() as f64;
        let total_mentions: i64 = domain_counts.iter().map(|(_, c)| c).sum();

        // Entropy of domain distribution
        let total = total_mentions as f64;
        let entropy: f64 = domain_counts
            .iter()
            .map(|(_, c)| {
                let p = *c as f64 / total;
                if p > 0.0 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();

        // IDF: penalize very common entities
        let mention_count: i64 = conn
            .query_row(
                "SELECT mention_count FROM entities WHERE name = ?1",
                params![entity],
                |row| row.get(0),
            )
            .unwrap_or(1);
        let idf = (total_entities / (1.0 + mention_count as f64))
            .ln()
            .max(0.1);

        let bridge_score = domain_count.ln() * entropy * idf;

        let domains_detail: Vec<serde_json::Value> = domain_counts
            .iter()
            .map(|(d, c)| serde_json::json!({"domain": d, "count": c}))
            .collect();

        patterns.push(RawPattern {
            pattern_type: "entity_bridge".to_string(),
            confidence: (bridge_score / 5.0).min(1.0), // normalize
            description: format!(
                "'{entity}' bridges {} domains ({} total mentions)",
                domain_counts.len(),
                total_mentions
            ),
            evidence_rids: vec![],
            entity_names: vec![entity.clone()],
            context: serde_json::json!({
                "domains": domains_detail,
                "bridge_score": bridge_score,
                "entropy": entropy,
                "idf": idf,
            }),
            dedup_key: format!("entity_bridge:{entity}"),
        });
    }

    // Sort by score descending, keep top 10
    patterns.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
    patterns.truncate(10);

    Ok(patterns)
}

// ── Pattern persistence ──

/// Upsert a pattern: insert new or update existing (by dedup_key via pattern_type + entities/rids).
fn upsert_pattern(db: &YantrikDB, raw: &RawPattern, ts: f64) -> Result<bool> {
    let conn = db.conn();

    // Check for existing pattern with same dedup key (stored in context)
    let existing: Option<(String, i64)> = conn
        .query_row(
            "SELECT pattern_id, occurrence_count FROM patterns \
             WHERE pattern_type = ?1 \
             AND json_extract(context, '$.dedup_key') = ?2 \
             AND status = 'active'",
            params![raw.pattern_type, raw.dedup_key],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .ok();

    if let Some((pattern_id, count)) = existing {
        // Update existing
        conn.execute(
            // v0.25: the description is refreshed too. Without this, a row
            // mined before a rendering fix (the weekday label in #326, the
            // snippet here) keeps the old text forever, because dedup matches
            // the existing row and only the counters moved.
            "UPDATE patterns SET confidence = MAX(confidence, ?1), \
             last_confirmed = ?2, occurrence_count = ?3, description = ?5 \
             WHERE pattern_id = ?4",
            params![raw.confidence, ts, count + 1, pattern_id, raw.description],
        )?;
        Ok(false) // not new
    } else {
        let pattern_id = crate::id::new_id();
        let hlc = db.tick_hlc();
        let hlc_bytes = hlc.to_bytes().to_vec();
        let actor_id = db.actor_id().to_string();
        let evidence_json = serde_json::to_string(&raw.evidence_rids)?;
        let entity_json = serde_json::to_string(&raw.entity_names)?;
        let mut context = raw.context.clone();
        if let serde_json::Value::Object(ref mut map) = context {
            map.insert("dedup_key".to_string(), serde_json::json!(raw.dedup_key));
        }
        let context_json = serde_json::to_string(&context)?;

        conn.execute(
            "INSERT INTO patterns \
             (pattern_id, pattern_type, status, confidence, description, \
              evidence_rids, entity_names, context, first_seen, last_confirmed, \
              occurrence_count, hlc, origin_actor) \
             VALUES (?1, ?2, 'active', ?3, ?4, ?5, ?6, ?7, ?8, ?8, 1, ?9, ?10)",
            params![
                pattern_id,
                raw.pattern_type,
                raw.confidence,
                raw.description,
                evidence_json,
                entity_json,
                context_json,
                ts,
                hlc_bytes,
                actor_id,
            ],
        )?;

        // Dual-write to join tables
        for rid in &raw.evidence_rids {
            conn.execute(
                "INSERT OR IGNORE INTO pattern_evidence (pattern_id, rid) VALUES (?1, ?2)",
                params![pattern_id, rid],
            )?;
        }
        for entity_name in &raw.entity_names {
            conn.execute(
                "INSERT OR IGNORE INTO pattern_entities (pattern_id, entity_name) VALUES (?1, ?2)",
                params![pattern_id, entity_name],
            )?;
        }

        // Drop conn before log_op (which also acquires the conn lock)
        drop(conn);

        // Log to oplog for replication
        db.log_op(
            "pattern_upsert",
            Some(&pattern_id),
            &serde_json::json!({
                "pattern_id": pattern_id,
                "pattern_type": raw.pattern_type,
                "status": "active",
                "confidence": raw.confidence,
                "description": raw.description,
                "evidence_rids": raw.evidence_rids,
                "entity_names": raw.entity_names,
                "context": context,
                "first_seen": ts,
                "last_confirmed": ts,
                "occurrence_count": 1,
            }),
            None,
        )?;

        Ok(true) // new pattern
    }
}

/// Mark patterns not confirmed in `max_age_secs` as stale.
pub fn expire_stale_patterns(db: &YantrikDB, ts: f64, max_age_secs: f64) -> Result<usize> {
    let conn = db.conn();
    let changes = conn.execute(
        "UPDATE patterns SET status = 'stale' \
         WHERE status = 'active' AND last_confirmed < ?1",
        params![ts - max_age_secs],
    )?;
    Ok(changes)
}

/// Run all pattern mining algorithms.
pub fn mine_patterns(db: &YantrikDB, config: &PatternConfig) -> Result<PatternMiningResult> {
    let ts = now();
    let mut new_count = 0;
    let mut updated_count = 0;

    let co_occurrences = mine_co_occurrence(db, config)?;
    let temporal = mine_temporal_clusters(db, config)?;
    let valence = mine_valence_trends(db, config)?;
    let topics = mine_topic_clusters(db, config)?;
    let hubs = mine_entity_hubs(db, config)?;

    // Cross-domain mining (V13)
    let cross_domain = if config.run_cross_domain {
        mine_cross_domain_patterns(db, config)?
    } else {
        vec![]
    };
    let entity_bridges = if config.run_cross_domain {
        mine_entity_bridges(db, config)?
    } else {
        vec![]
    };

    let all_raw: Vec<&RawPattern> = co_occurrences
        .iter()
        .chain(temporal.iter())
        .chain(valence.iter())
        .chain(topics.iter())
        .chain(hubs.iter())
        .chain(cross_domain.iter())
        .chain(entity_bridges.iter())
        .collect();

    for raw in all_raw.into_iter().take(config.max_patterns) {
        let is_new = upsert_pattern(db, raw, ts)?;
        if is_new {
            new_count += 1;
        } else {
            updated_count += 1;
        }
    }

    let stale_count = expire_stale_patterns(db, ts, 30.0 * 86400.0)?;

    Ok(PatternMiningResult {
        new_patterns: new_count,
        updated_patterns: updated_count,
        stale_patterns: stale_count,
    })
}

/// Query patterns from the database.
pub fn get_patterns(
    db: &YantrikDB,
    pattern_type: Option<&str>,
    status: Option<&str>,
    limit: usize,
) -> Result<Vec<Pattern>> {
    let conn = db.conn();

    let mut sql = String::from(
        "SELECT pattern_id, pattern_type, status, confidence, description, \
         evidence_rids, entity_names, context, first_seen, last_confirmed, occurrence_count \
         FROM patterns WHERE 1=1",
    );
    let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    if let Some(pt) = pattern_type {
        sql.push_str(&format!(" AND pattern_type = ?{}", param_values.len() + 1));
        param_values.push(Box::new(pt.to_string()));
    }
    if let Some(s) = status {
        sql.push_str(&format!(" AND status = ?{}", param_values.len() + 1));
        param_values.push(Box::new(s.to_string()));
    }
    sql.push_str(&format!(
        " ORDER BY confidence DESC LIMIT ?{}",
        param_values.len() + 1
    ));
    param_values.push(Box::new(limit as i64));

    let mut stmt = conn.prepare(&sql)?;
    let params_ref: Vec<&dyn rusqlite::types::ToSql> =
        param_values.iter().map(|p| p.as_ref()).collect();
    let rows = stmt
        .query_map(params_ref.as_slice(), |row| {
            let evidence_str: String = row.get("evidence_rids")?;
            let entity_str: String = row.get("entity_names")?;
            let context_str: String = row.get("context")?;
            Ok(Pattern {
                pattern_id: row.get("pattern_id")?,
                pattern_type: row.get("pattern_type")?,
                status: row.get("status")?,
                confidence: row.get("confidence")?,
                description: row.get("description")?,
                evidence_rids: serde_json::from_str(&evidence_str).unwrap_or_default(),
                entity_names: serde_json::from_str(&entity_str).unwrap_or_default(),
                context: serde_json::from_str(&context_str)
                    .unwrap_or(serde_json::Value::Object(Default::default())),
                first_seen: row.get("first_seen")?,
                last_confirmed: row.get("last_confirmed")?,
                occurrence_count: row.get("occurrence_count")?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_seed(seed: f32, dim: usize) -> Vec<f32> {
        let raw: Vec<f32> = (0..dim)
            .map(|i| (seed * (i as f32 + 1.0) * 1.7).sin() + (seed * (i as f32 + 2.0) * 0.3).cos())
            .collect();
        let norm: f32 = raw.iter().map(|x| x * x).sum::<f32>().sqrt();
        raw.iter().map(|x| x / norm).collect()
    }

    #[test]
    fn test_uuid_shaped_recognises_record_ids_and_spares_real_names() {
        assert!(crate::knowledge::graph::is_uuid_shaped(
            "01a21c67-9585-7268-ac2b-ed34ae58a8a8"
        ));
        assert!(crate::knowledge::graph::is_uuid_shaped(
            "01A21C67-9585-7268-AC2B-ED34AE58A8A8"
        ));
        assert!(crate::knowledge::graph::is_uuid_shaped(
            "01a21c6795857268ac2bed34ae58a8a8"
        ));
        // Real names that a naive hex rule would eat.
        assert!(!crate::knowledge::graph::is_uuid_shaped("ed25519"));
        assert!(!crate::knowledge::graph::is_uuid_shaped("face2face"));
        assert!(!crate::knowledge::graph::is_uuid_shaped("A380"));
        assert!(!crate::knowledge::graph::is_uuid_shaped("C0FFEE"));
        assert!(!crate::knowledge::graph::is_uuid_shaped(
            "01a21c67_9585_7268_ac2b_ed34ae58a8a8"
        ));
        assert!(!crate::knowledge::graph::is_uuid_shaped(
            "01a21c67-9585-7268-ac2b-ed34ae58a8"
        ));
    }

    #[test]
    fn test_entity_label_shows_a_snippet_for_a_record_and_the_name_otherwise() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rid = db
            .record(
                "Redis is the cache layer for the API gateway and it runs on port 6379.",
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec_seed(900.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        let conn = db.conn();
        let label = entity_label(&db, &conn, &rid);
        assert!(
            label.contains("Redis is the cache layer"),
            "a record id must render as readable text, got {label:?}"
        );
        assert!(
            !label.contains(&rid),
            "the description must not print the raw id"
        );
        drop(conn);
        let conn2 = db.conn();
        assert_eq!(entity_label(&db, &conn2, "API"), "API");
    }

    #[test]
    fn test_remine_refreshes_an_existing_mislabelled_pattern_in_place() {
        // The #326 defect left already-persisted rows with the old weekday
        // label, because dedup matched them and only the counters moved.
        // After the description refresh, a re-mine must correct the text of
        // the SAME row: same pattern id, one row, evidence untouched.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        for i in 0..9 {
            let rid = db
                .record(
                    &format!("remine {i}"),
                    "episodic",
                    0.5,
                    0.0,
                    604800.0,
                    &serde_json::json!({}),
                    &vec_seed(1100.0 + i as f32, 8),
                    "default",
                    0.8,
                    "general",
                    "user",
                    None,
                )
                .unwrap();
            db.conn()
                .execute(
                    "UPDATE memories SET created_at = ?1 WHERE rid = ?2",
                    params![1_795_651_200.0 + (i as f64) * 60.0, rid],
                )
                .unwrap();
        }

        let mined = mine_temporal_clusters(&db, &PatternConfig::default()).unwrap();
        assert_eq!(mined.len(), 1);
        let raw = &mined[0];
        let dedup = serde_json::json!({"dedup_key": raw.dedup_key}).to_string();

        // A row from before the fix: right dedup key, wrong text, fixed id.
        let conn = db.conn();
        conn.execute(
            "INSERT INTO patterns (pattern_id, pattern_type, status, confidence, \
             description, evidence_rids, entity_names, context, first_seen, \
             last_confirmed, occurrence_count, hlc, origin_actor) \
             VALUES ('stale-row', 'temporal_cluster', 'active', 0.1, \
                     'STALE LABEL from before the rendering fix (999 events)', '[]', '[]', \
                     ?1, 0.0, 0.0, 1, x'0000000000000000', 'test')",
            params![dedup],
        )
        .unwrap();
        // Seed REAL evidence rows. Without them the "unchanged" assertion
        // below compares 0 with 0 and passes even if the refresh deletes
        // evidence — a vacuous guard, found by review.
        let mut seeded_rids: Vec<String> = conn
            .prepare("SELECT rid FROM memories ORDER BY rid LIMIT 3")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        seeded_rids.truncate(2);
        for rid in &seeded_rids {
            conn.execute(
                "INSERT OR IGNORE INTO pattern_evidence (pattern_id, rid) VALUES ('stale-row', ?1)",
                params![rid],
            )
            .unwrap();
        }
        let before_evidence: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pattern_evidence WHERE pattern_id = 'stale-row'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            before_evidence, 2,
            "fixture check: evidence must exist to be preserved"
        );
        drop(conn);

        let created = upsert_pattern(&db, raw, 1_800_000_000.0).unwrap();
        assert!(!created, "the existing row must be matched, not duplicated");

        let conn = db.conn();
        let rows: Vec<(String, String)> = conn
            .prepare(
                "SELECT pattern_id, description FROM patterns \
                 WHERE json_extract(context, '$.dedup_key') = json_extract(?1, '$.dedup_key')",
            )
            .unwrap()
            .query_map(params![dedup], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(rows.len(), 1, "a re-mine must not create a duplicate row");
        assert_eq!(rows[0].0, "stale-row", "the pattern id must survive");
        assert_ne!(
            raw.description, "STALE LABEL from before the rendering fix (999 events)",
            "fixture check: the seeded text must differ from what mining produces, otherwise this cannot fail"
        );
        assert_eq!(
            rows[0].1, raw.description,
            "the stale label must be replaced by the freshly mined one"
        );
        let after_evidence: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM pattern_evidence WHERE pattern_id = 'stale-row'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            before_evidence, after_evidence,
            "refreshing a description must not change evidence semantics"
        );
        assert_eq!(
            before_evidence, 2,
            "the seeded evidence must still be there"
        );
        // Read back every column the matched path writes, so a future
        // parameter-order mistake cannot go unnoticed (review HF-04).
        let (count, conf): (i64, f64) = conn
            .query_row(
                "SELECT occurrence_count, confidence FROM patterns WHERE pattern_id = 'stale-row'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(count, 2, "the matched path increments the occurrence count");
        assert!(
            conf >= raw.confidence,
            "confidence is raised, never lowered"
        );
    }

    #[test]
    fn test_forgotten_text_never_reappears_in_a_label() {
        // forget() does not scrub `memories.text`; it only flips
        // consolidation_status to 'tombstoned'. An unfiltered read in
        // `entity_label` therefore printed forgotten content into a pattern
        // description, which defeats the forget.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let secret = "The api key rotates every ninety days.";
        let rid = db
            .record(
                secret,
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec_seed(1300.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        // Positive control: while active, the label really is the snippet.
        let conn = db.conn();
        let label = entity_label(&db, &conn, &rid);
        assert!(
            label.contains("api key rotates"),
            "positive control: an active record must render its text, got {label:?}"
        );
        drop(conn);

        db.forget(&rid).unwrap();

        let conn = db.conn();
        let label = entity_label(&db, &conn, &rid);
        assert!(
            !label.contains("api key rotates"),
            "the forgotten text must not reappear in a label, got {label:?}"
        );
        assert_eq!(label, rid, "an unresolvable record falls back to its name");
        drop(conn);

        // Re-mining must not resurrect it either.
        let _ = mine_patterns(&db, &PatternConfig::default()).unwrap();
        for p in get_patterns(&db, None, None, 200).unwrap() {
            assert!(
                !p.description.contains("api key rotates"),
                "a mined pattern must not quote a forgotten record: {:?}",
                p.description
            );
        }
    }

    #[test]
    fn test_encrypted_store_labels_show_plaintext_never_ciphertext() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("enc.db");
        let mut key = [0u8; 32];
        key[0] = 7;
        let db = YantrikDB::new_encrypted(path.to_str().unwrap(), 8, &key).unwrap();
        let secret = "Rotate the signing key before the quarter ends.";
        let rid = db
            .record(
                secret,
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec_seed(1400.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        let conn = db.conn();
        let stored: String = conn
            .query_row(
                "SELECT text FROM memories WHERE rid = ?1",
                params![rid],
                |r| r.get(0),
            )
            .unwrap();
        assert_ne!(
            stored, secret,
            "the fixture must actually be encrypted at rest"
        );

        let label = entity_label(&db, &conn, &rid);
        assert!(
            label.contains("signing key"),
            "positive control: the label must show the decrypted text, got {label:?}"
        );
        assert!(
            !label.contains(&stored[..8.min(stored.len())]),
            "the label must never print ciphertext, got {label:?}"
        );
    }

    #[test]
    fn test_mining_pass_survives_an_undecryptable_record() {
        // HF-05: `entity_label` falls back on a decryption failure, but the
        // public pass used to abort in `mine_topic_clusters`, which
        // `?`-propagated the same error. One unreadable row must not kill the
        // whole run.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("enc2.db");
        let mut key = [0u8; 32];
        key[0] = 9;
        let db = YantrikDB::new_encrypted(path.to_str().unwrap(), 8, &key).unwrap();
        let mut rids = Vec::new();
        for i in 0..6 {
            let body = format!("Signing key rotation note {i} for the gateway.");
            rids.push(
                db.record(
                    &body,
                    "episodic",
                    0.5,
                    0.0,
                    604800.0,
                    &serde_json::json!({}),
                    &vec_seed(1500.0 + i as f32, 8),
                    "default",
                    0.8,
                    "general",
                    "user",
                    None,
                )
                .unwrap(),
            );
        }
        db.conn()
            .execute(
                "UPDATE memories SET text = '!!!not-base64!!!' WHERE rid = ?1",
                params![rids[0]],
            )
            .unwrap();
        assert!(
            db.decrypt_text("!!!not-base64!!!").is_err(),
            "fixture check: this payload must really fail to decrypt"
        );

        let res = mine_patterns(&db, &PatternConfig::default());
        assert!(
            res.is_ok(),
            "one unreadable record must not abort the pass: {res:?}"
        );

        for p in get_patterns(&db, None, None, 200).unwrap() {
            assert!(
                !p.description.contains("not-base64"),
                "ciphertext must never reach a description: {:?}",
                p.description
            );
        }
    }

    #[test]
    fn test_persisted_description_is_corrected_after_forget_and_remine() {
        // HF-06: the maintainer's case — a description that already quotes a
        // record, the record is forgotten, and the same row is re-mined.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let secret = "The api key rotates every ninety days.";
        let rid = db
            .record(
                secret,
                "episodic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec_seed(1600.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        // Three shared neighbours so co_occurrence fires between the record
        // and a plain name.
        for t in ["Hub A", "Hub B", "Hub C"] {
            db.relate(&rid, t, "related_to", 1.0).unwrap();
            db.relate("Partner", t, "related_to", 1.0).unwrap();
        }

        let _ = mine_patterns(&db, &PatternConfig::default()).unwrap();
        let before = get_patterns(&db, Some("co_occurrence"), None, 50).unwrap();
        let row = before
            .iter()
            .find(|p| p.description.contains("api key rotates"))
            .expect("the first mine must quote the record's text");
        let pid = row.pattern_id.clone();

        db.forget(&rid).unwrap();
        let _ = mine_patterns(&db, &PatternConfig::default()).unwrap();

        let after = get_patterns(&db, Some("co_occurrence"), None, 50).unwrap();
        let same: Vec<_> = after.iter().filter(|p| p.pattern_id == pid).collect();
        assert_eq!(same.len(), 1, "the re-mine must not duplicate the row");
        assert!(
            !same[0].description.contains("api key rotates"),
            "the stale description must be corrected, got {:?}",
            same[0].description
        );
        assert!(
            same[0].description.contains(&rid),
            "the corrected label falls back to the rid, got {:?}",
            same[0].description
        );
    }

    #[test]
    fn test_mine_patterns_empty_db() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let result = mine_patterns(&db, &PatternConfig::default()).unwrap();
        assert_eq!(result.new_patterns, 0);
        assert_eq!(result.updated_patterns, 0);
        assert_eq!(result.stale_patterns, 0);
    }

    #[test]
    fn test_entity_hub_detection() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        // Create a hub with 6 edges
        for i in 0..6 {
            db.relate("Alice", &format!("target_{i}"), &format!("rel_{i}"), 1.0)
                .unwrap();
        }

        let config = PatternConfig {
            entity_hub_min_degree: 5,
            ..Default::default()
        };
        let hubs = mine_entity_hubs(&db, &config).unwrap();
        assert!(!hubs.is_empty());
        assert!(hubs[0].description.contains("Alice"));
    }

    #[test]
    fn test_co_occurrence_basic() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        // Create shared connections: Alice and Bob both connected to 3 memories
        for i in 0..3 {
            let target = format!("mem_{i}");
            db.relate("Alice", &target, "is_about", 1.0).unwrap();
            db.relate("Bob", &target, "is_about", 1.0).unwrap();
        }

        let config = PatternConfig {
            co_occurrence_min_count: 3,
            ..Default::default()
        };
        let co = mine_co_occurrence(&db, &config).unwrap();
        assert!(!co.is_empty());
        assert!(co[0].description.contains("Alice") || co[0].description.contains("Bob"));
    }

    #[test]
    fn test_pattern_upsert_updates_existing() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let raw = RawPattern {
            pattern_type: "entity_hub".to_string(),
            confidence: 0.7,
            description: "test pattern".to_string(),
            evidence_rids: vec![],
            entity_names: vec!["Alice".to_string()],
            context: serde_json::json!({}),
            dedup_key: "entity_hub:Alice".to_string(),
        };

        let ts = now();
        let is_new = upsert_pattern(&db, &raw, ts).unwrap();
        assert!(is_new);

        // Second upsert should update, not create
        let is_new2 = upsert_pattern(&db, &raw, ts + 1.0).unwrap();
        assert!(!is_new2);

        // Verify occurrence_count incremented
        let patterns = get_patterns(&db, Some("entity_hub"), None, 10).unwrap();
        assert_eq!(patterns.len(), 1);
        assert_eq!(patterns[0].occurrence_count, 2);
    }

    #[test]
    fn test_stale_pattern_expiry() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let raw = RawPattern {
            pattern_type: "entity_hub".to_string(),
            confidence: 0.7,
            description: "old pattern".to_string(),
            evidence_rids: vec![],
            entity_names: vec!["OldEntity".to_string()],
            context: serde_json::json!({}),
            dedup_key: "entity_hub:OldEntity".to_string(),
        };

        let old_ts = now() - 86400.0 * 60.0; // 60 days ago
        upsert_pattern(&db, &raw, old_ts).unwrap();

        let stale = expire_stale_patterns(&db, now(), 30.0 * 86400.0).unwrap();
        assert_eq!(stale, 1);

        let patterns = get_patterns(&db, None, Some("stale"), 10).unwrap();
        assert_eq!(patterns.len(), 1);
    }

    #[test]
    fn test_valence_trend_detection() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let ts = now();

        // Create baseline memories (weeks 2-4) with neutral valence
        for i in 0..10 {
            let rid = db
                .record(
                    &format!("baseline {i}"),
                    "episodic",
                    0.5,
                    0.0,
                    604800.0,
                    &serde_json::json!({}),
                    &vec_seed(i as f32, 8),
                    "default",
                    0.8,
                    "general",
                    "user",
                    None,
                )
                .unwrap();
            let age = 86400.0 * (14.0 + i as f64); // 14-24 days ago
            db.conn()
                .execute(
                    "UPDATE memories SET created_at = ?1 WHERE rid = ?2",
                    params![ts - age, rid],
                )
                .unwrap();
        }

        // Create recent memories (last 7 days) with strongly negative valence
        for i in 0..5 {
            let rid = db
                .record(
                    &format!("recent negative {i}"),
                    "episodic",
                    0.5,
                    -0.8,
                    604800.0,
                    &serde_json::json!({}),
                    &vec_seed(100.0 + i as f32, 8),
                    "default",
                    0.8,
                    "general",
                    "user",
                    None,
                )
                .unwrap();
            let age = 86400.0 * (1.0 + i as f64); // 1-5 days ago
            db.conn()
                .execute(
                    "UPDATE memories SET created_at = ?1 WHERE rid = ?2",
                    params![ts - age, rid],
                )
                .unwrap();
        }

        let config = PatternConfig::default();
        let trends = mine_valence_trends(&db, &config).unwrap();
        assert!(!trends.is_empty());
        assert_eq!(trends[0].pattern_type, "valence_trend");
    }
}
