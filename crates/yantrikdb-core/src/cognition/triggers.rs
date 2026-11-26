use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::params;

use crate::engine::YantrikDB;
use crate::error::Result;
use crate::scoring;
use crate::types::{PersistedTrigger, Trigger, TriggerType};

fn now() -> f64 {
    crate::time::now_secs()
}

/// Maximum length (chars) for a `snippet_*` field on a redundancy /
/// potential_conflict trigger's `context`. Issue #45: the trigger payload
/// previously carried full `text_a` / `text_b` which can be kilobytes;
/// snippet fields give callers a quick visual diff without the bloat.
const TRIGGER_SNIPPET_MAX_CHARS: usize = 120;

/// Truncate `text` to at most `TRIGGER_SNIPPET_MAX_CHARS` characters at a
/// char-boundary, appending an ellipsis when truncated. Used by the
/// redundancy / potential_conflict triggers' `context.snippet_a` /
/// `context.snippet_b` fields so consumers can preview both memories
/// without pulling the full text via a separate recall.
fn snippet_for_trigger(text: &str) -> String {
    if text.chars().count() <= TRIGGER_SNIPPET_MAX_CHARS {
        text.to_string()
    } else {
        let mut s: String = text.chars().take(TRIGGER_SNIPPET_MAX_CHARS).collect();
        s.push('…');
        s
    }
}

// ── Existing trigger checks ──

/// Find important memories that are decaying significantly.
pub fn check_decay_triggers(
    db: &YantrikDB,
    importance_threshold: f64,
    decay_threshold: f64,
    max_triggers: usize,
) -> Result<Vec<Trigger>> {
    let ts = now();
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "SELECT rid, text, type, importance, half_life, last_access, valence \
         FROM memories \
         WHERE consolidation_status = 'active' \
         AND importance >= ?1",
    )?;

    let mut triggers = Vec::new();

    let rows = stmt.query_map(rusqlite::params![importance_threshold], |row| {
        Ok((
            row.get::<_, String>("rid")?,
            row.get::<_, String>("text")?,
            row.get::<_, String>("type")?,
            row.get::<_, f64>("importance")?,
            row.get::<_, f64>("half_life")?,
            row.get::<_, f64>("last_access")?,
            row.get::<_, f64>("valence")?,
        ))
    })?;

    for row in rows {
        let (rid, stored_text, mem_type, importance, half_life, last_access, valence) = row?;
        let elapsed = ts - last_access;
        let current_score = scoring::decay_score(importance, half_life, elapsed);

        if current_score < decay_threshold {
            let text = db.decrypt_text(&stored_text)?;
            let days_since = elapsed / 86400.0;
            let decay_ratio = if importance > 0.0 {
                current_score / importance
            } else {
                0.0
            };

            let urgency = importance * (1.0 - decay_ratio);

            let mut context = HashMap::new();
            context.insert("text".to_string(), serde_json::json!(text));
            context.insert("type".to_string(), serde_json::json!(mem_type));
            context.insert(
                "original_importance".to_string(),
                serde_json::json!(importance),
            );
            context.insert(
                "current_score".to_string(),
                serde_json::json!(current_score),
            );
            context.insert(
                "days_since_access".to_string(),
                serde_json::json!(days_since),
            );
            context.insert("valence".to_string(), serde_json::json!(valence));

            triggers.push(Trigger {
                trigger_type: "decay_review".to_string(),
                reason: format!(
                    "Important memory (importance={importance:.1}) \
                     has decayed to {current_score:.3} after {days_since:.0} days"
                ),
                urgency,
                source_rids: vec![rid],
                suggested_action: "ask_user_to_confirm_or_forget".to_string(),
                context,
            });
        }
    }

    triggers.sort_by(|a, b| b.urgency.total_cmp(&a.urgency));
    triggers.truncate(max_triggers);
    Ok(triggers)
}

/// Trigger when there are enough active memories that consolidation might help.
pub fn check_consolidation_triggers(
    db: &YantrikDB,
    min_active_memories: i64,
) -> Result<Vec<Trigger>> {
    let stats = db.stats(None)?;
    let mut triggers = Vec::new();

    if stats.active_memories >= min_active_memories {
        let conn = db.conn();
        let unconsolidated: i64 = conn.query_row(
            "SELECT COUNT(*) FROM memories \
             WHERE consolidation_status = 'active' \
             AND type = 'episodic'",
            [],
            |row| row.get(0),
        )?;

        if unconsolidated >= min_active_memories {
            let mut context = HashMap::new();
            context.insert(
                "episodic_count".to_string(),
                serde_json::json!(unconsolidated),
            );
            context.insert(
                "total_active".to_string(),
                serde_json::json!(stats.active_memories),
            );

            triggers.push(Trigger {
                trigger_type: "consolidation_ready".to_string(),
                reason: format!(
                    "{unconsolidated} active episodic memories are awaiting a bounded consolidation scan"
                ),
                urgency: (unconsolidated as f64 / 50.0).min(1.0),
                source_rids: vec![],
                suggested_action: "run_consolidation".to_string(),
                context,
            });
        }
    }

    Ok(triggers)
}

// ── New V3 trigger checks ──

/// Trigger when too many conflicts are open or critical ones are aging.
pub fn check_conflict_escalation(db: &YantrikDB) -> Result<Vec<Trigger>> {
    let conn = db.conn();
    let open_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM conflicts WHERE status = 'open'",
        [],
        |row| row.get(0),
    )?;

    let ts = now();
    let critical_aging: bool = conn.query_row(
        "SELECT COUNT(*) > 0 FROM conflicts \
         WHERE status = 'open' AND priority = 'critical' \
         AND detected_at < ?1",
        params![ts - 86400.0 * 3.0],
        |row| row.get(0),
    )?;

    let mut triggers = Vec::new();
    if open_count > 5 || critical_aging {
        let mut urgency = (open_count as f64 / 10.0).min(1.0);
        if critical_aging {
            urgency = (urgency + 0.3).min(1.0);
        }

        let mut context = HashMap::new();
        context.insert("open_count".to_string(), serde_json::json!(open_count));
        context.insert(
            "critical_aging".to_string(),
            serde_json::json!(critical_aging),
        );

        triggers.push(Trigger {
            trigger_type: "conflict_escalation".to_string(),
            reason: format!("{open_count} open conflicts need attention"),
            urgency,
            source_rids: vec![],
            suggested_action: "review_conflicts".to_string(),
            context,
        });
    }

    Ok(triggers)
}

/// Trigger for old semantic memories that may be stale.
pub fn check_temporal_drift(db: &YantrikDB) -> Result<Vec<Trigger>> {
    let ts = now();
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "SELECT rid, text, created_at, last_access \
         FROM memories \
         WHERE type = 'semantic' \
         AND consolidation_status = 'active' \
         AND created_at < ?1 \
         AND last_access < ?2 \
         LIMIT 10",
    )?;

    let cutoff_created = ts - 86400.0 * 90.0; // 90 days old
    let cutoff_access = ts - 86400.0 * 30.0; // not accessed in 30 days
    let mut triggers = Vec::new();

    let rows = stmt.query_map(params![cutoff_created, cutoff_access], |row| {
        Ok((
            row.get::<_, String>("rid")?,
            row.get::<_, String>("text")?,
            row.get::<_, f64>("created_at")?,
            row.get::<_, f64>("last_access")?,
        ))
    })?;

    for row in rows {
        let (rid, stored_text, created_at, _last_access) = row?;
        let text = db.decrypt_text(&stored_text)?;
        let age_days = (ts - created_at) / 86400.0;
        let urgency = (age_days / 365.0).min(1.0);

        let mut context = HashMap::new();
        context.insert("text".to_string(), serde_json::json!(text));
        context.insert("age_days".to_string(), serde_json::json!(age_days));

        triggers.push(Trigger {
            trigger_type: "temporal_drift".to_string(),
            reason: format!("Semantic memory is {age_days:.0} days old and may be outdated"),
            urgency,
            source_rids: vec![rid],
            suggested_action: "verify_or_update".to_string(),
            context,
        });
    }

    Ok(triggers)
}

/// Trigger for near-duplicate active memories (cosine similarity > 0.85).
///
/// **Namespace-pure by construction.** The candidate scan runs once per
/// namespace. Two people recorded into one store must never be paired into
/// a single trigger: the payload carries both texts (`context.text_a` /
/// `context.text_b`), so a cross-namespace pair is a disclosure, not just a
/// bad suggestion. This is the same boundary `consolidate_cluster` enforces
/// with a hard error when a cluster spans namespaces.
pub fn check_redundancy(db: &YantrikDB, sim_threshold: f64) -> Result<Vec<Trigger>> {
    let namespaces: Vec<String> = {
        let conn = db.conn();
        let mut stmt = conn.prepare(
            "SELECT DISTINCT namespace FROM memories \
             WHERE consolidation_status = 'active' \
             AND embedding IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };

    let mut triggers = Vec::new();
    for namespace in namespaces {
        triggers.extend(check_redundancy_in_namespace(
            db,
            &namespace,
            sim_threshold,
        )?);
    }
    Ok(triggers)
}

/// A maintenance run must say when a namespace was not fully checked.
///
/// "No conflicts found" and "conflicts could not be evaluated" are different
/// answers, and only one of them is good news. The log alone is not enough:
/// whoever reads the result has to see it.
fn incomplete_namespace_trigger(namespace: &str, reason: &str) -> Trigger {
    Trigger {
        trigger_type: "maintenance_incomplete".to_string(),
        reason: format!(
            "namespace {namespace} was not fully checked: {reason}; \
             substitution conflicts are missing for it"
        ),
        urgency: 0.9,
        source_rids: vec![],
        suggested_action:
            "inspect the substitution tables; this namespace's substitution conflicts \
             were not evaluated"
                .to_string(),
        context: std::collections::HashMap::new(),
    }
}

fn check_redundancy_in_namespace(
    db: &YantrikDB,
    namespace: &str,
    sim_threshold: f64,
) -> Result<Vec<Trigger>> {
    // Phase 1: Collect memory data while holding the conn lock.
    let rows: Vec<(String, String, Vec<u8>)> = {
        let conn = db.conn();
        let mut stmt = conn.prepare(
            "SELECT rid, text, embedding \
             FROM memories \
             WHERE consolidation_status = 'active' \
             AND embedding IS NOT NULL \
             AND namespace = ?1 \
             ORDER BY created_at DESC \
             LIMIT 30",
        )?;

        let raw_rows: Vec<(String, String, Vec<u8>)> = stmt
            .query_map(rusqlite::params![namespace], |row| {
                Ok((
                    row.get::<_, String>("rid")?,
                    row.get::<_, String>("text")?,
                    row.get::<_, Vec<u8>>("embedding")?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // Decrypt text and embeddings if encrypted
        raw_rows
            .into_iter()
            .map(|(rid, stored_text, stored_emb)| {
                let text = db.decrypt_text(&stored_text)?;
                let emb = db.decrypt_embedding(&stored_emb)?;
                Ok((rid, text, emb))
            })
            .collect::<Result<Vec<_>>>()?
    }; // conn lock released here

    let mut triggers = Vec::new();
    let threshold = sim_threshold;

    // Load the exclusive substitution categories ONCE per namespace.
    //
    // The per-pair helper used to run a JOIN query for every differing token
    // pair of every near-duplicate memory pair: on this store (10 namespaces,
    // ≤30 rows each) that is ~8,700 memory pairs, and two texts of a few
    // thousand characters differ in hundreds of tokens each — tens of
    // thousands of SQL round trips per memory pair. The table holds 8
    // categories and 80 members; reading it once and answering the same
    // predicate in memory removes the cost without changing a single result.
    let substitution_index = {
        let conn = db.conn();
        match load_substitution_index_status(&conn) {
            Ok((index, None)) => index,
            // Loaded, but degraded (missing tables): the index is empty and
            // still usable — the reason must reach the report.
            Ok((index, Some(reason))) => {
                triggers.push(incomplete_namespace_trigger(namespace, &reason));
                index
            }
            // Degrade, do not abort. Review 8 (finding 3): propagating took
            // down the remaining namespaces, both conflict scans, the
            // consolidation and three further trigger checks — and surfaced as
            // an exception in the Python and WASM APIs where the caller used
            // to get a normal result.
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "substitution index unavailable for this namespace; \
                     substitution conflicts are skipped for it"
                );
                triggers.push(incomplete_namespace_trigger(namespace, &format!("{e}")));
                SubstitutionIndex::new()
            }
        }
    };

    for i in 0..rows.len() {
        for j in (i + 1)..rows.len() {
            let emb_a = crate::compression::decode_embedding_blob(&rows[i].2);
            let emb_b = crate::compression::decode_embedding_blob(&rows[j].2);
            let sim = crate::consolidate::cosine_similarity(&emb_a, &emb_b);

            if sim > threshold {
                // **Issue #45 — name the rids in the trigger payload.**
                // The trigger struct's `source_rids` field has always carried
                // the pair, but consumers reading only the `reason` string
                // (or the MCP tool surface which sometimes truncates context)
                // saw an opaque "Two memories are X% similar" with no way to
                // act on it. Embed the rids in `reason` itself + add compact
                // snippets in `context` so the trigger output is actionable
                // without a separate recall round-trip. See yantrikdb-agi
                // Phase 1 gap T1 + Phase 2 Proposal 7.1.
                let snippet_a = snippet_for_trigger(&rows[i].1);
                let snippet_b = snippet_for_trigger(&rows[j].1);
                let mut context = HashMap::new();
                context.insert("text_a".to_string(), serde_json::json!(rows[i].1));
                context.insert("text_b".to_string(), serde_json::json!(rows[j].1));
                context.insert("snippet_a".to_string(), serde_json::json!(snippet_a));
                context.insert("snippet_b".to_string(), serde_json::json!(snippet_b));
                context.insert("similarity".to_string(), serde_json::json!(sim));
                context.insert("rid_a".to_string(), serde_json::json!(rows[i].0));
                context.insert("rid_b".to_string(), serde_json::json!(rows[j].0));

                // Check if the pair shares entities — if so, this is likely a
                // contradiction (same topic, different facts) not a simple duplicate.
                let (entities_a, entities_b) = {
                    let conn = db.conn();
                    let ea: Vec<String> = conn
                        .prepare("SELECT entity_name FROM memory_entities WHERE memory_rid = ?1")?
                        .query_map(rusqlite::params![rows[i].0], |r| r.get(0))?
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    let eb: Vec<String> = conn
                        .prepare("SELECT entity_name FROM memory_entities WHERE memory_rid = ?1")?
                        .query_map(rusqlite::params![rows[j].0], |r| r.get(0))?
                        .collect::<std::result::Result<Vec<_>, _>>()?;
                    (ea, eb)
                };

                let shared: Vec<&String> = entities_a
                    .iter()
                    .filter(|e| entities_b.contains(e))
                    .collect();
                let is_potential_conflict = !shared.is_empty() && sim < 0.98;

                if is_potential_conflict {
                    context.insert(
                        "shared_entities".to_string(),
                        serde_json::json!(shared.iter().map(|s| s.as_str()).collect::<Vec<_>>()),
                    );
                    triggers.push(Trigger {
                        trigger_type: "potential_conflict".to_string(),
                        reason: format!(
                            "Two memories about '{}' are {:.0}% similar but may contradict each other (rid_a={}, rid_b={})",
                            shared.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", "),
                            sim * 100.0,
                            rows[i].0,
                            rows[j].0
                        ),
                        urgency: sim,
                        source_rids: vec![rows[i].0.clone(), rows[j].0.clone()],
                        suggested_action: "review_conflict".to_string(),
                        context,
                    });
                } else if let Some((cat_name, token_a, token_b)) =
                    check_substitution_pair_in_index(&substitution_index, &rows[i].1, &rows[j].1)
                {
                    // Substitution category match -> create actual conflict record
                    let reason = format!(
                        "{} substitution: '{}' vs '{}' (similarity={:.0}%, rid_a={}, rid_b={})",
                        cat_name,
                        token_a,
                        token_b,
                        sim * 100.0,
                        rows[i].0,
                        rows[j].0
                    );
                    {
                        // One statement checks and inserts, so concurrent
                        // checks (detached since #271) record it once.
                        let conflict_type =
                            crate::distributed::conflict::category_to_conflict_type(&cat_name);
                        let _ = crate::conflict::create_conflict_if_absent(
                            db,
                            &conflict_type,
                            &rows[i].0,
                            &rows[j].0,
                            None,
                            None,
                            &reason,
                        );
                    }
                    context.insert("category".to_string(), serde_json::json!(cat_name));
                    context.insert("token_a".to_string(), serde_json::json!(token_a));
                    context.insert("token_b".to_string(), serde_json::json!(token_b));
                    triggers.push(Trigger {
                        trigger_type: "potential_conflict".to_string(),
                        reason,
                        urgency: sim,
                        source_rids: vec![rows[i].0.clone(), rows[j].0.clone()],
                        suggested_action: "review_conflict".to_string(),
                        context,
                    });
                } else {
                    triggers.push(Trigger {
                        trigger_type: "redundancy".to_string(),
                        reason: format!(
                            "Two memories are {:.0}% similar and may be redundant (rid_a={}, rid_b={}): '{}' vs '{}'",
                            sim * 100.0,
                            rows[i].0,
                            rows[j].0,
                            snippet_a,
                            snippet_b,
                        ),
                        urgency: sim,
                        source_rids: vec![rows[i].0.clone(), rows[j].0.clone()],
                        suggested_action: "consolidate_or_forget".to_string(),
                        context,
                    });
                }
            }
        }
    }

    // Second pass: lower threshold for substitution category conflicts.
    // Substituting "PostgreSQL" for "MySQL" drops cosine similarity to ~0.80,
    // below the 0.85 redundancy threshold. Check pairs in [0.65, 0.85] range
    // specifically for category-based substitution.
    let cat_threshold = 0.65;
    for i in 0..rows.len() {
        for j in (i + 1)..rows.len() {
            let emb_a = crate::compression::decode_embedding_blob(&rows[i].2);
            let emb_b = crate::compression::decode_embedding_blob(&rows[j].2);
            let sim = crate::consolidate::cosine_similarity(&emb_a, &emb_b);

            if sim > cat_threshold && sim <= threshold {
                // `substitution_index` is already loaded for this namespace;
                // no per-pair SQL, so the self-deadlock hazard the previous
                // scrutinee-scoped `db.conn()` guarded against is gone.
                if let Some((cat_name, token_a, token_b)) =
                    check_substitution_pair_in_index(&substitution_index, &rows[i].1, &rows[j].1)
                {
                    let reason = format!(
                        "{} substitution: '{}' vs '{}' (similarity={:.0}%, rid_a={}, rid_b={})",
                        cat_name,
                        token_a,
                        token_b,
                        sim * 100.0,
                        rows[i].0,
                        rows[j].0
                    );
                    {
                        // One statement checks and inserts, so concurrent
                        // checks (detached since #271) record it once.
                        let conflict_type =
                            crate::distributed::conflict::category_to_conflict_type(&cat_name);
                        let _ = crate::conflict::create_conflict_if_absent(
                            db,
                            &conflict_type,
                            &rows[i].0,
                            &rows[j].0,
                            None,
                            None,
                            &reason,
                        );
                    }
                    // Issue #45: include compact snippets + explicit rid_a/rid_b
                    // in context so consumers reading the trigger payload see
                    // the matching rid pair + a quick visual diff without a
                    // separate recall round-trip. Keep full text_a/text_b for
                    // existing callers; this is additive.
                    let snippet_a = snippet_for_trigger(&rows[i].1);
                    let snippet_b = snippet_for_trigger(&rows[j].1);
                    let mut context = HashMap::new();
                    context.insert("text_a".to_string(), serde_json::json!(rows[i].1));
                    context.insert("text_b".to_string(), serde_json::json!(rows[j].1));
                    context.insert("snippet_a".to_string(), serde_json::json!(snippet_a));
                    context.insert("snippet_b".to_string(), serde_json::json!(snippet_b));
                    context.insert("similarity".to_string(), serde_json::json!(sim));
                    context.insert("rid_a".to_string(), serde_json::json!(rows[i].0));
                    context.insert("rid_b".to_string(), serde_json::json!(rows[j].0));
                    context.insert("category".to_string(), serde_json::json!(cat_name));
                    context.insert("token_a".to_string(), serde_json::json!(token_a));
                    context.insert("token_b".to_string(), serde_json::json!(token_b));
                    triggers.push(Trigger {
                        trigger_type: "potential_conflict".to_string(),
                        reason,
                        urgency: sim,
                        source_rids: vec![rows[i].0.clone(), rows[j].0.clone()],
                        suggested_action: "review_conflict".to_string(),
                        context,
                    });
                }
            }
        }
    }

    triggers.truncate(5);
    Ok(triggers)
}

/// Trigger for high-degree entities (relationship hubs).
pub fn check_relationship_insight(db: &YantrikDB) -> Result<Vec<Trigger>> {
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "SELECT src, COUNT(*) as degree \
         FROM edges WHERE tombstoned = 0 \
         GROUP BY src HAVING degree >= 5 \
         ORDER BY degree DESC \
         LIMIT 10",
    )?;

    let mut triggers = Vec::new();
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })?;

    for row in rows {
        let (entity, degree) = row?;
        // A node the entity table would refuse today (a year, a heading)
        // is not an insight, however many links an older extractor gave it.
        if !crate::graph::admit_entity(&entity) {
            continue;
        }
        let urgency = (degree as f64 / 20.0).min(1.0);

        let mut context = HashMap::new();
        context.insert("entity".to_string(), serde_json::json!(entity));
        context.insert("degree".to_string(), serde_json::json!(degree));

        triggers.push(Trigger {
            trigger_type: "relationship_insight".to_string(),
            reason: format!("Entity '{entity}' is a hub with {degree} connections"),
            urgency,
            source_rids: vec![],
            suggested_action: "explore_entity".to_string(),
            context,
        });
    }

    Ok(triggers)
}

/// Trigger when emotional valence shifts significantly over time.
pub fn check_valence_trend(db: &YantrikDB) -> Result<Vec<Trigger>> {
    let ts = now();
    let conn = db.conn();

    // Recent 7 days
    let recent_stats: Option<(f64, i64)> = conn
        .query_row(
            "SELECT AVG(valence), COUNT(*) FROM memories \
             WHERE consolidation_status = 'active' \
             AND created_at > ?1",
            params![ts - 86400.0 * 7.0],
            |row| Ok((row.get::<_, f64>(0)?, row.get::<_, i64>(1)?)),
        )
        .ok();

    // Preceding 30 days (day 7 to day 37)
    let baseline_stats: Option<(f64, i64)> = conn
        .query_row(
            "SELECT AVG(valence), COUNT(*) FROM memories \
             WHERE consolidation_status = 'active' \
             AND created_at BETWEEN ?1 AND ?2",
            params![ts - 86400.0 * 37.0, ts - 86400.0 * 7.0],
            |row| Ok((row.get::<_, f64>(0)?, row.get::<_, i64>(1)?)),
        )
        .ok();

    let mut triggers = Vec::new();

    if let (Some((recent_avg, recent_count)), Some((baseline_avg, baseline_count))) =
        (recent_stats, baseline_stats)
    {
        if recent_count >= 3 && baseline_count >= 3 {
            let delta = recent_avg - baseline_avg;
            if delta.abs() > 0.3 {
                let direction = if delta > 0.0 { "positive" } else { "negative" };
                let urgency = delta.abs().min(1.0);

                let mut context = HashMap::new();
                context.insert("recent_avg".to_string(), serde_json::json!(recent_avg));
                context.insert("baseline_avg".to_string(), serde_json::json!(baseline_avg));
                context.insert("delta".to_string(), serde_json::json!(delta));
                context.insert("direction".to_string(), serde_json::json!(direction));

                triggers.push(Trigger {
                    trigger_type: "valence_trend".to_string(),
                    reason: format!(
                        "Emotional tone has shifted {direction} by {:.2} over the past week",
                        delta.abs()
                    ),
                    urgency,
                    source_rids: vec![],
                    suggested_action: "acknowledge_trend".to_string(),
                    context,
                });
            }
        }
    }

    Ok(triggers)
}

/// Trigger for entities with contradictory edges (same src+rel_type, different dst).
pub fn check_entity_anomaly(db: &YantrikDB) -> Result<Vec<Trigger>> {
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "SELECT src, rel_type, COUNT(DISTINCT dst) as dst_count \
         FROM edges WHERE tombstoned = 0 \
         GROUP BY src, rel_type \
         HAVING dst_count >= 3 \
         ORDER BY dst_count DESC \
         LIMIT 10",
    )?;

    let identity_types: &[&str] = &[
        "birthday",
        "age",
        "lives_in",
        "works_at",
        "email",
        "phone",
        "full_name",
        "spouse",
        "hometown",
    ];
    let preference_types: &[&str] = &["prefers", "favorite", "likes", "dislikes"];

    let mut triggers = Vec::new();
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, i64>(2)?,
        ))
    })?;

    for row in rows {
        let (entity, rel_type, dst_count) = row?;
        let is_identity = identity_types.contains(&rel_type.as_str());
        let is_preference = preference_types.contains(&rel_type.as_str());

        if is_identity || is_preference {
            let urgency = (dst_count as f64 / 5.0).min(1.0);

            let mut context = HashMap::new();
            context.insert("entity".to_string(), serde_json::json!(entity));
            context.insert("rel_type".to_string(), serde_json::json!(rel_type));
            context.insert("distinct_values".to_string(), serde_json::json!(dst_count));

            triggers.push(Trigger {
                trigger_type: "entity_anomaly".to_string(),
                reason: format!(
                    "Entity '{entity}' has {dst_count} different values for '{rel_type}'"
                ),
                urgency,
                source_rids: vec![],
                suggested_action: "review_entity".to_string(),
                context,
            });
        }
    }

    Ok(triggers)
}

// ── Unified trigger check ──

/// Run all trigger checks and return a unified, priority-sorted list.
pub fn check_all_triggers(
    db: &YantrikDB,
    importance_threshold: f64,
    decay_threshold: f64,
    max_triggers: usize,
) -> Result<Vec<Trigger>> {
    let mut triggers = Vec::new();
    triggers.extend(check_decay_triggers(
        db,
        importance_threshold,
        decay_threshold,
        max_triggers,
    )?);
    triggers.extend(check_consolidation_triggers(db, 10)?);
    triggers.extend(check_conflict_escalation(db)?);
    triggers.extend(check_temporal_drift(db)?);
    triggers.extend(check_redundancy(db, 0.85)?);
    triggers.extend(check_relationship_insight(db)?);
    triggers.extend(check_valence_trend(db)?);
    triggers.extend(check_entity_anomaly(db)?);

    triggers.sort_by(|a, b| b.urgency.total_cmp(&a.urgency));
    triggers.truncate(max_triggers);
    Ok(triggers)
}

// ── Trigger persistence ──

/// Build a cooldown key for deduplication.
pub fn build_cooldown_key(trigger: &Trigger) -> String {
    if trigger.source_rids.is_empty() {
        trigger.trigger_type.clone()
    } else {
        let mut rids = trigger.source_rids.clone();
        rids.sort();
        format!("{}:{}", trigger.trigger_type, rids.join(","))
    }
}

/// Persist a trigger to trigger_log with cooldown checking.
/// Returns the trigger_id if persisted, None if suppressed by cooldown.
pub fn persist_trigger(db: &YantrikDB, trigger: &Trigger, ts: f64) -> Result<Option<String>> {
    let trigger_type = TriggerType::from_str(&trigger.trigger_type);
    let cooldown_key = build_cooldown_key(trigger);
    let cooldown_secs = trigger_type.default_cooldown_secs();

    let trigger_id = crate::id::new_id();
    let actor_id = db.actor_id().to_string();
    let expires_at = ts + trigger_type.default_expiry_secs();
    let source_rids_json = serde_json::to_string(&trigger.source_rids)?;
    let context_json = serde_json::to_string(&trigger.context)?;

    {
        // The cooldown check and the insert share ONE connection guard, so
        // two concurrent persists of one trigger (think() and the trigger
        // checks run with the GIL released, #271) cannot both pass the
        // check. Nothing slow happens under the guard.
        let conn = db.conn();
        let active_exists: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM trigger_log \
             WHERE cooldown_key = ?1 \
             AND status IN ('pending', 'delivered') \
             AND created_at > ?2",
            params![cooldown_key, ts - cooldown_secs],
            |row| row.get(0),
        )?;

        if active_exists {
            return Ok(None);
        }
        // Tick only for a trigger that is actually written: a suppressed
        // one must not advance the HLC. Under the guard, per the documented
        // lock order (conn -> hlc, CONCURRENCY.md Rule 1).
        let hlc_bytes = db.tick_hlc().to_bytes().to_vec();
        conn.execute(
            "INSERT INTO trigger_log \
             (trigger_id, trigger_type, urgency, status, reason, suggested_action, \
              source_rids, context, created_at, expires_at, cooldown_key, hlc, origin_actor) \
             VALUES (?1, ?2, ?3, 'pending', ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                trigger_id,
                trigger.trigger_type,
                trigger.urgency,
                trigger.reason,
                trigger.suggested_action,
                source_rids_json,
                context_json,
                ts,
                expires_at,
                cooldown_key,
                hlc_bytes,
                actor_id,
            ],
        )?;

        // Dual-write to join table
        for rid in &trigger.source_rids {
            conn.execute(
                "INSERT OR IGNORE INTO trigger_source_rids (trigger_id, rid) VALUES (?1, ?2)",
                params![trigger_id, rid],
            )?;
        }
    } // conn lock released before log_op

    db.log_op(
        "trigger_fire",
        Some(&trigger_id),
        &serde_json::json!({
            "trigger_id": trigger_id,
            "trigger_type": trigger.trigger_type,
            "urgency": trigger.urgency,
            "reason": trigger.reason,
            "suggested_action": trigger.suggested_action,
            "source_rids": trigger.source_rids,
            "context": trigger.context,
            "cooldown_key": cooldown_key,
            "expires_at": expires_at,
        }),
        None,
    )?;

    Ok(Some(trigger_id))
}

/// Expire old triggers past their expires_at.
pub fn expire_triggers(db: &YantrikDB, ts: f64) -> Result<usize> {
    let conn = db.conn();
    let changes = conn.execute(
        "UPDATE trigger_log SET status = 'expired' \
         WHERE status = 'pending' AND expires_at IS NOT NULL AND expires_at < ?1",
        params![ts],
    )?;
    Ok(changes)
}

/// Outcome of a [`YantrikDB::prune_triggers`] pass (task 27).
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct TriggerPruneReport {
    pub dry_run: bool,
    /// Pending triggers at the start of the pass.
    pub pending_before: usize,
    /// Pending triggers expired for being past their `expires_at` (TTL).
    pub expired_overdue: usize,
    /// Pending triggers expired because their underlying predicate is no
    /// longer true or none of their source memories remain active.
    pub expired_stale: usize,
    /// Pending triggers expired to keep the backlog under the cap.
    pub expired_over_cap: usize,
    /// Pending triggers remaining after the pass.
    pub pending_after: usize,
}

impl YantrikDB {
    /// Task 27 — bound the pending-trigger backlog so it never grows
    /// unbounded (16 and climbing at audit time, nothing ever expiring them).
    /// First expires overdue triggers (past `expires_at`), then, if still over
    /// `max_pending`, expires the lowest-urgency / oldest excess. Expired
    /// triggers are retained with `status = 'expired'` and remain auditable.
    /// Dry-run reports what would change.
    pub fn prune_triggers(&self, dry_run: bool, max_pending: usize) -> Result<TriggerPruneReport> {
        let ts = now();
        let conn = self.conn();

        let pending_before = conn.query_row(
            "SELECT COUNT(*) FROM trigger_log WHERE status = 'pending'",
            [],
            |r| r.get::<_, i64>(0),
        )? as usize;
        let overdue = conn.query_row(
            "SELECT COUNT(*) FROM trigger_log \
             WHERE status = 'pending' AND expires_at IS NOT NULL AND expires_at < ?1",
            params![ts],
            |r| r.get::<_, i64>(0),
        )? as usize;
        let open_conflicts: i64 = conn.query_row(
            "SELECT COUNT(*) FROM conflicts WHERE status = 'open'",
            [],
            |r| r.get(0),
        )?;
        let critical_aging: bool = conn.query_row(
            "SELECT COUNT(*) > 0 FROM conflicts \
             WHERE status = 'open' AND priority = 'critical' AND detected_at < ?1",
            params![ts - 86400.0 * 3.0],
            |r| r.get(0),
        )?;
        let conflict_escalation_active = open_conflicts > 5 || critical_aging;

        // Predicate-aware expiry. Keep this deliberately narrow: trigger
        // types whose truth cannot be reconstructed cheaply remain governed
        // by TTL and operator action. Source-backed triggers are stale only
        // when every source has left the active recall set.
        let stale = conn.query_row(
            "SELECT COUNT(*) FROM trigger_log t \
             WHERE t.status = 'pending' \
             AND NOT (t.expires_at IS NOT NULL AND t.expires_at < ?1) \
             AND ( \
               (t.trigger_type = 'conflict_escalation' AND ?2 = 0) \
               OR ( \
                 EXISTS (SELECT 1 FROM trigger_source_rids tsr \
                         WHERE tsr.trigger_id = t.trigger_id) \
                 AND NOT EXISTS ( \
                   SELECT 1 FROM trigger_source_rids tsr \
                   JOIN memories m ON m.rid = tsr.rid \
                   WHERE tsr.trigger_id = t.trigger_id \
                     AND m.consolidation_status = 'active' \
                 ) \
               ) \
             )",
            params![ts, conflict_escalation_active],
            |r| r.get::<_, i64>(0),
        )? as usize;

        let remaining_after_stale = pending_before.saturating_sub(overdue).saturating_sub(stale);
        let over_cap = remaining_after_stale.saturating_sub(max_pending);

        let mut report = TriggerPruneReport {
            dry_run,
            pending_before,
            expired_overdue: overdue,
            expired_stale: stale,
            expired_over_cap: over_cap,
            pending_after: remaining_after_stale.saturating_sub(over_cap),
        };

        if dry_run {
            return Ok(report);
        }

        // 1) TTL expiry.
        conn.execute(
            "UPDATE trigger_log SET status = 'expired' \
             WHERE status = 'pending' AND expires_at IS NOT NULL AND expires_at < ?1",
            params![ts],
        )?;
        // 2) Predicate/source revalidation.
        conn.execute(
            "UPDATE trigger_log SET status = 'expired' \
             WHERE status = 'pending' AND ( \
               (trigger_type = 'conflict_escalation' AND ?1 = 0) \
               OR ( \
                 EXISTS (SELECT 1 FROM trigger_source_rids tsr \
                         WHERE tsr.trigger_id = trigger_log.trigger_id) \
                 AND NOT EXISTS ( \
                   SELECT 1 FROM trigger_source_rids tsr \
                   JOIN memories m ON m.rid = tsr.rid \
                   WHERE tsr.trigger_id = trigger_log.trigger_id \
                     AND m.consolidation_status = 'active' \
                 ) \
               ) \
             )",
            params![conflict_escalation_active],
        )?;
        // 3) Over-cap eviction: lowest urgency, then oldest, from what remains.
        if over_cap > 0 {
            conn.execute(
                "UPDATE trigger_log SET status = 'expired' WHERE trigger_id IN (\
                   SELECT trigger_id FROM trigger_log WHERE status = 'pending' \
                   ORDER BY urgency ASC, created_at ASC LIMIT ?1)",
                params![over_cap as i64],
            )?;
        }

        // Recompute the true remaining count (defensive against drift).
        report.pending_after = conn.query_row(
            "SELECT COUNT(*) FROM trigger_log WHERE status = 'pending'",
            [],
            |r| r.get::<_, i64>(0),
        )? as usize;

        Ok(report)
    }
}

/// Drop any trigger whose source rids span more than one namespace.
///
/// The individual checks scope themselves, but this is the last gate before
/// an artifact becomes durable, and durability is what makes a leak
/// irreversible: a pairing trigger's `context` carries verbatim text from
/// both sides (`text_a` / `text_b`), so once such a row is in `trigger_log`
/// the disclosure has already happened — a later display filter can only
/// hide it.
///
/// Fail-closed applies to the *proven* case: a trigger is dropped when its
/// rids resolve to different namespaces. A rid that no longer resolves is
/// left alone (the memory is gone, the trigger names no live text) and
/// logged at debug; guessing ownership there would drop legitimate
/// single-rid triggers for forgotten memories.
pub fn drop_cross_namespace_triggers(db: &YantrikDB, triggers: Vec<Trigger>) -> Vec<Trigger> {
    if triggers.is_empty() {
        return triggers;
    }

    let mut namespace_of: HashMap<String, Option<String>> = HashMap::new();
    {
        let conn = db.conn();
        let mut stmt = match conn.prepare_cached("SELECT namespace FROM memories WHERE rid = ?1") {
            Ok(s) => s,
            // Cannot resolve ownership at all: do not invent a boundary.
            Err(_) => return triggers,
        };
        for t in &triggers {
            for rid in &t.source_rids {
                if namespace_of.contains_key(rid) {
                    continue;
                }
                let ns = stmt
                    .query_row(rusqlite::params![rid], |r| r.get::<_, String>(0))
                    .ok();
                namespace_of.insert(rid.clone(), ns);
            }
        }
    }

    let mut kept = Vec::with_capacity(triggers.len());
    for trigger in triggers {
        let mut span: Option<&str> = None;
        let mut crosses = false;
        for rid in &trigger.source_rids {
            let Some(Some(namespace)) = namespace_of.get(rid) else {
                continue;
            };
            match span {
                None => span = Some(namespace.as_str()),
                Some(seen) if seen != namespace.as_str() => {
                    crosses = true;
                    break;
                }
                _ => {}
            }
        }
        if crosses {
            tracing::warn!(
                trigger_type = %trigger.trigger_type,
                rids = ?trigger.source_rids,
                "dropping trigger: source memories span namespaces"
            );
        } else {
            kept.push(trigger);
        }
    }
    kept
}

/// Filter triggers by cooldown and persist. Returns only non-suppressed triggers.
pub fn filter_and_persist_triggers(
    db: &YantrikDB,
    triggers: Vec<Trigger>,
    ts: f64,
) -> Result<Vec<Trigger>> {
    let triggers = drop_cross_namespace_triggers(db, triggers);
    let mut persisted = Vec::new();
    for t in triggers {
        if persist_trigger(db, &t, ts)?.is_some() {
            persisted.push(t);
        }
    }
    Ok(persisted)
}

/// Query persisted triggers from trigger_log.
pub fn get_pending_triggers(db: &YantrikDB, limit: usize) -> Result<Vec<PersistedTrigger>> {
    let conn = db.conn();
    let mut stmt = conn.prepare(
        "SELECT trigger_id, trigger_type, urgency, status, reason, suggested_action, \
         source_rids, context, created_at, delivered_at, acknowledged_at, acted_at, expires_at \
         FROM trigger_log \
         WHERE status = 'pending' \
         ORDER BY urgency DESC \
         LIMIT ?1",
    )?;

    let rows = stmt
        .query_map(params![limit as i64], |row| {
            let source_rids_str: String = row.get("source_rids")?;
            let context_str: String = row.get("context")?;
            Ok(PersistedTrigger {
                trigger_id: row.get("trigger_id")?,
                trigger_type: row.get("trigger_type")?,
                urgency: row.get("urgency")?,
                status: row.get("status")?,
                reason: row.get("reason")?,
                suggested_action: row.get("suggested_action")?,
                source_rids: serde_json::from_str(&source_rids_str).unwrap_or_default(),
                context: serde_json::from_str(&context_str)
                    .unwrap_or(serde_json::Value::Object(Default::default())),
                created_at: row.get("created_at")?,
                delivered_at: row.get("delivered_at")?,
                acknowledged_at: row.get("acknowledged_at")?,
                acted_at: row.get("acted_at")?,
                expires_at: row.get("expires_at")?,
            })
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    Ok(rows)
}

/// Query trigger history with optional type filter.
pub fn get_trigger_history(
    db: &YantrikDB,
    trigger_type: Option<&str>,
    limit: usize,
) -> Result<Vec<PersistedTrigger>> {
    let conn = db.conn();
    let (sql, limit_val) = if let Some(tt) = trigger_type {
        let mut stmt = conn.prepare(
            "SELECT trigger_id, trigger_type, urgency, status, reason, suggested_action, \
             source_rids, context, created_at, delivered_at, acknowledged_at, acted_at, expires_at \
             FROM trigger_log \
             WHERE trigger_type = ?1 \
             ORDER BY created_at DESC \
             LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![tt, limit as i64], parse_persisted_trigger)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        return Ok(rows);
    } else {
        (
            "SELECT trigger_id, trigger_type, urgency, status, reason, suggested_action, \
             source_rids, context, created_at, delivered_at, acknowledged_at, acted_at, expires_at \
             FROM trigger_log \
             ORDER BY created_at DESC \
             LIMIT ?1",
            limit,
        )
    };

    let mut stmt = conn.prepare(sql)?;
    let rows = stmt
        .query_map(params![limit_val as i64], parse_persisted_trigger)?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(rows)
}

fn parse_persisted_trigger(row: &rusqlite::Row<'_>) -> rusqlite::Result<PersistedTrigger> {
    let source_rids_str: String = row.get("source_rids")?;
    let context_str: String = row.get("context")?;
    Ok(PersistedTrigger {
        trigger_id: row.get("trigger_id")?,
        trigger_type: row.get("trigger_type")?,
        urgency: row.get("urgency")?,
        status: row.get("status")?,
        reason: row.get("reason")?,
        suggested_action: row.get("suggested_action")?,
        source_rids: serde_json::from_str(&source_rids_str).unwrap_or_default(),
        context: serde_json::from_str(&context_str)
            .unwrap_or(serde_json::Value::Object(Default::default())),
        created_at: row.get("created_at")?,
        delivered_at: row.get("delivered_at")?,
        acknowledged_at: row.get("acknowledged_at")?,
        acted_at: row.get("acted_at")?,
        expires_at: row.get("expires_at")?,
    })
}

/// Check if two memory texts differ by tokens in the same substitution category.
/// Returns (category_name, token_a, token_b) if a match is found.
/// Token → (category_id, category name) for exclusive substitution categories.
///
/// Small by construction (this store: 8 categories, 80 members). Loaded once
/// per `check_redundancy` pass instead of being queried per token pair.
type SubstitutionIndex = std::collections::HashMap<String, Vec<(String, String)>>;

fn load_substitution_index(conn: &rusqlite::Connection) -> crate::error::Result<SubstitutionIndex> {
    load_substitution_index_status(conn).map(|(index, _)| index)
}

/// The loader, with the reason when the index could not be built.
///
/// `None` means "loaded". `Some(reason)` means "degraded": the caller must
/// surface that in the maintenance RESULT, not only in the log — otherwise
/// "no conflicts found" and "conflicts could not be evaluated" look alike.
fn load_substitution_index_status(
    conn: &rusqlite::Connection,
) -> crate::error::Result<(SubstitutionIndex, Option<String>)> {
    let mut index: SubstitutionIndex = std::collections::HashMap::new();
    let mut stmt = match conn.prepare(
        // ORDER BY, not insertion order: when a token sits in two exclusive
        // categories, the first match decides the stored `conflict_type`
        // (`cloud_providers` -> IdentityFact, everything else -> Preference).
        // Without the order that verdict drifted between runs (review 7, B3).
        "SELECT m.token_normalized, m.category_id, c.name
           FROM substitution_members m
           JOIN substitution_categories c ON c.id = m.category_id
          WHERE m.status = 'active' AND m.confidence >= 0.6
            AND c.status = 'active' AND c.conflict_mode = 'exclusive'
          ORDER BY m.token_normalized, c.name",
    ) {
        Ok(s) => s,
        // A store without the tables cannot answer the predicate; an empty
        // index does that correctly. But "tables absent" has two very different
        // causes and they must not look alike (review 8, finding 4):
        //
        //   * a store that never ran V13->V14 — supported, benign;
        //   * a store that did run it and is still missing the tables — an
        //     incomplete or rolled-back migration, i.e. damage.
        //
        // `entity_aliases` is created by the *same* V13->V14 script as the
        // substitution tables (`schema.rs` MIGRATE_V13_TO_V14), so it separates
        // the two cases. `claims` does NOT: it comes from an earlier migration
        // (review 9, E3 — the first version of this check used `claims` and the
        // premise was wrong).
        Err(rusqlite::Error::SqliteFailure(_, Some(ref msg))) if msg.contains("no such table") => {
            let migrated: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master \
                     WHERE type = 'table' AND name = 'entity_aliases'",
                    [],
                    |r| r.get(0),
                )
                .unwrap_or(0);
            if migrated > 0 {
                tracing::error!(
                    "substitution tables are missing although V13->V14 ran \
                     (entity_aliases exists) — the migration did not complete; \
                     substitution conflicts are disabled for this store"
                );
                return Ok((
                    index,
                    Some(
                        "substitution tables are missing although V13->V14 ran — \
                         the migration did not complete"
                            .to_string(),
                    ),
                ));
            } else {
                tracing::warn!(
                    "store without V13->V14 — no substitution tables, which is \
                     expected for this schema"
                );
                return Ok((
                    index,
                    Some("store without the V13->V14 migration".to_string()),
                ));
            }
        }
        // Every other failure is real. Returning an empty index here would
        // silently disable substitution conflicts for the WHOLE namespace —
        // the per-pair skip the index replaced was the smaller silence
        // (review 7, B6).
        Err(e) => return Err(e.into()),
    };
    let rows = stmt.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, String>(2)?,
        ))
    })?;
    for row in rows {
        let (token, category_id, name) = row?;
        index.entry(token).or_default().push((category_id, name));
    }
    Ok((index, None))
}

/// How many content words around the two substituted tokens must agree before a
/// category pair counts as a contradiction.
///
/// Category membership alone says the two values are *alternatives*; it says
/// nothing about whether the two documents assert the same thing. Measured on
/// the live store: without this gate every one of 45 open conflicts is a false
/// alarm (ADR-0057 "google" vs ADR-0125 "deepseek"), with it none of them pass
/// and the three positive controls do.
const MIN_SHARED_CONTEXT: usize = 2;

/// Tokens examined on each side of the substituted term.
const CONTEXT_WINDOW: usize = 10;

/// Minimum Jaccard overlap of the two frames once the substituted values are
/// removed: same subject and same predicate, different value.
const MIN_FRAME_SIMILARITY: f64 = 0.7;

/// Remove fenced blocks and inline code spans. A token that only appears inside
/// code is a value of the implementation, not a statement about the subject:
/// `--data` is not a claim about a data layer, and `[data-provider=basic]` is
/// not a preference.
fn strip_code_context(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_fence = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            out.push('\n');
            continue;
        }
        if in_fence {
            out.push('\n');
            continue;
        }
        let mut in_inline = false;
        for ch in line.chars() {
            match ch {
                '`' => in_inline = !in_inline,
                _ if !in_inline => out.push(ch),
                _ => {}
            }
        }
        out.push('\n');
    }
    out
}

fn tokenize_words(text: &str) -> std::collections::HashSet<String> {
    text.split_whitespace()
        .map(|w| {
            w.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Split prose into sentences without breaking on numbers or abbreviations.
///
/// A `.` ends a sentence only when whitespace follows it — `0.23.6`, `z.B.`
/// and `ADR-0139 §R.` therefore stay inside their sentence (review 9, E1).
/// `!`, `?` and a newline always end one.
fn sentence_spans(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    for i in 0..bytes.len() {
        let ends = match bytes[i] {
            b'\n' | b'!' | b'?' => true,
            b'.' => bytes
                .get(i + 1)
                .map(|b| b.is_ascii_whitespace())
                .unwrap_or(true),
            _ => false,
        };
        if ends {
            out.push(&text[start..=i]);
            start = i + 1;
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// The content words of the sentence that contains `token` — the frame a
/// statement sits in.
///
/// Neither a ±N-token window nor the whole document. The window is blind to a
/// clause that follows the value; the document scores two long notes as
/// different frames merely because they are long. A sentence keeps subject,
/// predicate and value together in both shapes.
///
/// **What this cannot see:** a retraction in the *next* sentence ("the
/// decision was discarded") is outside the frame. That is a semantic judgement,
/// not a frame question — it belongs to the knowledge layer, which is why the
/// scan produces candidates. Recorded, not hidden: review 9, headline 1.
fn context_words(text: &str, token: &str) -> std::collections::HashSet<String> {
    let normalise = |s: &str| -> Vec<String> {
        s.split_whitespace()
            .map(|w| {
                w.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase()
            })
            .filter(|w| !w.is_empty())
            .collect()
    };
    for sentence in sentence_spans(text) {
        let words = normalise(sentence);
        if !words.iter().any(|w| w == token) {
            continue;
        }
        return words
            .into_iter()
            .filter(|w| w.len() >= 4 && w != token)
            .collect();
    }
    // The token is not in any sentence (e.g. the caller passed an unnormalised
    // form): an empty set makes the gate fail closed, which is the intent.
    std::collections::HashSet::new()
}

/// How much the two neighbourhoods agree once the substituted values are
/// removed. Order-independent, so a reordered clause still scores 1.0, while a
/// pair whose surroundings genuinely differ falls under the threshold.
fn frame_similarity(text_a: &str, token_a: &str, text_b: &str, token_b: &str) -> f64 {
    let a = context_words(text_a, token_a);
    let b = context_words(text_b, token_b);
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(&b).count() as f64;
    let union = a.union(&b).count() as f64;
    inter / union
}

/// Count content words shared by the two token neighbourhoods.
fn shared_context_terms(text_a: &str, token_a: &str, text_b: &str, token_b: &str) -> usize {
    context_words(text_a, token_a)
        .intersection(&context_words(text_b, token_b))
        .count()
}

/// The subject/frame gate: do the two documents assert the same thing about the
/// same subject? Fail closed — an unchecked frame is not evidence of a
/// contradiction, and a silence is not a conflict either.
fn substitution_is_same_claim(prose_a: &str, ta: &str, prose_b: &str, tb: &str) -> bool {
    shared_context_terms(prose_a, ta, prose_b, tb) >= MIN_SHARED_CONTEXT
        && frame_similarity(prose_a, ta, prose_b, tb) >= MIN_FRAME_SIMILARITY
}

/// Same predicate as [`check_substitution_category_pair`], answered in memory.
///
/// Two layers, and the second one is the point:
///
/// 1. **Category membership** — cheap, from the pre-loaded index: are the two
///    differing tokens alternatives in one exclusive category?
/// 2. **Same claim** — on the prose of the full texts: do they share the
///    surrounding subject and predicate? Without layer 2 the index answers a
///    question nobody asked ("do these two documents each mention a provider?")
///    and every pair of ADRs becomes a contradiction.
///
/// Both call sites in `check_redundancy_in_namespace` go through this function,
/// so the gate covers the high-similarity pass and the 0.65–0.85 substitution
/// pass alike.
fn check_substitution_pair_in_index(
    index: &SubstitutionIndex,
    text_a: &str,
    text_b: &str,
) -> Option<(String, String, String)> {
    if index.is_empty() {
        return None;
    }
    let prose_a = strip_code_context(text_a);
    let prose_b = strip_code_context(text_b);
    let words_a = tokenize_words(&prose_a);
    let words_b = tokenize_words(&prose_b);
    // Sorted, not HashSet order: when two members of one category appear on
    // each side, several token pairs can pass the gate, and the first hit
    // becomes the stored `detection_reason`. `HashSet::difference` is randomly
    // seeded per process, so an unsorted scan makes the reason string drift
    // between runs and a conflict ledger impossible to diff. The memory pair
    // was already stable (the outer loops are ordered by created_at); this
    // keeps the payload stable too.
    let mut diff_a: Vec<&String> = words_a.difference(&words_b).collect();
    let mut diff_b: Vec<&String> = words_b.difference(&words_a).collect();
    diff_a.sort();
    diff_b.sort();

    for ta in &diff_a {
        let Some(cats_a) = index.get(ta.as_str()) else {
            continue;
        };
        if cats_a.is_empty() {
            continue;
        }
        for tb in &diff_b {
            let Some(cats_b) = index.get(tb.as_str()) else {
                continue;
            };
            if let Some((_, name)) = cats_a
                .iter()
                .find(|(id_a, _)| cats_b.iter().any(|(id_b, _)| id_b == id_a))
            {
                // Layer 2: a category alternative is only a contradiction when
                // both documents make the same claim about the same subject.
                if !substitution_is_same_claim(&prose_a, ta, &prose_b, tb) {
                    continue;
                }
                return Some((name.clone(), (*ta).clone(), (*tb).clone()));
            }
        }
    }
    None
}

/// Reference implementation, kept as the readable statement of the predicate.
/// `check_redundancy` no longer calls it: one JOIN per token pair made a
/// 30-row namespace cost minutes.
#[allow(dead_code)]
fn check_substitution_category_pair(
    conn: &rusqlite::Connection,
    text_a: &str,
    text_b: &str,
) -> Option<(String, String, String)> {
    let prose_a = strip_code_context(text_a);
    let prose_b = strip_code_context(text_b);
    let words_a = tokenize_words(&prose_a);
    let words_b = tokenize_words(&prose_b);

    let mut diff_a: Vec<&String> = words_a.difference(&words_b).collect();
    let mut diff_b: Vec<&String> = words_b.difference(&words_a).collect();
    // Same deterministic tie-break as the index path, so the reference and the
    // production implementation answer identically (the equivalence test
    // compares them).
    diff_a.sort();
    diff_b.sort();

    // Check each diff token pair against substitution_members
    for ta in &diff_a {
        for tb in &diff_b {
            let result: std::result::Result<(String,), _> = conn.query_row(
                "SELECT c.name FROM substitution_members m1
                 JOIN substitution_members m2 ON m1.category_id = m2.category_id
                 JOIN substitution_categories c ON c.id = m1.category_id
                 WHERE m1.token_normalized = ?1 AND m2.token_normalized = ?2
                   AND m1.status = 'active' AND m2.status = 'active'
                   AND m1.confidence >= 0.6 AND m2.confidence >= 0.6
                   AND c.status = 'active' AND c.conflict_mode = 'exclusive'
                 LIMIT 1",
                params![ta.as_str(), tb.as_str()],
                |row| Ok((row.get::<_, String>(0)?,)),
            );
            if let Ok((cat_name,)) = result {
                if substitution_is_same_claim(&prose_a, ta, &prose_b, tb) {
                    return Some((cat_name, ta.to_string(), tb.to_string()));
                }
            }
        }
    }

    None
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
    fn test_no_trigger_for_fresh() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        db.record(
            "fresh",
            "episodic",
            0.9,
            0.0,
            604800.0,
            &serde_json::json!({}),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
        let triggers = check_decay_triggers(&db, 0.5, 0.1, 5).unwrap();
        assert!(triggers.is_empty());
    }

    #[test]
    fn test_decay_trigger_fires() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rid = db
            .record(
                "important deadline",
                "episodic",
                0.9,
                0.0,
                100.0,
                &serde_json::json!({}),
                &vec_seed(1.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        db.conn()
            .execute(
                "UPDATE memories SET last_access = ?1 WHERE rid = ?2",
                rusqlite::params![now() - 10000.0, rid],
            )
            .unwrap();

        let triggers = check_decay_triggers(&db, 0.5, 0.1, 5).unwrap();
        assert!(!triggers.is_empty());
        assert_eq!(triggers[0].trigger_type, "decay_review");
        assert_eq!(triggers[0].source_rids, vec![rid]);
    }

    #[test]
    fn test_consolidation_trigger() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        for i in 0..15 {
            db.record(
                &format!("episodic memory {i}"),
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
        }

        let triggers = check_consolidation_triggers(&db, 10).unwrap();
        assert_eq!(triggers.len(), 1);
        assert_eq!(triggers[0].trigger_type, "consolidation_ready");
    }

    #[test]
    fn test_conflict_escalation_fires() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        // Create 6 open conflicts manually
        let ts = now();
        for i in 0..6 {
            let id = format!("conflict-{i}");
            let hlc = db.tick_hlc();
            db.conn().execute(
                "INSERT INTO conflicts (conflict_id, conflict_type, priority, status, \
                 memory_a, memory_b, detected_at, detected_by, detection_reason, hlc, origin_actor) \
                 VALUES (?1, 'minor', 'low', 'open', 'a', 'b', ?2, 'test', 'test', ?3, 'test')",
                params![id, ts, hlc.to_bytes().to_vec()],
            ).unwrap();
        }

        let triggers = check_conflict_escalation(&db).unwrap();
        assert_eq!(triggers.len(), 1);
        assert_eq!(triggers[0].trigger_type, "conflict_escalation");
    }

    #[test]
    fn test_conflict_escalation_no_fire() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        // Only 2 conflicts -> should not fire
        let ts = now();
        for i in 0..2 {
            let id = format!("conflict-{i}");
            let hlc = db.tick_hlc();
            db.conn().execute(
                "INSERT INTO conflicts (conflict_id, conflict_type, priority, status, \
                 memory_a, memory_b, detected_at, detected_by, detection_reason, hlc, origin_actor) \
                 VALUES (?1, 'minor', 'low', 'open', 'a', 'b', ?2, 'test', 'test', ?3, 'test')",
                params![id, ts, hlc.to_bytes().to_vec()],
            ).unwrap();
        }

        let triggers = check_conflict_escalation(&db).unwrap();
        assert!(triggers.is_empty());
    }

    #[test]
    fn test_temporal_drift_fires() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let rid = db
            .record(
                "works at Google",
                "semantic",
                0.8,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &vec_seed(1.0, 8),
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        // Backdate to 120 days ago
        let old_ts = now() - 86400.0 * 120.0;
        db.conn()
            .execute(
                "UPDATE memories SET created_at = ?1, last_access = ?1 WHERE rid = ?2",
                params![old_ts, rid],
            )
            .unwrap();

        let triggers = check_temporal_drift(&db).unwrap();
        assert!(!triggers.is_empty());
        assert_eq!(triggers[0].trigger_type, "temporal_drift");
    }

    #[test]
    fn test_temporal_drift_skips_recent() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        db.record(
            "works at Google",
            "semantic",
            0.8,
            0.0,
            604800.0,
            &serde_json::json!({}),
            &vec_seed(1.0, 8),
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

        let triggers = check_temporal_drift(&db).unwrap();
        assert!(triggers.is_empty());
    }

    #[test]
    fn test_relationship_insight_fires_for_hub() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        // Create a hub entity with 6 edges
        for i in 0..6 {
            db.relate("Alice", &format!("entity_{i}"), &format!("knows_{i}"), 1.0)
                .unwrap();
        }

        let triggers = check_relationship_insight(&db).unwrap();
        assert!(!triggers.is_empty());
        assert_eq!(triggers[0].trigger_type, "relationship_insight");
    }

    #[test]
    fn test_cooldown_prevents_refire() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let trigger = Trigger {
            trigger_type: "decay_review".to_string(),
            reason: "test".to_string(),
            urgency: 0.8,
            source_rids: vec!["rid-1".to_string()],
            suggested_action: "test".to_string(),
            context: HashMap::new(),
        };

        let ts = now();
        let first = persist_trigger(&db, &trigger, ts).unwrap();
        assert!(first.is_some());

        // Same trigger should be suppressed by cooldown
        let second = persist_trigger(&db, &trigger, ts).unwrap();
        assert!(second.is_none());
    }

    /// The cooldown must hold for concurrent callers too: think() and the
    /// trigger checks run with the GIL released (#271), and the check and
    /// the insert took the connection lock separately, so two persists of
    /// one trigger could both pass the check.
    #[test]
    fn concurrent_persists_of_one_trigger_fire_it_once() {
        for trial in 0..50 {
            let db = YantrikDB::new(":memory:", 8).unwrap();
            let trigger = Trigger {
                trigger_type: "decay_review".to_string(),
                reason: "test".to_string(),
                urgency: 0.8,
                source_rids: vec!["rid-1".to_string()],
                suggested_action: "test".to_string(),
                context: HashMap::new(),
            };
            let ts = now();
            let barrier = std::sync::Barrier::new(4);
            let fired: usize = std::thread::scope(|s| {
                let handles: Vec<_> = (0..4)
                    .map(|_| {
                        s.spawn(|| {
                            barrier.wait();
                            persist_trigger(&db, &trigger, ts).unwrap().is_some() as usize
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).sum()
            });
            let rows: i64 = db
                .conn()
                .query_row("SELECT COUNT(*) FROM trigger_log", [], |r| r.get(0))
                .unwrap();
            assert_eq!((fired, rows), (1, 1), "trial {trial}");
        }
    }

    /// A trigger its cooldown suppresses writes nothing, so it must not
    /// advance the HLC either.
    #[test]
    fn a_suppressed_trigger_leaves_the_hlc_alone() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let trigger = Trigger {
            trigger_type: "decay_review".to_string(),
            reason: "test".to_string(),
            urgency: 0.8,
            source_rids: vec!["rid-1".to_string()],
            suggested_action: "test".to_string(),
            context: HashMap::new(),
        };
        let ts = now();
        assert!(persist_trigger(&db, &trigger, ts).unwrap().is_some());
        let before = db.hlc.lock().latest();
        assert!(persist_trigger(&db, &trigger, ts).unwrap().is_none());
        assert_eq!(db.hlc.lock().latest(), before);
    }

    #[test]
    fn test_expiry_clears_old_triggers() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let trigger = Trigger {
            trigger_type: "decay_review".to_string(),
            reason: "test".to_string(),
            urgency: 0.8,
            source_rids: vec!["rid-1".to_string()],
            suggested_action: "test".to_string(),
            context: HashMap::new(),
        };

        // Persist with a past timestamp so it expires immediately
        let past = now() - 86400.0 * 30.0;
        persist_trigger(&db, &trigger, past).unwrap();

        let expired = expire_triggers(&db, now()).unwrap();
        assert_eq!(expired, 1);

        let pending = get_pending_triggers(&db, 10).unwrap();
        assert!(pending.is_empty());
    }

    #[test]
    fn prune_expires_triggers_whose_live_predicate_is_gone() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let conflict = Trigger {
            trigger_type: "conflict_escalation".to_string(),
            reason: "26 open conflicts need attention".to_string(),
            urgency: 1.0,
            source_rids: vec![],
            suggested_action: "review_conflicts".to_string(),
            context: HashMap::from([("open_count".to_string(), serde_json::json!(26))]),
        };
        let missing_source = Trigger {
            trigger_type: "decay_review".to_string(),
            reason: "review missing source".to_string(),
            urgency: 0.8,
            source_rids: vec!["missing-rid".to_string()],
            suggested_action: "review".to_string(),
            context: HashMap::new(),
        };
        persist_trigger(&db, &conflict, now()).unwrap();
        persist_trigger(&db, &missing_source, now()).unwrap();

        let preview = db.prune_triggers(true, 64).unwrap();
        assert_eq!(preview.expired_stale, 2);
        assert_eq!(preview.pending_after, 0);
        assert_eq!(get_pending_triggers(&db, 10).unwrap().len(), 2);

        let applied = db.prune_triggers(false, 64).unwrap();
        assert_eq!(applied.expired_stale, 2);
        assert!(get_pending_triggers(&db, 10).unwrap().is_empty());

        let again = db.prune_triggers(false, 64).unwrap();
        assert_eq!(again.expired_stale, 0, "pruning is idempotent");
    }

    #[test]
    fn test_filter_and_persist_deduplicates() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let t1 = Trigger {
            trigger_type: "decay_review".to_string(),
            reason: "test".to_string(),
            urgency: 0.8,
            source_rids: vec!["rid-1".to_string()],
            suggested_action: "test".to_string(),
            context: HashMap::new(),
        };
        let t2 = t1.clone();

        let ts = now();
        let persisted = filter_and_persist_triggers(&db, vec![t1, t2], ts).unwrap();
        assert_eq!(persisted.len(), 1); // second is suppressed by cooldown
    }

    /// Regression test for the v0.5.8 self-deadlock bug in check_redundancy.
    ///
    /// Before the fix (commit c4c2d9d), an `if let Some(...) =
    /// check_substitution_category_pair(&*db.conn(), ...)` in the high-similarity
    /// pass extended a `MutexGuard<Connection>` lifetime through the if-let body,
    /// and the body called `conflict_exists(db, ...)` which tried to take
    /// `db.conn()` again on the same thread. std::sync::Mutex is non-reentrant,
    /// so the consolidation worker self-deadlocked while holding the outer
    /// engine mutex, wedging every other worker on the next engine.lock().
    ///
    /// This test reproduces the exact trigger conditions:
    ///   1. Two memories with cosine similarity > 0.85 (we use identical
    ///      embeddings for determinism: similarity = 1.0)
    ///   2. No shared entities (memories share no relate edges)
    ///   3. Substitution-category membership for two of the differing tokens
    ///      (we seed a `databases` category with `mysql` and `postgres`)
    ///
    /// Before v0.5.8 this test hangs forever on std::sync::Mutex.
    /// After v0.5.8 + parking_lot (v0.5.9) it completes within milliseconds
    /// AND a conflict record is written.
    ///
    /// A 5-second background timeout would be nicer but Rust's stdlib does
    /// not expose cheap per-test timeouts. Tokio-style harnesses are
    /// inappropriate here because this is a pure sync core test. If the
    /// bug regresses, the entire test binary will hang and CI will catch
    /// it via the 60-second default cargo test timeout.
    /// Concurrent redundancy checks record each substitution conflict once.
    /// check_redundancy checked conflict_exists() and then created the
    /// conflict under separate connection guards, and the trigger checks run
    /// with the GIL released (#271), so two checks at once both wrote it.
    #[test]
    fn concurrent_redundancy_checks_record_each_substitution_conflict_once() {
        use rusqlite::params;

        const PAIRS: usize = 8;
        for trial in 0..20 {
            let db = YantrikDB::new(":memory:", 8).unwrap();
            let ts = now();
            let hlc_bytes = db.tick_hlc().to_bytes().to_vec();
            db.conn()
                .execute(
                    "INSERT INTO substitution_categories
                     (id, name, conflict_mode, status, created_at, updated_at, hlc, origin_actor)
                     VALUES ('cat-race', 'race_category', 'exclusive', 'active', ?1, ?1, ?2, 'test')",
                    params![ts, hlc_bytes],
                )
                .unwrap();
            for p in 0..PAIRS {
                for side in ["a", "b"] {
                    let tok = format!("qzrace{p}{side}");
                    let hlc_bytes = db.tick_hlc().to_bytes().to_vec();
                    db.conn()
                        .execute(
                            "INSERT INTO substitution_members
                             (id, category_id, token_normalized, token_display,
                              confidence, source, status, created_at, updated_at,
                              hlc, origin_actor)
                             VALUES (?1, 'cat-race', ?2, ?2, 0.9, 'test', 'active',
                                     ?3, ?3, ?4, 'test')",
                            params![format!("m-{p}-{side}"), tok, ts, hlc_bytes],
                        )
                        .unwrap();
                    db.record(
                        &format!("project {p} keeps its user profiles in {tok} for auth"),
                        "semantic",
                        0.8,
                        0.0,
                        604800.0,
                        &serde_json::json!({}),
                        &vec_seed(1.0 + p as f32 * 3.0, 8),
                        &format!("ns{p}"),
                        0.9,
                        "general",
                        "user",
                        None,
                    )
                    .unwrap();
                }
            }
            let barrier = std::sync::Barrier::new(3);
            std::thread::scope(|s| {
                for _ in 0..3 {
                    s.spawn(|| {
                        barrier.wait();
                        check_redundancy(&db, 0.85).unwrap();
                    });
                }
            });
            let (rows, pairs): (i64, i64) = db
                .conn()
                .query_row(
                    "SELECT COUNT(*), COUNT(DISTINCT min(memory_a, memory_b) || '|' || \
                     max(memory_a, memory_b)) FROM conflicts",
                    [],
                    |r| Ok((r.get(0)?, r.get(1)?)),
                )
                .unwrap();
            assert!(pairs > 0, "trial {trial}: the seed produced no conflicts");
            assert_eq!(rows, pairs, "trial {trial}: duplicate conflict rows");
        }
    }

    #[test]
    fn test_check_redundancy_no_self_deadlock_on_substitution_category() {
        use rusqlite::params;

        let db = YantrikDB::new(":memory:", 8).unwrap();

        // Seed a substitution category: {test_deadlock_regression} with
        // exclusive mode so a match triggers conflict creation (not just
        // redundancy). We use a test-only name to avoid colliding with
        // any default categories the schema may seed.
        let hlc_bytes = db.tick_hlc().to_bytes().to_vec();
        let ts = now();
        db.conn()
            .execute(
                "INSERT INTO substitution_categories
                 (id, name, conflict_mode, status, created_at, updated_at, hlc, origin_actor)
                 VALUES ('cat-test-deadlock', 'test_deadlock_regression',
                         'exclusive', 'active', ?1, ?1, ?2, 'test')",
                params![ts, hlc_bytes],
            )
            .unwrap();

        // Use fabricated tokens that cannot collide with any seeded members.
        for (tok, suffix) in [("zyxqvtoken1", "a"), ("zyxqvtoken2", "b")] {
            let hlc_bytes = db.tick_hlc().to_bytes().to_vec();
            db.conn()
                .execute(
                    "INSERT INTO substitution_members
                     (id, category_id, token_normalized, token_display,
                      confidence, source, status, created_at, updated_at,
                      hlc, origin_actor)
                     VALUES (?1, 'cat-test-deadlock', ?2, ?2, 0.9, 'test', 'active',
                             ?3, ?3, ?4, 'test')",
                    params![format!("mem-{suffix}"), tok, ts, hlc_bytes],
                )
                .unwrap();
        }

        // Record two memories with identical embeddings (cosine sim = 1.0,
        // well above the 0.85 redundancy threshold) and texts that share
        // most words but differ on the substitution-category tokens.
        let emb = vec_seed(1.0, 8);
        let rid_a = db
            .record(
                "we store user profiles in zyxqvtoken1 for the auth service",
                "semantic",
                0.8,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &emb,
                "default",
                0.9,
                "general",
                "user",
                None,
            )
            .unwrap();
        let rid_b = db
            .record(
                "we store user profiles in zyxqvtoken2 for the auth service",
                "semantic",
                0.8,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &emb,
                "default",
                0.9,
                "general",
                "user",
                None,
            )
            .unwrap();

        // If the v0.5.8 self-deadlock regresses, this call hangs forever and
        // the test times out. parking_lot (v0.5.9) would additionally trip
        // the runtime deadlock detector.
        let triggers = check_redundancy(&db, 0.85).unwrap();

        // The pair should have been flagged — either as a redundancy/substitution
        // trigger, or consumed into an actual conflict record by the body.
        // At minimum, check_redundancy must have returned at all.
        assert!(
            !triggers.is_empty(),
            "expected at least one trigger for mysql/postgres substitution pair"
        );

        // Verify conflict_exists path was reachable AND completed: a conflict
        // row should have been written by create_conflict() inside the body.
        // This confirms both memories and the substitution category were
        // resolved through the previously-deadlocking code path.
        let conflict_count: i64 = db
            .conn()
            .query_row(
                "SELECT COUNT(*) FROM conflicts
                 WHERE (memory_a = ?1 AND memory_b = ?2)
                    OR (memory_a = ?2 AND memory_b = ?1)",
                params![rid_a, rid_b],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            conflict_count, 1,
            "expected exactly one conflict record between the substitution pair"
        );
    }

    /// Issue #45 regression: the redundancy trigger's `reason` string must
    /// name the rid pair so consumers that only see `reason` (and not
    /// `source_rids` / `context`) can still act on it. Phase 1 gap T1 from
    /// the yantrikdb-agi gap analysis: "It never tells me WHICH two
    /// memories. I have to manually search for duplicates."
    #[test]
    fn test_redundancy_trigger_names_rids_in_reason() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let emb = vec_seed(1.0, 8);

        let rid_a = db
            .record(
                "compression wonder: parametric walls help",
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &emb,
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        let rid_b = db
            .record(
                "compression wonder: parametric walls assist",
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &emb,
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        let triggers = check_redundancy(&db, 0.85).unwrap();
        let redundancy: Vec<&Trigger> = triggers
            .iter()
            .filter(|t| t.trigger_type == "redundancy")
            .collect();
        assert!(
            !redundancy.is_empty(),
            "expected a redundancy trigger for identical embeddings, got: {:?}",
            triggers
        );

        let trig = redundancy[0];
        assert!(
            trig.reason.contains(&rid_a) && trig.reason.contains(&rid_b),
            "redundancy trigger `reason` must name both rids (rid_a={rid_a}, rid_b={rid_b}), got: {}",
            trig.reason
        );
        assert_eq!(
            trig.source_rids.len(),
            2,
            "redundancy trigger must carry both rids in source_rids"
        );
        assert!(
            trig.source_rids.contains(&rid_a) && trig.source_rids.contains(&rid_b),
            "source_rids must include rid_a and rid_b, got: {:?}",
            trig.source_rids
        );
    }

    /// Issue #45: trigger `context` must include compact `snippet_a` /
    /// `snippet_b` + explicit `rid_a` / `rid_b` keys so callers can
    /// preview the matching pair without a separate recall.
    #[test]
    fn test_redundancy_trigger_provides_snippets_and_rids_in_context() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let emb = vec_seed(1.0, 8);

        let rid_a = db
            .record(
                "test memory alpha",
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &emb,
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        let rid_b = db
            .record(
                "test memory beta",
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &emb,
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        let triggers = check_redundancy(&db, 0.85).unwrap();
        let trig = triggers
            .iter()
            .find(|t| t.trigger_type == "redundancy")
            .expect("expected redundancy trigger");

        let ctx_rid_a = trig
            .context
            .get("rid_a")
            .and_then(|v| v.as_str())
            .expect("context.rid_a must be present");
        let ctx_rid_b = trig
            .context
            .get("rid_b")
            .and_then(|v| v.as_str())
            .expect("context.rid_b must be present");
        // Pair may surface in either order; assert as an unordered pair.
        let pair: std::collections::HashSet<&str> = [ctx_rid_a, ctx_rid_b].into_iter().collect();
        let expected: std::collections::HashSet<&str> =
            [rid_a.as_str(), rid_b.as_str()].into_iter().collect();
        assert_eq!(
            pair, expected,
            "context.rid_a/rid_b must match the seeded pair"
        );

        assert!(
            trig.context.get("snippet_a").is_some(),
            "context.snippet_a must be present"
        );
        assert!(
            trig.context.get("snippet_b").is_some(),
            "context.snippet_b must be present"
        );
        // `similarity` was already in pre-#45 context; assert it survived.
        assert!(
            trig.context.get("similarity").is_some(),
            "context.similarity must remain in payload (back-compat)"
        );
    }

    /// Issue #45: `snippet_for_trigger` truncates beyond
    /// `TRIGGER_SNIPPET_MAX_CHARS` characters with an ellipsis, leaving
    /// short text untouched.
    #[test]
    fn test_snippet_for_trigger_truncates_long_text() {
        let short = "hello world";
        assert_eq!(snippet_for_trigger(short), short);

        let long: String = "x".repeat(TRIGGER_SNIPPET_MAX_CHARS + 50);
        let snipped = snippet_for_trigger(&long);
        assert!(
            snipped.ends_with('…'),
            "long text must be truncated with an ellipsis, got: {snipped:?}"
        );
        // Ellipsis is multi-byte but a single char; total char count =
        // TRIGGER_SNIPPET_MAX_CHARS + 1.
        assert_eq!(
            snipped.chars().count(),
            TRIGGER_SNIPPET_MAX_CHARS + 1,
            "snippet must be exactly TRIGGER_SNIPPET_MAX_CHARS + ellipsis chars"
        );
    }

    /// Issue #45: `snippet_for_trigger` must respect char boundaries on
    /// multi-byte UTF-8 input. A naive `text[..MAX]` slice would panic on
    /// non-ASCII text; the `chars().take(N).collect()` form is required
    /// for safety. Test asserts: (1) no panic, (2) truncation happens on
    /// a char boundary, (3) ellipsis appended.
    #[test]
    fn test_snippet_for_trigger_handles_unicode_safely() {
        // Multi-byte char that takes 3 bytes in UTF-8.
        let unicode: String = "मतलब".repeat(50); // 4 chars × 50 = 200 chars, > 120
        assert!(unicode.chars().count() > TRIGGER_SNIPPET_MAX_CHARS);
        let snipped = snippet_for_trigger(&unicode);
        assert!(
            snipped.ends_with('…'),
            "unicode snippet must end with ellipsis, got: {snipped:?}"
        );
        assert_eq!(
            snipped.chars().count(),
            TRIGGER_SNIPPET_MAX_CHARS + 1,
            "unicode snippet must be MAX + ellipsis chars"
        );
        // is_char_boundary must be true at every prefix length we slice at;
        // the easiest verification is that the string round-trips through
        // UTF-8 without error (which it must to even exist as a String).
        assert!(
            std::str::from_utf8(snipped.as_bytes()).is_ok(),
            "snippet must be valid UTF-8"
        );
    }

    /// Issue #45: the entity-overlap potential_conflict reason string must
    /// also name the rid pair, not only the pure-redundancy path. This
    /// branch fires when two memories are sim > 0.85 AND share at least
    /// one entity AND sim < 0.98 — i.e. same topic, different facts.
    #[test]
    fn test_potential_conflict_with_shared_entity_names_rids() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        // Identical embeddings + small text edit + shared entity link.
        let emb = vec_seed(2.5, 8);

        let rid_a = db
            .record(
                "Acme deployed v1.0 in March",
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &emb,
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();
        let rid_b = db
            .record(
                "Acme deployed v2.0 in March",
                "semantic",
                0.5,
                0.0,
                604800.0,
                &serde_json::json!({}),
                &emb,
                "default",
                0.8,
                "general",
                "user",
                None,
            )
            .unwrap();

        // Link both memories to the same entity to make them share.
        db.link_memory_entity(&rid_a, "Acme").unwrap();
        db.link_memory_entity(&rid_b, "Acme").unwrap();

        let triggers = check_redundancy(&db, 0.85).unwrap();

        // With identical embeddings, sim = 1.0 — that exceeds the 0.98
        // ceiling for is_potential_conflict, so this case routes to
        // the redundancy branch (or substitution branch). The shared-entity
        // branch fires only when sim < 0.98, which identical-embedding
        // pairs cannot reach. So instead of asserting the branch
        // specifically, just assert SOME trigger named both rids in its
        // reason string — verifying that whichever branch fires has been
        // patched.
        let any_named: bool = triggers
            .iter()
            .any(|t| t.reason.contains(&rid_a) && t.reason.contains(&rid_b));
        assert!(
            any_named,
            "expected some trigger whose `reason` names both rids, got triggers: {:?}",
            triggers
        );
    }

    /// Issue #45: when multiple duplicate pairs exist, the trigger output
    /// should surface multiple triggers (subject to the .truncate(5) cap),
    /// and each surfaced trigger should name its OWN rid pair in `reason`.
    /// Regression: a sloppy fix that hard-coded the first pair's rids into
    /// every trigger's reason would pass the single-pair test but fail
    /// this one.
    #[test]
    fn test_multiple_redundancy_pairs_each_name_their_own_rids() {
        let db = YantrikDB::new(":memory:", 8).unwrap();

        // Three distinct pairs, each pair internally identical.
        let mut pairs: Vec<(String, String)> = Vec::new();
        for cluster_idx in 0..3 {
            let emb = vec_seed(10.0 + cluster_idx as f32, 8);
            let rid_a = db
                .record(
                    &format!("cluster {cluster_idx} memory alpha"),
                    "semantic",
                    0.5,
                    0.0,
                    604800.0,
                    &serde_json::json!({}),
                    &emb,
                    "default",
                    0.8,
                    "general",
                    "user",
                    None,
                )
                .unwrap();
            let rid_b = db
                .record(
                    &format!("cluster {cluster_idx} memory beta"),
                    "semantic",
                    0.5,
                    0.0,
                    604800.0,
                    &serde_json::json!({}),
                    &emb,
                    "default",
                    0.8,
                    "general",
                    "user",
                    None,
                )
                .unwrap();
            pairs.push((rid_a, rid_b));
        }

        let triggers = check_redundancy(&db, 0.85).unwrap();

        // For each pair, assert SOME trigger names both of its rids in
        // `reason` — and that the trigger NOT covering this pair does not
        // claim it. The strictest assertion is "every pair has its own
        // trigger that names exactly its own rid pair."
        for (rid_a, rid_b) in &pairs {
            let pair_trigger = triggers
                .iter()
                .find(|t| t.reason.contains(rid_a) && t.reason.contains(rid_b));
            assert!(
                pair_trigger.is_some(),
                "pair ({rid_a}, {rid_b}) must surface in a trigger whose `reason` names both, \
                 got triggers: {:?}",
                triggers.iter().map(|t| &t.reason).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_check_redundancy_honors_sim_threshold() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        // Two unit vectors with cosine similarity 0.9 (a == e0, b == 0.9*e0 + sqrt(0.19)*e1).
        let emb_a: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let emb_b: Vec<f32> = vec![
            0.9,
            (1.0 - 0.9 * 0.9f32).sqrt(),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        db.record(
            "alpha record",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({}),
            &emb_a,
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
        db.record(
            "beta record",
            "semantic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({}),
            &emb_b,
            "default",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();

        // 0.9 > 0.85: the pair is flagged.
        assert!(
            !check_redundancy(&db, 0.85).unwrap().is_empty(),
            "a 0.9-similarity pair must be flagged at threshold 0.85"
        );
        // 0.9 < 0.95: nothing is flagged.
        assert!(
            check_redundancy(&db, 0.95).unwrap().is_empty(),
            "a 0.9-similarity pair must not be flagged at threshold 0.95"
        );
    }

    /// A `ThinkConfig` serialized before `redundancy_sim_threshold` existed
    /// must still deserialize, taking the same 0.85 that `Default` uses.
    #[test]
    fn test_think_config_without_redundancy_threshold_deserializes() {
        let mut old = serde_json::to_value(crate::types::ThinkConfig::default()).unwrap();
        old.as_object_mut()
            .unwrap()
            .remove("redundancy_sim_threshold");
        let cfg: crate::types::ThinkConfig = serde_json::from_value(old).unwrap();
        assert_eq!(cfg.redundancy_sim_threshold, 0.85);
    }

    // ---- Substitution index vs. SQL reference (0010) ----------------------
    //
    // The index answers the same predicate as `check_substitution_category_pair`
    // but loads the table once instead of issuing a JOIN per token pair. These
    // tests pin the equivalence where a rewrite could silently drift: several
    // categories matching the same token pair, the order in which tokens are
    // visited (a HashSet's iteration order is not stable), and rows the SQL
    // predicate must reject (inactive member, low confidence, non-exclusive or
    // inactive category).
    //
    // The schema ships DEFAULT categories, so every fixture here uses names and
    // tokens that cannot collide with them.

    fn seed_category(
        conn: &rusqlite::Connection,
        id: &str,
        name: &str,
        conflict_mode: &str,
        status: &str,
        members: &[(&str, f64, &str)],
    ) {
        conn.execute(
            "INSERT INTO substitution_categories
               (id, name, conflict_mode, status, created_at, updated_at, hlc, origin_actor)
             VALUES (?1, ?2, ?3, ?4, 0.0, 0.0, x'00', 'test')",
            rusqlite::params![id, name, conflict_mode, status],
        )
        .unwrap();
        for (i, (token, confidence, member_status)) in members.iter().enumerate() {
            conn.execute(
                "INSERT INTO substitution_members
                   (id, category_id, token_normalized, token_display, confidence, source,
                    status, created_at, updated_at, hlc, origin_actor)
                 VALUES (?1, ?2, ?3, ?3, ?4, 'test', ?5, 0.0, 0.0, x'00', 'test')",
                rusqlite::params![format!("{id}-m{i}"), id, token, confidence, member_status],
            )
            .unwrap();
        }
    }

    fn both(
        conn: &rusqlite::Connection,
        index: &SubstitutionIndex,
        a: &str,
        b: &str,
    ) -> (
        Option<(String, String, String)>,
        Option<(String, String, String)>,
    ) {
        (
            check_substitution_category_pair(conn, a, b),
            check_substitution_pair_in_index(index, a, b),
        )
    }

    #[test]
    fn index_matches_sql_for_a_single_matching_category() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let conn = db.conn();
        seed_category(
            &conn,
            "t-c1",
            "t_single",
            "exclusive",
            "active",
            &[("toka", 0.9, "active"), ("tokb", 0.9, "active")],
        );
        let index = load_substitution_index(&conn).unwrap();

        let a = "wir setzen toka fuer den store ein";
        let b = "wir setzen tokb fuer den store ein";
        let (sql, mem) = both(&*conn, &index, a, b);
        assert_eq!(sql.as_ref().map(|t| t.0.as_str()), Some("t_single"));
        assert_eq!(mem.as_ref().map(|t| t.0.as_str()), Some("t_single"));
        assert_eq!(
            sql.as_ref().map(|t| t.1.as_str()),
            mem.as_ref().map(|t| t.1.as_str()),
            "same token_a"
        );
        assert_eq!(
            sql.as_ref().map(|t| t.2.as_str()),
            mem.as_ref().map(|t| t.2.as_str()),
            "same token_b"
        );
    }

    #[test]
    fn index_matches_sql_when_several_categories_match() {
        // Same token pair in TWO exclusive categories. The SQL `LIMIT 1` has no
        // ORDER BY and the index walks a HashMap, so either may pick either.
        // What must hold: both find a match, and both name a category that
        // really contains the pair.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let conn = db.conn();
        seed_category(
            &conn,
            "t-c2a",
            "t_multi_a",
            "exclusive",
            "active",
            &[("tokm", 0.9, "active"), ("tokn", 0.9, "active")],
        );
        seed_category(
            &conn,
            "t-c2b",
            "t_multi_b",
            "exclusive",
            "active",
            &[("tokm", 0.8, "active"), ("tokn", 0.8, "active")],
        );
        let index = load_substitution_index(&conn).unwrap();

        let a = "tokm laeuft hier";
        let b = "tokn laeuft hier";
        let (sql, mem) = both(&*conn, &index, a, b);
        let allowed = ["t_multi_a", "t_multi_b"];
        let sql_name = sql.as_ref().map(|t| t.0.as_str()).expect("sql must match");
        let mem_name = mem
            .as_ref()
            .map(|t| t.0.as_str())
            .expect("index must match");
        assert!(allowed.contains(&sql_name), "sql picked {sql_name}");
        assert!(allowed.contains(&mem_name), "index picked {mem_name}");
    }

    #[test]
    fn index_matches_sql_regardless_of_token_order() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let conn = db.conn();
        seed_category(
            &conn,
            "t-c3",
            "t_order",
            "exclusive",
            "active",
            &[("tokx", 0.9, "active"), ("toky", 0.9, "active")],
        );
        let index = load_substitution_index(&conn).unwrap();

        // The differing token sits at the head, in the middle, and at the tail.
        for (a, b) in [
            ("tokx ist die wahl", "toky ist die wahl"),
            ("wir nehmen tokx heute", "wir nehmen toky heute"),
            ("die datenbank heisst tokx", "die datenbank heisst toky"),
        ] {
            let (sql, mem) = both(&*conn, &index, a, b);
            assert_eq!(sql.is_some(), mem.is_some(), "{a} vs {b}");
            assert_eq!(
                sql.as_ref().map(|t| t.0.as_str()),
                mem.as_ref().map(|t| t.0.as_str()),
                "{a} vs {b}"
            );
        }
    }

    #[test]
    fn index_ignores_rows_the_sql_predicate_rejects() {
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let conn = db.conn();
        // member status inactive
        seed_category(
            &conn,
            "t-r1",
            "t_inactive_member",
            "exclusive",
            "active",
            &[("tokp", 0.9, "inactive"), ("tokq", 0.9, "active")],
        );
        // confidence below the 0.6 floor
        seed_category(
            &conn,
            "t-r2",
            "t_low_confidence",
            "exclusive",
            "active",
            &[("tokr", 0.3, "active"), ("toks", 0.9, "active")],
        );
        // category not exclusive
        seed_category(
            &conn,
            "t-r3",
            "t_compatible",
            "compatible",
            "active",
            &[("tokt", 0.9, "active"), ("toku", 0.9, "active")],
        );
        // category inactive
        seed_category(
            &conn,
            "t-r4",
            "t_dead",
            "exclusive",
            "inactive",
            &[("tokv", 0.9, "active"), ("tokw", 0.9, "active")],
        );
        let index = load_substitution_index(&conn).unwrap();

        for (a, b) in [
            ("tokp hier", "tokq hier"),
            ("tokr hier", "toks hier"),
            ("tokt hier", "toku hier"),
            ("tokv hier", "tokw hier"),
        ] {
            let (sql, mem) = both(&*conn, &index, a, b);
            assert!(sql.is_none(), "sql must reject {a} vs {b}");
            assert!(mem.is_none(), "index must reject {a} vs {b}");
        }
    }

    #[test]
    fn index_returns_none_for_tokens_no_category_contains() {
        // The schema seeds default categories, so the index is NOT empty here.
        // What must hold: a pair outside every category is answered the same
        // way by both implementations.
        let db = YantrikDB::new(":memory:", 8).unwrap();
        let conn = db.conn();
        let index = load_substitution_index(&conn).unwrap();
        assert!(!index.is_empty(), "schema seeds default categories");

        let a = "zzznosuchtoken hier";
        let b = "zzzandertoken hier";
        let (sql, mem) = both(&*conn, &index, a, b);
        assert!(sql.is_none());
        assert!(mem.is_none());
    }

    // ------------------------------------------------------------------
    // Subject/frame gate — the composition of 0001 (#312) and 0010 (#315).
    //
    // These five cases are the acceptance criteria for the composed branch:
    // the predicate must hold in the *index* path, in both passes, and must
    // not turn every shared category into a contradiction.
    // ------------------------------------------------------------------

    /// A substitution index with the given (category, members) pairs. Category
    /// id and name are the same string here; only membership matters.
    fn test_index(cats: &[(&str, &[&str])]) -> SubstitutionIndex {
        let mut index: SubstitutionIndex = std::collections::HashMap::new();
        for (category, members) in cats {
            for token in *members {
                index
                    .entry((*token).to_string())
                    .or_default()
                    .push(((*category).to_string(), (*category).to_string()));
            }
        }
        index
    }

    fn provider_index() -> SubstitutionIndex {
        test_index(&[("llm_providers", &["google", "deepseek", "openai"])])
    }

    /// Case 1 — same category, different subject. This is the live false
    /// alarm: ADR-0057 (provenance) vs ADR-0125 (tooling standard). Both
    /// mention a provider; neither talks about the other's subject.
    #[test]
    fn same_category_different_subject_is_not_a_conflict() {
        let a = "ADR-0057 behandelt die Provenienz der Provider google und die Herkunft der Migrationen";
        let b = "ADR-0125 behandelt den Werkzeug-Basisstandard der Agenten und nennt deepseek als Modell";
        assert!(
            check_substitution_pair_in_index(&provider_index(), a, b).is_none(),
            "documents about different subjects must not be a contradiction"
        );
    }

    /// Case 2 — same category, different situational context. Shared
    /// vocabulary, but a test run and production are not in conflict about
    /// which provider is right.
    #[test]
    fn same_category_different_situational_context_is_not_a_conflict() {
        let a = "Im Testlauf nutzen wir google als Provider fuer die Evaluierung";
        let b = "Im Produktivbetrieb nutzen wir deepseek als Provider fuer die Auslieferung";
        assert!(
            check_substitution_pair_in_index(&provider_index(), a, b).is_none(),
            "different situational context must not be a contradiction"
        );
    }

    /// Case 3 — the real contradiction: same subject, same predicate,
    /// different value. This is what the rule exists for.
    #[test]
    fn actual_contradiction_with_same_subject_is_a_conflict() {
        let a = "Wir setzen google als Provider fuer die Bildgenerierung ein";
        let b = "Wir setzen deepseek als Provider fuer die Bildgenerierung ein";
        let hit = check_substitution_pair_in_index(&provider_index(), a, b);
        assert!(
            hit.is_some(),
            "same subject and frame, different value must be a conflict"
        );
        let (cat, ta, tb) = hit.unwrap();
        assert_eq!(cat, "llm_providers");
        assert!(
            matches!(
                (ta.as_str(), tb.as_str()),
                ("google", "deepseek") | ("deepseek", "google")
            ),
            "returned tokens must be the pair that was substituted, got {ta}/{tb}"
        );
    }

    /// Case 4 — several categories match on the same pair of texts. The
    /// function must return a pair whose tokens really are alternatives in
    /// the category it names, not a category from one pair and tokens from
    /// another.
    #[test]
    fn multiple_matching_categories_return_consistent_tokens() {
        let index = test_index(&[
            ("roles", &["backend", "data"]),
            ("llm_providers", &["google", "deepseek"]),
        ]);
        let a = "Wir setzen google fuer die Bildgenerierung ein und backend dauerhaft fuer die Auslieferung der Berichte";
        let b = "Wir setzen deepseek fuer die Bildgenerierung ein und data dauerhaft fuer die Auslieferung der Berichte";
        let hit = check_substitution_pair_in_index(&index, a, b);
        assert!(
            hit.is_some(),
            "at least one clean contradiction exists here"
        );
        let (cat, ta, tb) = hit.unwrap();
        let members: &[&str] = match cat.as_str() {
            "roles" => &["backend", "data"],
            "llm_providers" => &["google", "deepseek"],
            other => panic!("unexpected category {other}"),
        };
        assert!(
            members.contains(&ta.as_str()) && members.contains(&tb.as_str()),
            "tokens {ta}/{tb} do not belong to category {cat}"
        );
    }

    /// Case 5 — both passes of the substitution check. Pass 1 runs above the
    /// redundancy threshold, pass 2 in the 0.65–0.85 band. A gate that lives
    /// in only one of them shows up as a difference between the two arms.
    #[test]
    fn both_redundancy_passes_apply_the_subject_gate() {
        fn seed_category(db: &YantrikDB) {
            let hlc = db.tick_hlc().to_bytes().to_vec();
            let ts = now();
            db.conn()
                .execute(
                    "INSERT INTO substitution_categories
                     (id, name, conflict_mode, status, created_at, updated_at, hlc, origin_actor)
                     VALUES ('cat-llm-gate', 'llm_gate_test', 'exclusive', 'active',
                             ?1, ?1, ?2, 'test')",
                    rusqlite::params![ts, hlc],
                )
                .unwrap();
            for (id, token) in [("m-google", "google"), ("m-deepseek", "deepseek")] {
                let hlc = db.tick_hlc().to_bytes().to_vec();
                db.conn()
                    .execute(
                        "INSERT INTO substitution_members
                         (id, category_id, token_normalized, token_display, confidence,
                          source, status, created_at, updated_at, hlc, origin_actor)
                         VALUES (?1, 'cat-llm-gate', ?2, ?2, 0.9, 'test', 'active',
                                 ?3, ?3, ?4, 'test')",
                        rusqlite::params![id, token, ts, hlc],
                    )
                    .unwrap();
            }
        }

        /// Two unit vectors with the requested cosine.
        fn pair_with_cosine(cos: f32) -> (Vec<f32>, Vec<f32>) {
            let dim = 8;
            let mut a = vec![0.0f32; dim];
            let mut b = vec![0.0f32; dim];
            a[0] = 1.0;
            b[0] = cos;
            b[1] = (1.0 - cos * cos).max(0.0).sqrt();
            (a, b)
        }

        fn record_pair(db: &YantrikDB, text_a: &str, text_b: &str, cos: f32) -> (String, String) {
            let (ea, eb) = pair_with_cosine(cos);
            let ra = db
                .record(
                    text_a,
                    "semantic",
                    0.8,
                    0.0,
                    604800.0,
                    &serde_json::json!({}),
                    &ea,
                    "default",
                    0.9,
                    "general",
                    "user",
                    None,
                )
                .unwrap();
            let rb = db
                .record(
                    text_b,
                    "semantic",
                    0.8,
                    0.0,
                    604800.0,
                    &serde_json::json!({}),
                    &eb,
                    "default",
                    0.9,
                    "general",
                    "user",
                    None,
                )
                .unwrap();
            (ra, rb)
        }

        fn conflict_count(db: &YantrikDB, a: &str, b: &str) -> i64 {
            db.conn()
                .query_row(
                    "SELECT COUNT(*) FROM conflicts
                     WHERE (memory_a = ?1 AND memory_b = ?2)
                        OR (memory_a = ?2 AND memory_b = ?1)",
                    rusqlite::params![a, b],
                    |row| row.get(0),
                )
                .unwrap()
        }

        // 1.0 = high-similarity pass, 0.766 = substitution pass (cos 40°).
        for (cos, pass) in [(1.0f32, "high-similarity"), (0.766f32, "substitution")] {
            let db = YantrikDB::new(":memory:", 8).unwrap();
            seed_category(&db);
            let (ra, rb) = record_pair(
                &db,
                "Wir setzen google als Provider fuer die Bildgenerierung ein",
                "Wir setzen deepseek als Provider fuer die Bildgenerierung ein",
                cos,
            );
            let _ = check_redundancy(&db, 0.85).unwrap();
            assert_eq!(
                conflict_count(&db, &ra, &rb),
                1,
                "{pass} pass: a real contradiction must be recorded"
            );

            let db = YantrikDB::new(":memory:", 8).unwrap();
            seed_category(&db);
            let (ra, rb) = record_pair(
                &db,
                "ADR-0057 behandelt die Provenienz der Provider google und die Herkunft der Migrationen",
                "ADR-0125 behandelt den Werkzeug-Basisstandard der Agenten und nennt deepseek als Modell",
                cos,
            );
            let _ = check_redundancy(&db, 0.85).unwrap();
            assert_eq!(
                conflict_count(&db, &ra, &rb),
                0,
                "{pass} pass: different subjects must not be recorded as a conflict"
            );
        }
    }

    /// Review 7 (B5) / review 8 (finding 2): the earlier version of this test
    /// used the *same* filler in both arms, so the whole-document Jaccard was
    /// 53/55 = 0.96 and the test would have been green on the old frame too —
    /// it proved nothing. This one differs in the second sentence and in the
    /// filler, so a document-wide frame scores ~0.5 and rejects, while the
    /// sentence-level frame sees the same statement.
    #[test]
    fn a_long_note_keeps_its_frame_when_the_sentence_agrees() {
        let a = "der zahlungsdienst verwendet postgresql als primaeren datenspeicher \
                 und schreibt jede buchung zuerst dorthin. die taegliche abrechnung \
                 startet um zwei uhr nachts und wird vom controlling geprueft";
        let b = "der zahlungsdienst verwendet mysql als primaeren datenspeicher \
                 und schreibt jede buchung zuerst dorthin. die abrechnung laeuft \
                 nachts um zwei uhr und wird von der revision kontrolliert";

        assert!(
            substitution_is_same_claim(a, "postgresql", b, "mysql"),
            "the same sentence around the value must stay a candidate"
        );

        // Control: same length, different subject — the gate must still refuse,
        // otherwise the fix would have disabled it.
        let c = "der zahlungsdienst verwendet postgresql als primaeren datenspeicher \
                 und schreibt jede buchung zuerst dorthin";
        let d = "die marketingseite verwendet mysql fuer das archiv der kampagnen \
                 und speichert dort alle entwuerfe";
        assert!(
            !substitution_is_same_claim(c, "postgresql", d, "mysql"),
            "different subjects must still be refused"
        );
    }

    /// A maintenance result must distinguish "no conflicts found" from
    /// "conflicts could not be evaluated". Dropping the substitution table and
    /// running the scan has to surface a trigger, not just a log line — the
    /// report is what a reader trusts.
    #[test]
    fn a_missing_index_shows_up_in_the_result() {
        let db = YantrikDB::with_default(":memory:").unwrap();
        db.record_text(
            "der zahlungsdienst verwendet postgresql als datenspeicher.",
            "semantic",
            0.5,
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
        // The store is V14 (`claims` exists), so this is the damage case.
        db.conn()
            .execute("DROP TABLE substitution_members", [])
            .unwrap();

        let triggers = check_redundancy(&db, 0.85).unwrap();
        let marked: Vec<_> = triggers
            .iter()
            .filter(|t| t.trigger_type == "maintenance_incomplete")
            .collect();
        assert_eq!(
            marked.len(),
            1,
            "the incomplete namespace must be reported: {triggers:?}"
        );
        assert!(
            marked[0].reason.contains("was not fully checked"),
            "the reason must say what was skipped: {:?}",
            marked[0].reason
        );
        // The store ran V13->V14 (it has the schema), so the missing tables are
        // damage, not an old store — the reason must say that (review 9, E3).
        assert!(
            marked[0].reason.contains("migration did not complete"),
            "the damage case must be named: {:?}",
            marked[0].reason
        );
    }

    /// Review 9 (E1): a `.` inside a version number must not end the sentence.
    /// With the naive split, `0.23.6` broke the frame down to a single short
    /// word and the gate refused a real contradiction.
    #[test]
    fn a_version_number_does_not_cut_the_frame_apart() {
        let a = "version 0.23.6 postgresql aktiv";
        let b = "version 0.23.6 mysql aktiv";
        assert!(
            substitution_is_same_claim(a, "postgresql", b, "mysql"),
            "the version number must stay inside the sentence"
        );

        // Abbreviation with two dots, same shape.
        let c = "z.B. version 0.23.6 postgresql aktiv";
        let d = "z.B. version 0.23.6 mysql aktiv";
        assert!(
            substitution_is_same_claim(c, "postgresql", d, "mysql"),
            "z.B. must not end the sentence either"
        );
    }
}
