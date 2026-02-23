use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::engine::AIDB;
use crate::error::Result;
use crate::scoring;
use crate::types::Trigger;

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

/// Find important memories that are decaying significantly.
///
/// Triggers when:
///   - Original importance >= importance_threshold (it was important)
///   - Current effective score < decay_threshold (it's fading)
pub fn check_decay_triggers(
    db: &AIDB,
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
        let (rid, text, mem_type, importance, half_life, last_access, valence) = row?;
        let elapsed = ts - last_access;
        let current_score = scoring::decay_score(importance, half_life, elapsed);

        if current_score < decay_threshold {
            let days_since = elapsed / 86400.0;
            let decay_ratio = if importance > 0.0 {
                current_score / importance
            } else {
                0.0
            };

            // Urgency: higher for more important memories that decayed more
            let urgency = importance * (1.0 - decay_ratio);

            let mut context = HashMap::new();
            context.insert("text".to_string(), serde_json::json!(text));
            context.insert("type".to_string(), serde_json::json!(mem_type));
            context.insert("original_importance".to_string(), serde_json::json!(importance));
            context.insert("current_score".to_string(), serde_json::json!(current_score));
            context.insert("days_since_access".to_string(), serde_json::json!(days_since));
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

    // Sort by urgency, return top N
    triggers.sort_by(|a, b| {
        b.urgency
            .partial_cmp(&a.urgency)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    triggers.truncate(max_triggers);
    Ok(triggers)
}

/// Trigger when there are enough active memories that consolidation might help.
pub fn check_consolidation_triggers(
    db: &AIDB,
    min_active_memories: i64,
) -> Result<Vec<Trigger>> {
    let stats = db.stats()?;
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
                reason: format!("{unconsolidated} episodic memories could be consolidated"),
                urgency: (unconsolidated as f64 / 50.0).min(1.0),
                source_rids: vec![],
                suggested_action: "run_consolidation".to_string(),
                context,
            });
        }
    }

    Ok(triggers)
}

/// Run all trigger checks and return a unified, priority-sorted list.
pub fn check_all_triggers(
    db: &AIDB,
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

    triggers.sort_by(|a, b| {
        b.urgency
            .partial_cmp(&a.urgency)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    triggers.truncate(max_triggers);
    Ok(triggers)
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
        let db = AIDB::new(":memory:", 8).unwrap();
        db.record("fresh", "episodic", 0.9, 0.0, 604800.0, &serde_json::json!({}), &vec_seed(1.0, 8)).unwrap();
        let triggers = check_decay_triggers(&db, 0.5, 0.1, 5).unwrap();
        assert!(triggers.is_empty());
    }

    #[test]
    fn test_decay_trigger_fires() {
        let db = AIDB::new(":memory:", 8).unwrap();
        let rid = db.record("important deadline", "episodic", 0.9, 0.0, 100.0, &serde_json::json!({}), &vec_seed(1.0, 8)).unwrap();

        // Backdate last_access
        db.conn().execute(
            "UPDATE memories SET last_access = ?1 WHERE rid = ?2",
            rusqlite::params![now() - 10000.0, rid],
        ).unwrap();

        let triggers = check_decay_triggers(&db, 0.5, 0.1, 5).unwrap();
        assert!(!triggers.is_empty());
        assert_eq!(triggers[0].trigger_type, "decay_review");
        assert_eq!(triggers[0].source_rids, vec![rid]);
    }

    #[test]
    fn test_consolidation_trigger() {
        let db = AIDB::new(":memory:", 8).unwrap();
        for i in 0..15 {
            db.record(
                &format!("episodic memory {i}"),
                "episodic", 0.5, 0.0, 604800.0,
                &serde_json::json!({}),
                &vec_seed(i as f32, 8),
            ).unwrap();
        }

        let triggers = check_consolidation_triggers(&db, 10).unwrap();
        assert_eq!(triggers.len(), 1);
        assert_eq!(triggers[0].trigger_type, "consolidation_ready");
    }
}
