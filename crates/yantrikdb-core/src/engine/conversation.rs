//! Conversation working-memory ring buffer (v0.9.0).
//!
//! A cheap, verbatim, bounded FIFO of raw both-sides conversation turns, scoped
//! per namespace. This is the *short-term working memory* an agent needs — "what
//! were the last few things said" — which is a different need from the semantic
//! long-term store (`record_text` / `recall`):
//!
//! - **Not embedded.** Turns are stored verbatim; no vector is computed. This is
//!   what makes it nearly free and keeps raw chatter out of semantic recall.
//! - **Not kept forever.** On each append the buffer is pruned to the last `N`
//!   turns for that namespace (default [`DEFAULT_TURN_WINDOW`] = 10,
//!   configurable per call).
//! - **Encrypted at rest** like memory text, so it inherits the same privacy.
//!
//! If a turn worth remembering long-term scrolls out of the window, the caller
//! (or a consolidation pass) can promote it into a real memory via
//! `record_text`. The buffer itself stays cheap.

use rusqlite::params;

use crate::error::Result;

use super::{now, YantrikDB};

/// Default number of turns retained per namespace.
pub const DEFAULT_TURN_WINDOW: usize = 10;

/// One stored conversation turn.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Turn {
    pub role: String,
    pub content: String,
    pub created_at: f64,
}

impl YantrikDB {
    /// Append one raw conversation turn to `namespace`'s working-memory buffer,
    /// then prune to the last `max_turns` (use [`DEFAULT_TURN_WINDOW`] for the
    /// default of 10). Verbatim and un-embedded — cheap. Returns the turn's id.
    pub fn record_turn(
        &self,
        namespace: &str,
        role: &str,
        content: &str,
        max_turns: usize,
    ) -> Result<i64> {
        let ns = super::record::normalize_namespace(namespace);
        let stored = self.encrypt_text(content)?;
        let conn = self.conn();
        conn.execute(
            "INSERT INTO conversation_turns (namespace, role, content, created_at) \
             VALUES (?1, ?2, ?3, ?4)",
            params![ns, role, stored, now()],
        )?;
        let id = conn.last_insert_rowid();
        if max_turns > 0 {
            // Keep only the most-recent `max_turns` for this namespace.
            conn.execute(
                "DELETE FROM conversation_turns WHERE namespace = ?1 AND id NOT IN (\
                   SELECT id FROM conversation_turns WHERE namespace = ?1 \
                   ORDER BY id DESC LIMIT ?2)",
                params![ns, max_turns as i64],
            )?;
        }
        Ok(id)
    }

    /// The last `limit` turns for `namespace`, **oldest-first** (ready to
    /// prepend to a prompt as recent context).
    pub fn recent_turns(&self, namespace: &str, limit: usize) -> Result<Vec<Turn>> {
        let ns = super::record::normalize_namespace(namespace);
        // Read raw under the conn lock, then decrypt outside it.
        let raw: Vec<(String, String, f64)> = {
            let conn = self.conn();
            let mut stmt = conn.prepare(
                "SELECT role, content, created_at FROM conversation_turns \
                 WHERE namespace = ?1 ORDER BY id DESC LIMIT ?2",
            )?;
            let collected = stmt
                .query_map(params![ns, limit as i64], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, f64>(2)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            collected
        };
        let mut turns = Vec::with_capacity(raw.len());
        for (role, content, created_at) in raw {
            turns.push(Turn {
                role,
                content: self.decrypt_text(&content)?,
                created_at,
            });
        }
        turns.reverse(); // newest-first query → oldest-first for prompting
        Ok(turns)
    }

    /// Clear the conversation buffer for a namespace. Returns rows removed.
    pub fn clear_turns(&self, namespace: &str) -> Result<usize> {
        let ns = super::record::normalize_namespace(namespace);
        let conn = self.conn();
        Ok(conn.execute(
            "DELETE FROM conversation_turns WHERE namespace = ?1",
            params![ns],
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "bundled-embedder")]
    #[test]
    fn ring_buffer_keeps_last_n_oldest_first() {
        let db = YantrikDB::with_default(":memory:").unwrap();
        for i in 0..15 {
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            db.record_turn("chat", role, &format!("turn {i}"), 10)
                .unwrap();
        }
        let turns = db.recent_turns("chat", 10).unwrap();
        assert_eq!(turns.len(), 10, "pruned to the window");
        // Oldest-first: the window is turns 5..=14.
        assert_eq!(turns.first().unwrap().content, "turn 5");
        assert_eq!(turns.last().unwrap().content, "turn 14");
        // Roles preserved verbatim, both sides: turn 5 is odd -> assistant,
        // turn 14 is even -> user.
        assert_eq!(turns.first().unwrap().role, "assistant");
        assert_eq!(turns.last().unwrap().role, "user");
    }

    #[cfg(feature = "bundled-embedder")]
    #[test]
    fn buffers_are_namespace_isolated_and_clearable() {
        let db = YantrikDB::with_default(":memory:").unwrap();
        db.record_turn("a", "user", "in a", 10).unwrap();
        db.record_turn("b", "user", "in b", 10).unwrap();
        assert_eq!(db.recent_turns("a", 10).unwrap().len(), 1);
        assert_eq!(db.recent_turns("b", 10).unwrap().len(), 1);
        assert_eq!(db.clear_turns("a").unwrap(), 1);
        assert_eq!(db.recent_turns("a", 10).unwrap().len(), 0);
        assert_eq!(
            db.recent_turns("b", 10).unwrap().len(),
            1,
            "other ns untouched"
        );
    }
}
