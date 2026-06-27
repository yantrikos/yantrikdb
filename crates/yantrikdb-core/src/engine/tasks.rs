//! Task / chore store (v0.9.0).
//!
//! A minimal, general operational task primitive so an agent can keep its
//! chores in the same substrate as its memory — cheaply. A task is a flat
//! record with a `status`, a `priority`, and an optional `parent_id` (for
//! subtasks); the title is encrypted like memory text and the store is not
//! embedded (chores are queried relationally, not semantically).
//!
//! Deliberately NOT a Saga-style project/epic/task hierarchy: that opinionated
//! project-management model is a *convention* a consumer can layer on top
//! (e.g. via parent chains or its own tags), not something the general engine
//! imposes — the same line we hold for every domain model. What lives here is
//! only the cheap, universal "list of things to do, with state."
//!
//! (For *auto-detected* open loops — unanswered questions, pending commitments
//! — see the `agenda` system; this store is for *authored* chores.)

use rusqlite::{params, OptionalExtension};

use crate::error::Result;

use super::{now, YantrikDB};

/// A stored task / chore.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Task {
    pub id: String,
    pub namespace: String,
    pub title: String,
    pub status: String,
    pub priority: String,
    pub parent_id: Option<String>,
    pub created_at: f64,
    pub updated_at: f64,
}

impl YantrikDB {
    /// Create a task. `priority` is one of low|medium|high|critical;
    /// `parent_id` makes it a subtask. Returns the new task id.
    pub fn task_add(
        &self,
        namespace: &str,
        title: &str,
        priority: &str,
        parent_id: Option<&str>,
    ) -> Result<String> {
        let ns = super::record::normalize_namespace(namespace);
        let id = crate::id::new_id();
        let stored = self.encrypt_text(title)?;
        let ts = now();
        let conn = self.conn();
        conn.execute(
            "INSERT INTO tasks (id, namespace, title, status, priority, parent_id, \
             created_at, updated_at) VALUES (?1, ?2, ?3, 'open', ?4, ?5, ?6, ?6)",
            params![id, ns, stored, priority, parent_id, ts],
        )?;
        Ok(id)
    }

    /// Update a task's `status` and/or `priority` (only the provided fields).
    /// Returns whether the task existed.
    pub fn task_update(
        &self,
        id: &str,
        status: Option<&str>,
        priority: Option<&str>,
    ) -> Result<bool> {
        let conn = self.conn();
        let mut changed = false;
        if let Some(s) = status {
            changed |= conn.execute(
                "UPDATE tasks SET status = ?1, updated_at = ?2 WHERE id = ?3",
                params![s, now(), id],
            )? > 0;
        }
        if let Some(p) = priority {
            changed |= conn.execute(
                "UPDATE tasks SET priority = ?1, updated_at = ?2 WHERE id = ?3",
                params![p, now(), id],
            )? > 0;
        }
        Ok(changed)
    }

    /// List tasks in a namespace, optionally filtered by `status`, ordered by
    /// priority (critical first) then recency.
    pub fn task_list(&self, namespace: &str, status: Option<&str>) -> Result<Vec<Task>> {
        let ns = super::record::normalize_namespace(namespace);
        let order = "ORDER BY CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 \
                     WHEN 'medium' THEN 2 ELSE 3 END, created_at DESC";
        let raw: Vec<(String, String, String, String, Option<String>, f64, f64)> = {
            let conn = self.conn();
            let collected = if let Some(s) = status {
                let mut stmt = conn.prepare(&format!(
                    "SELECT id, title, status, priority, parent_id, created_at, updated_at \
                     FROM tasks WHERE namespace = ?1 AND status = ?2 {order}"
                ))?;
                let rows = stmt
                    .query_map(params![ns, s], Self::map_task_row)?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                rows
            } else {
                let mut stmt = conn.prepare(&format!(
                    "SELECT id, title, status, priority, parent_id, created_at, updated_at \
                     FROM tasks WHERE namespace = ?1 {order}"
                ))?;
                let rows = stmt
                    .query_map(params![ns], Self::map_task_row)?
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                rows
            };
            collected
        };
        let ns_owned = ns.to_string();
        let mut tasks = Vec::with_capacity(raw.len());
        for (id, title, status, priority, parent_id, created_at, updated_at) in raw {
            tasks.push(Task {
                id,
                namespace: ns_owned.clone(),
                title: self.decrypt_text(&title)?,
                status,
                priority,
                parent_id,
                created_at,
                updated_at,
            });
        }
        Ok(tasks)
    }

    /// Get one task by id.
    pub fn task_get(&self, id: &str) -> Result<Option<Task>> {
        let conn = self.conn();
        let row = conn
            .query_row(
                "SELECT id, namespace, title, status, priority, parent_id, created_at, updated_at \
                 FROM tasks WHERE id = ?1",
                params![id],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                        r.get::<_, String>(3)?,
                        r.get::<_, String>(4)?,
                        r.get::<_, Option<String>>(5)?,
                        r.get::<_, f64>(6)?,
                        r.get::<_, f64>(7)?,
                    ))
                },
            )
            .optional()?;
        match row {
            None => Ok(None),
            Some((id, namespace, title, status, priority, parent_id, created_at, updated_at)) => {
                Ok(Some(Task {
                    id,
                    namespace,
                    title: self.decrypt_text(&title)?,
                    status,
                    priority,
                    parent_id,
                    created_at,
                    updated_at,
                }))
            }
        }
    }

    /// Delete a task. Returns whether a row was removed.
    pub fn task_delete(&self, id: &str) -> Result<bool> {
        let conn = self.conn();
        Ok(conn.execute("DELETE FROM tasks WHERE id = ?1", params![id])? > 0)
    }

    #[allow(clippy::type_complexity)]
    fn map_task_row(
        r: &rusqlite::Row,
    ) -> rusqlite::Result<(String, String, String, String, Option<String>, f64, f64)> {
        Ok((
            r.get(0)?,
            r.get(1)?,
            r.get(2)?,
            r.get(3)?,
            r.get(4)?,
            r.get(5)?,
            r.get(6)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "bundled-embedder")]
    #[test]
    fn task_lifecycle_and_listing() {
        let db = YantrikDB::with_default(":memory:").unwrap();
        let a = db
            .task_add("chores", "rotate the API keys", "high", None)
            .unwrap();
        let _b = db
            .task_add("chores", "tidy the inbox", "low", None)
            .unwrap();
        let sub = db
            .task_add("chores", "draft the rotation runbook", "medium", Some(&a))
            .unwrap();

        // Listed, priority-ordered (high before low), with subtask parent set.
        let all = db.task_list("chores", None).unwrap();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0].title, "rotate the API keys"); // high first
        let sub_task = db.task_get(&sub).unwrap().unwrap();
        assert_eq!(sub_task.parent_id.as_deref(), Some(a.as_str()));

        // Update status; open filter shrinks.
        assert!(db.task_update(&a, Some("done"), None).unwrap());
        let open = db.task_list("chores", Some("open")).unwrap();
        assert_eq!(open.len(), 2, "completed task drops out of open");
        assert!(open.iter().all(|t| t.status == "open"));
        assert_eq!(db.task_get(&a).unwrap().unwrap().status, "done");

        // Namespace isolation + delete.
        assert!(db.task_list("other", None).unwrap().is_empty());
        assert!(db.task_delete(&a).unwrap());
        assert_eq!(db.task_list("chores", None).unwrap().len(), 2);
    }
}
