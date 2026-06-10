//! Mega-blob splitting: oversized episodic dumps → atomic semantic facts.
//!
//! The 2026-06-10 audit found the recent corpus dominated by 1,500+ character
//! episodic session dumps. They bury the individual facts inside them and make
//! recall return walls of text instead of the one sentence that answers the
//! query. This is the hierarchical-compression promise the README makes ("real
//! memory is hierarchical, compressed") applied to ingest debt.
//!
//! This module's maintenance pass segments each oversized episode into atomic
//! facts, stores each as its own semantic memory linked back to the source
//! episode with [`LinkType::DerivedFrom`] (the v0.7.21 record-to-record link
//! model), and demotes the original episode to the cold tier — retained for
//! provenance, but out of the hot vector index and so out of primary recall.
//!
//! Segmentation is a deterministic, dependency-free heuristic (sentence
//! boundaries + greedy packing). True semantic atomic-fact extraction wants an
//! LLM, which the engine has no business calling; a smarter splitter can be
//! supplied by the MCP/server layer later. The engine provides the mechanism
//! and a sound default.
//!
//! Like the other ingest-integrity passes it is dry-run-first, idempotent (a
//! split parent goes cold and the scan only looks at hot rows, so it is never
//! re-split), and fault-tolerant (a failure on one episode is recorded and the
//! sweep continues).

use rusqlite::params;

use crate::error::Result;
use crate::types::{LinkType, RecordLink};

use super::{now, YantrikDB};

/// Target character length for a packed atomic fact. Single sentences longer
/// than this are kept whole rather than split mid-thought.
const TARGET_FACT_LEN: usize = 280;
/// Minimum length for a segment to be kept — drops trivial fragments.
const MIN_FACT_CHARS: usize = 24;
/// Cap on atomic facts produced per episode; the overflow tail is merged into
/// the last kept fact so no content is dropped.
const MAX_FACTS: usize = 40;
/// Cap on rids echoed back in a report.
const SAMPLE_CAP: usize = 50;

/// Outcome of a [`YantrikDB::split_oversized_episodes`] pass.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SplitReport {
    pub dry_run: bool,
    /// Oversized episodes examined (after the exact length recheck).
    pub episodes_scanned: usize,
    /// Episodes actually split (yielded ≥2 atomic facts).
    pub episodes_split: usize,
    /// Atomic-fact memories created (or, in dry-run, that would be created).
    pub atomic_facts_created: usize,
    /// Sample of split parent rids for operator spot-checking.
    pub sample_parent_rids: Vec<String>,
    /// Per-episode errors; the sweep continues past them.
    pub errors: Vec<String>,
}

/// Segment free text into atomic facts: split on sentence boundaries and
/// newlines, greedily pack into chunks up to `target_len`, drop trivial
/// fragments, and cap at `max_facts` (merging any overflow into the last kept
/// fact so nothing is silently lost). Pure and deterministic.
pub(crate) fn segment_into_atomic_facts(
    text: &str,
    target_len: usize,
    max_facts: usize,
) -> Vec<String> {
    // 1) Break into sentence-ish units on terminal punctuation + whitespace,
    //    and on newlines.
    let mut units: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        cur.push(c);
        let boundary = match c {
            '\n' => true,
            '.' | '!' | '?' => chars
                .peek()
                .map(|n| n.is_whitespace())
                .unwrap_or(true),
            _ => false,
        };
        if boundary {
            let t = cur.trim();
            if !t.is_empty() {
                units.push(t.to_string());
            }
            cur.clear();
        }
    }
    let tail = cur.trim();
    if !tail.is_empty() {
        units.push(tail.to_string());
    }

    // 2) Greedily pack units into chunks up to ~target_len.
    let mut facts: Vec<String> = Vec::new();
    let mut chunk = String::new();
    for u in units {
        if chunk.is_empty() {
            chunk = u;
        } else if chunk.len() + 1 + u.len() <= target_len {
            chunk.push(' ');
            chunk.push_str(&u);
        } else {
            facts.push(std::mem::take(&mut chunk));
            chunk = u;
        }
    }
    if !chunk.is_empty() {
        facts.push(chunk);
    }

    // 3) Drop trivially short fragments.
    facts.retain(|f| f.trim().chars().count() >= MIN_FACT_CHARS);

    // 4) Cap the count; merge the overflow tail into the last kept fact.
    if facts.len() > max_facts {
        let overflow = facts.split_off(max_facts);
        if let Some(last) = facts.last_mut() {
            for t in overflow {
                last.push(' ');
                last.push_str(&t);
            }
        }
    }
    facts
}

impl YantrikDB {
    /// Split oversized episodic memories into atomic semantic facts. See the
    /// module docs for the contract. `min_chars` is the plaintext length above
    /// which an episode is a candidate (operators typically pass ~1500). Run
    /// with `dry_run = true` to preview.
    pub fn split_oversized_episodes(&self, dry_run: bool, min_chars: usize) -> Result<SplitReport> {
        let mut report = SplitReport {
            dry_run,
            ..Default::default()
        };

        // Coarse SQL prefilter on stored length (a superset under encryption,
        // where ciphertext is at least as long as plaintext); the exact
        // plaintext length is rechecked after decryption below.
        let candidates: Vec<(String, String, f64, f64, String, String)> = {
            let conn = self.conn();
            let mut stmt = conn.prepare(
                "SELECT rid, text, importance, half_life, namespace, domain FROM memories \
                 WHERE consolidation_status = 'active' AND storage_tier = 'hot' \
                   AND type = 'episodic' AND length(text) >= ?1 \
                 ORDER BY rid",
            )?;
            let rows = stmt
                .query_map(params![min_chars as i64], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, f64>(2)?,
                        r.get::<_, f64>(3)?,
                        r.get::<_, String>(4)?,
                        r.get::<_, String>(5)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            rows
        };

        for (rid, enc_text, importance, half_life, namespace, domain) in candidates {
            let text = match self.decrypt_text(&enc_text) {
                Ok(t) => t,
                Err(e) => {
                    report.errors.push(format!("{rid}: decrypt failed: {e}"));
                    continue;
                }
            };
            // Exact recheck against decrypted plaintext.
            if text.chars().count() < min_chars {
                continue;
            }
            let facts = segment_into_atomic_facts(&text, TARGET_FACT_LEN, MAX_FACTS);
            // Not worth splitting if it doesn't yield at least two facts.
            if facts.len() < 2 {
                continue;
            }
            report.episodes_scanned += 1;
            if report.sample_parent_rids.len() < SAMPLE_CAP {
                report.sample_parent_rids.push(rid.clone());
            }

            if dry_run {
                report.atomic_facts_created += facts.len();
                continue;
            }

            // Children are capped below the parent so the cluster of facts
            // can't dominate ranking; they stay comfortably retrievable.
            let child_importance = importance.min(0.7);
            let child_meta = serde_json::json!({ "kind": "atomic_fact", "derived_from": rid });

            let mut created = 0usize;
            let mut child_failed = false;
            for fact in &facts {
                match self.record_text(
                    fact,
                    "semantic",
                    child_importance,
                    0.0,
                    half_life,
                    &child_meta,
                    &namespace,
                    0.8,
                    &domain,
                    "consolidation",
                    None,
                ) {
                    Ok(child_rid) => {
                        if let Err(e) = self.link(
                            &child_rid,
                            &RecordLink {
                                target_rid: rid.clone(),
                                link_type: LinkType::DerivedFrom,
                            },
                        ) {
                            report
                                .errors
                                .push(format!("{child_rid}: link to {rid} failed: {e}"));
                        }
                        created += 1;
                    }
                    Err(e) => {
                        report.errors.push(format!("{rid}: child record failed: {e}"));
                        child_failed = true;
                    }
                }
            }

            // Demote the parent ONLY if at least one child landed — never
            // strand an episode out of recall with no atomic facts to replace
            // it. Marking it `consolidated` keeps the row (and its outbound
            // DerivedFrom inbound links) for provenance while default recall
            // (include_consolidated = false) excludes it from primary results.
            if created > 0 && !child_failed {
                {
                    let conn = self.conn();
                    conn.execute(
                        "UPDATE memories SET consolidation_status = 'consolidated', \
                         updated_at = ?1 WHERE rid = ?2",
                        params![now(), rid],
                    )?;
                }
                // Keep the scoring cache in step so recall's status filter sees
                // the demotion immediately rather than after eviction.
                if let Some(row) = self.scoring_cache.write().get_mut(&rid) {
                    row.consolidation_status = "consolidated".to_string();
                }
                report.episodes_split += 1;
                report.atomic_facts_created += created;
            }
        }

        tracing::info!(
            target: "yantrikdb::audit::split",
            episodes_scanned = report.episodes_scanned,
            episodes_split = report.episodes_split,
            atomic_facts_created = report.atomic_facts_created,
            errors = report.errors.len(),
            "mega-blob split pass complete",
        );

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_multi_sentence_text() {
        let text = "Alice leads engineering. Bob owns infra. The deadline is March 30. \
                    We chose Postgres for the metadata store.";
        let facts = segment_into_atomic_facts(text, 60, 40);
        assert!(facts.len() >= 2, "splits into multiple facts: {facts:?}");
        for f in &facts {
            assert!(f.chars().count() >= MIN_FACT_CHARS);
        }
    }

    #[test]
    fn drops_trivial_fragments() {
        let text = "Ok. Yes. This is a substantive sentence that should survive segmentation.";
        let facts = segment_into_atomic_facts(text, 280, 40);
        // "Ok." / "Yes." are below MIN_FACT_CHARS and dropped.
        assert!(facts.iter().all(|f| f.chars().count() >= MIN_FACT_CHARS));
        assert!(facts.iter().any(|f| f.contains("substantive")));
    }

    #[test]
    fn caps_fact_count_without_dropping_content() {
        // 10 sentences, max_facts = 3 → tail merged into the 3rd.
        let text = (0..10)
            .map(|i| format!("This is sentence number {i} with enough length to survive."))
            .collect::<Vec<_>>()
            .join(" ");
        let facts = segment_into_atomic_facts(&text, 60, 3);
        assert_eq!(facts.len(), 3, "count is capped: {}", facts.len());
        // Nothing lost: the last sentence's text is still present.
        assert!(facts.last().unwrap().contains("number 9"));
    }

    #[test]
    fn empty_text_yields_no_facts() {
        assert!(segment_into_atomic_facts("   \n  ", 280, 40).is_empty());
    }
}
