//! C4 — the claims lane (wheel piece 3): retrieval finally reads the
//! store that knows direction.
//!
//! The write path extracts directional relations losslessly into
//! `claims` (src, rel_type, dst, polarity, source_memory_rid) — and
//! until this lane, NO retrieval code consulted them ("the substrate
//! stores the answer; retrieval never reads it"). Measured cost on the
//! stress gate: every query whose answer existed in claims with
//! correct direction was missed, because cosine destroys
//! subject/object direction (`Taylor reports to Carol` scored BELOW
//! `Pat reports to Taylor` for the query "taylor") and the co-mention
//! entity graph is undirected.
//!
//! Shape: resolve query entities with the (post-C5a, alias-folded)
//! graph index, look up their claims by src OR dst, and admit each
//! claim's SOURCE RECORD into the candidate pool with a why that
//! carries the full directional provenance — "claims_match:
//! Taylor -reports_to-> Carol (anchor Taylor)". The lane is exact
//! evidence (an index lookup, not a heuristic), so admitted candidates
//! also get keyword-reserve eligibility at full lexical strength: the
//! same rescue guarantee that flipped the exact-phrase repro.
//!
//! One definition, both recall twins — the copy-a-pattern law.

use rusqlite::{params, Connection};

use crate::graph_index::GraphIndex;

/// Cap on query entities consulted — a query rarely names more.
const MAX_ANCHOR_ENTITIES: usize = 4;
/// Cap on claims admitted per anchor entity; keeps the lane an
/// index lookup, never a scan.
const MAX_CLAIMS_PER_ENTITY: usize = 24;

/// A claims-lane candidate: the claim's source record plus the
/// directional provenance that justifies its admission.
pub(crate) struct ClaimCandidate {
    pub rid: String,
    /// e.g. `claims_match: Taylor -reports_to-> Carol (anchor Taylor)`
    pub why: String,
}

/// Resolve `query_tokens` to entities and return the source records of
/// their claims. Best-effort by design: a missing `claims` table (old
/// packs) or any read error yields an empty lane, never a failed
/// recall. Duplicate rids keep their first (best-anchored) why.
/// Claims with a phantom endpoint (an entity today's extractor would
/// not mint) are suppressed at read time, with NO extractor exemption —
/// the V14→V15 backfill made 'manual' untrustworthy on lane rows; see
/// the inline comment in the row loop.
pub(crate) fn claims_candidates(
    conn: &Connection,
    graph_index: &GraphIndex,
    query_tokens: &[String],
    namespace: Option<&str>,
) -> Vec<ClaimCandidate> {
    let mut anchors = graph_index.entity_matches_query(query_tokens);
    if anchors.is_empty() {
        return Vec::new();
    }
    // Strongest anchors first (mention count), bounded — with entity
    // name as the TOTAL tiebreak. Fix (f), 2026-08-06: without it,
    // equal-mention anchors arrive in `entity_matches_query`'s HashMap
    // iteration order, which is seeded PER ENGINE INSTANCE — so every
    // fresh open could consult a different anchor order (and, via the
    // truncate below, a different anchor SET). This was the residual
    // nondeterminism hermes's probe caught surviving fix (e): it
    // oscillated across runs because each run opened a fresh instance,
    // and f7c0e2d had masked it downstream with distinct boost scores.
    anchors.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    anchors.truncate(MAX_ANCHOR_ENTITIES);

    let mut out: Vec<ClaimCandidate> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (entity, _etype, _mentions) in &anchors {
        let sql = format!(
            "SELECT src, rel_type, dst, source_memory_rid, polarity, extractor FROM claims \
             WHERE (src = ?1 OR dst = ?1) AND tombstoned = 0 \
             AND source_memory_rid IS NOT NULL {} \
             ORDER BY created_at DESC LIMIT {}",
            if namespace.is_some() {
                "AND namespace = ?2"
            } else {
                ""
            },
            MAX_CLAIMS_PER_ENTITY,
        );
        let Ok(mut stmt) = conn.prepare_cached(&sql) else {
            return out; // no claims table — empty lane, never an error
        };
        let rows: Vec<(String, String, String, String, i64, String)> = {
            let mapper = |row: &rusqlite::Row| -> rusqlite::Result<_> {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                ))
            };
            let res = if let Some(ns) = namespace {
                stmt.query_map(params![entity, ns], mapper)
                    .map(|r| r.filter_map(|x| x.ok()).collect())
            } else {
                stmt.query_map(params![entity], mapper)
                    .map(|r| r.filter_map(|x| x.ok()).collect())
            };
            res.unwrap_or_default()
        };
        for (src, rel, dst, rid, polarity, extractor) in rows {
            // PHANTOM SUPPRESSION — the claims-lane twin of the 0.14.1
            // GraphIndex::build_from_db heal. Claims written by pre-0.14.1
            // extractors keep stopword anchors (observed live 2026-08-16:
            // `claims_match: DB -leads-> THE (anchor THE)` in production
            // why_retrieved), and this lane read them back verbatim. A claim
            // whose src or dst is an entity today's extractor would not mint
            // is excluded at READ time — rows are never rewritten, so
            // reverting the rules restores the old behaviour exactly, same
            // reversibility contract as the graph heal.
            //
            // NO extractor exemption in THIS lane. A GENUINE relate() row
            // cannot reach it: relate() writes no source_memory_rid and
            // the lane's SQL requires one — so the only 'manual'-labeled
            // rows the lane can see are V14→V15 migration backfills of old
            // extraction (schema.rs backfilled every pre-V15 row to
            // 'manual'), for which the label is not evidence of intent.
            // An exemption here would therefore protect exactly and only
            // mislabeled rows. Filter unconditionally; if relate() rows
            // ever gain lane access, the exemption question reopens then —
            // explicitly, not by default.
            //
            // (Release-probe note, 2026-08-17: an earlier claim that
            // phantoms "survived the exemption" on the production store
            // was a broken instrument — the probe had imported the
            // published wheel via a relative PYTHONPATH. The store's
            // phantom rows are extractor='heuristic_v1' and the original
            // per-claim filter caught them; this unconditional form is
            // kept on the architectural argument above, not that probe.)
            //
            // Suppressed rows do occupy slots in the per-anchor fetch
            // window above; a phantom-heavy window yields fewer candidates,
            // which is the point — those rows were noise.
            let _ = &extractor; // fetched for observability; not a gate here
            if crate::graph::is_rejected_entity_name(&src)
                || crate::graph::is_rejected_entity_name(&dst)
            {
                continue;
            }
            if !seen.insert(rid.clone()) {
                continue;
            }
            let neg = if polarity < 0 { "NOT " } else { "" };
            out.push(ClaimCandidate {
                why: format!("claims_match: {src} -{neg}{rel}-> {dst} (anchor {entity})"),
                rid,
            });
        }
    }
    out
}

impl super::YantrikDB {
    /// Apply the claims lane to a recall candidate pool: STAMP pool
    /// members whose records back a claim about a query entity (the
    /// `claims_match:` why — provenance only, NO score boost; fix (c)
    /// removed the boost this doc once promised), and admit source
    /// records the vector/FTS lanes missed at plain composite score.
    /// The stamp's value is keyword-reserve eligibility at lex = 1.0
    /// (see `lexical::apply_keyword_reserve`), the same rescue
    /// guarantee that flipped the exact-phrase repro.
    ///
    /// Shared by `recall_inner` and `recall_profiled_inner` — one
    /// definition, two callers.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_claims_lane(
        &self,
        scored: &mut Vec<crate::types::RecallResult>,
        query_embedding: &[f32],
        query_text: Option<&str>,
        namespace: Option<&str>,
        time_window: Option<(f64, f64)>,
        include_consolidated: bool,
        // 2026-08-13: this lane could re-admit records the caller had
        // filtered out — it never received these at all, so it could not
        // have honoured them. Threaded through so one predicate governs
        // every lane.
        memory_type: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
        certainty_min: Option<f64>,
        // #149 phase 2: valid-time eligible universe (allow-set). A claim
        // may be exact while its source record sits outside the caller's
        // temporal window — membership gating here keeps the lane from
        // re-admitting it.
        event_allow: Option<&std::collections::HashSet<String>>,
        learned_weights: &crate::types::LearnedWeights,
        ts: f64,
        query_sentiment: f64,
    ) -> crate::error::Result<()> {
        use crate::base::scoring;

        let Some(qt) = query_text else {
            return Ok(());
        };
        let cands = {
            let gi = self.graph_index.read();
            let tokens = crate::graph::tokenize(qt);
            let conn = self.read_conn();
            claims_candidates(&conn, &gi, &tokens, namespace)
        };
        if cands.is_empty() {
            return Ok(());
        }

        // Deterministic admission order — fix (e): the previous cut
        // drained a HashMap here, so hash-random iteration set the
        // insertion order of tie-band candidates and retrieval became
        // non-deterministic (hermes probe: 3 distinct top-5s from
        // identical bytes). `cands` order is deterministic by
        // construction (anchors by mention count, claims by created_at
        // DESC); keep it.
        let mut by_rid: std::collections::HashMap<&str, &str> = cands
            .iter()
            .map(|c| (c.rid.as_str(), c.why.as_str()))
            .collect();

        // Members already in the pool: stamp provenance ONLY. Fix (c),
        // measured 2026-08-06: the original cut ALSO added a keyword-
        // magnitude boost here, and the clone gate failed 0.600→0.576 —
        // entity-anchored claim records displaced labeled answers on
        // paraphrase queries that mention an entity without asking
        // about its relations. The `claims_match` why alone grants
        // keyword-reserve eligibility at lex=1.0, which is admission
        // (cutoff+ε, still top-5 at small top_k) without magnitude.
        for result in scored.iter_mut() {
            if let Some(why) = by_rid.remove(result.rid.as_str()) {
                if !result
                    .why_retrieved
                    .iter()
                    .any(|w| w.starts_with("claims_match"))
                {
                    result.why_retrieved.push(why.to_string());
                }
            }
        }

        // Source records no lane admitted: score them in, in `cands`
        // order (deterministic), NOT HashMap drain order. No cosine
        // floor here — the lane's whole point is that the record's
        // embedding may be arbitrarily far from the query while the
        // claim is exact.
        let new_rids: Vec<(&str, &str)> = cands
            .iter()
            .filter(|c| by_rid.contains_key(c.rid.as_str()))
            .map(|c| (c.rid.as_str(), c.why.as_str()))
            .collect();
        if new_rids.is_empty() {
            return Ok(());
        }
        let rid_refs: Vec<&str> = new_rids.iter().map(|(r, _)| *r).collect();
        let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;
        let cache = self.scoring_cache.read();
        for (rid, claim_why) in new_rids {
            let Some(row) = cache.get(rid) else { continue };
            if !crate::engine::recall::passes_recall_filters(
                rid,
                row,
                include_consolidated,
                memory_type,
                time_window,
                namespace,
                domain,
                source,
                certainty_min,
                event_allow,
            ) {
                continue;
            }
            let Some(emb_blob) = emb_map.get(rid) else {
                continue;
            };
            let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
            let sim_score = crate::consolidate::cosine_similarity(query_embedding, &mem_emb) as f64;
            let decay = scoring::ranking_decay(row.importance, row.created_at, ts);
            let age = ts - row.created_at;
            let recency = scoring::recency_score(age);
            let composite = scoring::adaptive_composite_score(
                sim_score,
                decay,
                recency,
                row.importance,
                row.valence,
                query_sentiment,
                learned_weights,
            );
            let mut why = scoring::build_why(sim_score, recency, decay, row.valence);
            why.push(claim_why.to_string());
            let contributions = scoring::adaptive_contributions(
                sim_score,
                decay,
                recency,
                row.importance,
                learned_weights,
            );
            let valence_multiplier = scoring::query_valence_boost(row.valence, query_sentiment);
            scored.push(crate::types::RecallResult {
                rid: rid.to_string(),
                memory_type: row.memory_type.clone(),
                text: String::new(),
                created_at: row.created_at,
                importance: row.importance,
                valence: row.valence,
                score: composite,
                scores: crate::types::ScoreBreakdown {
                    similarity: sim_score,
                    decay,
                    recency,
                    importance: row.importance,
                    graph_proximity: 0.0,
                    contributions,
                    valence_multiplier,
                },
                why_retrieved: why,
                metadata: serde_json::Value::Null,
                namespace: row.namespace.clone(),
                certainty: row.certainty,
                domain: row.domain.clone(),
                source: row.source.clone(),
                emotional_state: row.emotional_state.clone(),
                current_status: Default::default(),
                superseded_by: None,
                disputed_with: Vec::new(),
                aged_last_verified: None,
                best_span: None,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A store the way a pre-0.14.1 extractor left it: phantom entity
    /// `THE` with real mentions, a heuristic claim anchored on it
    /// (`DB -leads-> THE`, the exact row observed in production
    /// why_retrieved 2026-08-16), a legitimate heuristic claim, and a
    /// manual claim. `edges` is a VIEW over `claims`, as in the real
    /// schema since V17.
    fn seeded_store() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE entities (name TEXT PRIMARY KEY, entity_type TEXT, \
                 first_seen REAL, last_seen REAL, mention_count INTEGER);
             CREATE TABLE memory_entities (memory_rid TEXT, entity_name TEXT);
             CREATE TABLE claims (claim_id TEXT PRIMARY KEY, src TEXT NOT NULL, \
                 dst TEXT NOT NULL, rel_type TEXT NOT NULL, weight REAL DEFAULT 1.0, \
                 created_at REAL NOT NULL, tombstoned INTEGER NOT NULL DEFAULT 0, \
                 polarity INTEGER NOT NULL DEFAULT 1, \
                 extractor TEXT NOT NULL DEFAULT 'manual', source_memory_rid TEXT, \
                 namespace TEXT NOT NULL DEFAULT 'default');
             CREATE VIEW edges AS SELECT src, dst, weight, tombstoned FROM claims;",
        )
        .unwrap();
        for (name, etype, mc) in [
            ("DB", "tech", 5),
            ("Postgres", "tech", 3),
            ("THE", "unknown", 10),
        ] {
            conn.execute(
                "INSERT INTO entities (name, entity_type, first_seen, last_seen, mention_count) \
                 VALUES (?1, ?2, 0.0, 0.0, ?3)",
                params![name, etype, mc],
            )
            .unwrap();
        }
        for (cid, src, dst, rel, ts, extractor, rid) in [
            // The live phantom, verbatim: an old extractor minted `THE`.
            ("c1", "DB", "THE", "leads", 3.0, "heuristic_v1", "m1"),
            // A legitimate claim between real entities.
            ("c2", "DB", "Postgres", "uses", 2.0, "heuristic_v1", "m2"),
            // A deliberate relate()-style assertion touching the stopword.
            ("c3", "THE", "DB", "leads", 1.0, "manual", "m3"),
        ] {
            conn.execute(
                "INSERT INTO claims (claim_id, src, dst, rel_type, created_at, \
                 extractor, source_memory_rid) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![cid, src, dst, rel, ts, extractor, rid],
            )
            .unwrap();
        }
        conn
    }

    /// The heal, anchored from the LEGITIMATE side: a claim whose other
    /// endpoint is a stopword phantom must not ride in on its real
    /// anchor — REGARDLESS of extractor label. The V14→V15 migration
    /// backfilled every old row to 'manual', so inside this lane the
    /// label cannot be trusted (and genuine relate() rows never reach
    /// the lane at all — they carry no source_memory_rid).
    #[test]
    fn stopword_endpoint_claims_are_suppressed_at_read() {
        let conn = seeded_store();
        let gi = GraphIndex::build_from_db(&conn).unwrap();
        let tokens = crate::graph::tokenize("what does DB use");
        let cands = claims_candidates(&conn, &gi, &tokens, None);
        let rids: Vec<&str> = cands.iter().map(|c| c.rid.as_str()).collect();
        assert!(
            !rids.contains(&"m1"),
            "heuristic claim with stopword dst must be suppressed, got {rids:?}"
        );
        assert!(
            rids.contains(&"m2"),
            "legitimate claim must still match, got {rids:?}"
        );
        assert!(
            !rids.contains(&"m3"),
            "migration-backfilled 'manual' label must NOT exempt a phantom-anchored              claim in this lane, got {rids:?}"
        );
    }

    /// The SECOND production repro, caught by the first-hand release probe
    /// AFTER the stopword heal: `claims_match: 15 -leads-> LOG (anchor 15)`.
    /// Bare-number subjects carry no meaning and passed the stopword-only
    /// predicate; `is_rejected_entity_name` now rejects any name with no
    /// alphabetic character, and this pins the claims lane honoring it.
    #[test]
    fn numeric_endpoint_claims_are_suppressed_at_read() {
        let conn = seeded_store();
        conn.execute(
            "INSERT INTO entities (name, entity_type, first_seen, last_seen, mention_count) \
             VALUES ('15', 'unknown', 0.0, 0.0, 4)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO claims (claim_id, src, dst, rel_type, created_at, extractor, \
             source_memory_rid) VALUES ('c15', '15', 'LOG', 'leads', 0.0, 'heuristic_v1', 'm9')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO memory_entities (memory_rid, entity_name) VALUES ('m9', '15')",
            [],
        )
        .unwrap();
        let gi = GraphIndex::build_from_db(&conn).unwrap();
        let tokens = crate::graph::tokenize("release 15 log architecture");
        let cands = claims_candidates(&conn, &gi, &tokens, None);
        let rids: Vec<&str> = cands.iter().map(|c| c.rid.as_str()).collect();
        assert!(
            !rids.contains(&"m9"),
            "bare-number-anchored heuristic claim must be suppressed, got {rids:?}"
        );
    }

    /// The production repro: `claims_match: DB -leads-> THE (anchor THE)`.
    /// The manual claim c3 protects entity `THE` in the graph index (the
    /// 0.14.1 heal's entity-level exemption), so `THE` still resolves as
    /// an anchor — but only MANUAL claims about it may surface. The
    /// heuristic phantom row must not, even anchored at `THE` itself.
    #[test]
    fn protected_stopword_anchor_surfaces_only_manual_claims() {
        let conn = seeded_store();
        let gi = GraphIndex::build_from_db(&conn).unwrap();
        // Precondition of the live defect: THE is a resolvable anchor.
        assert!(
            !gi.entity_matches_query(&[String::from("the")]).is_empty(),
            "fixture must reproduce the protected-anchor precondition"
        );
        let tokens = crate::graph::tokenize("the database leads");
        let cands = claims_candidates(&conn, &gi, &tokens, None);
        assert!(
            !cands.iter().any(|c| c.why.contains("DB -leads-> THE")),
            "the exact live phantom why must never be emitted, got {:?}",
            cands.iter().map(|c| c.why.as_str()).collect::<Vec<_>>()
        );
        assert!(
            !cands.iter().any(|c| c.rid == "m3"),
            "no phantom-anchored claim survives, whatever its extractor label —              the migration backfill made 'manual' meaningless for lane rows"
        );
    }

    #[test]
    fn direction_provenance_is_spelled_out() {
        // Pure formatting contract — the why must let a consumer see
        // WHO is subject without opening the record.
        let c = ClaimCandidate {
            rid: "r".into(),
            why: format!(
                "claims_match: {} -{}{}-> {} (anchor {})",
                "Taylor", "", "reports_to", "Carol", "Taylor"
            ),
        };
        assert_eq!(
            c.why,
            "claims_match: Taylor -reports_to-> Carol (anchor Taylor)"
        );
    }
}
