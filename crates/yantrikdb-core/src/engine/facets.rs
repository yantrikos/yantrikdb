//! Typed facets over the synthesis substrate.
//!
//! First facet: `standing_instruction` — a user-authored durable directive
//! ("Always ...") persisted as an atomic synthesis record so recall can give
//! it deliberate salience instead of hoping a vector hit surfaces it.
//!
//! Contract: `docs/standing_instruction_facet_design.md` (behavior-normative).
//! The mechanism was promoted by two preregistered frozen-40 AMB runs
//! (discovery +0.100 CI [.019,.200]; fresh-session replication +0.144
//! CI [.056,.244]) before this implementation existed; the acceptance gate is
//! that a product-backed replay reproduces those panels hash-for-hash.
//!
//! Design laws inherited from the contract (each traceable to a measured
//! failure earlier in the arc):
//! - Recall-critical type information lives in real columns
//!   (`synthesis_axis` / `synthesis_granularity`), never only in metadata
//!   JSON (#149: stored-but-unreachable).
//! - Evidence is engine-resolved through `synthesis_dependencies`, never
//!   trusted from caller metadata (the provenance-laundering lesson).
//! - The detector is an admission-candidate generator, not an authority
//!   oracle: provenance (`source == "user"`) is checked against the stored
//!   record, and correction/forget invalidate through the EXISTING synthesis
//!   lifecycle — no parallel machinery to drift.
//! - v1 is deliberately narrow: first lexical token `Always`, complete user
//!   turns only. `Never`/`Stop`/preferences need contradiction semantics and
//!   their own benchmark gate.

use crate::base::error::Result;

/// Detector identity stamped into every facet and idempotency key. Bump the
/// version when detection rules change: same source revision + same detector
/// version must always produce the same facet (idempotent replay), so a rule
/// change under the same version would be a silent contract violation.
pub const STANDING_INSTRUCTION_DETECTOR: &str = "explicit_always_v1";

/// Axis value for standing-instruction facets in `memories.synthesis_axis`.
pub const STANDING_INSTRUCTION_AXIS: &str = "standing_instruction";

/// The v1 detection verdict for a single candidate user turn.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FacetDetection {
    /// The full directive text to persist (never a paraphrase).
    StandingInstruction { directive: String },
    /// Not a directive; the reason is audit vocabulary, not error text.
    Rejected(&'static str),
}

/// `explicit_always_v1`: decide whether one complete user-authored turn is a
/// standing instruction.
///
/// Rules (contract §Write-Time Detection, all four burned into tests before
/// this function existed):
/// 1. Caller must have verified speaker provenance; this function only sees
///    text and judges FORM. Provenance rejection happens at the extraction
///    pass against the stored record's `source` column.
/// 2. Ignore leading whitespace; the first lexical token must be `Always`,
///    case-insensitively, as a standalone word.
/// 3. Store the complete directive without paraphrasing.
/// 4. Reject titles ("Always Sunny in..."), negation prose ("I do not
///    always..."), quoted/embedded occurrences, mid-sentence `always`, and
///    empty directives ("Always." / "Always").
pub(crate) fn detect_standing_instruction_v1(turn_text: &str) -> FacetDetection {
    let trimmed = turn_text.trim();
    if trimmed.is_empty() {
        return FacetDetection::Rejected("empty_turn");
    }

    // First lexical token must be exactly `always` (case-insensitive). Any
    // prefix — "I always", quotes, bullets — is a rejection, not a repair:
    // v1 never guesses intent from surrounding prose.
    let mut tokens = trimmed.split_whitespace();
    let first = tokens.next().unwrap_or("");
    let first_word = first.trim_matches(|c: char| !c.is_alphanumeric());
    if !first_word.eq_ignore_ascii_case("always") {
        return FacetDetection::Rejected("non_directive");
    }
    // A quoted or bracketed opening ("Always..." / ["Always...) means the
    // occurrence is embedded, not the speaker's own directive form.
    if !first.starts_with(|c: char| c.is_alphabetic()) {
        return FacetDetection::Rejected("embedded_occurrence");
    }

    // There must be a directive body beyond the keyword.
    let rest: Vec<&str> = tokens.collect();
    if rest.is_empty() {
        return FacetDetection::Rejected("empty_directive");
    }

    // Title heuristic, v2 — v1's "any lowercase token in the body" fired on
    // "Always Sunny in Philadelphia" ("in" qualified); the corpus caught it
    // on this implementation's first run. The discriminating position is the
    // FIRST body token: directives lead with a lowercase verb or connective
    // ("use", "reply", "keep", "when"), while titles capitalize the next word
    // ("Sunny", "Coca-Cola", "BE"). Known v1 narrowing, documented: a
    // directive whose first body word is a capitalized token ("Always CC
    // Priya on invoices") does not fire — acceptable for explicit_always_v1,
    // revisit with evidence.
    let first_body = rest[0].trim_matches(|c: char| !c.is_alphanumeric());
    let leads_lowercase = first_body.chars().next().is_some_and(|c| c.is_lowercase());
    if !leads_lowercase {
        return FacetDetection::Rejected("title_form");
    }

    FacetDetection::StandingInstruction {
        directive: trimmed.to_string(),
    }
}

/// Normalized form used for the logical key / idempotency: casefold and
/// collapse whitespace. The PERSISTED text stays verbatim (contract rule 3);
/// only the identity key normalizes, so "Always  reply in French" and
/// "always reply in French" are one facet, not two.
pub(crate) fn normalize_directive(directive: &str) -> String {
    directive
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Dry-run / live audit counters, names fixed by the contract §False-Fire
/// Audit — these exact fields are the observable surface tests pin.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct FacetExtractionAudit {
    pub user_turns_scanned: u64,
    pub candidates: u64,
    pub accepted: u64,
    pub rejected_unverified_provenance: u64,
    pub rejected_non_directive: u64,
    pub duplicate_candidates: u64,
    pub would_write: u64,
    /// RIDs written (live runs only; dry runs leave this empty).
    pub written_rids: Vec<String>,
    pub detector_version: String,
    pub namespace: String,
    pub dry_run: bool,
}

impl crate::YantrikDB {
    /// Extract standing-instruction facets from user-authored records in
    /// `namespace`.
    ///
    /// Bounded, resumable, safe to retry: idempotency is carried by the
    /// synthesis logical key (normalized directive + detector version), so a
    /// rerun re-derives the same facets and writes nothing new. `dry_run`
    /// never writes records, dependencies, idempotency claims, or cursor
    /// progress — it only counts (contract §False-Fire Audit).
    ///
    /// Provenance: only records whose stored `source` is exactly `user` are
    /// candidates. The stored column is the ingestion boundary's verdict; text
    /// content never overrides it.
    pub fn extract_standing_instructions(
        &self,
        namespace: &str,
        dry_run: bool,
    ) -> Result<FacetExtractionAudit> {
        let mut audit = FacetExtractionAudit {
            detector_version: STANDING_INSTRUCTION_DETECTOR.to_string(),
            namespace: namespace.to_string(),
            dry_run,
            ..Default::default()
        };

        // Snapshot candidate rows first; the write path below re-locks per
        // facet. Ordinary user records only — synthesized rows can never be
        // evidence for further synthesis in v1 (no derivation chains).
        let candidates: Vec<(String, String, String, f64)> = {
            let conn = self.conn();
            let mut stmt = conn.prepare(
                "SELECT rid, text, source, created_at FROM memories \
                 WHERE namespace = ?1 \
                   AND synthesis_axis IS NULL \
                   AND consolidation_status != 'forgotten' \
                 ORDER BY created_at ASC",
            )?;
            let rows = stmt.query_map(rusqlite::params![namespace], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, f64>(3)?,
                ))
            })?;
            rows.collect::<std::result::Result<Vec<_>, _>>()?
        };

        let mut seen_keys: std::collections::HashSet<String> = {
            // Existing facets in this namespace: replays must dedupe against
            // what is already persisted, not just within this batch.
            let conn = self.conn();
            let mut stmt = conn.prepare(
                "SELECT synthesis_logical_key FROM memories \
                 WHERE namespace = ?1 AND synthesis_axis = ?2 \
                   AND synthesis_logical_key IS NOT NULL",
            )?;
            let rows = stmt.query_map(
                rusqlite::params![namespace, STANDING_INSTRUCTION_AXIS],
                |r| r.get::<_, String>(0),
            )?;
            rows.collect::<std::result::Result<_, _>>()?
        };

        for (rid, text, source, _created_at) in candidates {
            audit.user_turns_scanned += 1;
            if source != "user" {
                // Not user-authored at the ingestion boundary: an assistant
                // echo of a directive must never become an instruction.
                match detect_standing_instruction_v1(&text) {
                    FacetDetection::StandingInstruction { .. } => {
                        audit.rejected_unverified_provenance += 1;
                    }
                    FacetDetection::Rejected(_) => {}
                }
                continue;
            }
            match detect_standing_instruction_v1(&text) {
                FacetDetection::Rejected(_) => {
                    audit.rejected_non_directive += 1;
                }
                FacetDetection::StandingInstruction { directive } => {
                    audit.candidates += 1;
                    let key = format!(
                        "{STANDING_INSTRUCTION_DETECTOR}:{}",
                        normalize_directive(&directive)
                    );
                    if !seen_keys.insert(key.clone()) {
                        audit.duplicate_candidates += 1;
                        continue;
                    }
                    audit.accepted += 1;
                    if dry_run {
                        audit.would_write += 1;
                        continue;
                    }

                    // The existing synthesis writer supplies everything the
                    // contract demands: engine-resolved dependency closure,
                    // evidence hashing, idempotent retry on the logical key,
                    // and generation supersession when evidence changes.
                    let metadata = serde_json::json!({
                        "facet_type": STANDING_INSTRUCTION_AXIS,
                        "detector_version": STANDING_INSTRUCTION_DETECTOR,
                        "source_actor": "user",
                    });
                    let written = crate::cognition::consolidate::record_synthesis(
                        self,
                        &[rid.clone()],
                        &directive,
                        None,
                        STANDING_INSTRUCTION_AXIS,
                        "atomic",
                        &metadata,
                        &key,
                    )?;
                    audit.would_write += 1;
                    // Same key precedence the substrate's own tests use.
                    if let Some(r) = written
                        .get("consolidated_rid")
                        .and_then(|v| v.as_str())
                        .or_else(|| written.get("rid").and_then(|v| v.as_str()))
                    {
                        audit.written_rids.push(r.to_string());
                    }
                }
            }
        }
        Ok(audit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // == the false-fire corpus: written before the detector, every case from
    // == the contract's adversarial list. These are permanent regression
    // == tests per the contract: false-fires found later get appended here.

    fn fires(text: &str) -> bool {
        matches!(
            detect_standing_instruction_v1(text),
            FacetDetection::StandingInstruction { .. }
        )
    }

    #[test]
    fn accepts_genuine_directives() {
        for t in [
            "Always reply to me in French.",
            "always use the metric system",
            "  Always keep my calendar entries private.",
            "Always, when summarizing, keep exact dates.",
            "ALWAYS cite the source turn when you answer.",
        ] {
            assert!(fires(t), "should fire: {t:?}");
        }
    }

    #[test]
    fn rejects_titles_and_proper_noun_forms() {
        // "Always Sunny in Philadelphia" caught the first title heuristic on
        // this implementation's very first corpus run ("in" satisfied an
        // any-lowercase-token rule). Permanent regression case.
        for t in [
            "Always Sunny in Philadelphia",
            "Always Coca-Cola",
            "Always BE CLOSING",
        ] {
            assert!(!fires(t), "title form must not fire: {t:?}");
        }
        // Documented v1 narrowing, not a bug: a genuine directive leading
        // with a capitalized token does not fire under explicit_always_v1.
        assert!(!fires("Always CC Priya on the invoices."));
    }

    #[test]
    fn rejects_negation_and_mid_sentence_always() {
        for t in [
            "I do not always remember to save my work.",
            "I always forget her birthday.",
            "We should always have been more careful.",
            "It's not always about the money.",
        ] {
            assert!(!fires(t), "prose must not fire: {t:?}");
        }
    }

    #[test]
    fn rejects_quoted_and_embedded_occurrences() {
        for t in [
            "\"Always reply in French\" is what she told me.",
            "'Always check twice' was his motto.",
            "(Always draft first) — that's the rule they use.",
        ] {
            assert!(!fires(t), "embedded occurrence must not fire: {t:?}");
        }
    }

    #[test]
    fn rejects_empty_directives() {
        for t in ["Always", "Always.", "  always  ", ""] {
            assert!(!fires(t), "empty directive must not fire: {t:?}");
        }
    }

    #[test]
    fn persists_verbatim_and_normalizes_key_separately() {
        let d = match detect_standing_instruction_v1("Always  reply   in French.") {
            FacetDetection::StandingInstruction { directive } => directive,
            other => panic!("expected fire, got {other:?}"),
        };
        // Verbatim text (trimmed only), no whitespace collapsing in storage.
        assert_eq!(d, "Always  reply   in French.");
        // Identity key collapses whitespace and case.
        assert_eq!(normalize_directive(&d), "always reply in french.");
        assert_eq!(
            normalize_directive("ALWAYS REPLY IN FRENCH."),
            "always reply in french."
        );
    }

    #[test]
    fn rejection_reasons_are_audit_vocabulary() {
        assert_eq!(
            detect_standing_instruction_v1("I always forget."),
            FacetDetection::Rejected("non_directive")
        );
        assert_eq!(
            detect_standing_instruction_v1("Always"),
            FacetDetection::Rejected("empty_directive")
        );
        assert_eq!(
            detect_standing_instruction_v1("Always Sunny In Philadelphia"),
            FacetDetection::Rejected("title_form")
        );
        assert_eq!(
            detect_standing_instruction_v1("\"Always reply in French\""),
            FacetDetection::Rejected("embedded_occurrence")
        );
    }
}

#[cfg(all(test, feature = "bundled-embedder"))]
mod integration_tests {
    use super::*;
    use crate::YantrikDB;

    fn meta() -> serde_json::Value {
        serde_json::json!({})
    }

    fn record_user_turn(db: &YantrikDB, text: &str, source: &str) -> String {
        db.record_text(
            text,
            "episodic",
            0.5,
            0.0,
            604800.0,
            &meta(),
            "n",
            0.8,
            "general",
            source,
            None,
        )
        .unwrap()
    }

    /// One store exercising every audit counter: two genuine directives (one
    /// duplicated in a different case/whitespace form), adversarial
    /// negatives, and an assistant-authored directive that must be rejected
    /// on PROVENANCE even though its text would fire.
    fn seeded_store() -> YantrikDB {
        let db = YantrikDB::with_default(":memory:").unwrap();
        record_user_turn(&db, "Always reply to me in French.", "user");
        record_user_turn(&db, "always  reply to me in FRENCH.", "user"); // dup by key
        record_user_turn(&db, "Always keep my calendar entries private.", "user");
        record_user_turn(&db, "I always forget her birthday.", "user"); // negation prose
        record_user_turn(&db, "Always Sunny in Philadelphia", "user"); // title
        record_user_turn(&db, "What's the weather like today?", "user");
        // Assistant echo: correct text form, wrong author. The stored source
        // column is the ingestion boundary's verdict; the text never wins.
        record_user_turn(
            &db,
            "Always send the weekly report on Fridays.",
            "inference",
        );
        db
    }

    #[test]
    fn dry_run_counts_everything_and_writes_nothing() {
        let db = seeded_store();
        let audit = db.extract_standing_instructions("n", true).unwrap();
        assert!(audit.dry_run);
        assert_eq!(audit.user_turns_scanned, 7);
        assert_eq!(audit.candidates, 3, "two directives + one duplicate form");
        assert_eq!(audit.accepted, 2);
        assert_eq!(audit.duplicate_candidates, 1);
        assert_eq!(audit.rejected_unverified_provenance, 1);
        assert_eq!(audit.rejected_non_directive, 3);
        assert_eq!(audit.would_write, 2);
        assert!(audit.written_rids.is_empty(), "dry run must not write");

        // The store is untouched: no synthesis rows, no dependencies.
        let conn = db.conn();
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memories WHERE synthesis_axis = ?1",
                rusqlite::params![STANDING_INSTRUCTION_AXIS],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 0, "dry run wrote a synthesis row");
        let d: i64 = conn
            .query_row("SELECT COUNT(*) FROM synthesis_dependencies", [], |r| {
                r.get(0)
            })
            .unwrap();
        assert_eq!(d, 0, "dry run wrote a dependency");
    }

    #[test]
    fn live_run_persists_facets_with_evidence_and_replays_idempotently() {
        let db = seeded_store();
        let audit = db.extract_standing_instructions("n", false).unwrap();
        assert_eq!(audit.accepted, 2, "full audit: {audit:?}");
        assert_eq!(audit.written_rids.len(), 2);

        // Facets are real synthesis rows with engine-resolved dependencies.
        {
            let conn = db.conn();
            let n: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM memories WHERE synthesis_axis = ?1 \
                     AND synthesis_granularity = 'atomic' AND namespace = 'n'",
                    rusqlite::params![STANDING_INSTRUCTION_AXIS],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(n, 2);
            let deps: i64 = conn
                .query_row("SELECT COUNT(*) FROM synthesis_dependencies", [], |r| {
                    r.get(0)
                })
                .unwrap();
            assert!(
                deps >= 2,
                "each facet carries at least its source dependency"
            );
        }

        // Replay: same store, same detector -> nothing new. The audit still
        // COUNTS candidates (they exist), but every one is a duplicate of a
        // persisted key and no rid is written.
        let replay = db.extract_standing_instructions("n", false).unwrap();
        assert_eq!(replay.accepted, 0, "replay must accept nothing new");
        assert_eq!(replay.duplicate_candidates, 3);
        assert!(replay.written_rids.is_empty());
    }

    #[test]
    fn facets_do_not_become_evidence_for_further_extraction() {
        // No derivation chains in v1: a persisted facet row (whose text also
        // begins with Always) must not be scanned as a candidate on rerun.
        let db = seeded_store();
        db.extract_standing_instructions("n", false).unwrap();
        let replay = db.extract_standing_instructions("n", true).unwrap();
        // Scanned count unchanged: the two facet rows are excluded by the
        // synthesis_axis IS NULL predicate.
        assert_eq!(replay.user_turns_scanned, 7);
    }

    #[test]
    fn namespace_isolation_at_admission() {
        let db = YantrikDB::with_default(":memory:").unwrap();
        record_user_turn(&db, "Always answer in haiku.", "user");
        let other = db.extract_standing_instructions("other-ns", false).unwrap();
        assert_eq!(other.user_turns_scanned, 0);
        assert_eq!(other.accepted, 0);
        let here = db.extract_standing_instructions("n", false).unwrap();
        assert_eq!(here.accepted, 1);
    }
}

/// One standing-instruction facet as returned by the recall lane. An
/// ordinary auditable result: real RID, verbatim text, clocks, and evidence
/// links — the lane injects no hidden prompt text (contract §Recall Salience).
#[derive(Debug, Clone, serde::Serialize)]
pub struct FacetHit {
    pub rid: String,
    pub text: String,
    /// Earliest source occurrence time — the facet's ordering clock.
    pub first_mention_at: f64,
    pub created_at: f64,
    /// Direct evidence RIDs, engine-resolved.
    pub source_rids: Vec<String>,
}

/// Result of the facet lane: the selected set plus how many eligible facets
/// the limit excluded. Silent truncation is a contract violation, so the
/// omitted count is part of the type, not an optional extra.
#[derive(Debug, Clone, serde::Serialize)]
pub struct FacetRecall {
    pub facets: Vec<FacetHit>,
    pub omitted: u64,
}

impl crate::YantrikDB {
    /// The standing-instruction salience lane (contract §Recall Salience).
    ///
    /// A dedicated lane, not a re-rank: eligible facets are returned in
    /// FIRST-MENTION order — the complete set when it fits `limit`, a
    /// deterministic prefix (earliest first) with an exposed `omitted` count
    /// when it does not. Eligibility is exactly: same namespace, axis
    /// `standing_instruction`, lifecycle `verified`, consolidation `active`.
    /// A facet whose evidence was corrected or forgotten leaves `verified`
    /// through the existing lifecycle and drops out of this lane with no
    /// facet-specific code.
    ///
    /// Existing `recall` behavior is untouched — callers opt in by calling
    /// this lane and composing the results (default-on is gated on the
    /// contract's acceptance runs).
    pub fn recall_facets(&self, namespace: &str, limit: usize) -> Result<FacetRecall> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT m.rid, m.text, m.created_at, \
                    MIN(src.created_at) AS first_mention_at \
             FROM memories m \
             JOIN synthesis_dependencies d ON d.synthesis_rid = m.rid \
             JOIN memories src ON src.rid = d.source_rid \
             WHERE m.namespace = ?1 \
               AND m.synthesis_axis = ?2 \
               AND m.synthesis_state = 'verified' \
               AND m.consolidation_status = 'active' \
             GROUP BY m.rid, m.text, m.created_at \
             ORDER BY first_mention_at ASC, m.rid ASC",
        )?;
        let rows: Vec<(String, String, f64, f64)> = stmt
            .query_map(
                rusqlite::params![namespace, STANDING_INSTRUCTION_AXIS],
                |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, f64>(2)?,
                        r.get::<_, f64>(3)?,
                    ))
                },
            )?
            .collect::<std::result::Result<_, _>>()?;

        let total = rows.len();
        let mut facets = Vec::with_capacity(total.min(limit));
        for (rid, text, created_at, first_mention_at) in rows.into_iter().take(limit) {
            let mut dep = conn.prepare(
                "SELECT source_rid FROM synthesis_dependencies \
                 WHERE synthesis_rid = ?1 AND is_direct = 1 \
                 ORDER BY source_rid",
            )?;
            let source_rids: Vec<String> = dep
                .query_map(rusqlite::params![rid], |r| r.get::<_, String>(0))?
                .collect::<std::result::Result<_, _>>()?;
            drop(dep);
            facets.push(FacetHit {
                rid,
                text,
                first_mention_at,
                created_at,
                source_rids,
            });
        }
        Ok(FacetRecall {
            facets,
            omitted: total.saturating_sub(limit).max(0) as u64,
        })
    }
}

#[cfg(all(test, feature = "bundled-embedder"))]
mod lane_tests {
    use super::*;
    use crate::YantrikDB;

    fn seed(db: &YantrikDB, text: &str) {
        db.record_text(
            text,
            "episodic",
            0.5,
            0.0,
            604800.0,
            &serde_json::json!({}),
            "n",
            0.8,
            "general",
            "user",
            None,
        )
        .unwrap();
    }

    #[test]
    fn lane_returns_complete_set_in_first_mention_order() {
        let db = YantrikDB::with_default(":memory:").unwrap();
        // Insertion order IS first-mention order here; the lane must
        // preserve it even though extraction scans in the same order.
        seed(&db, "Always reply to me in French.");
        seed(&db, "Always keep my calendar entries private.");
        seed(&db, "Always cite the source turn in answers.");
        db.extract_standing_instructions("n", false).unwrap();

        let out = db.recall_facets("n", 8).unwrap();
        assert_eq!(out.facets.len(), 3);
        assert_eq!(out.omitted, 0);
        assert!(out.facets[0].text.contains("French"));
        assert!(out.facets[1].text.contains("calendar"));
        assert!(out.facets[2].text.contains("cite the source"));
        // Ordering clock is monotone and every facet carries evidence.
        assert!(out.facets[0].first_mention_at <= out.facets[1].first_mention_at);
        for f in &out.facets {
            assert!(!f.source_rids.is_empty(), "facet without evidence: {f:?}");
        }
    }

    #[test]
    fn lane_truncates_deterministically_and_exposes_omitted() {
        let db = YantrikDB::with_default(":memory:").unwrap();
        seed(&db, "Always reply to me in French.");
        seed(&db, "Always keep my calendar entries private.");
        seed(&db, "Always cite the source turn in answers.");
        db.extract_standing_instructions("n", false).unwrap();

        let out = db.recall_facets("n", 2).unwrap();
        assert_eq!(out.facets.len(), 2);
        assert_eq!(out.omitted, 1, "truncation must be visible, never silent");
        // Deterministic prefix: earliest first-mention wins.
        assert!(out.facets[0].text.contains("French"));
    }

    #[test]
    fn forgetting_the_evidence_removes_the_facet_from_the_lane() {
        let db = YantrikDB::with_default(":memory:").unwrap();
        seed(&db, "Always reply to me in French.");
        db.extract_standing_instructions("n", false).unwrap();
        assert_eq!(db.recall_facets("n", 8).unwrap().facets.len(), 1);

        // Forget the SOURCE. The existing synthesis lifecycle must pull the
        // facet out of `verified`, and the lane must reflect that with no
        // facet-specific invalidation code.
        let source_rid: String = {
            let conn = db.conn();
            conn.query_row(
                "SELECT source_rid FROM synthesis_dependencies LIMIT 1",
                [],
                |r| r.get(0),
            )
            .unwrap()
        };
        db.forget(&source_rid).unwrap();

        let after = db.recall_facets("n", 8).unwrap();
        assert!(
            after.facets.is_empty(),
            "facet must leave the lane when its evidence is forgotten: {after:?}"
        );
    }

    #[test]
    fn lane_is_namespace_isolated() {
        let db = YantrikDB::with_default(":memory:").unwrap();
        seed(&db, "Always answer in haiku.");
        db.extract_standing_instructions("n", false).unwrap();
        assert!(db.recall_facets("other", 8).unwrap().facets.is_empty());
    }
}
