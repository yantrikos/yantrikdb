use std::collections::HashMap;

use rusqlite::{params, OptionalExtension};

use crate::error::{Result, YantrikDbError};
use crate::scoring;
use crate::types::*;

use super::{now, TextMetadataRow, YantrikDB};

/// Issue #46: how the final `recall()` top_k is ordered after MMR.
///
/// - `Relevance` (default): the existing behaviour — sort by composite
///   relevance score descending. This is what users have always seen.
/// - `Certainty`: re-sort the top_k by `certainty` descending. Useful for
///   "give me the most-confident matches first" when downstream consumers
///   want to weight their reasoning toward high-confidence claims.
/// - `Recency`: re-sort the top_k by `created_at` descending. Useful for
///   "what did I write about this recently?" The candidate set is still
///   the MMR-diverse relevance pool, but presentation is recency-first.
/// - `FirstMention`: re-sort the top_k by engine-owned
///   `metadata.first_mention_at` ascending, falling back to `created_at`.
///   This presents a relevance-selected synthesized-item set chronologically
///   without confusing evidence availability with first mention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecallOrder {
    Relevance,
    Certainty,
    Recency,
    FirstMention,
}

/// Parse the `order` string from the engine `recall()` surface. Accepts:
/// `"relevance"`, `"certainty"`, `"recency"`, `"first_mention"`, and the
/// `"chronological"` alias. `None` and `Some("relevance")` both map to
/// `Relevance` so the default is stable. Unknown strings return a typed error
/// so callers see the typo immediately.
pub(crate) fn parse_recall_order(order: Option<&str>) -> Result<RecallOrder> {
    match order {
        None | Some("relevance") => Ok(RecallOrder::Relevance),
        Some("certainty") => Ok(RecallOrder::Certainty),
        Some("recency") => Ok(RecallOrder::Recency),
        Some("first_mention" | "chronological") => Ok(RecallOrder::FirstMention),
        Some(other) => Err(YantrikDbError::InvalidInput(format!(
            "recall: invalid `order` value {other:?}; expected one of \
             \"relevance\" (default) | \"certainty\" | \"recency\" | \
             \"first_mention\" | \"chronological\""
        ))),
    }
}

pub(super) const MAX_OVERSAMPLED_RECALL_CANDIDATES: usize = 10_000;
pub(super) const MAX_TRACKED_RECALL_LIMIT_NAMESPACES: usize = 1_024;
const RECALL_LIMIT_NAMESPACE_OVERFLOW: &str = "<other>";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct RecallFetchPlan {
    pub(super) fetch_k: usize,
    pub(super) requested_candidates: usize,
    pub(super) candidate_cap: usize,
    pub(super) cap_bound: bool,
}

impl RecallFetchPlan {
    fn report(
        self,
        requested_top_k: usize,
        index_len: usize,
        has_post_filters: bool,
    ) -> crate::types::RetrievalLimits {
        crate::types::RetrievalLimits {
            requested_top_k,
            requested_candidates: self.requested_candidates,
            candidate_cap: self.candidate_cap,
            fetch_k: self.fetch_k,
            index_len,
            has_post_filters,
            cap_bound: self.cap_bound,
        }
    }
}

pub(super) fn recall_fetch_plan(
    top_k: usize,
    index_len: usize,
    has_post_filters: bool,
) -> RecallFetchPlan {
    if top_k == 0 {
        return RecallFetchPlan {
            fetch_k: 0,
            requested_candidates: 0,
            candidate_cap: MAX_OVERSAMPLED_RECALL_CANDIDATES.max(top_k),
            cap_bound: false,
        };
    }
    let ceiling = MAX_OVERSAMPLED_RECALL_CANDIDATES.max(top_k);
    let mut requested_candidates = top_k.saturating_mul(20);
    if has_post_filters {
        // Filtering happens after HNSW candidate generation. Exhaust small
        // indexes so a dense unqualified prefix cannot hide every eligible row.
        requested_candidates = requested_candidates.max(index_len);
    }
    let uncapped_fetch = requested_candidates.min(index_len);
    let fetch_k = uncapped_fetch.min(ceiling);
    RecallFetchPlan {
        fetch_k,
        requested_candidates,
        candidate_cap: ceiling,
        cap_bound: fetch_k < uncapped_fetch,
    }
}

/// Row mapper shared by every FTS phase: `(rid, bm25 rank)`. The raw
/// (negative) rank FTS5 already computes for the ORDER BY now also
/// ships to `lexical::lexical_strengths`, so the keyword lane can tell
/// a rare-term exact match from common-term noise.
fn rid_rank_row(row: &rusqlite::Row) -> rusqlite::Result<(String, f64)> {
    Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
}

/// Simple English suffix stripping for FTS5 query expansion.
///
/// Returns a stem suitable for FTS5 prefix matching (e.g., "reading" → "read",
/// used as `read*` in FTS5 MATCH). This is data-agnostic — it works for any
/// English text without hardcoded domain knowledge.
fn simple_stem(word: &str) -> Option<String> {
    if word.len() <= 4 {
        return None;
    }
    // Ordered longest-first so we strip the most specific suffix.
    let suffixes: &[(&str, usize)] = &[
        ("ations", 3),
        ("ation", 3),
        ("tions", 3),
        ("ments", 3),
        ("tion", 3),
        ("sion", 3),
        ("ment", 3),
        ("ence", 3),
        ("ance", 3),
        ("ness", 3),
        ("ible", 3),
        ("able", 3),
        ("ful", 3),
        ("ous", 3),
        ("ive", 3),
        ("ary", 3),
        ("ery", 3),
        ("ory", 3),
        ("ing", 3),
        ("ble", 3),
        ("ity", 3),
        ("ish", 3),
        ("ed", 3),
        ("er", 3),
        ("ly", 3),
        ("al", 3),
        ("es", 3),
        ("s", 3),
    ];
    for &(suffix, min_stem) in suffixes {
        if word.ends_with(suffix) && word.len() - suffix.len() >= min_stem {
            return Some(word[..word.len() - suffix.len()].to_string());
        }
    }
    None
}

/// Map of irregular verb forms → base form, used to expand FTS queries.
/// E.g., query contains "grow" → also search for "grew" and "grown".
/// Each entry: (form, &[all_forms]) — we map any form to all alternate forms.
const IRREGULAR_VERBS: &[(&str, &[&str])] = &[
    ("grow", &["grew", "grown", "growing"]),
    ("grew", &["grow", "grown", "growing"]),
    ("go", &["went", "gone", "going"]),
    ("went", &["go", "gone", "going"]),
    ("come", &["came", "coming"]),
    ("came", &["come", "coming"]),
    ("run", &["ran", "running"]),
    ("ran", &["run", "running"]),
    ("eat", &["ate", "eaten", "eating"]),
    ("ate", &["eat", "eaten", "eating"]),
    ("drink", &["drank", "drunk", "drinking"]),
    ("drank", &["drink", "drunk", "drinking"]),
    ("write", &["wrote", "written", "writing"]),
    ("wrote", &["write", "written", "writing"]),
    ("read", &["reading"]),
    ("speak", &["spoke", "spoken", "speaking"]),
    ("spoke", &["speak", "spoken", "speaking"]),
    ("think", &["thought", "thinking"]),
    ("thought", &["think", "thinking"]),
    ("buy", &["bought", "buying"]),
    ("bought", &["buy", "buying"]),
    ("teach", &["taught", "teaching"]),
    ("taught", &["teach", "teaching"]),
    ("feel", &["felt", "feeling"]),
    ("felt", &["feel", "feeling"]),
    ("keep", &["kept", "keeping"]),
    ("kept", &["keep", "keeping"]),
    ("leave", &["left", "leaving"]),
    ("left", &["leave", "leaving"]),
    ("meet", &["met", "meeting"]),
    ("met", &["meet", "meeting"]),
    ("take", &["took", "taken", "taking"]),
    ("took", &["take", "taken", "taking"]),
    ("give", &["gave", "given", "giving"]),
    ("gave", &["give", "given", "giving"]),
    ("know", &["knew", "known", "knowing"]),
    ("knew", &["know", "known", "knowing"]),
    ("see", &["saw", "seen", "seeing"]),
    ("saw", &["see", "seen", "seeing"]),
    ("begin", &["began", "begun", "beginning"]),
    ("began", &["begin", "begun", "beginning"]),
    ("break", &["broke", "broken", "breaking"]),
    ("broke", &["break", "broken", "breaking"]),
    ("drive", &["drove", "driven", "driving"]),
    ("drove", &["drive", "driven", "driving"]),
    ("sing", &["sang", "sung", "singing"]),
    ("sang", &["sing", "sung", "singing"]),
    ("swim", &["swam", "swum", "swimming"]),
    ("swam", &["swim", "swum", "swimming"]),
    ("choose", &["chose", "chosen", "choosing"]),
    ("chose", &["choose", "chosen", "choosing"]),
    ("lose", &["lost", "losing"]),
    ("lost", &["lose", "losing"]),
    ("win", &["won", "winning"]),
    ("won", &["win", "winning"]),
    ("sleep", &["slept", "sleeping"]),
    ("slept", &["sleep", "sleeping"]),
    ("build", &["built", "building"]),
    ("built", &["build", "building"]),
    ("send", &["sent", "sending"]),
    ("sent", &["send", "sending"]),
    ("spend", &["spent", "spending"]),
    ("spent", &["spend", "spending"]),
    ("fall", &["fell", "fallen", "falling"]),
    ("fell", &["fall", "fallen", "falling"]),
];

/// Get irregular verb alternate forms for a word (if any).
fn irregular_verb_forms(word: &str) -> Option<&'static [&'static str]> {
    for &(form, alts) in IRREGULAR_VERBS {
        if form == word {
            return Some(alts);
        }
    }
    None
}

/// Every filter the caller asked for, in ONE place.
///
/// # Why this exists (2026-08-13, found by external code review)
///
/// `recall()` takes `memory_type`, `namespace`, `domain`, `source`,
/// `certainty_min`, `time_window` and consolidation-status filters. The
/// vector lane applied all of them. The secondary lanes did not: the FTS
/// re-admission path rechecked only status and time, the claims lane
/// omitted type/domain/source/certainty, and graph-only admission checked
/// status/type/time/namespace. Any record excluded by the caller could
/// therefore be put back by a different lane.
///
/// That is not a ranking nit. Reproduced on a fresh store: a recall with
/// `domain="work"` returned a `domain="health"` record, and one with
/// `certainty_min=0.8` returned a `certainty=0.2` record. A caller who
/// separates sensitive domains was being handed the rows they excluded.
/// (Namespace happened to hold, because most lanes did check it — but by
/// coincidence of each site remembering, which is exactly the fragility
/// this function removes.)
///
/// Filtering is a property of the REQUEST, not of the lane that happened
/// to find the row, so there is one predicate and every lane calls it.
///
/// # Valid-time bounds (#149 phase 2)
///
/// `event_allow` is the ELIGIBLE UNIVERSE for a bounded recall: the rid
/// set the SQL pre-query over `idx_memories_event_time` returned for the
/// caller's `event_after`/`event_before` window. `Some(set)` means the
/// caller set at least one bound — a row is eligible only if its rid is
/// a member (rows with NULL `event_time_min` are never members, so the
/// NULL-excluded contract falls out of membership). `None` means no
/// bound was set and valid time does not constrain anything. Threading
/// it through THIS predicate (rather than each lane checking) keeps the
/// 2026-08-13 invariant: no lane can re-admit a row the request excluded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn passes_recall_filters(
    rid: &str,
    row: &crate::types::ScoringRow,
    include_consolidated: bool,
    memory_type: Option<&str>,
    time_window: Option<(f64, f64)>,
    namespace: Option<&str>,
    domain: Option<&str>,
    source: Option<&str>,
    certainty_min: Option<f64>,
    event_allow: Option<&std::collections::HashSet<String>>,
) -> bool {
    if let Some(allow) = event_allow {
        if !allow.contains(rid) {
            return false;
        }
    }
    if !synthesis_lifecycle_allows(row) {
        return false;
    }
    let status_ok = if include_consolidated {
        row.consolidation_status == "active" || row.consolidation_status == "consolidated"
    } else {
        row.consolidation_status == "active"
    };
    if !status_ok {
        return false;
    }
    if let Some(mt) = memory_type {
        if row.memory_type != mt {
            return false;
        }
    }
    if let Some((start, end)) = time_window {
        if row.created_at < start || row.created_at > end {
            return false;
        }
    }
    if let Some(ns) = namespace {
        if row.namespace != ns {
            return false;
        }
    }
    if let Some(d) = domain {
        if row.domain != d {
            return false;
        }
    }
    if let Some(s) = source {
        if row.source != s {
            return false;
        }
    }
    if let Some(min_cert) = certainty_min {
        if row.certainty < min_cert {
            return false;
        }
    }
    true
}

fn synthesis_lifecycle_allows(row: &crate::types::ScoringRow) -> bool {
    row.synthesis_state
        .as_deref()
        .is_none_or(|state| state == "verified")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SynthesisGranularityIntent {
    Atomic,
    Rollup,
    Conflict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SynthesisRepresentationIntent {
    axis: Option<&'static str>,
    granularity: Option<SynthesisGranularityIntent>,
}

fn query_contains_phrase(words: &[String], phrase: &[&str]) -> bool {
    words
        .windows(phrase.len())
        .any(|window| window.iter().map(String::as_str).eq(phrase.iter().copied()))
}

fn synthesis_representation_intent(
    query_text: Option<&str>,
) -> Option<SynthesisRepresentationIntent> {
    // Punctuation is a boundary by design. Apostrophized and hyphenated
    // compounds therefore become multiple words; add future patterns in that
    // tokenized form rather than expecting a punctuation-bearing literal.
    let words: Vec<String> = query_text?
        .split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect();
    if words.is_empty() {
        return None;
    }

    let has_word =
        |candidates: &[&str]| words.iter().any(|word| candidates.contains(&word.as_str()));
    let has_phrase = |phrase: &[&str]| query_contains_phrase(&words, phrase);

    let asks_for_atomic = has_word(&[
        "list", "order", "ordered", "sequence", "timeline", "stages", "items", "aspects",
    ]) || has_phrase(&["walk", "me", "through"]);
    let asks_for_rollup = has_word(&[
        "summarize",
        "summarise",
        "summary",
        "overview",
        "recap",
        "theme",
        "themes",
        "pattern",
        "patterns",
        "overall",
        "broadly",
    ]);
    let granularity = match (asks_for_atomic, asks_for_rollup) {
        (true, false) => Some(SynthesisGranularityIntent::Atomic),
        (false, true) => Some(SynthesisGranularityIntent::Rollup),
        // A mixed "summarize these items as a list" request does not give
        // enough evidence to prefer either stored tier.
        (true, true) => Some(SynthesisGranularityIntent::Conflict),
        (false, false) => None,
    };

    // The axes are intentionally phrase-led. A bare "ask" in "can you ask
    // the database" must not select the user's historical `asked` view.
    let axis = if has_phrase(&["who", "said"])
        || has_phrase(&["who", "told"])
        || has_phrase(&["who", "mentioned"])
        || has_phrase(&["who", "shared"])
    {
        Some("who_said")
    } else if has_phrase(&["i", "asked"])
        || has_phrase(&["i", "ask"])
        || has_phrase(&["did", "i", "ask"])
        || has_phrase(&["what", "i", "asked"])
        || has_phrase(&["questions", "i", "asked"])
        || has_phrase(&["my", "questions"])
    {
        Some("asked")
    } else if has_phrase(&["i", "brought", "up"])
        || has_phrase(&["i", "raised"])
        || has_phrase(&["i", "mentioned"])
        || has_phrase(&["i", "shared"])
        || has_phrase(&["i", "told"])
        || has_phrase(&["my", "contributions"])
        || has_phrase(&["my", "ideas"])
        || has_phrase(&["my", "input"])
        || has_word(&["contributions"])
    {
        Some("contributed")
    } else {
        None
    };

    (axis.is_some() || granularity.is_some())
        .then_some(SynthesisRepresentationIntent { axis, granularity })
}

/// Prefer the stored representation that matches the shape of the request.
///
/// This is a match-only preference, never a filter or mismatch penalty. Raw
/// memories and unlabelled queries keep their old scores exactly, while a bad
/// heuristic classification cannot suppress otherwise relevant evidence.
fn apply_synthesis_representation_preference(
    scored: &mut [RecallResult],
    cache: &HashMap<String, crate::types::ScoringRow>,
    query_text: Option<&str>,
) {
    const GRANULARITY_MATCH_MULTIPLIER: f64 = 1.06;
    const AXIS_MATCH_MULTIPLIER: f64 = 1.04;

    let Some(intent) = synthesis_representation_intent(query_text) else {
        return;
    };

    for result in scored {
        let Some(row) = cache.get(&result.rid) else {
            continue;
        };
        if row.synthesis_state.as_deref() != Some("verified") {
            continue;
        }

        let granularity_match = match intent.granularity {
            Some(SynthesisGranularityIntent::Atomic) => {
                row.synthesis_granularity.as_deref() == Some("atomic")
            }
            Some(SynthesisGranularityIntent::Rollup) => {
                row.synthesis_granularity.as_deref() == Some("rollup")
            }
            Some(SynthesisGranularityIntent::Conflict) => false,
            None => false,
        };
        if granularity_match {
            result.score *= GRANULARITY_MATCH_MULTIPLIER;
            result.why_retrieved.push(format!(
                "representation_match:granularity={}",
                row.synthesis_granularity.as_deref().unwrap_or_default()
            ));
        }

        if intent.axis.is_some() && row.synthesis_axis.as_deref() == intent.axis {
            result.score *= AXIS_MATCH_MULTIPLIER;
            result.why_retrieved.push(format!(
                "representation_match:axis={}",
                row.synthesis_axis.as_deref().unwrap_or_default()
            ));
        }
    }
}

/// Content words of a chunk, for novelty scoring.
///
/// Deliberately crude: lowercase alphanumeric tokens of 4+ characters, no
/// stemming and no stopword list beyond the length cut. The measurement that
/// motivated this used exactly this crudeness and still bought +7.0 coverage
/// points, so precision is not where the value lies.
fn content_tokens(text: &str) -> std::collections::HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 4)
        .map(|w| w.to_lowercase())
        .collect()
}

/// Re-select `top_k` to maximise the UNION of content covered, trading some
/// per-item score for set-level breadth.
///
/// # The failure this addresses
///
/// Some questions need several distinct parts of a history at once ("list the
/// order in which I raised each aspect", "summarise how this progressed").
/// Top-k by score answers those badly: in a compressed embedding space the
/// highest-scoring chunks are near-duplicates of each other, all describing
/// whichever phase matched best, so the other phases never appear. Measured
/// on BEAM event_ordering, an ORACLE selecting from the SAME pool reached
/// 0.965 rubric coverage where top-k reached 0.592 — the content was
/// retrieved and then discarded by selection.
///
/// # Why MMR does not already do this
///
/// MMR measures redundancy by EMBEDDING similarity at lambda 0.9 (relevance
/// weighted 9:1). In a space where the entire top-100 spans 1.286x, cosine
/// cannot separate a near-duplicate from a genuinely new record. Lexical
/// novelty can.
///
/// # Shape
///
/// ```text
/// value = (1 - w) * (score / best_score) + w * new_token_fraction
/// ```
///
/// Score is normalised against the best in the set so the two terms are
/// commensurable. An ABSOLUTE score would make `w` mean something different
/// in every store — the same calibration mistake this codebase already made
/// with raw-cosine bounds.
///
/// `w = 0` reproduces the input order exactly, so the default path is
/// untouched.
pub(crate) fn apply_novelty_selection(scored: &mut Vec<crate::types::RecallResult>, top_k: usize) {
    // `tuning()` is a process-wide OnceLock, so the weight is split out as a
    // parameter: a test cannot flip a OnceLock, and an untestable knob is how
    // four parameters previously reported "inert" while simply being unwired.
    apply_novelty_selection_w(scored, top_k, crate::base::tuning::tuning().novelty_weight)
}

pub(crate) fn apply_novelty_selection_w(
    scored: &mut Vec<crate::types::RecallResult>,
    top_k: usize,
    w: f64,
) {
    if w <= 0.0 || scored.len() <= 1 || top_k == 0 {
        return;
    }
    let best = scored.iter().map(|r| r.score).fold(f64::MIN, f64::max);
    if !(best > 0.0) {
        return;
    }
    let toks: Vec<std::collections::HashSet<String>> =
        scored.iter().map(|r| content_tokens(&r.text)).collect();

    let n = scored.len().min(top_k);
    let mut picked: Vec<usize> = Vec::with_capacity(n);
    let mut taken = vec![false; scored.len()];
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for _ in 0..n {
        let mut best_i: Option<usize> = None;
        let mut best_v = f64::MIN;
        for i in 0..scored.len() {
            if taken[i] {
                continue;
            }
            let novel = if toks[i].is_empty() {
                0.0
            } else {
                toks[i].iter().filter(|t| !seen.contains(*t)).count() as f64 / toks[i].len() as f64
            };
            let v = (1.0 - w) * (scored[i].score / best) + w * novel;
            // Strict > keeps ties on the lower index, i.e. the better
            // original rank, so selection stays deterministic.
            if v > best_v {
                best_v = v;
                best_i = Some(i);
            }
        }
        let Some(i) = best_i else { break };
        taken[i] = true;
        seen.extend(toks[i].iter().cloned());
        picked.push(i);
    }
    // Unpicked candidates keep their original relative order behind the
    // selection, so nothing is dropped and the tail stays sane.
    let mut rest: Vec<usize> = (0..scored.len()).filter(|i| !taken[*i]).collect();
    picked.append(&mut rest);
    let out: Vec<crate::types::RecallResult> =
        picked.into_iter().map(|i| scored[i].clone()).collect();
    *scored = out;
}

/// Which lane a candidate is ATTRIBUTED to for quota purposes.
///
/// A record can be found by several lanes; quotas need one owner per slot, so
/// attribution is by PROVENANCE PRIORITY: the vector lane owns anything it
/// found, because that is the lane whose ordering the quota exists to protect.
/// Everything else is attributed to the strongest non-vector signal that
/// surfaced it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum LaneOwner {
    Vector,
    Lexical,
    Claims,
    Graph,
    Exploration,
}

pub(crate) fn lane_owner(
    r: &crate::types::RecallResult,
    win_by_rid: &std::collections::HashMap<String, u32>,
) -> LaneOwner {
    // Attribution follows WHAT DRIVES THE SCORE, not what found the record
    // first. The earlier vector-first rule made quotas inert: win_by_rid holds
    // every raw HNSW result, so the vector lane owned nearly the whole pool.
    //
    // Measured on a real store: for one probe the target sat at cosine rank 4
    // and recall rank 39, and ALL TWELVE records above it carried
    // `keyword_match` — including ones at cosine 0.305 beating the target's
    // 0.534. Those records are ranked where they are because of the ADDITIVE
    // lexical boost, not because the vector lane found them, so the lexical
    // lane is what a quota must be able to cap.
    let why = &r.why_retrieved;
    if why
        .iter()
        .any(|w| w == "keyword_match" || w == "fts_sourced")
    {
        return LaneOwner::Lexical;
    }
    if win_by_rid.contains_key(&r.rid) {
        return LaneOwner::Vector;
    }
    if why.iter().any(|w| w.starts_with("claims_match")) {
        return LaneOwner::Claims;
    }
    if r.scores.graph_proximity > 0.0 || why.iter().any(|w| w.starts_with("graph-connected")) {
        return LaneOwner::Graph;
    }
    LaneOwner::Exploration
}

/// Cap how many of `top_k` any one lane may claim.
///
/// # Why a count and not a score adjustment
///
/// Every score-space bound in this engine is a raw-cosine ratio, and that unit
/// is not portable: on a real 5,035-record store the entire top 100 spanned
/// 1.286x, so a "1.30x budget" could reorder 120-252 records. A quota is a
/// COUNT — "at most 2 of 8 slots" means the same thing in any embedding space,
/// at any dimension, in any corpus.
///
/// # What it fixes
///
/// Measured on that store: records the embedding ranked 4, 8 and 36 came back
/// from recall at 42, 89 and >100, displaced by whatever else flooded the
/// slots above them. A ceiling stops a lane flooding. Crucially it does NOT
/// stop a lane helping: the same measurement showed other probes RESCUED from
/// cosine rank 60 to 4 and 20 to 9. A global constant must trade one for the
/// other; a per-lane ceiling keeps both.
///
/// Ceilings only. A lane with nothing good to offer contributes nothing rather
/// than being guaranteed a slot — a floor would manufacture mediocre results,
/// which is a new failure mode rather than a fix for the old one.
///
/// Runs on the already-ranked pool and preserves relative order within every
/// lane, so it never promotes anything; it only withholds slots from a lane
/// that has already taken its share. Anything displaced stays available to
/// fill the tail, so `top_k` is still filled whenever the pool can fill it.
pub(crate) fn apply_lane_quotas(
    scored: &mut Vec<crate::types::RecallResult>,
    win_by_rid: &std::collections::HashMap<String, u32>,
    top_k: usize,
) {
    let t = crate::base::tuning::tuning();
    // Fast path: quotas are unlimited by default, so an untuned engine pays
    // one comparison and nothing else.
    if t.quota_vector >= 1.0
        && t.quota_lexical >= 1.0
        && t.quota_claims >= 1.0
        && t.quota_graph >= 1.0
        && t.quota_exploration >= 1.0
    {
        return;
    }
    if scored.is_empty() {
        return;
    }
    let cap = |frac: f64| -> usize {
        if frac >= 1.0 {
            usize::MAX
        } else {
            // At least one slot for any lane with a nonzero quota: a fraction
            // that rounds to zero would silently DISABLE a lane, which is a
            // different decision from limiting it.
            ((frac * top_k as f64).floor() as usize).max(if frac > 0.0 { 1 } else { 0 })
        }
    };
    let (mut nv, mut nl_, mut nc, mut ng, mut ne) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let (cv, cl, cc, cg, ce) = (
        cap(t.quota_vector),
        cap(t.quota_lexical),
        cap(t.quota_claims),
        cap(t.quota_graph),
        cap(t.quota_exploration),
    );
    let mut kept: Vec<crate::types::RecallResult> = Vec::with_capacity(scored.len());
    for r in scored.drain(..) {
        let owner = lane_owner(&r, win_by_rid);
        let (used, limit) = match owner {
            LaneOwner::Vector => (&mut nv, cv),
            LaneOwner::Lexical => (&mut nl_, cl),
            LaneOwner::Claims => (&mut nc, cc),
            LaneOwner::Graph => (&mut ng, cg),
            LaneOwner::Exploration => (&mut ne, ce),
        };
        if *used < limit {
            *used += 1;
            kept.push(r);
        }
    }
    // Over-quota candidates are REMOVED, not moved to the tail. Deferring
    // them was a no-op: MMR selects from the whole pool by its own criterion,
    // so anything still present could be chosen regardless of position. A
    // quota only changes WHICH records are selected if the over-quota ones
    // are not there to select. The slots a capped lane gives up are then
    // filled by the next-best candidates from OTHER lanes, which is the
    // entire point.
    //
    // If the caps sum to less than top_k the result set is genuinely smaller.
    // That is what a ceiling means, and it is an explicit configuration
    // choice rather than a defect.
    *scored = kept;
}

/// The ranking half of `why_retrieved` (2026-08-13).
///
/// Counts the distinct retrieval lanes that independently surfaced each
/// candidate — vector (`win_by_rid` membership), lexical (`keyword_match` /
/// `fts_sourced`), claims (`claims_match…`), graph (proximity or
/// `graph-connected…`) — the same classification the explain path has
/// always shown, now allowed to touch the score through the bounded
/// [`agreement_mult`](scoring::agreement_mult) (max ~+3.5% at defaults:
/// exp(ln 1.30 x 0.13), shared inversion budget with freshness and graph). Runs before keyword-slot
/// reservation in BOTH recall paths so the reserve cutoff sees final
/// scores; tags boosted rows so the boost itself is explainable.
fn apply_lane_agreement(
    scored: &mut [crate::types::RecallResult],
    win_by_rid: &std::collections::HashMap<String, u32>,
) {
    for r in scored.iter_mut() {
        let why = &r.why_retrieved;
        let mut lanes = 0usize;
        if win_by_rid.contains_key(&r.rid) {
            lanes += 1;
        }
        if why
            .iter()
            .any(|w| w == "keyword_match" || w == "fts_sourced")
        {
            lanes += 1;
        }
        if why.iter().any(|w| w.starts_with("claims_match")) {
            lanes += 1;
        }
        if r.scores.graph_proximity > 0.0 || why.iter().any(|w| w.starts_with("graph-connected")) {
            lanes += 1;
        }
        let extra = lanes.saturating_sub(1);
        if extra > 0 {
            r.score *= scoring::agreement_mult(extra);
            r.why_retrieved
                .push(format!("multi-lane agreement ({lanes} lanes)"));
        }
    }
}

impl YantrikDB {
    pub(super) fn note_recall_candidate_cap_bound(&self, namespace: Option<&str>) {
        let key = namespace.unwrap_or("*");
        let mut counts = self.recall_candidate_cap_bound_since_boot.lock();
        if let Some(count) = counts.get_mut(key) {
            *count = count.saturating_add(1);
            return;
        }

        let tracked_namespaces = counts
            .keys()
            .filter(|candidate| {
                candidate.as_str() != "*" && candidate.as_str() != RECALL_LIMIT_NAMESPACE_OVERFLOW
            })
            .count();
        let bucket =
            if namespace.is_some() && tracked_namespaces >= MAX_TRACKED_RECALL_LIMIT_NAMESPACES {
                self.recall_candidate_cap_namespace_stats_truncated_since_boot
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                RECALL_LIMIT_NAMESPACE_OVERFLOW
            } else {
                key
            };
        let count = counts.entry(bucket.to_string()).or_insert(0);
        *count = count.saturating_add(1);
    }

    /// Retrieve memories using multi-signal fusion scoring.
    /// When `expand_entities` is true, graph edges are followed to pull in
    /// entity-connected memories that pure vector search would miss.
    ///
    /// **v0.10 Item 3 — correction seqlock (sol r4/r5/r6).** Thin retry
    /// wrapper around [`Self::recall_inner`]: reads an EVEN correction epoch
    /// before candidate generation and rechecks it (Acquire-fenced) after
    /// hydration. If a text-changing correction interleaved (epoch changed /
    /// odd), the result could pair a stale ranking vector with corrected
    /// text, so it is discarded and retried. EVERY attempt validates — a
    /// result (even an empty one) is never returned without a passing
    /// recheck. Corrections are rare, so a retry is rare; after the budget
    /// (`MAX_ATTEMPTS`) the wrapper returns the retryable `RecallContended`
    /// rather than ever accepting an unvalidated read.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip(self, query_embedding), fields(top_k, expand_entities, namespace))]
    pub fn recall(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
        query_text: Option<&str>,
        skip_reinforce: bool,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
        certainty_min: Option<f64>,
        order: Option<&str>,
        include_superseded: bool,
        // #149 phase 2: valid-time bounds (epoch seconds). When either is
        // set, the temporal window defines the ELIGIBLE UNIVERSE before
        // relevance ranking and top_k (filter-first); rows with NULL
        // event_time_min are excluded; interval overlap is inclusive.
        // `event_after > event_before` is a typed InvalidInput error.
        event_after: Option<f64>,
        event_before: Option<f64>,
    ) -> Result<Vec<RecallResult>> {
        self.recall_with_limits(
            query_embedding,
            top_k,
            time_window,
            memory_type,
            include_consolidated,
            expand_entities,
            query_text,
            skip_reinforce,
            namespace,
            domain,
            source,
            certainty_min,
            order,
            include_superseded,
            event_after,
            event_before,
        )
        .map(|(results, _)| results)
    }

    #[allow(clippy::too_many_arguments)]
    fn recall_with_limits(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
        query_text: Option<&str>,
        skip_reinforce: bool,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
        certainty_min: Option<f64>,
        order: Option<&str>,
        include_superseded: bool,
        event_after: Option<f64>,
        event_before: Option<f64>,
    ) -> Result<(Vec<RecallResult>, crate::types::RetrievalLimits)> {
        // v0.10 Item 3 seqlock (sol r5): EVERY attempt validates — a result
        // is never returned without a passing epoch recheck (coherence is
        // never traded for a result). After the budget, surface a retryable
        // busy error rather than an unvalidated read.
        const MAX_ATTEMPTS: u32 = 8;
        for attempt in 0..MAX_ATTEMPTS {
            let Some(epoch0) = self.correction_epoch_even() else {
                // Even-wait timed out under a sustained correction storm.
                // Report attempts completed so far (sol r6: accurate count).
                return Err(YantrikDbError::RecallContended { attempts: attempt });
            };
            if let Some((results, limits)) = self.recall_inner(
                query_embedding,
                top_k,
                time_window,
                memory_type,
                include_consolidated,
                expand_entities,
                query_text,
                skip_reinforce,
                namespace,
                domain,
                source,
                certainty_min,
                order,
                include_superseded,
                event_after,
                event_before,
                attempt == 0,
                epoch0,
                None,
            )? {
                return Ok((results, limits));
            }
        }
        Err(YantrikDbError::RecallContended {
            attempts: MAX_ATTEMPTS,
        })
    }

    /// v0.13.1 — `recall()` with the explain surface attached: the same
    /// pipeline, plus a [`crate::types::RecallExplain`] whose `pool` is
    /// the candidate set snapshotted post-boost/post-reserve,
    /// pre-MMR-truncation — the set that ENTERS final selection.
    /// Snapshotted earlier it would show a healthy vector lane and tell
    /// you nothing; later it would only show survivors again (a k=50
    /// survivors comparison once cleared a defect living at pool
    /// positions 51–99). Explain is DEFAULT-LANE ONLY: `recall_profiled`
    /// has no explain parameter, so it cannot silently return less.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip(self, query_embedding), fields(top_k, expand_entities, namespace))]
    pub fn recall_explained(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
        query_text: Option<&str>,
        skip_reinforce: bool,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
        certainty_min: Option<f64>,
        order: Option<&str>,
        include_superseded: bool,
        event_after: Option<f64>,
        event_before: Option<f64>,
    ) -> Result<(Vec<RecallResult>, crate::types::RecallExplain)> {
        const MAX_ATTEMPTS: u32 = 8;
        let mut explain: Option<crate::types::RecallExplain> = None;
        for attempt in 0..MAX_ATTEMPTS {
            let Some(epoch0) = self.correction_epoch_even() else {
                return Err(YantrikDbError::RecallContended { attempts: attempt });
            };
            if let Some((results, _limits)) = self.recall_inner(
                query_embedding,
                top_k,
                time_window,
                memory_type,
                include_consolidated,
                expand_entities,
                query_text,
                skip_reinforce,
                namespace,
                domain,
                source,
                certainty_min,
                order,
                include_superseded,
                event_after,
                event_before,
                attempt == 0,
                epoch0,
                Some(&mut explain),
            )? {
                return Ok((results, explain.unwrap_or_default()));
            }
        }
        Err(YantrikDbError::RecallContended {
            attempts: MAX_ATTEMPTS,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn recall_inner(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
        query_text: Option<&str>,
        skip_reinforce: bool,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
        // Issue #46: confidence first-class on recall.
        // `certainty_min` filters out candidates whose `certainty < min`
        // BEFORE scoring (saves work). `order` re-sorts the final top_k
        // AFTER MMR diversity selection so callers can request "most
        // recent matches first" or "most confident matches first" without
        // the engine abandoning its relevance-based candidate retrieval.
        // Naming note: engine-internal field stays `certainty`; the param
        // mirrors that. MCP-layer can re-expose as `confidence_min` /
        // `confidence` order if it wants the user-facing rename.
        certainty_min: Option<f64>,
        order: Option<&str>,
        // v0.10 Item 1: positive-named re-admission switch (per nuron's
        // consumer review). When the status read policy is active,
        // superseded records are EXCLUDED from result eligibility by
        // default; `include_superseded = true` re-admits them — stamped
        // with `current_status = Superseded` + `superseded_by` — for
        // history/archaeology queries ("show me what I used to believe").
        // On legacy-policy databases this flag is a no-op (everything is
        // already included).
        include_superseded: bool,
        // #149 phase 2: valid-time bounds. FILTER-FIRST — when either is
        // set the eligible universe is fetched from the indexed v48
        // columns up front, every lane is constrained to it (via
        // `passes_recall_filters`' allow-set), and a dedicated universe
        // lane direct-scores every eligible row so an in-window record is
        // reachable even when the unfiltered similarity pool would never
        // surface it.
        event_after: Option<f64>,
        event_before: Option<f64>,
        // Since-boot limit telemetry counts public calls, not discarded
        // seqlock attempts. Only the first attempt increments the counter;
        // every binding attempt still emits its structured trace.
        observe_limit_binding: bool,
        // v0.10 Item 3 seqlock: the even correction epoch snapshotted by the
        // wrapper before candidate generation, rechecked (with an Acquire
        // fence) after hydration. Returns Ok(None) on mismatch → wrapper
        // retries; a result is never returned without a passing recheck.
        epoch0: u64,
        // v0.13.1 explain surface: when Some, a RecallExplain is written
        // through on the SUCCESS path only (a retried attempt overwrites,
        // a discarded one never assigns). None costs nothing.
        mut explain_sink: Option<&mut Option<crate::types::RecallExplain>>,
    ) -> Result<Option<(Vec<RecallResult>, crate::types::RetrievalLimits)>> {
        // Validate `order` upfront so callers get a clear error rather
        // than silently falling back to relevance when they typo a value.
        let recall_order = parse_recall_order(order)?;
        // #149 phase 2: an inverted valid-time window is a caller error,
        // not an empty result — an empty result claims "nothing happened
        // then", which is a different (and false) statement. Non-finite
        // bounds are rejected first (repo-wide caller-scalar rule): NaN
        // makes every comparison false, so it would silently pass the
        // inversion check AND match nothing.
        if let Some(after) = event_after {
            crate::validate::validate_scalars("recall", &[("event_after", after)])?;
        }
        if let Some(before) = event_before {
            crate::validate::validate_scalars("recall", &[("event_before", before)])?;
        }
        if let (Some(after), Some(before)) = (event_after, event_before) {
            if after > before {
                return Err(YantrikDbError::InvalidInput(format!(
                    "event_after ({after}) must be <= event_before ({before})"
                )));
            }
        }
        // #149 phase 2 integration choice, documented per the contract:
        // of the two robust shapes — (a) restrict the vector search to an
        // allow-list, (b) direct-score the eligible set — this engine's
        // HNSW surface has no allow-list parameter, but the importance-
        // fallback lane already established the direct-scoring pattern
        // (fetch embeddings by rid, cosine against the query, push into
        // the pool pre-truncation). So: ONE SQL query over
        // idx_memories_event_time builds the eligible universe as a rid
        // set; `passes_recall_filters` gates every lane on membership
        // (post-filtering a bounded pool is forbidden — that recreates
        // the bounded-by-today's-top-k false negative this issue fixes);
        // and the universe lane below direct-scores any eligible row no
        // other lane admitted. Temporal windows are typically small, so
        // direct scoring is cheap where it matters.
        let event_allow_owned: Option<std::collections::HashSet<String>> =
            if event_after.is_some() || event_before.is_some() {
                Some(self.event_time_eligible_rids(namespace, event_after, event_before)?)
            } else {
                None
            };
        let event_allow = event_allow_owned.as_ref();
        let ts = now();

        // Load per-database learned weights (falls back to defaults if none learned yet)
        let learned_weights = self.load_learned_weights()?;

        // Detect query sentiment once for directional valence boosting
        let query_sentiment = query_text
            .map(scoring::detect_query_sentiment)
            .unwrap_or(0.0);

        // Step 1: Vector candidate generation via HNSW
        // **Issue #41 brainstorm-4 §1.** Snapshot SearchState once for
        // the full recall path so the HNSW search sees the same
        // generation-anchored index every time it's queried within
        // this call.
        let state = self.search_state.load_full();
        // Fetch a large pool so selective post-filters and MMR do not silently
        // underfill top_k. Namespace is the ordinary partition key, so making
        // it exhaustive would turn the common scoped-recall path into a near
        // full traversal; namespace-aware indexes are the durable fix there.
        let has_post_filters = time_window.is_some()
            || memory_type.is_some()
            || domain.is_some()
            || source.is_some()
            || certainty_min.is_some();
        let fetch_plan = recall_fetch_plan(top_k, state.vec_index.len(), has_post_filters);
        let retrieval_limits = fetch_plan.report(top_k, state.vec_index.len(), has_post_filters);
        if fetch_plan.cap_bound {
            if observe_limit_binding {
                self.note_recall_candidate_cap_bound(namespace);
            }
            tracing::debug!(
                target: "yantrikdb::recall",
                requested_top_k = top_k,
                requested_candidates = fetch_plan.requested_candidates,
                fetch_k = fetch_plan.fetch_k,
                candidate_cap = fetch_plan.candidate_cap,
                index_len = state.vec_index.len(),
                has_post_filters,
                counted_since_boot = observe_limit_binding,
                "recall candidate cap bound"
            );
        }
        let fetch_k = fetch_plan.fetch_k;
        // v0.9.3 contract gate: a caller-supplied NaN/wrong-dim QUERY vector
        // poisons every distance in the search. Typed rejection instead.
        // (recall_with_seq / recall_with_response / recall_refine all
        // delegate here, so this single gate covers them.)
        crate::validate::validate_embedding("recall", query_embedding, state.dim())?;
        let vec_results = {
            let _span = tracing::debug_span!("hnsw_search", fetch_k).entered();
            state
                .vec_index
                .search_with_windows(query_embedding, fetch_k)?
        };
        // Winning chunk window per candidate — consumed by snippet-span
        // stamping after hydration (engine/snippet.rs). Only trusted for
        // records that actually have chunk vectors (filtered there).
        let mut win_by_rid: std::collections::HashMap<String, u32> =
            std::collections::HashMap::with_capacity(vec_results.len());
        for (rid, _, w) in &vec_results {
            win_by_rid.insert(rid.clone(), *w);
        }
        let vec_results: Vec<(String, f64)> = vec_results
            .into_iter()
            .map(|(rid, dist, _)| (rid, dist))
            .collect();

        let capture_inst = self as *const Self as usize;
        // v0.13.1 explain: lane-admission tracking. Rows created by lanes
        // that stamp a why marker (fts_sourced / claims_match /
        // cold_memory / graph-connected) are attributed from the marker at
        // snapshot time; the three markerless admission paths (vector,
        // importance fallback, valence scan) are tracked by rid here.
        let explain_on = explain_sink.is_some();
        let mut explain_fallback_rids: std::collections::HashSet<String> = Default::default();
        let mut explain_valence_rids: std::collections::HashSet<String> = Default::default();
        // #149 phase 2: rows admitted by the valid-time universe lane
        // (direct-scored because no similarity lane surfaced them).
        let mut explain_event_universe_rids: std::collections::HashSet<String> = Default::default();
        // Pack rows live in their pack's scoring cache, not the host cache.
        // Track their provenance across the merged-pool lifecycle check; the
        // pack collector has already applied the same lifecycle predicate.
        let mut pack_rids: std::collections::HashSet<String> = Default::default();
        let mut explain_fts_ran = false;
        if crate::engine::capture::enabled() {
            crate::engine::capture::emit(
                capture_inst,
                "hnsw_pool",
                serde_json::json!(vec_results
                    .iter()
                    .map(|(rid, d)| (rid.as_str(), crate::engine::capture::bits(*d)))
                    .collect::<Vec<_>>()),
            );
        }

        // This short-circuit's premise is that the host index is the only
        // candidate source, which mounting a pack falsifies. Taking it
        // with a pack mounted would make the flagship case — a database
        // with few or no memories of its own mounting a knowledge pack —
        // return nothing at all. A valid-time-bounded recall (#149
        // phase 2) falsifies the premise the same way: its universe lane
        // sources candidates from SQL over the v48 columns, so an empty
        // vector pool (e.g. a freshly applied replication follower) must
        // still fall through to the universe lane.
        if vec_results.is_empty() && self.packs.read().is_empty() && event_allow.is_none() {
            // No candidates → no vector/text pairing to protect. Still
            // validate the epoch (sol r6): a successful recall must never be
            // returned during an unvalidated correction interval, even when
            // empty. On mismatch return None so the wrapper retries against a
            // stable epoch; a genuinely empty index re-searches empty and
            // validates once corrections quiesce.
            if !self.correction_epoch_validate(epoch0) {
                return Ok(None);
            }
            // Explain must not be silently empty on this path: the vector
            // lane RAN and found nothing, and no pack was mounted — say so.
            if let Some(sink) = explain_sink.as_deref_mut() {
                let mut lanes = std::collections::BTreeMap::new();
                lanes.insert(
                    "vector".to_string(),
                    crate::types::ExplainLaneReport {
                        status: "ran_empty".to_string(),
                        candidates: 0,
                        reason: None,
                    },
                );
                *sink = Some(crate::types::RecallExplain {
                    retrieval_limits: retrieval_limits.clone(),
                    comparator: "rank_cmp: score quantized at 1e-6 desc, rid asc".to_string(),
                    score_algebra: "empty index short-circuit — no scoring ran".to_string(),
                    query_sentiment: 0.0,
                    bm25_near_best_fraction: None,
                    lanes,
                    pool: Vec::new(),
                });
            }
            return Ok(Some((vec![], retrieval_limits)));
        }

        // Candidate count BEFORE filtering, for the gate diagnostic: it
        // separates "the vector layer returned few" from "the filters ate
        // them", which no amount of reading the selection code can.
        let vec_candidate_count = vec_results.len();
        // Step 2: Score from in-memory cache (replaces fetch_memories_by_rids)
        let mut scored: Vec<RecallResult> = Vec::new();
        {
            let cache = self.scoring_cache.read();
            for (rid, distance) in &vec_results {
                let Some(row) = cache.get(rid) else { continue };

                // Filter: consolidation_status
                let status_ok = if include_consolidated {
                    row.consolidation_status == "active"
                        || row.consolidation_status == "consolidated"
                } else {
                    row.consolidation_status == "active"
                };
                if !status_ok {
                    continue;
                }

                // Filter: memory_type
                if let Some(mt) = memory_type {
                    if row.memory_type != mt {
                        continue;
                    }
                }

                // Filter: time_window
                if let Some((start, end)) = time_window {
                    if row.created_at < start || row.created_at > end {
                        continue;
                    }
                }

                // Filter: namespace
                if let Some(ns) = namespace {
                    if row.namespace != ns {
                        continue;
                    }
                }

                // Filter: domain (V10)
                if let Some(d) = domain {
                    if row.domain != d {
                        continue;
                    }
                }

                // Filter: source (V10)
                if let Some(s) = source {
                    if row.source != s {
                        continue;
                    }
                }

                // Filter: certainty_min (Issue #46). Drop candidates whose
                // stored certainty falls below the requested floor. Done
                // before scoring so we don't waste work on rows that won't
                // make the result set.
                if let Some(min_cert) = certainty_min {
                    if row.certainty < min_cert {
                        continue;
                    }
                }

                // Filter: valid-time eligible universe (#149 phase 2).
                // Membership in the SQL-derived allow-set — NULL event
                // times are never members, so they are excluded whenever
                // a bound is set.
                if let Some(allow) = event_allow {
                    if !allow.contains(rid.as_str()) {
                        continue;
                    }
                }

                let sim_score = (1.0 - distance).max(0.0);
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
                    &learned_weights,
                );
                let why = scoring::build_why(sim_score, recency, decay, row.valence);
                let contributions = scoring::adaptive_contributions(
                    sim_score,
                    decay,
                    recency,
                    row.importance,
                    &learned_weights,
                );
                let valence_multiplier = scoring::query_valence_boost(row.valence, query_sentiment);

                scored.push(RecallResult {
                    rid: rid.clone(),
                    memory_type: row.memory_type.clone(),
                    text: String::new(), // hydrated after top_k selection
                    created_at: row.created_at,
                    importance: row.importance,
                    valence: row.valence,
                    score: composite,
                    scores: ScoreBreakdown {
                        similarity: sim_score,
                        decay,
                        recency,
                        importance: row.importance,
                        graph_proximity: 0.0,
                        contributions,
                        valence_multiplier,
                    },
                    why_retrieved: why,
                    metadata: serde_json::Value::Null, // hydrated after top_k selection
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
        } // drop cache borrow

        let explain_len_before_fallback = scored.len();
        // Step 2.5: High-importance memory fallback (similarity-gated)
        //
        // Anchor memories define the user's life story. HNSW approximate search
        // may miss them when many noise memories dominate the nearest-neighbor pool.
        // Include important memories if they have at least moderate similarity.
        //
        // Thresholds adapt to database size: large databases need more aggressive
        // fallback because HNSW approximate search degrades with more vectors.
        {
            let total_memories = self.scoring_cache.read().len();
            let high_imp_threshold = if total_memories > 5000 { 0.5 } else { 0.7 };
            let min_sim_for_fallback = if total_memories > 5000 { 0.15 } else { 0.20 };
            let existing_rids: std::collections::HashSet<&str> =
                scored.iter().map(|r| r.rid.as_str()).collect();
            let important_rids: Vec<String> = {
                let cache = self.scoring_cache.read();
                cache
                    .iter()
                    .filter(|(rid, row)| {
                        // EIGHTH admission path. It listed most filters by hand
                        // and omitted certainty_min, so a high-importance,
                        // low-certainty row the vector pool missed could be
                        // re-admitted against an explicit certainty floor.
                        // One predicate, like every other lane.
                        row.importance >= high_imp_threshold
                            && !existing_rids.contains(rid.as_str())
                            && passes_recall_filters(
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
                            )
                    })
                    .map(|(rid, _)| rid.clone())
                    .collect()
            };

            if !important_rids.is_empty() {
                let rid_refs: Vec<&str> = important_rids.iter().map(|r| r.as_str()).collect();
                let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;
                let cache = self.scoring_cache.read();
                for rid in &important_rids {
                    let Some(row) = cache.get(rid) else { continue };
                    let Some(emb_blob) = emb_map.get(rid.as_str()) else {
                        continue;
                    };
                    let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
                    let sim_score =
                        crate::consolidate::cosine_similarity(query_embedding, &mem_emb) as f64;

                    if sim_score < min_sim_for_fallback {
                        continue;
                    }

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
                        &learned_weights,
                    );
                    let why = scoring::build_why(sim_score, recency, decay, row.valence);
                    let contributions = scoring::adaptive_contributions(
                        sim_score,
                        decay,
                        recency,
                        row.importance,
                        &learned_weights,
                    );
                    let valence_multiplier =
                        scoring::query_valence_boost(row.valence, query_sentiment);

                    scored.push(RecallResult {
                        rid: rid.clone(),
                        memory_type: row.memory_type.clone(),
                        text: String::new(),
                        created_at: row.created_at,
                        importance: row.importance,
                        valence: row.valence,
                        score: composite,
                        scores: ScoreBreakdown {
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
            }
        }

        if explain_on {
            explain_fallback_rids.extend(
                scored[explain_len_before_fallback..]
                    .iter()
                    .map(|r| r.rid.clone()),
            );
        }

        // Step 2.6: valid-time UNIVERSE lane (#149 phase 2).
        //
        // When the caller set event_after/event_before, the temporal
        // window defines the eligible universe — and every member must be
        // REACHABLE, even one whose similarity rank sits far below the
        // vector pool's fetch_k. Post-filtering the bounded pool cannot
        // provide that (the bounded-by-today's-top-k false negative this
        // issue exists to fix), so any eligible row no lane has admitted
        // yet is direct-scored here: embed-once query cosine against the
        // row's stored embedding, same composite scoring as every other
        // lane, pushed into the pool BEFORE boosting/MMR/top_k
        // truncation. Ranking then happens strictly WITHIN the universe
        // (all other lanes are membership-gated on the same allow-set).
        // Temporal windows are typically small, so the extra embedding
        // fetch is proportional to the window, not the store.
        if let Some(allow) = event_allow {
            let missing_rids: Vec<String> = {
                let existing_rids: std::collections::HashSet<&str> =
                    scored.iter().map(|r| r.rid.as_str()).collect();
                let cache = self.scoring_cache.read();
                allow
                    .iter()
                    .filter(|rid| !existing_rids.contains(rid.as_str()))
                    .filter(|rid| {
                        cache.get(rid.as_str()).is_some_and(|row| {
                            passes_recall_filters(
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
                            )
                        })
                    })
                    .cloned()
                    .collect()
            };
            if !missing_rids.is_empty() {
                let rid_refs: Vec<&str> = missing_rids.iter().map(|r| r.as_str()).collect();
                let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;
                let cache = self.scoring_cache.read();
                for rid in &missing_rids {
                    let Some(row) = cache.get(rid) else { continue };
                    // Deliberately NO similarity floor: eligibility comes
                    // from the temporal window; similarity only RANKS
                    // within it. A floor here would silently shrink the
                    // universe — the exact false-negative class again. The
                    // same principle covers a row with no readable vector
                    // (a replication follower before vector sync): it is
                    // admitted at similarity 0.0 rather than dropped, so
                    // leader and follower agree on the eligible rid set.
                    let sim_score = match emb_map.get(rid.as_str()) {
                        Some(emb_blob) => {
                            let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
                            crate::consolidate::cosine_similarity(query_embedding, &mem_emb) as f64
                        }
                        None => 0.0,
                    };
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
                        &learned_weights,
                    );
                    let mut why = scoring::build_why(sim_score, recency, decay, row.valence);
                    why.push("event_time_window".to_string());
                    let contributions = scoring::adaptive_contributions(
                        sim_score,
                        decay,
                        recency,
                        row.importance,
                        &learned_weights,
                    );
                    let valence_multiplier =
                        scoring::query_valence_boost(row.valence, query_sentiment);

                    if explain_on {
                        explain_event_universe_rids.insert(rid.clone());
                    }
                    scored.push(RecallResult {
                        rid: rid.clone(),
                        memory_type: row.memory_type.clone(),
                        text: String::new(),
                        created_at: row.created_at,
                        importance: row.importance,
                        valence: row.valence,
                        score: composite,
                        scores: ScoreBreakdown {
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
            }
        }

        // rid → per-query lexical strength from FTS5 bm25 ranks (fusion:
        // see engine/lexical.rs). Filled by the keyword lane below, read
        // by the boost sites and the keyword reserve (step 3.5).
        let mut lex_by_rid: std::collections::HashMap<String, f64> =
            std::collections::HashMap::new();

        // Step 1.5: FTS5 keyword fallback
        //
        // Catches queries where keywords appear in memory text but HNSW
        // misses the match (e.g., "What books am I reading?" where "reading"
        // is in the text). Uses importance-weighted BM25 ranking so anchor
        // memories naturally surface above noise — no hardcoded thresholds.
        //
        // FTS limit scales dynamically with database size.
        // FTS5 only works when encryption is disabled (the default).
        if !self.is_encrypted() {
            // Wired to tuning: YANTRIKDB_FTS_MIN_SIM (default 0.05). This was a
            // local const while tuning parsed-and-fingerprinted the knob nobody
            // read — the exact unwired-parameter failure tuning.rs exists to end.
            let fts_min_sim: f64 = crate::base::tuning::tuning().fts_min_sim;

            const STOPWORDS: &[&str] = &[
                "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "what", "who",
                "how", "when", "where", "which", "why", "do", "did", "does", "have", "has", "had",
                "i", "me", "my", "mine", "we", "our", "you", "your", "to", "of", "in", "on", "at",
                "by", "for", "with", "from", "about", "tell", "and", "or", "but", "not", "no",
                "it", "its", "that", "this", "there", "s", "she", "her", "he", "his", "they",
                "them", "most", "each", "any", "all", "every", "been", "being", "up", "out", "so",
                "if", "than", "very", "just", "also",
                // v0.9.3 accuracy work: function words that were slipping
                // through and anchoring keyword_match boosts on unrelated
                // memories (eval diagnosis: "during" in "what did we learn
                // DURING the project?" boosted a memory-leak memory to rank 1
                // while the actual lesson memories missed top-10 entirely).
                "during", "while", "after", "before", "between", "into", "over", "under", "through",
                "against", "within", "without", "us", "as", "then", "some", "more", "other",
                "these", "those", "will", "would", "could", "should", "can", "may", "might",
                "must", "shall", "get", "got", "make", "made", "let",
            ];

            {
                if let Some(qt) = query_text {
                    let raw_keywords: Vec<String> = qt
                        .split(|c: char| !c.is_alphanumeric())
                        .filter(|s| !s.is_empty() && s.len() > 1)
                        .filter(|s| !STOPWORDS.contains(&s.to_lowercase().as_str()))
                        .map(|s| s.to_string())
                        .collect();

                    // Filter out person-type entity names from FTS keywords.
                    // Person names (e.g., "Priya", "Meera") appear in thousands
                    // of memories, flooding FTS results with entity-matching noise.
                    // Topic entities (e.g., "yoga", "reading") are kept — they're
                    // valuable FTS keywords that graph expansion may not cover.
                    let mut keywords: Vec<String> = {
                        let gi = self.graph_index.read();
                        let query_tokens = crate::graph::tokenize(qt);
                        let matched = gi.entity_matches_query(&query_tokens);
                        // Only filter entities with type "person" — these are the
                        // high-frequency names that cause FTS noise.
                        let person_matches: Vec<_> = matched
                            .into_iter()
                            .filter(|(_, etype, _)| etype == "person")
                            .collect();
                        if person_matches.is_empty() {
                            raw_keywords
                        } else {
                            let person_tokens: std::collections::HashSet<String> = person_matches
                                .into_iter()
                                .flat_map(|(name, _, _)| {
                                    let mut tokens = vec![name.to_lowercase()];
                                    for token in name.split_whitespace() {
                                        tokens.push(token.to_lowercase());
                                    }
                                    tokens
                                })
                                .collect();
                            let filtered: Vec<String> = raw_keywords
                                .iter()
                                .filter(|kw| !person_tokens.contains(&kw.to_lowercase()))
                                .cloned()
                                .collect();
                            if filtered.is_empty() {
                                raw_keywords
                            } else {
                                filtered
                            }
                        }
                    };

                    // Entity-seeded FTS for group/aggregation queries.
                    //
                    // For "Tell me about Priya's family", normal FTS keyword
                    // "family" won't match individual member memories ("My husband
                    // Arjun is a product manager"). Inject graph-connected person
                    // entity names as additional FTS keywords so those memories
                    // enter the scoring pool with keyword_match boost.
                    {
                        const GROUP_FTS_WORDS: &[&str] = &[
                            "team",
                            "group",
                            "colleagues",
                            "coworkers",
                            "friends",
                            "family",
                            "staff",
                            "members",
                            "people",
                        ];
                        let qt_lower = qt.to_lowercase();
                        if GROUP_FTS_WORDS.iter().any(|kw| qt_lower.contains(kw)) {
                            let gi = self.graph_index.read();
                            let query_tokens = crate::graph::tokenize(qt);
                            let matched = gi.entity_matches_query(&query_tokens);
                            if !matched.is_empty() {
                                let seed_names: Vec<&str> =
                                    matched.iter().map(|(n, _, _)| n.as_str()).collect();
                                let expanded = gi.expand_bfs(&seed_names, 2, 30);
                                let mut injected = 0usize;
                                for (name, hops, _) in &expanded {
                                    if *hops == 0 || injected >= 15 {
                                        continue;
                                    }
                                    if gi.entity_type(name).map_or(false, |t| t == "person") {
                                        for part in name.split_whitespace() {
                                            if part.len() > 1
                                                && !keywords
                                                    .iter()
                                                    .any(|k| k.eq_ignore_ascii_case(part))
                                            {
                                                keywords.push(part.to_string());
                                            }
                                        }
                                        injected += 1;
                                    }
                                }
                            }
                        }
                    }

                    if !keywords.is_empty() {
                        // Build FTS5 query with stemmed prefix expansion,
                        // irregular verb forms, and AND conjunction for
                        // multi-keyword selectivity.
                        //
                        // Each keyword becomes a group:
                        //   "reading" → ("reading" OR read*)
                        //   "grow"    → ("grow" OR "grew" OR "grown" OR "growing")
                        //
                        // Multiple groups are AND'd for selectivity:
                        //   ("books" OR book*) AND ("reading" OR read*)
                        //
                        // Falls back to OR if AND returns too few results.
                        let mut keyword_groups: Vec<String> = Vec::new();
                        for kw in &keywords {
                            let kw_lower = kw.to_lowercase();
                            let mut parts: Vec<String> = Vec::new();
                            parts.push(format!("\"{}\"", kw.replace('"', "")));
                            if let Some(stem) = simple_stem(&kw_lower) {
                                parts.push(format!("{}*", stem));
                            }
                            if let Some(alts) = irregular_verb_forms(&kw_lower) {
                                for alt in alts {
                                    parts.push(format!("\"{}\"", alt));
                                }
                            }
                            keyword_groups.push(if parts.len() == 1 {
                                parts[0].clone()
                            } else {
                                format!("({})", parts.join(" OR "))
                            });
                        }

                        // Use AND for selectivity when 2+ keyword groups,
                        // single keywords always use their group directly.
                        let fts_query_and = if keyword_groups.len() >= 2 {
                            Some(keyword_groups.join(" AND "))
                        } else {
                            None
                        };
                        let fts_query_or = keyword_groups.join(" OR ");
                        // Primary query: AND if available, else OR
                        let fts_query = fts_query_and.as_deref().unwrap_or(&fts_query_or);

                        // Adaptive FTS limit: scales with database size.
                        // Small DBs (<3K): 30 is enough. Large DBs (15K+): need 150+
                        // to surface important memories above keyword noise.
                        let total_memories = self.scoring_cache.read().len();
                        let fts_limit = (total_memories / 100).max(30).min(200);

                        // Dynamic importance threshold for Phase 2.
                        // Phase 2 ensures important keyword-matching memories surface
                        // even when noise exhausts Phase 1's BM25-ranked LIMIT.
                        // Use 70% of mean to catch relevant anchors with modest importance
                        // (e.g., yoga at imp=0.30 when mean≈0.35).
                        let mean_importance = {
                            let cache = self.scoring_cache.read();
                            if cache.is_empty() {
                                0.5
                            } else {
                                let sum: f64 = cache.values().map(|r| r.importance).sum();
                                let mean = sum / cache.len() as f64;
                                (mean * 0.7).max(0.25)
                            }
                        };

                        // Importance-weighted BM25 ranking.
                        // rank is negative in FTS5 (more negative = better BM25).
                        // Multiplying by (0.5 + importance) makes important memories
                        // more negative → sort higher.
                        let fts_sql = if memory_type.is_some() {
                            format!(
                                "SELECT m.rid, memories_fts.rank FROM memories m \
                                 JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                 WHERE memories_fts MATCH ?1 \
                                 AND m.consolidation_status = 'active' \
                                 AND m.type = ?2 \
                                 {} \
                                 ORDER BY rank * (0.5 + m.importance) \
                                 LIMIT {}",
                                if namespace.is_some() {
                                    "AND m.namespace = ?3"
                                } else {
                                    ""
                                },
                                fts_limit,
                            )
                        } else {
                            format!(
                                "SELECT m.rid, memories_fts.rank FROM memories m \
                                 JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                 WHERE memories_fts MATCH ?1 \
                                 AND m.consolidation_status = 'active' \
                                 {} \
                                 ORDER BY rank * (0.5 + m.importance) \
                                 LIMIT {}",
                                if namespace.is_some() {
                                    "AND m.namespace = ?2"
                                } else {
                                    ""
                                },
                                fts_limit,
                            )
                        };

                        // Helper closure to run an FTS query and collect RIDs.
                        let run_fts_phase1 = |q: &str| -> Vec<(String, f64)> {
                            let conn = self.read_conn();
                            let mut stmt = conn.prepare_cached(&fts_sql).ok();
                            if let Some(ref mut stmt) = stmt {
                                let result: std::result::Result<Vec<(String, f64)>, _> =
                                    if let Some(mt) = memory_type {
                                        if let Some(ns) = namespace {
                                            stmt.query_map(params![q, mt, ns], rid_rank_row)
                                                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                        } else {
                                            stmt.query_map(params![q, mt], rid_rank_row)
                                                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                        }
                                    } else if let Some(ns) = namespace {
                                        stmt.query_map(params![q, ns], rid_rank_row)
                                            .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    } else {
                                        stmt.query_map(params![q], rid_rank_row)
                                            .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    };
                                result.unwrap_or_default()
                            } else {
                                vec![]
                            }
                        };

                        // Run AND query first (more selective), fall back to OR
                        // if AND returns too few results.
                        let mut fts_hits = run_fts_phase1(fts_query);
                        if fts_hits.len() < 5 && fts_query_and.is_some() {
                            fts_hits = run_fts_phase1(&fts_query_or);
                        }
                        // (rid, bm25 rank) rows feeding the per-query lexical
                        // strengths; every later FTS phase appends its rows.
                        let mut lex_ranked: Vec<(String, f64)> = fts_hits.clone();
                        let mut fts_rids: Vec<String> =
                            fts_hits.into_iter().map(|(rid, _)| rid).collect();

                        // Phase 2: Importance-filtered FTS.
                        // Phase 1's BM25-ranked LIMIT can be exhausted by noise
                        // (e.g., "yoga" matches thousands of generated memories).
                        // Phase 2 ensures important memories with keyword matches
                        // always enter the scoring pool by filtering on importance
                        // and using a separate LIMIT.
                        //
                        // Uses AND first for selectivity, then falls back to OR
                        // when AND is too strict (e.g., "books AND reading" misses
                        // memories that only contain "reading" like "I'm reading Sapiens").
                        {
                            let imp_fts_sql = if memory_type.is_some() {
                                format!(
                                    "SELECT m.rid, memories_fts.rank FROM memories m \
                                     JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                     WHERE memories_fts MATCH ?1 \
                                     AND m.consolidation_status = 'active' \
                                     AND m.importance > ?2 \
                                     AND m.type = ?3 \
                                     {} \
                                     ORDER BY m.importance DESC \
                                     LIMIT 100",
                                    if namespace.is_some() {
                                        "AND m.namespace = ?4"
                                    } else {
                                        ""
                                    },
                                )
                            } else {
                                format!(
                                    "SELECT m.rid, memories_fts.rank FROM memories m \
                                     JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                     WHERE memories_fts MATCH ?1 \
                                     AND m.consolidation_status = 'active' \
                                     AND m.importance > ?2 \
                                     {} \
                                     ORDER BY m.importance DESC \
                                     LIMIT 100",
                                    if namespace.is_some() {
                                        "AND m.namespace = ?3"
                                    } else {
                                        ""
                                    },
                                )
                            };

                            // Helper to run Phase 2 with a given FTS query string.
                            let run_fts_phase2 = |q: &str| -> Vec<(String, f64)> {
                                let conn = self.read_conn();
                                let mut stmt = conn.prepare_cached(&imp_fts_sql).ok();
                                if let Some(ref mut stmt) = stmt {
                                    let result: std::result::Result<Vec<(String, f64)>, _> =
                                        if let Some(mt) = memory_type {
                                            if let Some(ns) = namespace {
                                                stmt.query_map(
                                                    params![q, mean_importance, mt, ns],
                                                    rid_rank_row,
                                                )
                                                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                            } else {
                                                stmt.query_map(
                                                    params![q, mean_importance, mt],
                                                    rid_rank_row,
                                                )
                                                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                            }
                                        } else if let Some(ns) = namespace {
                                            stmt.query_map(
                                                params![q, mean_importance, ns],
                                                rid_rank_row,
                                            )
                                            .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                        } else {
                                            stmt.query_map(
                                                params![q, mean_importance],
                                                rid_rank_row,
                                            )
                                            .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                        };
                                    result.unwrap_or_default()
                                } else {
                                    vec![]
                                }
                            };

                            // Run AND first (selective), fall back to OR if too few results.
                            let mut imp_hits = run_fts_phase2(fts_query);
                            if imp_hits.len() < 10 && fts_query_and.is_some() {
                                let or_hits = run_fts_phase2(&fts_query_or);
                                let existing: std::collections::HashSet<String> =
                                    imp_hits.iter().map(|(rid, _)| rid.clone()).collect();
                                imp_hits.extend(
                                    or_hits
                                        .into_iter()
                                        .filter(|(rid, _)| !existing.contains(rid)),
                                );
                            }

                            // Merge Phase 2 into Phase 1 results (dedup).
                            lex_ranked.extend(imp_hits.iter().cloned());
                            let existing_set: std::collections::HashSet<String> =
                                fts_rids.iter().cloned().collect();
                            fts_rids.extend(
                                imp_hits
                                    .into_iter()
                                    .map(|(rid, _)| rid)
                                    .filter(|rid| !existing_set.contains(rid)),
                            );
                        }

                        // Phase 2.5: Per-keyword anchor scan.
                        //
                        // When keywords match many memories (e.g., "reading"),
                        // Phase 1+2 may not surface the specific anchor memory
                        // among thousands of noise matches. Scan each individual
                        // keyword for the most important matching memories.
                        //
                        // Only targets anchor memories (importance > 0.5) and
                        // applies NO keyword_boost — they compete on pure scoring.
                        if keyword_groups.len() >= 2 {
                            let anchor_fts_sql = if memory_type.is_some() {
                                format!(
                                    "SELECT m.rid, memories_fts.rank FROM memories m \
                                     JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                     WHERE memories_fts MATCH ?1 \
                                     AND m.consolidation_status = 'active' \
                                     AND m.importance > 0.5 \
                                     AND m.type = ?2 \
                                     {} \
                                     ORDER BY m.importance DESC \
                                     LIMIT 10",
                                    if namespace.is_some() {
                                        "AND m.namespace = ?3"
                                    } else {
                                        ""
                                    },
                                )
                            } else {
                                format!(
                                    "SELECT m.rid, memories_fts.rank FROM memories m \
                                     JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                     WHERE memories_fts MATCH ?1 \
                                     AND m.consolidation_status = 'active' \
                                     AND m.importance > 0.5 \
                                     {} \
                                     ORDER BY m.importance DESC \
                                     LIMIT 10",
                                    if namespace.is_some() {
                                        "AND m.namespace = ?2"
                                    } else {
                                        ""
                                    },
                                )
                            };

                            let existing_fts: std::collections::HashSet<String> =
                                fts_rids.iter().cloned().collect();

                            for group in &keyword_groups {
                                let anchor_hits: Vec<(String, f64)> = {
                                    let conn = self.read_conn();
                                    let mut stmt = conn.prepare_cached(&anchor_fts_sql).ok();
                                    if let Some(ref mut stmt) = stmt {
                                        let result: std::result::Result<Vec<(String, f64)>, _> =
                                            if let Some(mt) = memory_type {
                                                if let Some(ns) = namespace {
                                                    stmt.query_map(
                                                        params![group, mt, ns],
                                                        rid_rank_row,
                                                    )
                                                    .map(|rows| {
                                                        rows.filter_map(|r| r.ok()).collect()
                                                    })
                                                } else {
                                                    stmt.query_map(params![group, mt], rid_rank_row)
                                                        .map(|rows| {
                                                            rows.filter_map(|r| r.ok()).collect()
                                                        })
                                                }
                                            } else if let Some(ns) = namespace {
                                                stmt.query_map(params![group, ns], rid_rank_row)
                                                    .map(|rows| {
                                                        rows.filter_map(|r| r.ok()).collect()
                                                    })
                                            } else {
                                                stmt.query_map(params![group], rid_rank_row).map(
                                                    |rows| rows.filter_map(|r| r.ok()).collect(),
                                                )
                                            };
                                        result.unwrap_or_default()
                                    } else {
                                        vec![]
                                    }
                                };

                                for (rid, rank) in anchor_hits {
                                    lex_ranked.push((rid.clone(), rank));
                                    if !existing_fts.contains(&rid) {
                                        fts_rids.push(rid);
                                    }
                                }
                            }
                        }

                        // Boost existing candidates that matched FTS5 keywords.
                        // Scale boost inversely with similarity: memories where vector
                        // search failed but keywords matched get more boost.
                        // Per-query lexical strengths from every phase's bm25
                        // rows — read by the keyword reserve, the only
                        // consumer left (see engine/lexical.rs).
                        lex_by_rid = crate::engine::lexical::lexical_strengths(&lex_ranked);
                        explain_fts_ran = true;

                        {
                            let fts_rid_set: std::collections::HashSet<&str> =
                                fts_rids.iter().map(|r| r.as_str()).collect();
                            for result in &mut scored {
                                if fts_rid_set.contains(result.rid.as_str())
                                    && !result.why_retrieved.iter().any(|w| w == "keyword_match")
                                {
                                    let sim = result.scores.similarity;
                                    let lex =
                                        lex_by_rid.get(result.rid.as_str()).copied().unwrap_or(1.0);
                                    let boost = crate::engine::lexical::keyword_lane_boost(
                                        learned_weights.keyword_boost,
                                        sim,
                                        lex,
                                    );
                                    result.score += boost;
                                    result.why_retrieved.push("keyword_match".to_string());
                                }
                            }
                        }

                        // Add new FTS candidates not already in the pool
                        let existing_rids: std::collections::HashSet<String> =
                            scored.iter().map(|r| r.rid.clone()).collect();
                        let new_fts_rids: Vec<String> = fts_rids
                            .into_iter()
                            .filter(|r| !existing_rids.contains(r))
                            .collect();

                        if !new_fts_rids.is_empty() {
                            let rid_refs: Vec<&str> =
                                new_fts_rids.iter().map(|r| r.as_str()).collect();
                            let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;

                            let cache = self.scoring_cache.read();
                            for rid in &new_fts_rids {
                                let Some(row) = cache.get(rid) else { continue };

                                // ONE predicate, same as the vector lane. This
                                // path used to check only status and time, so a
                                // record the caller excluded by domain, source or
                                // certainty could be re-admitted here.
                                if !passes_recall_filters(
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

                                let Some(emb_blob) = emb_map.get(rid.as_str()) else {
                                    continue;
                                };
                                let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
                                let sim_score = crate::consolidate::cosine_similarity(
                                    query_embedding,
                                    &mem_emb,
                                ) as f64;

                                let lex = lex_by_rid.get(rid.as_str()).copied().unwrap_or(0.0);
                                if sim_score < fts_min_sim
                                    && lex < crate::engine::lexical::LEX_STRONG
                                {
                                    // Below the cosine noise floor AND not a
                                    // near-best lexical match — noise. A top
                                    // bm25 match passes regardless: dilution
                                    // parks exact-phrase records at any sim.
                                    continue;
                                }

                                let decay =
                                    scoring::ranking_decay(row.importance, row.created_at, ts);
                                let age = ts - row.created_at;
                                let recency = scoring::recency_score(age);
                                let composite = scoring::adaptive_composite_score(
                                    sim_score,
                                    decay,
                                    recency,
                                    row.importance,
                                    row.valence,
                                    query_sentiment,
                                    &learned_weights,
                                );
                                let kw_boost = crate::engine::lexical::keyword_lane_boost(
                                    learned_weights.keyword_boost,
                                    sim_score,
                                    lex,
                                );
                                let mut why =
                                    scoring::build_why(sim_score, recency, decay, row.valence);
                                why.push("keyword_match".to_string());
                                why.push("fts_sourced".to_string());
                                let contributions = scoring::adaptive_contributions(
                                    sim_score,
                                    decay,
                                    recency,
                                    row.importance,
                                    &learned_weights,
                                );
                                let valence_multiplier =
                                    scoring::query_valence_boost(row.valence, query_sentiment);

                                scored.push(RecallResult {
                                    rid: rid.clone(),
                                    memory_type: row.memory_type.clone(),
                                    text: String::new(),
                                    created_at: row.created_at,
                                    importance: row.importance,
                                    valence: row.valence,
                                    score: composite + kw_boost,
                                    scores: ScoreBreakdown {
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
                        }
                    }
                }
            }
        }

        // Step 1.6 (C4): the claims lane — retrieval reads the one store
        // that knows relation DIRECTION (engine/claims_lane.rs). Runs on
        // encrypted databases too; needs only query text + the graph
        // index (post-C5a, alias-folded).
        self.apply_claims_lane(
            &mut scored,
            query_embedding,
            query_text,
            namespace,
            time_window,
            include_consolidated,
            memory_type,
            domain,
            source,
            certainty_min,
            event_allow,
            &learned_weights,
            ts,
            query_sentiment,
        )?;

        let explain_len_before_valence = scored.len();
        // Step 2.7: Valence-based retrieval for emotional queries
        //
        // For queries with strong sentiment (e.g., "stressful moments", "happiest times"),
        // scan the scoring cache for strongly-valenced memories that HNSW may miss due to
        // low semantic overlap. These memories are important for emotional retrieval.
        if query_sentiment.abs() > 0.5 {
            const VALENCE_SCAN_THRESHOLD: f64 = 0.4; // min |valence| to consider
            const VALENCE_SCAN_MAX: usize = 30; // max new candidates
                                                // Wired to tuning: YANTRIKDB_VALENCE_MIN_SIM (default 0.02) — very low
                                                // floor, valence is the signal.
            let valence_min_sim: f64 = crate::base::tuning::tuning().valence_min_sim;

            let existing_rids: std::collections::HashSet<&str> =
                scored.iter().map(|r| r.rid.as_str()).collect();

            // Find strongly-valenced memories matching query sentiment direction
            let valence_rids: Vec<String> = {
                let cache = self.scoring_cache.read();
                let mut candidates: Vec<(String, f64)> = cache
                    .iter()
                    .filter(|(rid, row)| {
                        row.consolidation_status == "active"
                            && !existing_rids.contains(rid.as_str())
                            && row.valence.abs() >= VALENCE_SCAN_THRESHOLD
                            // Match direction: negative query wants negative memories
                            && (query_sentiment * row.valence > 0.0
                                || (query_sentiment < 0.0 && row.valence < -0.2))
                            && row.importance >= 0.5 // only important memories
                            // Same predicate as every other lane: this one
                            // used to omit domain, source and certainty.
                            && passes_recall_filters(
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
                            )
                    })
                    .map(|(rid, row)| {
                        // Rank by |valence| * importance
                        let rank = row.valence.abs() * row.importance;
                        (rid.clone(), rank)
                    })
                    .collect();
                // Fix (k): this rank feeds a truncation, and on uniform
                // corpora it is a massive tie — the total order (quantized
                // rank desc, rid asc) is what keeps the admitted subset
                // identical across opens (the audit rule from fixes (e)/(f),
                // applied to every remaining rank-and-take site at once).
                candidates.sort_by(|a, b| {
                    crate::engine::lexical::quantize_score(b.1)
                        .total_cmp(&crate::engine::lexical::quantize_score(a.1))
                        .then_with(|| a.0.cmp(&b.0))
                });
                candidates
                    .into_iter()
                    .take(VALENCE_SCAN_MAX)
                    .map(|(rid, _)| rid)
                    .collect()
            };

            if !valence_rids.is_empty() {
                let rid_refs: Vec<&str> = valence_rids.iter().map(|r| r.as_str()).collect();
                let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;
                let cache = self.scoring_cache.read();
                for rid in &valence_rids {
                    let Some(row) = cache.get(rid) else { continue };
                    let Some(emb_blob) = emb_map.get(rid.as_str()) else {
                        continue;
                    };
                    let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
                    let sim_score =
                        crate::consolidate::cosine_similarity(query_embedding, &mem_emb) as f64;

                    if sim_score < valence_min_sim {
                        continue;
                    }

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
                        &learned_weights,
                    );
                    // Additive valence boost: helps valence-matched memories compete
                    // when cosine similarity is low. Scaled by |valence| * importance
                    // so only strongly-valenced important memories get meaningful lift.
                    // Bounded multiplicative lift, not an unbounded add:
                    // a strongly-valenced record still competes, but one
                    // with no semantic match cannot outrank a real answer.
                    let valence_lift = scoring::lane_lift_mult(row.valence.abs() * row.importance);
                    let mut why = scoring::build_why(sim_score, recency, decay, row.valence);
                    why.push("valence_match".to_string());
                    let contributions = scoring::adaptive_contributions(
                        sim_score,
                        decay,
                        recency,
                        row.importance,
                        &learned_weights,
                    );
                    let valence_multiplier =
                        scoring::query_valence_boost(row.valence, query_sentiment);

                    scored.push(RecallResult {
                        rid: rid.clone(),
                        memory_type: row.memory_type.clone(),
                        text: String::new(),
                        created_at: row.created_at,
                        importance: row.importance,
                        valence: row.valence,
                        score: composite * valence_lift,
                        scores: ScoreBreakdown {
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
            }
        }

        if explain_on {
            explain_valence_rids.extend(
                scored[explain_len_before_valence..]
                    .iter()
                    .map(|r| r.rid.clone()),
            );
        }

        // Step 2.9: Lexical rescue fallback
        //
        // When the vector lane's best score is weak, sweep FTS for records it
        // missed. Admission is a pure function of (query, corpus): the
        // activation threshold, the `cold_min_sim` floor, the dedup against
        // already-scored RIDs and `passes_recall_filters` do all the work.
        //
        // THIS LANE USED TO FILTER ON `access_count = 0` (2026-08-17). The
        // stated intent was "surface knowledge buried under frequently-accessed
        // noise", but `reinforce()` increments `access_count` on every recall
        // that RETURNS a record, so the predicate had two effects the intent
        // did not ask for:
        //
        //   1. Read history decided admission. A record surfaced once by an
        //      UNRELATED query lost its only rescue route permanently — two
        //      byte-identical stores gave different results because someone
        //      had read one of them. Same defect family as the currency bug
        //      (a prior computed from state a reader mutates), one layer up:
        //      that one leaked into scoring, this one into candidate selection.
        //   2. The lane SELF-DISABLED with use. On a mature store nearly every
        //      record has been returned at least once, so the eligible set
        //      drains toward empty — the rescue path stops working precisely
        //      when the corpus is large enough to need it.
        //
        // The lift below is keyed on `row.importance`, a property of the
        // record, so it was never part of this leak and is unchanged.
        if !self.is_encrypted() {
            if let Some(qt) = query_text {
                let best_score = scored.iter().map(|r| r.score).fold(0.0f64, f64::max);
                const COLD_ACTIVATION_THRESHOLD: f64 = 0.55;
                // Wired to tuning: YANTRIKDB_COLD_MIN_SIM (default 0.10).
                let cold_min_sim: f64 = crate::base::tuning::tuning().cold_min_sim;
                const COLD_MAX_CANDIDATES: usize = 30;

                if best_score < COLD_ACTIVATION_THRESHOLD {
                    // Extract keywords (reusing same logic as FTS5 step)
                    let cold_keywords: Vec<String> = qt
                        .split(|c: char| !c.is_alphanumeric())
                        .filter(|s| !s.is_empty() && s.len() > 1)
                        .filter(|s| {
                            const STOP: &[&str] = &[
                                "a", "an", "the", "is", "are", "am", "was", "were", "be", "been",
                                "what", "who", "how", "when", "where", "which", "why", "do", "did",
                                "does", "have", "has", "had", "i", "me", "my", "mine", "we", "our",
                                "you", "your", "to", "of", "in", "on", "at", "by", "for", "with",
                                "from", "about", "tell", "and", "or", "but", "not", "no", "it",
                                "its", "that", "this", "there", "s", "she", "her", "he", "his",
                                "they", "them",
                            ];
                            !STOP.contains(&s.to_lowercase().as_str())
                        })
                        .map(|s| s.to_string())
                        .collect();

                    if !cold_keywords.is_empty() {
                        // Build OR query (looser than AND for cold memories)
                        let mut fts_parts: Vec<String> = Vec::new();
                        for kw in &cold_keywords {
                            let kw_lower = kw.to_lowercase();
                            fts_parts.push(format!("\"{}\"", kw.replace('"', "")));
                            if let Some(stem) = simple_stem(&kw_lower) {
                                fts_parts.push(format!("{}*", stem));
                            }
                            if let Some(alts) = irregular_verb_forms(&kw_lower) {
                                for alt in alts {
                                    fts_parts.push(format!("\"{}\"", alt));
                                }
                            }
                        }
                        let cold_fts = fts_parts.join(" OR ");

                        // Lexical rescue over ALL active rows (see the lane comment)
                        let cold_sql = if memory_type.is_some() {
                            format!(
                                "SELECT m.rid FROM memories m \
                                 JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                 WHERE memories_fts MATCH ?1 \
                                 AND m.consolidation_status = 'active' \
                                 AND m.type = ?2 \
                                 {} \
                                 ORDER BY m.importance DESC \
                                 LIMIT {}",
                                if namespace.is_some() {
                                    "AND m.namespace = ?3"
                                } else {
                                    ""
                                },
                                COLD_MAX_CANDIDATES,
                            )
                        } else {
                            format!(
                                "SELECT m.rid FROM memories m \
                                 JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                 WHERE memories_fts MATCH ?1 \
                                 AND m.consolidation_status = 'active' \
                                 {} \
                                 ORDER BY m.importance DESC \
                                 LIMIT {}",
                                if namespace.is_some() {
                                    "AND m.namespace = ?2"
                                } else {
                                    ""
                                },
                                COLD_MAX_CANDIDATES,
                            )
                        };

                        let cold_rids: Vec<String> = {
                            let conn = self.read_conn();
                            let mut stmt = conn.prepare_cached(&cold_sql).ok();
                            if let Some(ref mut stmt) = stmt {
                                let result: std::result::Result<Vec<String>, _> = if let Some(mt) =
                                    memory_type
                                {
                                    if let Some(ns) = namespace {
                                        stmt.query_map(params![cold_fts, mt, ns], |row| {
                                            row.get::<_, String>(0)
                                        })
                                        .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    } else {
                                        stmt.query_map(params![cold_fts, mt], |row| {
                                            row.get::<_, String>(0)
                                        })
                                        .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    }
                                } else if let Some(ns) = namespace {
                                    stmt.query_map(params![cold_fts, ns], |row| {
                                        row.get::<_, String>(0)
                                    })
                                    .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                } else {
                                    stmt.query_map(params![cold_fts], |row| row.get::<_, String>(0))
                                        .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                };
                                result.unwrap_or_default()
                            } else {
                                vec![]
                            }
                        };

                        // Score cold candidates — filter to new RIDs only
                        let existing_rids: std::collections::HashSet<String> =
                            scored.iter().map(|r| r.rid.clone()).collect();
                        let new_cold: Vec<String> = cold_rids
                            .into_iter()
                            .filter(|r| !existing_rids.contains(r))
                            .collect();

                        if !new_cold.is_empty() {
                            let rid_refs: Vec<&str> = new_cold.iter().map(|r| r.as_str()).collect();
                            let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;
                            let cache = self.scoring_cache.read();

                            for rid in &new_cold {
                                let Some(row) = cache.get(rid) else { continue };
                                // The cold-lane SQL filters only status,
                                // access_count, type and namespace, so domain,
                                // source and certainty must be enforced here.
                                if !passes_recall_filters(
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
                                let Some(emb_blob) = emb_map.get(rid.as_str()) else {
                                    continue;
                                };
                                let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
                                let sim_score = crate::consolidate::cosine_similarity(
                                    query_embedding,
                                    &mem_emb,
                                ) as f64;

                                if sim_score < cold_min_sim {
                                    continue;
                                }

                                let decay =
                                    scoring::ranking_decay(row.importance, row.created_at, ts);
                                let age = ts - row.created_at;
                                let recency = scoring::recency_score(age);
                                let composite = scoring::adaptive_composite_score(
                                    sim_score,
                                    decay,
                                    recency,
                                    row.importance,
                                    row.valence,
                                    query_sentiment,
                                    &learned_weights,
                                );
                                // Rescue-lane lift, scaled by the record's own
                                // importance — NOT by whether it has been read.
                                // Bounded multiplicative lift (see
                                // LANE_LIFT_MAX): admission is the lane's
                                // job; it does not buy relevance.
                                let cold_lift = scoring::lane_lift_mult(row.importance);
                                let mut why =
                                    scoring::build_why(sim_score, recency, decay, row.valence);
                                why.push("lexical_rescue".to_string());
                                let contributions = scoring::adaptive_contributions(
                                    sim_score,
                                    decay,
                                    recency,
                                    row.importance,
                                    &learned_weights,
                                );
                                let valence_multiplier =
                                    scoring::query_valence_boost(row.valence, query_sentiment);

                                scored.push(RecallResult {
                                    rid: rid.clone(),
                                    memory_type: row.memory_type.clone(),
                                    text: String::new(),
                                    created_at: row.created_at,
                                    importance: row.importance,
                                    valence: row.valence,
                                    score: composite * cold_lift,
                                    scores: ScoreBreakdown {
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
                        }
                    }
                }
            }
        }

        // Step 3: Graph expansion (when enabled)
        if expand_entities {
            let _span = tracing::debug_span!("graph_expansion").entered();
            let gi = self.graph_index.read();
            let query_entities: Vec<(String, String, u32)> = if let Some(qt) = query_text {
                let query_tokens = crate::graph::tokenize(qt);
                gi.entity_matches_query(&query_tokens)
            } else {
                vec![]
            };

            let (mut base_boost, mut seed_entities, entity_idfs): (
                f64,
                Vec<String>,
                std::collections::HashMap<String, f64>,
            ) = if !query_entities.is_empty() {
                let has_person = query_entities.iter().any(|(_, etype, _)| etype == "person");
                let factor = if has_person {
                    0.20
                } else if query_entities.len() >= 2 {
                    0.15
                } else {
                    0.12
                };
                let idfs: std::collections::HashMap<String, f64> = query_entities
                    .iter()
                    .map(|(name, _, mc)| {
                        let idf = 1.0 / (1.0 + (*mc as f64).max(1.0).ln());
                        (name.to_lowercase(), idf)
                    })
                    .collect();
                let names: Vec<String> = query_entities.into_iter().map(|(n, _, _)| n).collect();
                (factor, names, idfs)
            } else if query_text.is_none() {
                // Embedding-only search (no query text): seed from top results
                let mut seed_sorted = scored.clone();
                seed_sorted.sort_by(crate::engine::lexical::rank_cmp);
                let seed_count = 3.min(seed_sorted.len());
                let seed_rids: Vec<&str> = seed_sorted[..seed_count]
                    .iter()
                    .map(|r| r.rid.as_str())
                    .collect();
                let seeds = gi.entities_for_memories(&seed_rids);
                (0.05, seeds, std::collections::HashMap::new())
            } else {
                (0.0, vec![], std::collections::HashMap::new())
            };

            // Group query expansion: if query mentions "team", "family", etc.,
            // seed expansion with person entities CONNECTED to query entities.
            //
            // Previous approach: entities_by_type("person").take(15) grabbed
            // arbitrary person entities, often filler characters unrelated to
            // the query. New approach: BFS from query entities finds people
            // actually connected to the subject (e.g., Priya → Arjun, Meera,
            // Appa, Amma for "family"; Priya → Deepa, Neha for "team").
            const GROUP_KEYWORDS: &[&str] = &[
                "team",
                "group",
                "colleagues",
                "coworkers",
                "friends",
                "family",
                "staff",
                "members",
                "people",
            ];
            if let Some(qt) = query_text {
                let qt_lower = qt.to_lowercase();
                if GROUP_KEYWORDS.iter().any(|kw| qt_lower.contains(kw)) {
                    if !seed_entities.is_empty() {
                        // BFS from query entities to find connected person entities
                        let query_seeds: Vec<&str> =
                            seed_entities.iter().map(|s| s.as_str()).collect();
                        let nearby = gi.expand_bfs(&query_seeds, 2, 50);
                        for (name, hops, _) in &nearby {
                            if *hops > 0
                                && gi.entity_type(name).map_or(false, |t| t == "person")
                                && !seed_entities.contains(&name.to_string())
                            {
                                seed_entities.push(name.clone());
                            }
                        }
                    } else {
                        // No query entities — fall back to type-based expansion
                        let person_entities = gi.entities_by_type("person");
                        for person in person_entities.into_iter().take(15) {
                            if !seed_entities.contains(&person) {
                                seed_entities.push(person);
                            }
                        }
                    }
                    base_boost = base_boost.max(0.20_f64);
                }
            }

            const MAX_BOOST_PER_MEMORY: f64 = 0.25;
            const MAX_GRAPH_FRACTION: f64 = 1.0;
            const MAX_SEED_ENTITIES: usize = 8;

            // Cap seed entities to prevent graph explosion with many entities
            if seed_entities.len() > MAX_SEED_ENTITIES {
                seed_entities.truncate(MAX_SEED_ENTITIES);
            }

            if !seed_entities.is_empty() && base_boost > 0.0 {
                let seed_refs: Vec<&str> = seed_entities.iter().map(|s| s.as_str()).collect();
                let expanded = gi.expand_bfs(&seed_refs, 2, 30);

                let expanded_map: std::collections::HashMap<String, (u8, f64)> = expanded
                    .iter()
                    .map(|(name, hops, weight)| (name.clone(), (*hops, *weight)))
                    .collect();

                // (a) IDF-weighted additive boost for existing results
                for result in &mut scored {
                    let prox = gi.graph_proximity(&result.rid, &expanded_map);
                    if prox > 0.0 {
                        let mem_entities: Vec<String> = gi
                            .entities_for_memory(&result.rid)
                            .into_iter()
                            .map(|s| s.to_string())
                            .collect();

                        let mut best_idf = 1.0f64;
                        let mut connecting_entity = String::new();
                        for entity in &mem_entities {
                            if expanded_map.contains_key(entity) {
                                let idf = entity_idfs
                                    .get(&entity.to_lowercase())
                                    .copied()
                                    .unwrap_or(1.0);
                                if connecting_entity.is_empty() || idf > best_idf {
                                    best_idf = idf;
                                    connecting_entity = entity.clone();
                                }
                            }
                        }

                        // Consolidation penalty: use consolidation_status as proxy
                        let cache = self.scoring_cache.read();
                        let consolidation_factor = cache
                            .get(&result.rid)
                            .map(|r| {
                                if r.consolidation_status == "consolidated" {
                                    0.5
                                } else {
                                    1.0
                                }
                            })
                            .unwrap_or(1.0);
                        drop(cache);

                        // MULTIPLICATIVE, not additive. This was
                        // `result.score += boost` with boost capped at 0.25 —
                        // on composites that typically run 0.2-0.6, a flat
                        // +0.25 could nearly double a weak score, so a graph
                        // edge promoted records relevance had ranked below
                        // them. Same wall as importance, freshness and the
                        // graph composite; this was the last additive site.
                        //
                        // The quality modifiers are kept and now shape the
                        // EVIDENCE rather than the magnitude: proximity,
                        // discounted by the connecting entity's inverse
                        // document frequency (a hub entity is weak evidence)
                        // and by the consolidation penalty, scaled by the
                        // expansion strength. GRAPH_SCALE alone sets the
                        // ceiling: ~+3.5% at defaults (ln 1.30 x 0.13).
                        let evidence = ((base_boost / MAX_BOOST_PER_MEMORY)
                            * prox
                            * best_idf
                            * consolidation_factor)
                            .clamp(0.0, 1.0);
                        result.scores.graph_proximity = prox;
                        result.score *= scoring::graph_mult(evidence);
                        if !connecting_entity.is_empty() {
                            result
                                .why_retrieved
                                .push(format!("graph-connected via {connecting_entity}"));
                        }
                    }
                }

                // (b) Graph-only candidates: score from cache + batch embedding fetch
                let max_graph_only = ((MAX_GRAPH_FRACTION * top_k as f64).ceil() as usize).max(1);
                let all_entity_names: Vec<&str> =
                    expanded.iter().map(|(n, _, _)| n.as_str()).collect();
                let graph_rids = gi.memories_for_entities(&all_entity_names);

                let existing_rids: std::collections::HashSet<&str> =
                    scored.iter().map(|r| r.rid.as_str()).collect();
                let new_rids: Vec<String> = graph_rids
                    .into_iter()
                    .filter(|r| !existing_rids.contains(r.as_str()))
                    .collect();

                // Filter graph-only candidates, rank by importance * graph_proximity.
                // This ensures memories directly linked to seed entities (prox≈1.0)
                // outrank memories linked through distant neighbors (prox≈0.25).
                let preselect_pool = max_graph_only * 5; // fetch more, let full scoring pick best
                let filtered_rids: Vec<String> = {
                    let cache = self.scoring_cache.read();
                    let mut candidates: Vec<(String, f64)> = new_rids
                        .into_iter()
                        .filter_map(|rid| {
                            let row = cache.get(&rid)?;
                            // Graph-only admission used to skip domain,
                            // source and certainty; one predicate now.
                            if !passes_recall_filters(
                                &rid,
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
                                return None;
                            }
                            let prox = gi.graph_proximity(&rid, &expanded_map);
                            let rank = row.importance * (0.3 + 0.7 * prox);
                            Some((rid, rank))
                        })
                        .collect();
                    // Fix (k): this rank feeds a truncation, and on uniform
                    // corpora it is a massive tie — the total order (quantized
                    // rank desc, rid asc) is what keeps the admitted subset
                    // identical across opens (the audit rule from fixes (e)/(f),
                    // applied to every remaining rank-and-take site at once).
                    candidates.sort_by(|a, b| {
                        crate::engine::lexical::quantize_score(b.1)
                            .total_cmp(&crate::engine::lexical::quantize_score(a.1))
                            .then_with(|| a.0.cmp(&b.0))
                    });
                    candidates
                        .into_iter()
                        .take(preselect_pool)
                        .map(|(rid, _)| rid)
                        .collect()
                };

                if !filtered_rids.is_empty() {
                    // Batch fetch embeddings for cosine similarity
                    let rid_refs: Vec<&str> = filtered_rids.iter().map(|r| r.as_str()).collect();
                    let embeddings = self.fetch_embeddings_by_rids(&rid_refs)?;

                    let cache = self.scoring_cache.read();
                    for rid in &filtered_rids {
                        let Some(row) = cache.get(rid) else { continue };
                        let Some(emb_blob_row) = embeddings.get(rid) else {
                            continue;
                        };

                        let mem_embedding = crate::serde_helpers::deserialize_f32(emb_blob_row);
                        let sim_score =
                            crate::consolidate::cosine_similarity(query_embedding, &mem_embedding)
                                as f64;

                        let decay = scoring::ranking_decay(row.importance, row.created_at, ts);
                        let age = ts - row.created_at;
                        let recency = scoring::recency_score(age);

                        let prox = gi.graph_proximity(rid, &expanded_map);
                        let composite = scoring::adaptive_graph_composite_score(
                            sim_score,
                            decay,
                            recency,
                            row.importance,
                            row.valence,
                            prox,
                            query_sentiment,
                            &learned_weights,
                        );
                        let contributions = scoring::adaptive_graph_contributions(
                            sim_score,
                            decay,
                            recency,
                            row.importance,
                            prox,
                            &learned_weights,
                        );
                        let valence_multiplier =
                            scoring::query_valence_boost(row.valence, query_sentiment);

                        let mut why = scoring::build_why(sim_score, recency, decay, row.valence);
                        let mem_entities: Vec<String> = gi
                            .entities_for_memory(rid)
                            .into_iter()
                            .map(|s| s.to_string())
                            .collect();
                        for entity in &mem_entities {
                            if expanded_map.contains_key(entity) {
                                why.push(format!("graph-connected via {entity}"));
                                break;
                            }
                        }

                        scored.push(RecallResult {
                            rid: rid.clone(),
                            memory_type: row.memory_type.clone(),
                            text: String::new(),
                            created_at: row.created_at,
                            importance: row.importance,
                            valence: row.valence,
                            score: composite,
                            scores: ScoreBreakdown {
                                similarity: sim_score,
                                decay,
                                recency,
                                importance: row.importance,
                                graph_proximity: prox,
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
                    drop(cache);
                }
            }
        }

        // Step 3.45 (packs): merge candidates from mounted packs.
        //
        // Placed after host candidate generation and BEFORE the status
        // filter so pack rows compete in the same pool as host rows for
        // everything that follows. Two properties depend on this exact
        // position:
        //
        // - Step 3.4 below removes pack rows that a host record
        //   supersedes, because `superseded_rids_among` matches on
        //   target rid regardless of which file the target lives in.
        //   That is the user-correction overlay, for free.
        // - MMR (step 4) runs once over the union, so a pack cannot
        //   flood top_k with near-duplicates.
        //
        // Pack rows arrive already hydrated: their text lives in the
        // pack file, and step 5 hydrates only from the host.
        //
        // #149 phase 2: bounded recall EXCLUDES all pack candidates.
        // Pack rows live outside the host's indexed event-time universe
        // (idx_memories_event_time), so their event time is effectively
        // unknown — NULL-excluded semantics extend to them. Post-filtering
        // PackFilters instead would still miss dated pack rows outside
        // pack_fetch_k, violating filter-first. Full pack event-time
        // support is a follow-up (the #164 v1.1 pack-lane pattern).
        if event_allow.is_none() {
            let pack_candidates = self.collect_pack_candidates(
                query_embedding,
                top_k,
                ts,
                &learned_weights,
                query_sentiment,
                &crate::engine::pack::PackFilters {
                    include_consolidated,
                    memory_type,
                    time_window,
                    namespace,
                    domain,
                    source,
                    certainty_min,
                },
            )?;
            if !pack_candidates.is_empty() {
                tracing::debug!(count = pack_candidates.len(), "merged pack candidates");
                pack_rids.extend(pack_candidates.iter().map(|r| r.rid.clone()));
                scored.extend(pack_candidates);
            }
        }

        // Step 3.4 (v0.10 Item 1): status-led eligibility filter.
        //
        // Eligibility, not demotion: when the status read policy is
        // active and the caller didn't ask for history, superseded
        // records leave the candidate pool HERE — before keyword slot
        // reservation computes its cutoff and before MMR competes for
        // top_k slots — so a stale fact can never outrank or crowd out
        // its own successor (trace contract T01's hard-zero at k=1).
        // Batched against the pool in one IN-clause query; databases
        // with no supersedes edges pay one indexed probe per candidate
        // and filter nothing.
        if self.status_read_policy() && !include_superseded && !scored.is_empty() {
            let pool_rids: Vec<&str> = scored.iter().map(|r| r.rid.as_str()).collect();
            let superseded = self.superseded_rids_among(&pool_rids)?;
            if !superseded.is_empty() {
                scored.retain(|r| !superseded.contains(&r.rid));
            }
        }

        // Step 3.5: Keyword slot reservation
        //
        // Topic-relevant keyword matches (e.g., "yoga", "reading") often have
        // moderate importance (0.30-0.40) that can't compete with high-importance
        // memories (0.70-1.00) in the composite score formula. Reserve up to 3
        // top_k slots for the best keyword-matched candidates — admitted and
        // ranked by bm25 lexical strength with cosine as the fallback door
        // (engine/lexical.rs; the sim-only door starved exact-phrase matches).
        //
        // The boost is minimal (just above cutoff). NOTE the survival
        // guarantee is PARTIAL: on the truncate branch reserved items sit at
        // cutoff+epsilon and survive; inside the MMR branch they compete like
        // any candidate (no exemption is implemented) and the 0.98 near-dup
        // skip can drop them. Exempting them inside MMR is a behavior change
        // that must be measured before shipping, not assumed here.
        if crate::engine::capture::enabled() {
            let mut lex: Vec<(&str, String)> = lex_by_rid
                .iter()
                .map(|(rid, s)| (rid.as_str(), crate::engine::capture::bits(*s)))
                .collect();
            lex.sort();
            crate::engine::capture::emit(capture_inst, "lex_by_rid", serde_json::json!(lex));
            crate::engine::capture::emit(
                capture_inst,
                "scored_pre_reserve",
                serde_json::json!(scored
                    .iter()
                    .map(|r| (
                        r.rid.as_str(),
                        crate::engine::capture::bits(r.score),
                        r.why_retrieved.join("|"),
                    ))
                    .collect::<Vec<_>>()),
            );
        }
        apply_lane_agreement(&mut scored, &win_by_rid);
        crate::engine::lexical::apply_keyword_reserve(&mut scored, &lex_by_rid, top_k);

        // Lifecycle eligibility is a result invariant, not a lane concern.
        // Recheck the merged pool so a candidate cannot leak through an older
        // lane that performed its own partial copy of the request filters.
        {
            let cache = self.scoring_cache.read();
            scored.retain(|result| {
                pack_rids.contains(&result.rid)
                    || cache
                        .get(&result.rid)
                        .is_some_and(synthesis_lifecycle_allows)
            });
            apply_synthesis_representation_preference(&mut scored, &cache, query_text);
        }

        // Step 4: MMR diversity selection
        //
        // Without diversity filtering, near-duplicate generated memories (e.g.,
        // "Arjun cooked X today" x50) dominate all K result slots. MMR ensures
        // each result adds new information by penalizing candidates too similar
        // to already-selected results.
        scored.sort_by(crate::engine::lexical::rank_cmp);
        // Quotas run AFTER the sort, not before: the sort re-orders the
        // whole pool by score, so a quota pass placed above it was silently
        // undone by the very next statement. The function was correct and
        // correctly called, and had no effect whatsoever.
        apply_lane_quotas(&mut scored, &win_by_rid, top_k);
        if crate::engine::capture::enabled() {
            crate::engine::capture::emit(
                capture_inst,
                "pool_post_reserve_sorted",
                serde_json::json!(scored
                    .iter()
                    .map(|r| (r.rid.as_str(), crate::engine::capture::bits(r.score)))
                    .collect::<Vec<_>>()),
            );
        }

        // v0.13.1 explain snapshot — post-boost/post-reserve,
        // pre-MMR-truncation: the set that ENTERS final selection.
        // Earlier would show a healthy vector lane and tell you nothing;
        // later would only show survivors again.
        let mut explain_report: Option<crate::types::RecallExplain> = None;
        if explain_on {
            let mut sim_order: Vec<usize> = (0..scored.len()).collect();
            sim_order.sort_by(|&a, &b| {
                scored[b]
                    .scores
                    .similarity
                    .total_cmp(&scored[a].scores.similarity)
                    .then_with(|| scored[a].rid.cmp(&scored[b].rid))
            });
            let mut pre_rank = vec![0usize; scored.len()];
            for (rank, &i) in sim_order.iter().enumerate() {
                pre_rank[i] = rank;
            }
            let pool: Vec<crate::types::ExplainPoolRow> = scored
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let why = &r.why_retrieved;
                    let mut lanes: Vec<String> = Vec::new();
                    if win_by_rid.contains_key(&r.rid) {
                        lanes.push("vector".into());
                    }
                    if why
                        .iter()
                        .any(|w| w == "keyword_match" || w == "fts_sourced")
                    {
                        lanes.push("fts".into());
                    }
                    if why.iter().any(|w| w.starts_with("claims_match")) {
                        lanes.push("claims".into());
                    }
                    if r.scores.graph_proximity > 0.0
                        || why.iter().any(|w| w.starts_with("graph-connected"))
                    {
                        lanes.push("graph".into());
                    }
                    if why.iter().any(|w| w == "lexical_rescue") {
                        lanes.push("lexical_rescue".into());
                    }
                    if why.iter().any(|w| w == "keyword_reserved") {
                        lanes.push("keyword_reserve".into());
                    }
                    if explain_fallback_rids.contains(&r.rid) {
                        lanes.push("importance_fallback".into());
                    }
                    if explain_valence_rids.contains(&r.rid) {
                        lanes.push("valence_scan".into());
                    }
                    if explain_event_universe_rids.contains(&r.rid) {
                        lanes.push("event_time_universe".into());
                    }
                    if pack_rids.contains(&r.rid) {
                        lanes.push("pack".into());
                    }
                    crate::types::ExplainPoolRow {
                        rid: r.rid.clone(),
                        score_q: crate::engine::lexical::quantize_score(r.score) / 1e6,
                        similarity: r.scores.similarity,
                        lex: lex_by_rid.get(r.rid.as_str()).copied(),
                        lanes_admitted: lanes,
                        rank_pre_fusion: pre_rank[i],
                        rank_post_fusion: i,
                        selected: false,
                    }
                })
                .collect();

            let mut lanes = std::collections::BTreeMap::new();
            let lane = |status: &str, candidates: usize, reason: Option<String>| {
                crate::types::ExplainLaneReport {
                    status: status.to_string(),
                    candidates,
                    reason,
                }
            };
            lanes.insert("vector".to_string(), lane("ran", win_by_rid.len(), None));
            lanes.insert(
                "fts".to_string(),
                if !explain_fts_ran {
                    lane(
                        "never_ran",
                        0,
                        Some(if query_text.is_none() {
                            "no query_text".to_string()
                        } else {
                            "no extractable keywords (or FTS unavailable)".to_string()
                        }),
                    )
                } else if lex_by_rid.is_empty() {
                    lane("ran_empty", 0, None)
                } else {
                    lane("ran", lex_by_rid.len(), None)
                },
            );
            let claims_count = pool
                .iter()
                .filter(|p| p.lanes_admitted.iter().any(|l| l == "claims"))
                .count();
            lanes.insert(
                "claims".to_string(),
                if query_text.is_none() {
                    lane("never_ran", 0, Some("no query_text".to_string()))
                } else if claims_count == 0 {
                    lane("ran_empty", 0, None)
                } else {
                    lane("ran", claims_count, None)
                },
            );
            let graph_count = pool
                .iter()
                .filter(|p| p.lanes_admitted.iter().any(|l| l == "graph"))
                .count();
            lanes.insert(
                "graph".to_string(),
                if !expand_entities {
                    lane("never_ran", 0, Some("expand_entities=false".to_string()))
                } else if graph_count == 0 {
                    lane("ran_empty", 0, None)
                } else {
                    lane("ran", graph_count, None)
                },
            );
            lanes.insert(
                "valence_scan".to_string(),
                if query_sentiment.abs() <= 0.5 {
                    lane(
                        "never_ran",
                        0,
                        Some(format!(
                            "query_sentiment {query_sentiment:.2} inside neutral band (|s| <= 0.5)"
                        )),
                    )
                } else {
                    lane(
                        if explain_valence_rids.is_empty() {
                            "ran_empty"
                        } else {
                            "ran"
                        },
                        explain_valence_rids.len(),
                        None,
                    )
                },
            );
            lanes.insert(
                "importance_fallback".to_string(),
                lane(
                    if explain_fallback_rids.is_empty() {
                        "ran_empty"
                    } else {
                        "ran"
                    },
                    explain_fallback_rids.len(),
                    None,
                ),
            );
            // #149 phase 2: reported only when the caller set a valid-time
            // bound, so unbounded recalls keep their pre-phase-2 explain
            // shape byte-for-byte.
            if let Some(allow) = event_allow {
                lanes.insert(
                    "event_time_universe".to_string(),
                    lane(
                        if explain_event_universe_rids.is_empty() {
                            "ran_empty"
                        } else {
                            "ran"
                        },
                        explain_event_universe_rids.len(),
                        Some(format!("eligible_universe={}", allow.len())),
                    ),
                );
            }
            lanes.insert(
                "pack".to_string(),
                lane(
                    if pack_rids.is_empty() {
                        "ran_empty"
                    } else {
                        "ran"
                    },
                    pack_rids.len(),
                    None,
                ),
            );

            let bm25_near_best_fraction = if lex_by_rid.is_empty() {
                None
            } else {
                let near_best = lex_by_rid.values().filter(|&&s| s >= 0.9).count();
                Some(near_best as f64 / lex_by_rid.len() as f64)
            };

            explain_report = Some(crate::types::RecallExplain {
                retrieval_limits: retrieval_limits.clone(),
                comparator: "rank_cmp: score quantized at 1e-6 desc, rid asc (every recall-path \
                             selection)"
                    .to_string(),
                score_algebra: "score = w_sim*similarity*freshness_mult(decay,recency) * \
                                importance_mult * valence_multiplier, plus additive lane boosts \
                                (keyword; graph is MULTIPLICATIVE via graph_mult since 2026-08-13) and reserve/claims lifts to cutoff+eps. \
                                Verified synthesis records may then receive match-only \
                                axis/granularity multipliers inferred from query text. \
                                scores.contributions are per-signal DIAGNOSTIC MAGNITUDES on \
                                these mixed terms and do NOT sum to score — do not derive \
                                arithmetic from them."
                    .to_string(),
                query_sentiment,
                bm25_near_best_fraction,
                lanes,
                pool,
            });
        }

        // Novelty selection REPLACES MMR rather than composing with it.
        // They are two diversity mechanisms with different notions of
        // redundancy (embedding vs lexical); running both in sequence would
        // make each one's weight mean something different depending on what
        // the other did first, which is unsweepable.
        let novelty_w = crate::base::tuning::tuning().novelty_weight;
        let min_pool_for_mmr = top_k.saturating_mul(3).max(20);
        // Set-level re-selection.
        //
        // Needs TEXT, and step 5 below hydrates text only for the final
        // top_k — by design, so a 100-candidate recall does not fetch 100
        // bodies. The first version of this ran here against the
        // pre-hydration structs, where every `text` is String::new(); every
        // novelty score was therefore 0, every candidate tied, and the
        // greedy reproduced score order EXACTLY. It looked like a working
        // feature with a weight that did nothing.
        //
        // So when novelty is enabled we hydrate the SELECTION POOL early —
        // capped, and only on this path, so the default costs nothing.
        if std::env::var("YANTRIKDB_DEBUG_NOVELTY").is_ok() {
            eprintln!(
                "[novelty-gate] w={:.3} vec_candidates={} fetch_k={} scored={} top_k={} enters={}",
                novelty_w,
                vec_candidate_count,
                fetch_k,
                scored.len(),
                top_k,
                novelty_w > 0.0 && scored.len() > top_k
            );
        }
        if novelty_w > 0.0 && scored.len() > top_k {
            let pool = scored
                .len()
                .min(top_k.saturating_mul(10).max(min_pool_for_mmr));
            scored.truncate(pool);
            let pool_rids: Vec<&str> = scored.iter().map(|r| r.rid.as_str()).collect();
            let pool_text = self.fetch_text_metadata_by_rids(&pool_rids)?;
            for r in &mut scored {
                if let Some(tm) = pool_text.get(&r.rid) {
                    r.text = tm.text.clone();
                }
            }
            // Diagnostic, off unless asked for: distinguishes "pool too
            // small to select from" and "text never arrived" from "policy
            // ran and chose this". Without it the three are indistinguishable
            // from the outside, which cost a full debugging cycle.
            if std::env::var("YANTRIKDB_DEBUG_NOVELTY").is_ok() {
                let hydrated = scored.iter().filter(|r| !r.text.is_empty()).count();
                let before: Vec<String> =
                    scored.iter().take(top_k).map(|r| r.rid.clone()).collect();
                apply_novelty_selection(&mut scored, top_k);
                let after: Vec<String> = scored.iter().take(top_k).map(|r| r.rid.clone()).collect();
                eprintln!(
                    "[novelty] w={:.3} pool={} hydrated={} top_k={} reordered={}",
                    novelty_w,
                    pool,
                    hydrated,
                    top_k,
                    before != after
                );
            } else {
                apply_novelty_selection(&mut scored, top_k);
            }
            scored.truncate(top_k);
        }

        if novelty_w <= 0.0 && scored.len() > top_k && scored.len() >= min_pool_for_mmr {
            // Fetch embeddings for top candidates to compute pairwise similarity
            let pool_size = scored.len().min(top_k.saturating_mul(10));
            scored.truncate(pool_size);

            let pool_rids: Vec<&str> = scored.iter().map(|r| r.rid.as_str()).collect();
            let emb_map = self.fetch_embeddings_by_rids(&pool_rids)?;

            // Parse embeddings for each candidate
            let pool_embeddings: Vec<Option<Vec<f32>>> = scored
                .iter()
                .map(|r| {
                    emb_map
                        .get(r.rid.as_str())
                        .map(|blob| crate::serde_helpers::deserialize_f32(blob))
                })
                .collect();

            // Greedy MMR: λ * relevance - (1-λ) * max_sim_to_selected
            //
            // λ = 0.9, raised from 0.7 (2026-08-05): measured on a 4,297-record
            // production clone with a paraphrase-labeled set, λ=0.7 cost
            // 0.062 MRR (0.566 → 0.504) by letting the diversity credit pull
            // far-ranked items over relevant ones; λ=0.9 keeps the
            // duplicate-flood defense (the 0.98 skip below is untouched) at
            // 40% of that cost (0.541). The labeled set structurally cannot
            // reward diversity (one right answer per query), so λ stays
            // below 1.0 on the strength of the near-duplicate UX case, not
            // that measurement.
            // Wired to tuning: YANTRIKDB_MMR_LAMBDA (default 0.9, clamped 0..=1
            // at parse). The hardcoded const made every λ sweep a silent no-op
            // while the fingerprint stamped the env value as if it governed.
            let lambda: f64 = crate::base::tuning::tuning().mmr_lambda;
            const SIM_THRESHOLD: f64 = 0.98; // skip only near-exact duplicates

            let mut selected: Vec<usize> = Vec::with_capacity(top_k);
            let mut selected_embeddings: Vec<&[f32]> = Vec::with_capacity(top_k);

            // Always pick the top-scored candidate first
            if !scored.is_empty() {
                selected.push(0);
                if let Some(Some(ref emb)) = pool_embeddings.first() {
                    selected_embeddings.push(emb);
                }
            }

            // Greedily select remaining candidates
            for _ in 1..top_k {
                let mut best_idx = None;
                let mut best_mmr = f64::NEG_INFINITY;

                for (idx, result) in scored.iter().enumerate() {
                    if selected.contains(&idx) {
                        continue;
                    }

                    let relevance = result.score;
                    let max_sim = if let Some(Some(ref cand_emb)) = pool_embeddings.get(idx) {
                        selected_embeddings
                            .iter()
                            .map(|sel_emb| {
                                crate::consolidate::cosine_similarity(cand_emb, sel_emb) as f64
                            })
                            .fold(0.0f64, f64::max)
                    } else {
                        0.0
                    };

                    // Skip near-duplicates entirely
                    if max_sim > SIM_THRESHOLD {
                        continue;
                    }

                    let mmr = lambda * relevance - (1.0 - lambda) * max_sim;
                    if mmr > best_mmr {
                        best_mmr = mmr;
                        best_idx = Some(idx);
                    }
                }

                match best_idx {
                    Some(idx) => {
                        if crate::engine::capture::enabled() {
                            crate::engine::capture::emit(
                                capture_inst,
                                "mmr_step",
                                serde_json::json!({
                                    "step": selected.len(),
                                    "chosen": scored[idx].rid.as_str(),
                                    "relevance_bits":
                                        crate::engine::capture::bits(scored[idx].score),
                                    "mmr_bits": crate::engine::capture::bits(best_mmr),
                                    "has_emb": pool_embeddings
                                        .get(idx)
                                        .map(|e| e.is_some())
                                        .unwrap_or(false),
                                }),
                            );
                        }
                        selected.push(idx);
                        if let Some(Some(ref emb)) = pool_embeddings.get(idx) {
                            selected_embeddings.push(emb);
                        }
                    }
                    None => break, // No more candidates pass the threshold
                }
            }

            // Rebuild scored from selected indices, preserving order
            let mut diverse_results = Vec::with_capacity(selected.len());
            for i in selected {
                diverse_results.push(scored[i].clone());
            }
            scored = diverse_results;
        } else {
            scored.truncate(top_k);
        }

        // Issue #46: optional reorder of the final top_k. Candidate
        // retrieval + MMR still operate on relevance — we re-sort the
        // selected diverse set here so the caller-requested presentation
        // order (most-confident-first, most-recent-first) does not
        // sacrifice diversity. Default (`Relevance`) leaves the existing
        // score-desc order untouched.
        match recall_order {
            RecallOrder::Relevance => {}
            RecallOrder::Certainty => {
                scored.sort_by(|a, b| b.certainty.total_cmp(&a.certainty));
            }
            RecallOrder::Recency => {
                scored.sort_by(|a, b| b.created_at.total_cmp(&a.created_at));
            }
            RecallOrder::FirstMention => {}
        }

        // Step 5: Hydrate final top_k with text + metadata from SQLite
        let final_rids: Vec<&str> = scored.iter().map(|r| r.rid.as_str()).collect();
        let text_meta = {
            let _span = tracing::debug_span!("hydrate", count = final_rids.len()).entered();
            self.fetch_text_metadata_by_rids(&final_rids)?
        };
        for result in &mut scored {
            if let Some(tm) = text_meta.get(&result.rid) {
                result.text = tm.text.clone();
                result.metadata = serde_json::from_str(&tm.metadata)
                    .unwrap_or(serde_json::Value::Object(Default::default()));
            }
        }

        if recall_order == RecallOrder::FirstMention {
            let first_mention_at = |result: &RecallResult| {
                result
                    .metadata
                    .get("first_mention_at")
                    .and_then(serde_json::Value::as_f64)
                    .filter(|value| value.is_finite())
                    .unwrap_or(result.created_at)
            };
            scored.sort_by(|a, b| {
                first_mention_at(a)
                    .total_cmp(&first_mention_at(b))
                    .then_with(|| a.created_at.total_cmp(&b.created_at))
                    .then_with(|| a.rid.cmp(&b.rid))
            });
        }

        if crate::engine::capture::enabled() {
            crate::engine::capture::emit(
                capture_inst,
                "final",
                serde_json::json!(scored
                    .iter()
                    .map(|r| (r.rid.as_str(), crate::engine::capture::bits(r.score)))
                    .collect::<Vec<_>>()),
            );
        }

        // Step 5.5: snippet spans — report WHERE in each long record the
        // match lives. The vector layer's winning window is trusted only
        // for records with chunk vectors; everything else (FTS-sourced,
        // graph-expanded, never-chunked installs) gets the query-term
        // scan inside stamp_best_spans.
        {
            let chunked = {
                let owned: Vec<String> = scored.iter().map(|r| r.rid.clone()).collect();
                let refs: Vec<&str> = owned.iter().map(|s| s.as_str()).collect();
                self.rids_with_chunks(&refs)
            };
            win_by_rid.retain(|rid, _| chunked.contains(rid));
            crate::engine::snippet::stamp_best_spans(
                &mut scored,
                &win_by_rid,
                query_text,
                self.snippet_window(),
            );
        }

        // **v0.10 Item 3 seqlock recheck (sol r4).** Vector candidate
        // generation (above) and text hydration (just now) read different
        // subsystems at different instants. If a text-changing correction
        // committed+published in between, `scored` pairs a stale ranking
        // vector with corrected text — violating "wholly old or wholly
        // new". Detect via the epoch and discard; the wrapper retries. The
        // check is BEFORE impressions/reinforcement so a discarded result
        // leaves no ledger/spaced-repetition side effects. The Acquire fence
        // inside `correction_epoch_validate` orders the search + hydration
        // reads above BEFORE this version check (sol r5 finding 3).
        if !self.correction_epoch_validate(epoch0) {
            return Ok(None);
        }

        // v0.10 Item 2: persist the impression ledger BEFORE reinforcement
        // mutates last_access/access_count — the learner reads features
        // from here, never from mutable memory state. skip_reinforce
        // recalls are engine-internal and leave no impressions (the
        // ranker's own traversals are not consumer evidence). Best-effort:
        // the read path must not fail because the ledger hiccuped.
        if !skip_reinforce {
            let generation = learned_weights.generation;
            let _ = self.log_recall_impressions(&scored, query_embedding, namespace, generation);
        }

        // Reinforce accessed memories (spaced repetition). Best-effort:
        // reinforcement is a side-effect of a READ, and a read that found
        // its results must return them — failing the recall over a
        // bookkeeping UPDATE (as `?` did here) turned a transient BUSY
        // into a lost answer.
        if !skip_reinforce {
            for r in &scored {
                if let Err(e) = self.reinforce(&r.rid) {
                    tracing::warn!(rid = %r.rid, error = %e, "reinforce failed; recall unaffected");
                }
            }
        }

        // Task 25: surface unresolved conflicts at the moment of use, so a
        // contradicted memory arrives visibly flagged rather than asserted
        // as current fact.
        self.stamp_open_conflicts(&mut scored)?;

        // Task 41: surface trust metadata (aged-unconfirmed, superseded) so a
        // stale fact arrives visibly hedged rather than asserted as current.
        self.stamp_trust_metadata(&mut scored)?;

        // v0.9.0: record demand (what was asked + how well it was answered) so
        // the substrate can surface its own knowledge gaps. Skipped for
        // internal / eval recalls; best-effort (never fails the recall).
        // v0.9.3: scoped to the recall's namespace filter (None = the global
        // bucket) and a no-op on encrypted databases — see engine/demand.rs.
        if !skip_reinforce {
            if let Some(qt) = query_text {
                let top = scored.first().map(|r| r.score).unwrap_or(0.0);
                let _ = self.record_recall_demand(namespace, qt, scored.len(), top);
            }
        }

        // v0.13.1 explain: mark survivors and hand the report through the
        // sink — on the success path only, AFTER the epoch recheck, so a
        // discarded attempt never leaks a half-built report.
        if let Some(sink) = explain_sink.as_deref_mut() {
            if let Some(mut report) = explain_report {
                let selected: std::collections::HashSet<&str> =
                    scored.iter().map(|r| r.rid.as_str()).collect();
                for row in &mut report.pool {
                    row.selected = selected.contains(row.rid.as_str());
                }
                *sink = Some(report);
            }
        }

        Ok(Some((scored, retrieval_limits)))
    }

    /// Task 41 — annotate recall hits with trust signals so staleness is
    /// visible at the moment of use: a memory that is old AND rarely
    /// re-accessed (low confirmation), or one that a newer record supersedes,
    /// gets a `why_retrieved` hedge. Batched against the result rids.
    fn stamp_trust_metadata(&self, results: &mut [RecallResult]) -> Result<()> {
        if results.is_empty() {
            return Ok(());
        }
        const STALE_AGE_DAYS: f64 = 90.0;
        const LOW_CONFIRMATION: i64 = 1;
        let now_ts = crate::time::now_secs();
        let conn = self.conn();
        // Read created_at/updated_at/access_count from the table (the source
        // of truth), not the RecallResult, whose created_at comes from the
        // scoring cache.
        let mut access_stmt = conn
            .prepare("SELECT created_at, updated_at, access_count FROM memories WHERE rid = ?1")?;
        // v0.10 Item 1: fetch the SUCCESSOR rid (not just a count) so the
        // typed status can name it.
        let mut superseded_stmt = conn.prepare(
            "SELECT source_rid FROM record_links WHERE target_rid = ?1 \
             AND link_type = 'supersedes' \
             AND status = 'active' AND selection_state = 'selected' \
             LIMIT 1",
        )?;
        for r in results.iter_mut() {
            let (created_at, updated_at, access_count): (f64, f64, i64) = access_stmt
                .query_row(params![r.rid], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })
                .unwrap_or((now_ts, now_ts, 0));
            let age_days = (now_ts - created_at).max(0.0) / 86_400.0;
            if age_days > STALE_AGE_DAYS && access_count <= LOW_CONFIRMATION {
                // v0.10 Item 1: typed aged flag (orthogonal to status —
                // aged is a retrieval-age signal, not a truth status).
                r.aged_last_verified = Some(updated_at);
                r.why_retrieved.push(format!(
                    "⚠ {age_days:.0}d old, rarely confirmed — verify it is still current"
                ));
            }
            let successor: Option<String> = superseded_stmt
                .query_row(params![r.rid], |row| row.get(0))
                .optional()
                .unwrap_or(None);
            if let Some(successor_rid) = successor {
                // v0.10 Item 1: exclusive chain-derived status, typed. The
                // prose stamp stays for one release.
                r.current_status = crate::types::RecordStatus::Superseded;
                r.superseded_by = Some(successor_rid);
                r.why_retrieved
                    .push("⚠ superseded by a newer record — likely outdated".to_string());
                // v0.10 Item 1 adoption nudge: a superseded result was
                // actually SERVED (legacy policy, or include_superseded).
                // Counted here — the single point every returned result
                // passes through — and surfaced in stats() so migrated
                // DBs can see what the status read policy would exclude.
                self.superseded_served_since_boot
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        Ok(())
    }

    /// **Phase 6 RYW** — `recall()` with strict read-your-writes guard.
    ///
    /// Waits up to ``timeout`` for ``visible_seq[namespace] >= min_seq``,
    /// then runs the standard ``recall()`` pipeline. If the watermark is
    /// reached, search proceeds normally; if the timeout expires before
    /// the watermark, returns ``Error::RyWaitTimeout`` rather than
    /// returning a possibly-incomplete result set.
    ///
    /// **When to use**: caller has just performed a write (record /
    /// record_with_rid) and wants to be SURE the next recall sees that
    /// write. The default ``recall()`` is "delta is always visible" via
    /// the search merge, but during a compaction-in-progress window or
    /// cluster follower-apply lag, that guarantee can have gaps. ``recall_with_seq``
    /// closes those gaps for callers that explicitly opt in.
    ///
    /// ``min_seq`` should come from the seq returned by the prior write
    /// (Phase 6 will expose seq on record() return — coming in v0.7.0).
    /// ``namespace`` MUST be the namespace the write went into; passing
    /// a different namespace will time out (visible_seq is per-namespace).
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip(self, query_embedding), fields(top_k, min_seq, namespace))]
    pub fn recall_with_seq(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
        query_text: Option<&str>,
        skip_reinforce: bool,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
        min_seq: u64,
        timeout: std::time::Duration,
    ) -> Result<Vec<RecallResult>> {
        let ns = namespace.unwrap_or("default");
        self.wait_for_visible_seq(ns, min_seq, timeout)?;
        self.recall(
            query_embedding,
            top_k,
            time_window,
            memory_type,
            include_consolidated,
            expand_entities,
            query_text,
            skip_reinforce,
            namespace,
            domain,
            source,
            None,  // certainty_min (#46) — recall_with_seq path defers to caller-side filtering
            None,  // order (#46) — default relevance
            false, // include_superseded (v0.10 Item 1) — policy default; use query() builder for history
            None,  // event_after (#149) — no valid-time bound
            None,  // event_before (#149)
        )
    }

    /// Execute a recall query built with the `RecallQuery` builder.
    ///
    /// ```rust,ignore
    /// let results = db.query(embedding)
    ///     .top_k(10)
    ///     .memory_type("episodic")
    ///     .namespace("work")
    ///     .execute(&db)?;
    /// ```
    pub fn query(&self, q: RecallQuery) -> Result<Vec<RecallResult>> {
        self.recall(
            &q.embedding,
            q.top_k,
            q.time_window,
            q.memory_type.as_deref(),
            q.include_consolidated,
            q.expand_entities,
            q.query_text.as_deref(),
            q.skip_reinforce,
            q.namespace.as_deref(),
            q.domain.as_deref(),
            q.source.as_deref(),
            q.certainty_min,
            q.order.as_deref(),
            q.include_superseded,
            None, // event_after (#149) — not yet surfaced on the builder
            None, // event_before (#149)
        )
    }

    /// Recall with full response including confidence scoring and refinement hints.
    ///
    /// Wraps `recall()` and computes a `RecallResponse` with confidence level,
    /// retrieval summary, and hints for the calling agent to refine queries.
    pub fn recall_with_response(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
        query_text: Option<&str>,
        skip_reinforce: bool,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
    ) -> Result<RecallResponse> {
        let (results, retrieval_limits) = self.recall_with_limits(
            query_embedding,
            top_k,
            time_window,
            memory_type,
            include_consolidated,
            expand_entities,
            query_text,
            skip_reinforce,
            namespace,
            domain,
            source,
            None,  // certainty_min (#46) — recall_with_response defers to caller-side filtering
            None,  // order (#46) — default relevance
            false, // include_superseded (v0.10 Item 1) — policy default; use query() builder for history
            None,  // event_after (#149) — no valid-time bound
            None,  // event_before (#149)
        )?;

        // Determine which retrieval sources were used
        let mut sources_used = vec!["hnsw".to_string()];
        if query_text.is_some() && !self.is_encrypted() {
            sources_used.push("fts5".to_string());
        }
        if expand_entities {
            sources_used.push("graph".to_string());
        }
        let query_sentiment = query_text
            .map(crate::scoring::detect_query_sentiment)
            .unwrap_or(0.0);
        if query_sentiment.abs() > 0.01 {
            sources_used.push("valence".to_string());
        }

        // Candidate count: approximate from scoring cache filtered by same criteria
        let candidate_count = {
            let cache = self.scoring_cache.read();
            cache
                .values()
                .filter(|row| {
                    let status_ok = if include_consolidated {
                        row.consolidation_status == "active"
                            || row.consolidation_status == "consolidated"
                    } else {
                        row.consolidation_status == "active"
                    };
                    status_ok
                        && memory_type.map_or(true, |mt| row.memory_type == mt)
                        && namespace.map_or(true, |ns| row.namespace == ns)
                        && domain.map_or(true, |d| row.domain == d)
                        && source.map_or(true, |s| row.source == s)
                })
                .count()
        };

        // Compute retrieval summary
        let top_similarity = results.first().map(|r| r.scores.similarity).unwrap_or(0.0);
        let score_spread = if results.len() >= 2 {
            results.first().unwrap().score - results.last().unwrap().score
        } else {
            0.0
        };

        let summary = RetrievalSummary {
            top_similarity,
            score_spread,
            sources_used: sources_used.clone(),
            candidate_count,
        };

        // Compute confidence from 4 signals with detailed reasons
        let signal_sim = top_similarity;
        let signal_gap = if results.len() >= 3 {
            results[0].score - results[2].score
        } else if results.len() == 2 {
            results[0].score - results[1].score
        } else {
            0.0
        };
        let signal_diversity = sources_used.len() as f64 / 4.0; // max 4 sources
                                                                // top_k.max(1): a caller-supplied top_k=0 would divide to NaN/Inf
                                                                // and poison the confidence signal (not a panic — f64 — but garbage).
        let signal_density = (results.len() as f64 / top_k.max(1) as f64).min(1.0);

        let confidence = (0.35 * signal_sim
            + 0.25 * signal_gap
            + 0.20 * signal_diversity
            + 0.20 * signal_density)
            .clamp(0.0, 1.0);

        // Build certainty reasons explaining the confidence score
        let mut certainty_reasons = Vec::new();
        if signal_sim >= 0.7 {
            certainty_reasons.push(format!(
                "Strong semantic match (top similarity: {:.0}%)",
                signal_sim * 100.0
            ));
        } else if signal_sim >= 0.4 {
            certainty_reasons.push(format!(
                "Moderate semantic match (top similarity: {:.0}%)",
                signal_sim * 100.0
            ));
        } else if signal_sim > 0.0 {
            certainty_reasons.push(format!(
                "Weak semantic match (top similarity: {:.0}%) — query may be outside stored knowledge",
                signal_sim * 100.0
            ));
        } else {
            certainty_reasons.push("No matching memories found".to_string());
        }

        if results.is_empty() {
            certainty_reasons.push(format!("No results from {} candidates", candidate_count));
        } else if signal_density < 0.5 {
            certainty_reasons.push(format!(
                "Sparse results: only {}/{} slots filled",
                results.len(),
                top_k
            ));
        }

        if signal_gap > 0.3 {
            certainty_reasons.push("Clear winner: top result stands out from the rest".to_string());
        } else if signal_gap < 0.05 && results.len() >= 2 {
            certainty_reasons.push(
                "Ambiguous: multiple results scored similarly — consider refining query"
                    .to_string(),
            );
        }

        if sources_used.contains(&"graph".to_string()) {
            certainty_reasons
                .push("Graph expansion contributed entity-linked memories".to_string());
        }

        // Check for stale results (last accessed > 30 days ago)
        let ts = now();
        let stale_count = results
            .iter()
            .filter(|r| {
                // Use created_at as a proxy since we don't have last_access in RecallResult
                ts - r.created_at > 30.0 * 86400.0
            })
            .count();
        if stale_count > results.len() / 2 && !results.is_empty() {
            certainty_reasons.push(format!(
                "Note: {}/{} results are older than 30 days — information may be outdated",
                stale_count,
                results.len()
            ));
        }

        // Check for low-certainty memories in results
        let low_certainty_count = results.iter().filter(|r| r.certainty < 0.5).count();
        if low_certainty_count > 0 {
            certainty_reasons.push(format!(
                "{}/{} results have low memory certainty (<50%) — treat with caution",
                low_certainty_count,
                results.len()
            ));
        }

        // Generate hints when confidence < 0.60
        let hints = if confidence < 0.60 {
            self.generate_hints(query_text, query_embedding, &results, &summary)
        } else {
            vec![]
        };

        // v0.10 Item 1b (trace T08): typed coverage. The relevance gate is
        // the per-database learned gate_tau — the same threshold the scorer
        // uses to decide when a hit is similar enough for importance
        // boosting to engage. Outcome classification:
        //   scope empty                       -> NoMatchingRecord
        //   results empty, or best under gate -> BelowThreshold
        //   otherwise                         -> Matched
        let threshold_tau = self.load_learned_weights()?.gate_tau;
        let outcome = if candidate_count == 0 {
            crate::types::CoverageOutcome::NoMatchingRecord
        } else if results.is_empty() || top_similarity < threshold_tau {
            crate::types::CoverageOutcome::BelowThreshold
        } else {
            crate::types::CoverageOutcome::Matched
        };
        // v0.10 Item 2: the piggyback label request — up to 2 served rids
        // nearest the relevance gate (most informative to grade), never
        // re-asked for the same query. Best-effort: the read path never
        // fails on the rider.
        let label_request = self
            .propose_label_requests(&results, query_embedding, threshold_tau)
            .unwrap_or_default();

        let coverage = crate::types::SearchCoverage {
            namespace: namespace.map(str::to_string),
            memory_type: memory_type.map(str::to_string),
            candidate_count,
            threshold_tau,
            top_similarity,
            outcome,
            label_request,
        };

        Ok(RecallResponse {
            results,
            confidence,
            certainty_reasons,
            retrieval_summary: summary,
            hints,
            coverage,
            retrieval_limits,
        })
    }

    /// Generate refinement hints based on the recall results and query context.
    fn generate_hints(
        &self,
        query_text: Option<&str>,
        _query_embedding: &[f32],
        results: &[RecallResult],
        summary: &RetrievalSummary,
    ) -> Vec<RefinementHint> {
        let mut hints = Vec::new();

        // Hint 0 (task 35): structural intent. Recency / enumeration / count
        // questions want an EXACT answer, not similarity ranking — `recall`
        // can never guarantee one. Detect the intent and point at the
        // structural path so the agent stops trusting a probabilistic recall
        // for a deterministic question.
        if let Some(qt) = query_text {
            let lq = qt.to_lowercase();
            let recency = [
                "most recent",
                "latest",
                "last entry",
                "newest",
                "chain head",
            ]
            .iter()
            .any(|k| lq.contains(k));
            let enumeration = [
                "list all",
                "all records",
                "all memories",
                "enumerate",
                "every record",
            ]
            .iter()
            .any(|k| lq.contains(k));
            let counting =
                lq.contains("how many") || lq.starts_with("count ") || lq.contains("number of");
            if recency || enumeration || counting {
                let suggestion = if counting {
                    "This looks like a count. recall returns a ranked sample, not a total — \
                     enumerate with list_records(...) and count."
                } else if recency {
                    "This looks like a recency question. recall ranks by similarity, not time — \
                     for the exact latest entry use chain_head(namespace) or \
                     list_records(order=\"desc\", limit=1)."
                } else {
                    "This looks like an enumeration. recall returns a ranked top-k, not the full \
                     set — use list_records(kind/namespace/..., order, since_rid) to page all \
                     matching records exactly."
                };
                hints.push(RefinementHint {
                    hint_type: "structural".to_string(),
                    suggestion: suggestion.to_string(),
                    related_entities: vec![],
                });
            }
        }

        // Hint 1: Specificity — if query is very short, suggest adding detail
        if let Some(qt) = query_text {
            let word_count = qt.split_whitespace().count();
            if word_count <= 3 {
                hints.push(RefinementHint {
                    hint_type: "specificity".to_string(),
                    suggestion: "Try adding more context — who, when, or where?".to_string(),
                    related_entities: vec![],
                });
            }
        }

        // Hint 2: Entity hints — find entities in the graph near the query
        if let Some(qt) = query_text {
            let query_tokens = crate::graph::tokenize(qt);
            let gi = self.graph_index.read();
            let matched = gi.entity_matches_query(&query_tokens);
            // Suggest entities that matched the query but whose memories aren't in results
            let result_rids: std::collections::HashSet<&str> =
                results.iter().map(|r| r.rid.as_str()).collect();
            let mut entity_suggestions = Vec::new();
            for (name, _etype, _score) in &matched {
                // Find memories linked to this entity
                let linked = gi.memories_for_entities(&[name.as_str()]);
                let has_result = linked.iter().any(|rid| result_rids.contains(rid.as_str()));
                if !has_result && entity_suggestions.len() < 5 {
                    entity_suggestions.push(name.clone());
                }
            }
            if !entity_suggestions.is_empty() {
                hints.push(RefinementHint {
                    hint_type: "entity".to_string(),
                    suggestion: format!(
                        "Related entities found but not in results: {}. Try mentioning them.",
                        entity_suggestions.join(", ")
                    ),
                    related_entities: entity_suggestions,
                });
            }
        }

        // Hint 3: Time range — if results span a wide time range
        if results.len() >= 2 {
            let min_ts = results
                .iter()
                .map(|r| r.created_at)
                .fold(f64::INFINITY, f64::min);
            let max_ts = results
                .iter()
                .map(|r| r.created_at)
                .fold(f64::NEG_INFINITY, f64::max);
            let span_days = (max_ts - min_ts) / 86400.0;
            if span_days > 30.0 {
                hints.push(RefinementHint {
                    hint_type: "time_range".to_string(),
                    suggestion: "Results span a wide time range. Try specifying a time period."
                        .to_string(),
                    related_entities: vec![],
                });
            }
        }

        // Hint 4: Low similarity — if even the best result has low similarity
        if summary.top_similarity < 0.25 {
            hints.push(RefinementHint {
                hint_type: "keyword".to_string(),
                suggestion: "The query may use different words than the stored memories. Try rephrasing with synonyms or related terms.".to_string(),
                related_entities: vec![],
            });
        }

        // Hint 5: Domain diversity — if all results from one domain but DB has others
        if results.len() >= 3 {
            let result_domains: std::collections::HashSet<&str> =
                results.iter().map(|r| r.domain.as_str()).collect();
            if result_domains.len() == 1 {
                // Check if other domains exist in the DB
                let cache = self.scoring_cache.read();
                let all_domains: std::collections::HashSet<&str> =
                    cache.values().map(|r| r.domain.as_str()).collect();
                let other_domains: Vec<&str> = all_domains
                    .difference(&result_domains)
                    .filter(|d| **d != "general")
                    .copied()
                    .take(5)
                    .collect();
                if !other_domains.is_empty() {
                    hints.push(RefinementHint {
                        hint_type: "domain".to_string(),
                        suggestion: format!(
                            "Results only from '{}' domain. Other domains available: {}. \
                             Consider cross-domain search if relevant.",
                            result_domains.iter().next().unwrap_or(&"unknown"),
                            other_domains.join(", ")
                        ),
                        related_entities: vec![],
                    });
                }
            }
        }

        // Hint 6: Procedural memory available — if query looks like a "how to" question
        if let Some(qt) = query_text {
            let qt_lower = qt.to_lowercase();
            let is_procedural_query = qt_lower.starts_with("how ")
                || qt_lower.contains("how do")
                || qt_lower.contains("how to")
                || qt_lower.contains("best way")
                || qt_lower.contains("approach")
                || qt_lower.contains("strategy");
            if is_procedural_query {
                let has_procedural = results.iter().any(|r| r.memory_type == "procedural");
                if !has_procedural {
                    // Check if procedural memories exist at all
                    let cache = self.scoring_cache.read();
                    let procedural_count = cache
                        .values()
                        .filter(|r| r.memory_type == "procedural")
                        .count();
                    if procedural_count > 0 {
                        hints.push(RefinementHint {
                            hint_type: "memory_type".to_string(),
                            suggestion: format!(
                                "{} procedural memories exist but weren't retrieved. \
                                 Try filtering by memory_type='procedural'.",
                                procedural_count
                            ),
                            related_entities: vec![],
                        });
                    }
                }
            }
        }

        hints
    }

    /// Refine a previous recall by combining original + refinement embeddings.
    ///
    /// The AI agent calls this after receiving hints from `recall_with_response()`.
    /// It combines the original query embedding with a refinement text embedding
    /// (weighted: 0.4 original + 0.6 refinement) and excludes already-seen RIDs.
    pub fn recall_refine(
        &self,
        original_query_embedding: &[f32],
        refinement_embedding: &[f32],
        original_rids: &[String],
        top_k: usize,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
    ) -> Result<RecallResponse> {
        // Combine embeddings: 0.4 * original + 0.6 * refinement
        let dim = original_query_embedding
            .len()
            .min(refinement_embedding.len());
        let mut combined = vec![0.0f32; dim];
        for i in 0..dim {
            combined[i] = 0.4 * original_query_embedding[i] + 0.6 * refinement_embedding[i];
        }
        // Normalize
        let norm: f32 = combined.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for v in &mut combined {
                *v /= norm;
            }
        }

        // Run recall with combined embedding
        let mut response = self.recall_with_response(
            &combined, top_k, None, None, false, false, None,
            true, // skip_reinforce=true for refinement
            namespace, domain, source,
        )?;

        // Exclude already-seen RIDs
        let exclude: std::collections::HashSet<&str> =
            original_rids.iter().map(|s| s.as_str()).collect();
        response
            .results
            .retain(|r| !exclude.contains(r.rid.as_str()));

        Ok(response)
    }

    /// Profiled version of recall() that returns per-phase timing breakdown.
    ///
    /// Mirrors `recall()` exactly but wraps each phase in timing
    /// instrumentation — including the v0.10 Item 3 correction seqlock
    /// (sol r4): a thin retry wrapper around `recall_profiled_inner`.
    #[cfg(feature = "profiling")]
    #[allow(clippy::too_many_arguments)]
    pub fn recall_profiled(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
        query_text: Option<&str>,
        skip_reinforce: bool,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
    ) -> Result<RecallProfiledResult> {
        const MAX_ATTEMPTS: u32 = 8;
        for attempt in 0..MAX_ATTEMPTS {
            let Some(epoch0) = self.correction_epoch_even() else {
                // Report attempts completed so far (sol r6: accurate count).
                return Err(YantrikDbError::RecallContended { attempts: attempt });
            };
            if let Some(r) = self.recall_profiled_inner(
                query_embedding,
                top_k,
                time_window,
                memory_type,
                include_consolidated,
                expand_entities,
                query_text,
                skip_reinforce,
                namespace,
                domain,
                source,
                None, // certainty_min — the public profiled surface has no
                // certainty param; None = no floor, matching recall()
                // when the caller sets none.
                epoch0,
            )? {
                return Ok(r);
            }
        }
        Err(YantrikDbError::RecallContended {
            attempts: MAX_ATTEMPTS,
        })
    }

    #[cfg(feature = "profiling")]
    #[allow(clippy::too_many_arguments)]
    fn recall_profiled_inner(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        time_window: Option<(f64, f64)>,
        memory_type: Option<&str>,
        include_consolidated: bool,
        expand_entities: bool,
        query_text: Option<&str>,
        skip_reinforce: bool,
        namespace: Option<&str>,
        domain: Option<&str>,
        source: Option<&str>,
        // Added 2026-08-15: the 08-13 filter-integrity fix pasted
        // passes_recall_filters(..certainty_min) calls into this copy while
        // its signature never carried the parameter — the profiled build
        // (--features profiling, built by no CI job) shipped uncompilable
        // and stayed broken through several later commits. The duplicated
        // pipeline receives fixes only when someone remembers; nothing
        // checks. Until the copies are collapsed, profiling MUST be in CI.
        certainty_min: Option<f64>,
        epoch0: u64,
    ) -> Result<Option<RecallProfiledResult>> {
        use std::time::Instant;
        let t_start = Instant::now();

        let learned_weights = self.load_learned_weights()?;

        let query_sentiment = query_text
            .map(scoring::detect_query_sentiment)
            .unwrap_or(0.0);

        // ── Phase 1: Vector search (HNSW) ──
        let t_vec = Instant::now();
        let ts = now();
        // **Issue #41 brainstorm-4 §1.** SearchState snapshot for the
        // profiled-recall variant. Same generation-anchoring contract
        // as `recall_ranked`.
        let state = self.search_state.load_full();
        let has_post_filters = time_window.is_some()
            || memory_type.is_some()
            || domain.is_some()
            || source.is_some()
            || certainty_min.is_some();
        let fetch_plan = recall_fetch_plan(top_k, state.vec_index.len(), has_post_filters);
        if fetch_plan.cap_bound {
            self.note_recall_candidate_cap_bound(namespace);
            tracing::debug!(
                target: "yantrikdb::recall",
                requested_top_k = top_k,
                requested_candidates = fetch_plan.requested_candidates,
                fetch_k = fetch_plan.fetch_k,
                candidate_cap = fetch_plan.candidate_cap,
                index_len = state.vec_index.len(),
                has_post_filters,
                profiled = true,
                "recall candidate cap bound"
            );
        }
        let fetch_k = fetch_plan.fetch_k;
        // v0.9.3 contract gate — same as `recall` (profiled duplicate).
        crate::validate::validate_embedding("recall_profiled", query_embedding, state.dim())?;
        let vec_results = state
            .vec_index
            .search_with_windows(query_embedding, fetch_k)?;
        // Mirrors recall(): winning chunk window per candidate for
        // snippet-span stamping after hydration.
        let mut win_by_rid: std::collections::HashMap<String, u32> =
            std::collections::HashMap::with_capacity(vec_results.len());
        for (rid, _, w) in &vec_results {
            win_by_rid.insert(rid.clone(), *w);
        }
        let vec_results: Vec<(String, f64)> = vec_results
            .into_iter()
            .map(|(rid, dist, _)| (rid, dist))
            .collect();
        let vec_search_ms = t_vec.elapsed().as_secs_f64() * 1000.0;
        let candidate_count = vec_results.len();

        if vec_results.is_empty() {
            // Validate the epoch even for an empty result (sol r6) — a
            // successful recall is never returned during an unvalidated
            // correction interval. On mismatch the wrapper retries.
            if !self.correction_epoch_validate(epoch0) {
                return Ok(None);
            }
            return Ok(Some(RecallProfiledResult {
                results: vec![],
                timings: RecallTimings {
                    vec_search_ms,
                    cache_score_ms: 0.0,
                    fetch_ms: 0.0,
                    scoring_ms: 0.0,
                    graph_ms: 0.0,
                    reinforce_ms: 0.0,
                    sort_truncate_ms: 0.0,
                    total_ms: t_start.elapsed().as_secs_f64() * 1000.0,
                    candidate_count: 0,
                    graph_expansion_count: 0,
                },
            }));
        }

        // ── Phase 2: Score from in-memory cache ──
        let t_cache_score = Instant::now();
        let mut scored: Vec<RecallResult> = Vec::new();
        {
            let cache = self.scoring_cache.read();
            for (rid, distance) in &vec_results {
                let Some(row) = cache.get(rid) else { continue };

                let status_ok = if include_consolidated {
                    row.consolidation_status == "active"
                        || row.consolidation_status == "consolidated"
                } else {
                    row.consolidation_status == "active"
                };
                if !status_ok {
                    continue;
                }
                if let Some(mt) = memory_type {
                    if row.memory_type != mt {
                        continue;
                    }
                }
                if let Some((start, end)) = time_window {
                    if row.created_at < start || row.created_at > end {
                        continue;
                    }
                }
                if let Some(ns) = namespace {
                    if row.namespace != ns {
                        continue;
                    }
                }
                if let Some(d) = domain {
                    if row.domain != d {
                        continue;
                    }
                }
                if let Some(s) = source {
                    if row.source != s {
                        continue;
                    }
                }

                let sim_score = (1.0 - distance).max(0.0);
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
                    &learned_weights,
                );
                let why = scoring::build_why(sim_score, recency, decay, row.valence);
                let contributions = scoring::adaptive_contributions(
                    sim_score,
                    decay,
                    recency,
                    row.importance,
                    &learned_weights,
                );
                let valence_multiplier = scoring::query_valence_boost(row.valence, query_sentiment);

                scored.push(RecallResult {
                    rid: rid.clone(),
                    memory_type: row.memory_type.clone(),
                    text: String::new(),
                    created_at: row.created_at,
                    importance: row.importance,
                    valence: row.valence,
                    score: composite,
                    scores: ScoreBreakdown {
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
        }
        let cache_score_ms = t_cache_score.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 2.5: High-importance memory fallback (similarity-gated) ──
        let t_fallback = Instant::now();
        {
            let total_memories = self.scoring_cache.read().len();
            let high_imp_threshold = if total_memories > 5000 { 0.5 } else { 0.7 };
            let min_sim_for_fallback = if total_memories > 5000 { 0.15 } else { 0.20 };
            let existing_rids: std::collections::HashSet<&str> =
                scored.iter().map(|r| r.rid.as_str()).collect();
            let important_rids: Vec<String> = {
                let cache = self.scoring_cache.read();
                cache
                    .iter()
                    .filter(|(rid, row)| {
                        // EIGHTH admission path. It listed most filters by hand
                        // and omitted certainty_min, so a high-importance,
                        // low-certainty row the vector pool missed could be
                        // re-admitted against an explicit certainty floor.
                        // One predicate, like every other lane.
                        row.importance >= high_imp_threshold
                            && !existing_rids.contains(rid.as_str())
                            && passes_recall_filters(
                                rid,
                                row,
                                include_consolidated,
                                memory_type,
                                time_window,
                                namespace,
                                domain,
                                source,
                                certainty_min,
                                None,
                            )
                    })
                    .map(|(rid, _)| rid.clone())
                    .collect()
            };

            if !important_rids.is_empty() {
                let rid_refs: Vec<&str> = important_rids.iter().map(|r| r.as_str()).collect();
                let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;
                let cache = self.scoring_cache.read();
                for rid in &important_rids {
                    let Some(row) = cache.get(rid) else { continue };
                    let Some(emb_blob) = emb_map.get(rid.as_str()) else {
                        continue;
                    };
                    let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
                    let sim_score =
                        crate::consolidate::cosine_similarity(query_embedding, &mem_emb) as f64;
                    if sim_score < min_sim_for_fallback {
                        continue;
                    }

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
                        &learned_weights,
                    );
                    let why = scoring::build_why(sim_score, recency, decay, row.valence);
                    let contributions = scoring::adaptive_contributions(
                        sim_score,
                        decay,
                        recency,
                        row.importance,
                        &learned_weights,
                    );
                    let valence_multiplier =
                        scoring::query_valence_boost(row.valence, query_sentiment);

                    scored.push(RecallResult {
                        rid: rid.clone(),
                        memory_type: row.memory_type.clone(),
                        text: String::new(),
                        created_at: row.created_at,
                        importance: row.importance,
                        valence: row.valence,
                        score: composite,
                        scores: ScoreBreakdown {
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
            }
        }
        let fallback_ms = t_fallback.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 1.5: FTS5 keyword fallback ──
        // (mirrors recall() Step 1.5 — see comments there for full rationale)
        let mut lex_by_rid: std::collections::HashMap<String, f64> =
            std::collections::HashMap::new();
        let t_fts = Instant::now();
        if !self.is_encrypted() {
            // Wired to tuning: YANTRIKDB_FTS_MIN_SIM (default 0.05). This was a
            // local const while tuning parsed-and-fingerprinted the knob nobody
            // read — the exact unwired-parameter failure tuning.rs exists to end.
            let fts_min_sim: f64 = crate::base::tuning::tuning().fts_min_sim;

            const STOPWORDS: &[&str] = &[
                "a", "an", "the", "is", "are", "am", "was", "were", "be", "been", "what", "who",
                "how", "when", "where", "which", "why", "do", "did", "does", "have", "has", "had",
                "i", "me", "my", "mine", "we", "our", "you", "your", "to", "of", "in", "on", "at",
                "by", "for", "with", "from", "about", "tell", "and", "or", "but", "not", "no",
                "it", "its", "that", "this", "there", "s", "she", "her", "he", "his", "they",
                "them", "most", "each", "any", "all", "every", "been", "being", "up", "out", "so",
                "if", "than", "very", "just", "also",
                // v0.9.3 accuracy work: function words that were slipping
                // through and anchoring keyword_match boosts on unrelated
                // memories (eval diagnosis: "during" in "what did we learn
                // DURING the project?" boosted a memory-leak memory to rank 1
                // while the actual lesson memories missed top-10 entirely).
                "during", "while", "after", "before", "between", "into", "over", "under", "through",
                "against", "within", "without", "us", "as", "then", "some", "more", "other",
                "these", "those", "will", "would", "could", "should", "can", "may", "might",
                "must", "shall", "get", "got", "make", "made", "let",
            ];

            {
                if let Some(qt) = query_text {
                    let raw_keywords: Vec<String> = qt
                        .split(|c: char| !c.is_alphanumeric())
                        .filter(|s| !s.is_empty() && s.len() > 1)
                        .filter(|s| !STOPWORDS.contains(&s.to_lowercase().as_str()))
                        .map(|s| s.to_string())
                        .collect();

                    // (mirrors recall() — see comments there for full rationale)
                    let mut keywords: Vec<String> = {
                        let gi = self.graph_index.read();
                        let query_tokens = crate::graph::tokenize(qt);
                        let matched = gi.entity_matches_query(&query_tokens);
                        let person_matches: Vec<_> = matched
                            .into_iter()
                            .filter(|(_, etype, _)| etype == "person")
                            .collect();
                        if person_matches.is_empty() {
                            raw_keywords
                        } else {
                            let person_tokens: std::collections::HashSet<String> = person_matches
                                .into_iter()
                                .flat_map(|(name, _, _)| {
                                    let mut tokens = vec![name.to_lowercase()];
                                    for token in name.split_whitespace() {
                                        tokens.push(token.to_lowercase());
                                    }
                                    tokens
                                })
                                .collect();
                            let filtered: Vec<String> = raw_keywords
                                .iter()
                                .filter(|kw| !person_tokens.contains(&kw.to_lowercase()))
                                .cloned()
                                .collect();
                            if filtered.is_empty() {
                                raw_keywords
                            } else {
                                filtered
                            }
                        }
                    };

                    // Entity-seeded FTS for group/aggregation queries (mirrors recall())
                    {
                        const GROUP_FTS_WORDS: &[&str] = &[
                            "team",
                            "group",
                            "colleagues",
                            "coworkers",
                            "friends",
                            "family",
                            "staff",
                            "members",
                            "people",
                        ];
                        let qt_lower = qt.to_lowercase();
                        if GROUP_FTS_WORDS.iter().any(|kw| qt_lower.contains(kw)) {
                            let gi = self.graph_index.read();
                            let query_tokens = crate::graph::tokenize(qt);
                            let matched = gi.entity_matches_query(&query_tokens);
                            if !matched.is_empty() {
                                let seed_names: Vec<&str> =
                                    matched.iter().map(|(n, _, _)| n.as_str()).collect();
                                let expanded = gi.expand_bfs(&seed_names, 2, 30);
                                let mut injected = 0usize;
                                for (name, hops, _) in &expanded {
                                    if *hops == 0 || injected >= 15 {
                                        continue;
                                    }
                                    if gi.entity_type(name).map_or(false, |t| t == "person") {
                                        for part in name.split_whitespace() {
                                            if part.len() > 1
                                                && !keywords
                                                    .iter()
                                                    .any(|k| k.eq_ignore_ascii_case(part))
                                            {
                                                keywords.push(part.to_string());
                                            }
                                        }
                                        injected += 1;
                                    }
                                }
                            }
                        }
                    }

                    if !keywords.is_empty() {
                        // Build FTS5 query with AND conjunction (mirrors recall())
                        let mut keyword_groups: Vec<String> = Vec::new();
                        for kw in &keywords {
                            let kw_lower = kw.to_lowercase();
                            let mut parts: Vec<String> = Vec::new();
                            parts.push(format!("\"{}\"", kw.replace('"', "")));
                            if let Some(stem) = simple_stem(&kw_lower) {
                                parts.push(format!("{}*", stem));
                            }
                            if let Some(alts) = irregular_verb_forms(&kw_lower) {
                                for alt in alts {
                                    parts.push(format!("\"{}\"", alt));
                                }
                            }
                            keyword_groups.push(if parts.len() == 1 {
                                parts[0].clone()
                            } else {
                                format!("({})", parts.join(" OR "))
                            });
                        }
                        let fts_query_and = if keyword_groups.len() >= 2 {
                            Some(keyword_groups.join(" AND "))
                        } else {
                            None
                        };
                        let fts_query_or = keyword_groups.join(" OR ");
                        let fts_query = fts_query_and.as_deref().unwrap_or(&fts_query_or);

                        let total_memories = self.scoring_cache.read().len();
                        let fts_limit = (total_memories / 100).max(30).min(200);

                        let mean_importance = {
                            let cache = self.scoring_cache.read();
                            if cache.is_empty() {
                                0.5
                            } else {
                                let sum: f64 = cache.values().map(|r| r.importance).sum();
                                let mean = sum / cache.len() as f64;
                                (mean * 0.7).max(0.25)
                            }
                        };

                        let fts_sql = if memory_type.is_some() {
                            format!(
                                "SELECT m.rid, memories_fts.rank FROM memories m \
                                 JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                 WHERE memories_fts MATCH ?1 \
                                 AND m.consolidation_status = 'active' \
                                 AND m.type = ?2 \
                                 {} \
                                 ORDER BY rank * (0.5 + m.importance) \
                                 LIMIT {}",
                                if namespace.is_some() {
                                    "AND m.namespace = ?3"
                                } else {
                                    ""
                                },
                                fts_limit,
                            )
                        } else {
                            format!(
                                "SELECT m.rid, memories_fts.rank FROM memories m \
                                 JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                 WHERE memories_fts MATCH ?1 \
                                 AND m.consolidation_status = 'active' \
                                 {} \
                                 ORDER BY rank * (0.5 + m.importance) \
                                 LIMIT {}",
                                if namespace.is_some() {
                                    "AND m.namespace = ?2"
                                } else {
                                    ""
                                },
                                fts_limit,
                            )
                        };

                        let run_fts_phase1 = |q: &str| -> Vec<(String, f64)> {
                            let conn = self.read_conn();
                            let mut stmt = conn.prepare_cached(&fts_sql).ok();
                            if let Some(ref mut stmt) = stmt {
                                let result: std::result::Result<Vec<(String, f64)>, _> =
                                    if let Some(mt) = memory_type {
                                        if let Some(ns) = namespace {
                                            stmt.query_map(params![q, mt, ns], rid_rank_row)
                                                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                        } else {
                                            stmt.query_map(params![q, mt], rid_rank_row)
                                                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                        }
                                    } else if let Some(ns) = namespace {
                                        stmt.query_map(params![q, ns], rid_rank_row)
                                            .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    } else {
                                        stmt.query_map(params![q], rid_rank_row)
                                            .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    };
                                result.unwrap_or_default()
                            } else {
                                vec![]
                            }
                        };

                        let mut fts_hits = run_fts_phase1(fts_query);
                        if fts_hits.len() < 5 && fts_query_and.is_some() {
                            fts_hits = run_fts_phase1(&fts_query_or);
                        }
                        // (rid, bm25 rank) rows feeding the per-query lexical
                        // strengths; every later FTS phase appends its rows.
                        let mut lex_ranked: Vec<(String, f64)> = fts_hits.clone();
                        let mut fts_rids: Vec<String> =
                            fts_hits.into_iter().map(|(rid, _)| rid).collect();

                        // Phase 2: Importance-filtered FTS (mirrors recall())
                        // Uses AND first, falls back to OR when AND is too strict.
                        {
                            let imp_fts_sql = if memory_type.is_some() {
                                format!(
                                    "SELECT m.rid, memories_fts.rank FROM memories m \
                                     JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                     WHERE memories_fts MATCH ?1 \
                                     AND m.consolidation_status = 'active' \
                                     AND m.importance > ?2 \
                                     AND m.type = ?3 \
                                     {} \
                                     ORDER BY m.importance DESC \
                                     LIMIT 100",
                                    if namespace.is_some() {
                                        "AND m.namespace = ?4"
                                    } else {
                                        ""
                                    },
                                )
                            } else {
                                format!(
                                    "SELECT m.rid, memories_fts.rank FROM memories m \
                                     JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                     WHERE memories_fts MATCH ?1 \
                                     AND m.consolidation_status = 'active' \
                                     AND m.importance > ?2 \
                                     {} \
                                     ORDER BY m.importance DESC \
                                     LIMIT 100",
                                    if namespace.is_some() {
                                        "AND m.namespace = ?3"
                                    } else {
                                        ""
                                    },
                                )
                            };

                            let run_fts_phase2 = |q: &str| -> Vec<(String, f64)> {
                                let conn = self.read_conn();
                                let mut stmt = conn.prepare_cached(&imp_fts_sql).ok();
                                if let Some(ref mut stmt) = stmt {
                                    let result: std::result::Result<Vec<(String, f64)>, _> =
                                        if let Some(mt) = memory_type {
                                            if let Some(ns) = namespace {
                                                stmt.query_map(
                                                    params![q, mean_importance, mt, ns],
                                                    rid_rank_row,
                                                )
                                                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                            } else {
                                                stmt.query_map(
                                                    params![q, mean_importance, mt],
                                                    rid_rank_row,
                                                )
                                                .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                            }
                                        } else if let Some(ns) = namespace {
                                            stmt.query_map(
                                                params![q, mean_importance, ns],
                                                rid_rank_row,
                                            )
                                            .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                        } else {
                                            stmt.query_map(
                                                params![q, mean_importance],
                                                rid_rank_row,
                                            )
                                            .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                        };
                                    result.unwrap_or_default()
                                } else {
                                    vec![]
                                }
                            };

                            // Run AND first, fall back to OR if too few results.
                            let mut imp_hits = run_fts_phase2(fts_query);
                            if imp_hits.len() < 10 && fts_query_and.is_some() {
                                let or_hits = run_fts_phase2(&fts_query_or);
                                let existing: std::collections::HashSet<String> =
                                    imp_hits.iter().map(|(rid, _)| rid.clone()).collect();
                                imp_hits.extend(
                                    or_hits
                                        .into_iter()
                                        .filter(|(rid, _)| !existing.contains(rid)),
                                );
                            }

                            lex_ranked.extend(imp_hits.iter().cloned());
                            // Own the set so it doesn't borrow `fts_rids` —
                            // we push into `fts_rids` below, which requires
                            // a mutable borrow that conflicts with any live
                            // immutable borrow. (E0502 caught by clippy
                            // --all-features in CI 2026-05-14.)
                            let existing_set: std::collections::HashSet<String> =
                                fts_rids.iter().cloned().collect();
                            for (rid, _) in imp_hits {
                                if !existing_set.contains(&rid) {
                                    fts_rids.push(rid);
                                }
                            }
                        }

                        // Phase 2.5: Per-keyword anchor scan (mirrors recall())
                        if keyword_groups.len() >= 2 {
                            let anchor_fts_sql = if memory_type.is_some() {
                                format!(
                                    "SELECT m.rid, memories_fts.rank FROM memories m \
                                     JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                     WHERE memories_fts MATCH ?1 \
                                     AND m.consolidation_status = 'active' \
                                     AND m.importance > 0.5 \
                                     AND m.type = ?2 \
                                     {} \
                                     ORDER BY m.importance DESC \
                                     LIMIT 10",
                                    if namespace.is_some() {
                                        "AND m.namespace = ?3"
                                    } else {
                                        ""
                                    },
                                )
                            } else {
                                format!(
                                    "SELECT m.rid, memories_fts.rank FROM memories m \
                                     JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                     WHERE memories_fts MATCH ?1 \
                                     AND m.consolidation_status = 'active' \
                                     AND m.importance > 0.5 \
                                     {} \
                                     ORDER BY m.importance DESC \
                                     LIMIT 10",
                                    if namespace.is_some() {
                                        "AND m.namespace = ?2"
                                    } else {
                                        ""
                                    },
                                )
                            };

                            let existing_fts: std::collections::HashSet<String> =
                                fts_rids.iter().cloned().collect();

                            for group in &keyword_groups {
                                let anchor_hits: Vec<(String, f64)> = {
                                    let conn = self.read_conn();
                                    let mut stmt = conn.prepare_cached(&anchor_fts_sql).ok();
                                    if let Some(ref mut stmt) = stmt {
                                        let result: std::result::Result<Vec<(String, f64)>, _> =
                                            if let Some(mt) = memory_type {
                                                if let Some(ns) = namespace {
                                                    stmt.query_map(
                                                        params![group, mt, ns],
                                                        rid_rank_row,
                                                    )
                                                    .map(|rows| {
                                                        rows.filter_map(|r| r.ok()).collect()
                                                    })
                                                } else {
                                                    stmt.query_map(params![group, mt], rid_rank_row)
                                                        .map(|rows| {
                                                            rows.filter_map(|r| r.ok()).collect()
                                                        })
                                                }
                                            } else if let Some(ns) = namespace {
                                                stmt.query_map(params![group, ns], rid_rank_row)
                                                    .map(|rows| {
                                                        rows.filter_map(|r| r.ok()).collect()
                                                    })
                                            } else {
                                                stmt.query_map(params![group], rid_rank_row).map(
                                                    |rows| rows.filter_map(|r| r.ok()).collect(),
                                                )
                                            };
                                        result.unwrap_or_default()
                                    } else {
                                        vec![]
                                    }
                                };

                                for (rid, rank) in anchor_hits {
                                    lex_ranked.push((rid.clone(), rank));
                                    if !existing_fts.contains(&rid) {
                                        fts_rids.push(rid);
                                    }
                                }
                            }
                        }

                        // Per-query lexical strengths from every phase's bm25
                        // rows — read by the keyword reserve, the only
                        // consumer left (see engine/lexical.rs).
                        lex_by_rid = crate::engine::lexical::lexical_strengths(&lex_ranked);

                        {
                            let fts_rid_set: std::collections::HashSet<&str> =
                                fts_rids.iter().map(|r| r.as_str()).collect();
                            for result in &mut scored {
                                if fts_rid_set.contains(result.rid.as_str())
                                    && !result.why_retrieved.iter().any(|w| w == "keyword_match")
                                {
                                    let sim = result.scores.similarity;
                                    let lex =
                                        lex_by_rid.get(result.rid.as_str()).copied().unwrap_or(1.0);
                                    let boost = crate::engine::lexical::keyword_lane_boost(
                                        learned_weights.keyword_boost,
                                        sim,
                                        lex,
                                    );
                                    result.score += boost;
                                    result.why_retrieved.push("keyword_match".to_string());
                                }
                            }
                        }

                        let existing_rids: std::collections::HashSet<String> =
                            scored.iter().map(|r| r.rid.clone()).collect();
                        let new_fts_rids: Vec<String> = fts_rids
                            .into_iter()
                            .filter(|r| !existing_rids.contains(r))
                            .collect();

                        if !new_fts_rids.is_empty() {
                            let rid_refs: Vec<&str> =
                                new_fts_rids.iter().map(|r| r.as_str()).collect();
                            let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;

                            let cache = self.scoring_cache.read();
                            for rid in &new_fts_rids {
                                let Some(row) = cache.get(rid) else { continue };
                                // ONE predicate, same as the vector lane. This
                                // path used to check only status and time, so a
                                // record the caller excluded by domain, source or
                                // certainty could be re-admitted here.
                                if !passes_recall_filters(
                                    rid,
                                    row,
                                    include_consolidated,
                                    memory_type,
                                    time_window,
                                    namespace,
                                    domain,
                                    source,
                                    certainty_min,
                                    None,
                                ) {
                                    continue;
                                }

                                let Some(emb_blob) = emb_map.get(rid.as_str()) else {
                                    continue;
                                };
                                let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
                                let sim_score = crate::consolidate::cosine_similarity(
                                    query_embedding,
                                    &mem_emb,
                                ) as f64;

                                let lex = lex_by_rid.get(rid.as_str()).copied().unwrap_or(0.0);
                                if sim_score < fts_min_sim
                                    && lex < crate::engine::lexical::LEX_STRONG
                                {
                                    // Below the cosine noise floor AND not a
                                    // near-best lexical match — noise. A top
                                    // bm25 match passes regardless: dilution
                                    // parks exact-phrase records at any sim.
                                    continue;
                                }

                                let decay =
                                    scoring::ranking_decay(row.importance, row.created_at, ts);
                                let age = ts - row.created_at;
                                let recency = scoring::recency_score(age);
                                let composite = scoring::adaptive_composite_score(
                                    sim_score,
                                    decay,
                                    recency,
                                    row.importance,
                                    row.valence,
                                    query_sentiment,
                                    &learned_weights,
                                );
                                let kw_boost = crate::engine::lexical::keyword_lane_boost(
                                    learned_weights.keyword_boost,
                                    sim_score,
                                    lex,
                                );
                                let mut why =
                                    scoring::build_why(sim_score, recency, decay, row.valence);
                                why.push("keyword_match".to_string());
                                why.push("fts_sourced".to_string());
                                let contributions = scoring::adaptive_contributions(
                                    sim_score,
                                    decay,
                                    recency,
                                    row.importance,
                                    &learned_weights,
                                );
                                let valence_multiplier =
                                    scoring::query_valence_boost(row.valence, query_sentiment);

                                scored.push(RecallResult {
                                    rid: rid.clone(),
                                    memory_type: row.memory_type.clone(),
                                    text: String::new(),
                                    created_at: row.created_at,
                                    importance: row.importance,
                                    valence: row.valence,
                                    score: composite + kw_boost,
                                    scores: ScoreBreakdown {
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
                        }
                    }
                }
            }
        }
        let fts_ms = t_fts.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 1.6 (C4): claims lane (mirrors recall() Step 1.6) ──
        self.apply_claims_lane(
            &mut scored,
            query_embedding,
            query_text,
            namespace,
            time_window,
            include_consolidated,
            memory_type,
            domain,
            source,
            certainty_min,
            None, // #149: valid-time bounds are default-lane only (like explain)
            &learned_weights,
            ts,
            query_sentiment,
        )?;

        // ── Phase 2.7: Valence-based retrieval for emotional queries ──
        // (mirrors recall() Step 2.7)
        if query_sentiment.abs() > 0.5 {
            const VALENCE_SCAN_THRESHOLD: f64 = 0.4;
            const VALENCE_SCAN_MAX: usize = 30;
            let valence_min_sim: f64 = crate::base::tuning::tuning().valence_min_sim;

            let existing_rids: std::collections::HashSet<&str> =
                scored.iter().map(|r| r.rid.as_str()).collect();

            let valence_rids: Vec<String> = {
                let cache = self.scoring_cache.read();
                let mut candidates: Vec<(String, f64)> = cache
                    .iter()
                    .filter(|(rid, row)| {
                        row.consolidation_status == "active"
                            && !existing_rids.contains(rid.as_str())
                            && row.valence.abs() >= VALENCE_SCAN_THRESHOLD
                            && (query_sentiment * row.valence > 0.0
                                || (query_sentiment < 0.0 && row.valence < -0.2))
                            && row.importance >= 0.5
                            && memory_type.map_or(true, |mt| row.memory_type == mt)
                            && time_window
                                .map_or(true, |(s, e)| row.created_at >= s && row.created_at <= e)
                            && namespace.map_or(true, |ns| row.namespace == ns)
                    })
                    .map(|(rid, row)| {
                        let rank = row.valence.abs() * row.importance;
                        (rid.clone(), rank)
                    })
                    .collect();
                // Fix (k): this rank feeds a truncation, and on uniform
                // corpora it is a massive tie — the total order (quantized
                // rank desc, rid asc) is what keeps the admitted subset
                // identical across opens (the audit rule from fixes (e)/(f),
                // applied to every remaining rank-and-take site at once).
                candidates.sort_by(|a, b| {
                    crate::engine::lexical::quantize_score(b.1)
                        .total_cmp(&crate::engine::lexical::quantize_score(a.1))
                        .then_with(|| a.0.cmp(&b.0))
                });
                candidates
                    .into_iter()
                    .take(VALENCE_SCAN_MAX)
                    .map(|(rid, _)| rid)
                    .collect()
            };

            if !valence_rids.is_empty() {
                let rid_refs: Vec<&str> = valence_rids.iter().map(|r| r.as_str()).collect();
                let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;
                let cache = self.scoring_cache.read();
                for rid in &valence_rids {
                    let Some(row) = cache.get(rid) else { continue };
                    let Some(emb_blob) = emb_map.get(rid.as_str()) else {
                        continue;
                    };
                    let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
                    let sim_score =
                        crate::consolidate::cosine_similarity(query_embedding, &mem_emb) as f64;

                    if sim_score < valence_min_sim {
                        continue;
                    }

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
                        &learned_weights,
                    );
                    // Bounded multiplicative lift, not an unbounded add:
                    // a strongly-valenced record still competes, but one
                    // with no semantic match cannot outrank a real answer.
                    let valence_lift = scoring::lane_lift_mult(row.valence.abs() * row.importance);
                    let mut why = scoring::build_why(sim_score, recency, decay, row.valence);
                    why.push("valence_match".to_string());
                    let contributions = scoring::adaptive_contributions(
                        sim_score,
                        decay,
                        recency,
                        row.importance,
                        &learned_weights,
                    );
                    let valence_multiplier =
                        scoring::query_valence_boost(row.valence, query_sentiment);

                    scored.push(RecallResult {
                        rid: rid.clone(),
                        memory_type: row.memory_type.clone(),
                        text: String::new(),
                        created_at: row.created_at,
                        importance: row.importance,
                        valence: row.valence,
                        score: composite * valence_lift,
                        scores: ScoreBreakdown {
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
            }
        }

        // ── Step 2.9: Lexical rescue fallback (mirrors recall()) ──
        if !self.is_encrypted() {
            if let Some(qt) = query_text {
                let best_score = scored.iter().map(|r| r.score).fold(0.0f64, f64::max);
                const COLD_ACTIVATION_THRESHOLD: f64 = 0.55;
                // Wired to tuning: YANTRIKDB_COLD_MIN_SIM (default 0.10).
                let cold_min_sim: f64 = crate::base::tuning::tuning().cold_min_sim;
                const COLD_MAX_CANDIDATES: usize = 30;

                if best_score < COLD_ACTIVATION_THRESHOLD {
                    let cold_keywords: Vec<String> = qt
                        .split(|c: char| !c.is_alphanumeric())
                        .filter(|s| !s.is_empty() && s.len() > 1)
                        .filter(|s| {
                            const STOP: &[&str] = &[
                                "a", "an", "the", "is", "are", "am", "was", "were", "be", "been",
                                "what", "who", "how", "when", "where", "which", "why", "do", "did",
                                "does", "have", "has", "had", "i", "me", "my", "mine", "we", "our",
                                "you", "your", "to", "of", "in", "on", "at", "by", "for", "with",
                                "from", "about", "tell", "and", "or", "but", "not", "no", "it",
                                "its", "that", "this", "there", "s", "she", "her", "he", "his",
                                "they", "them",
                            ];
                            !STOP.contains(&s.to_lowercase().as_str())
                        })
                        .map(|s| s.to_string())
                        .collect();

                    if !cold_keywords.is_empty() {
                        let mut fts_parts: Vec<String> = Vec::new();
                        for kw in &cold_keywords {
                            let kw_lower = kw.to_lowercase();
                            fts_parts.push(format!("\"{}\"", kw.replace('"', "")));
                            if let Some(stem) = simple_stem(&kw_lower) {
                                fts_parts.push(format!("{}*", stem));
                            }
                            if let Some(alts) = irregular_verb_forms(&kw_lower) {
                                for alt in alts {
                                    fts_parts.push(format!("\"{}\"", alt));
                                }
                            }
                        }
                        let cold_fts = fts_parts.join(" OR ");

                        let cold_sql = if memory_type.is_some() {
                            format!(
                                "SELECT m.rid FROM memories m \
                                 JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                 WHERE memories_fts MATCH ?1 \
                                 AND m.consolidation_status = 'active' \
                                 AND m.type = ?2 \
                                 {} \
                                 ORDER BY m.importance DESC \
                                 LIMIT {}",
                                if namespace.is_some() {
                                    "AND m.namespace = ?3"
                                } else {
                                    ""
                                },
                                COLD_MAX_CANDIDATES,
                            )
                        } else {
                            format!(
                                "SELECT m.rid FROM memories m \
                                 JOIN memories_fts ON memories_fts.rowid = m.rowid \
                                 WHERE memories_fts MATCH ?1 \
                                 AND m.consolidation_status = 'active' \
                                 {} \
                                 ORDER BY m.importance DESC \
                                 LIMIT {}",
                                if namespace.is_some() {
                                    "AND m.namespace = ?2"
                                } else {
                                    ""
                                },
                                COLD_MAX_CANDIDATES,
                            )
                        };

                        let cold_rids: Vec<String> = {
                            let conn = self.read_conn();
                            let mut stmt = conn.prepare_cached(&cold_sql).ok();
                            if let Some(ref mut stmt) = stmt {
                                let result: std::result::Result<Vec<String>, _> = if let Some(mt) =
                                    memory_type
                                {
                                    if let Some(ns) = namespace {
                                        stmt.query_map(params![cold_fts, mt, ns], |row| {
                                            row.get::<_, String>(0)
                                        })
                                        .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    } else {
                                        stmt.query_map(params![cold_fts, mt], |row| {
                                            row.get::<_, String>(0)
                                        })
                                        .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                    }
                                } else if let Some(ns) = namespace {
                                    stmt.query_map(params![cold_fts, ns], |row| {
                                        row.get::<_, String>(0)
                                    })
                                    .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                } else {
                                    stmt.query_map(params![cold_fts], |row| row.get::<_, String>(0))
                                        .map(|rows| rows.filter_map(|r| r.ok()).collect())
                                };
                                result.unwrap_or_default()
                            } else {
                                vec![]
                            }
                        };

                        let existing_rids: std::collections::HashSet<String> =
                            scored.iter().map(|r| r.rid.clone()).collect();
                        let new_cold: Vec<String> = cold_rids
                            .into_iter()
                            .filter(|r| !existing_rids.contains(r))
                            .collect();

                        if !new_cold.is_empty() {
                            let rid_refs: Vec<&str> = new_cold.iter().map(|r| r.as_str()).collect();
                            let emb_map = self.fetch_embeddings_by_rids(&rid_refs)?;
                            let cache = self.scoring_cache.read();

                            for rid in &new_cold {
                                let Some(row) = cache.get(rid) else { continue };
                                // The cold-lane SQL filters only status,
                                // access_count, type and namespace, so domain,
                                // source and certainty must be enforced here.
                                if !passes_recall_filters(
                                    rid,
                                    row,
                                    include_consolidated,
                                    memory_type,
                                    time_window,
                                    namespace,
                                    domain,
                                    source,
                                    certainty_min,
                                    None,
                                ) {
                                    continue;
                                }
                                let Some(emb_blob) = emb_map.get(rid.as_str()) else {
                                    continue;
                                };
                                let mem_emb = crate::serde_helpers::deserialize_f32(emb_blob);
                                let sim_score = crate::consolidate::cosine_similarity(
                                    query_embedding,
                                    &mem_emb,
                                ) as f64;

                                if sim_score < cold_min_sim {
                                    continue;
                                }

                                let decay =
                                    scoring::ranking_decay(row.importance, row.created_at, ts);
                                let age = ts - row.created_at;
                                let recency = scoring::recency_score(age);
                                let composite = scoring::adaptive_composite_score(
                                    sim_score,
                                    decay,
                                    recency,
                                    row.importance,
                                    row.valence,
                                    query_sentiment,
                                    &learned_weights,
                                );
                                // Bounded multiplicative lift (see
                                // LANE_LIFT_MAX): admission is the lane's
                                // job; it does not buy relevance.
                                let cold_lift = scoring::lane_lift_mult(row.importance);
                                let mut why =
                                    scoring::build_why(sim_score, recency, decay, row.valence);
                                why.push("lexical_rescue".to_string());
                                let contributions = scoring::adaptive_contributions(
                                    sim_score,
                                    decay,
                                    recency,
                                    row.importance,
                                    &learned_weights,
                                );
                                let valence_multiplier =
                                    scoring::query_valence_boost(row.valence, query_sentiment);

                                scored.push(RecallResult {
                                    rid: rid.clone(),
                                    memory_type: row.memory_type.clone(),
                                    text: String::new(),
                                    created_at: row.created_at,
                                    importance: row.importance,
                                    valence: row.valence,
                                    score: composite * cold_lift,
                                    scores: ScoreBreakdown {
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
                        }
                    }
                }
            }
        }

        // ── Phase 3: Graph expansion ──
        let t_graph = Instant::now();
        let mut graph_expansion_count = 0usize;
        if expand_entities {
            let gi = self.graph_index.read();
            let query_entities: Vec<(String, String, u32)> = if let Some(qt) = query_text {
                let query_tokens = crate::graph::tokenize(qt);
                gi.entity_matches_query(&query_tokens)
            } else {
                vec![]
            };

            let (mut base_boost, mut seed_entities, entity_idfs): (
                f64,
                Vec<String>,
                std::collections::HashMap<String, f64>,
            ) = if !query_entities.is_empty() {
                let has_person = query_entities.iter().any(|(_, etype, _)| etype == "person");
                let factor = if has_person {
                    0.20
                } else if query_entities.len() >= 2 {
                    0.15
                } else {
                    0.12
                };
                let idfs: std::collections::HashMap<String, f64> = query_entities
                    .iter()
                    .map(|(name, _, mc)| {
                        let idf = 1.0 / (1.0 + (*mc as f64).max(1.0).ln());
                        (name.to_lowercase(), idf)
                    })
                    .collect();
                let names: Vec<String> = query_entities.into_iter().map(|(n, _, _)| n).collect();
                (factor, names, idfs)
            } else if query_text.is_none() {
                // Embedding-only search (no query text): seed from top results
                let mut seed_sorted = scored.clone();
                seed_sorted.sort_by(crate::engine::lexical::rank_cmp);
                let seed_count = 3.min(seed_sorted.len());
                let seed_rids: Vec<&str> = seed_sorted[..seed_count]
                    .iter()
                    .map(|r| r.rid.as_str())
                    .collect();
                let seeds = gi.entities_for_memories(&seed_rids);
                (0.05, seeds, std::collections::HashMap::new())
            } else {
                (0.0, vec![], std::collections::HashMap::new())
            };

            // Group query expansion: if query mentions "team", "family", etc.,
            // seed expansion with person entities CONNECTED to query entities.
            //
            // Previous approach: entities_by_type("person").take(15) grabbed
            // arbitrary person entities, often filler characters unrelated to
            // the query. New approach: BFS from query entities finds people
            // actually connected to the subject (e.g., Priya → Arjun, Meera,
            // Appa, Amma for "family"; Priya → Deepa, Neha for "team").
            const GROUP_KEYWORDS: &[&str] = &[
                "team",
                "group",
                "colleagues",
                "coworkers",
                "friends",
                "family",
                "staff",
                "members",
                "people",
            ];
            if let Some(qt) = query_text {
                let qt_lower = qt.to_lowercase();
                if GROUP_KEYWORDS.iter().any(|kw| qt_lower.contains(kw)) {
                    if !seed_entities.is_empty() {
                        // BFS from query entities to find connected person entities
                        let query_seeds: Vec<&str> =
                            seed_entities.iter().map(|s| s.as_str()).collect();
                        let nearby = gi.expand_bfs(&query_seeds, 2, 50);
                        for (name, hops, _) in &nearby {
                            if *hops > 0
                                && gi.entity_type(name).map_or(false, |t| t == "person")
                                && !seed_entities.contains(&name.to_string())
                            {
                                seed_entities.push(name.clone());
                            }
                        }
                    } else {
                        // No query entities — fall back to type-based expansion
                        let person_entities = gi.entities_by_type("person");
                        for person in person_entities.into_iter().take(15) {
                            if !seed_entities.contains(&person) {
                                seed_entities.push(person);
                            }
                        }
                    }
                    base_boost = base_boost.max(0.20_f64);
                }
            }

            const MAX_BOOST_PER_MEMORY: f64 = 0.25;
            const MAX_GRAPH_FRACTION: f64 = 1.0;
            const MAX_SEED_ENTITIES: usize = 8;

            // Cap seed entities to prevent graph explosion with many entities
            if seed_entities.len() > MAX_SEED_ENTITIES {
                seed_entities.truncate(MAX_SEED_ENTITIES);
            }

            if !seed_entities.is_empty() && base_boost > 0.0 {
                let seed_refs: Vec<&str> = seed_entities.iter().map(|s| s.as_str()).collect();
                let expanded = gi.expand_bfs(&seed_refs, 2, 30);
                let expanded_map: std::collections::HashMap<String, (u8, f64)> = expanded
                    .iter()
                    .map(|(name, hops, weight)| (name.clone(), (*hops, *weight)))
                    .collect();

                for result in &mut scored {
                    let prox = gi.graph_proximity(&result.rid, &expanded_map);
                    if prox > 0.0 {
                        let mem_entities: Vec<String> = gi
                            .entities_for_memory(&result.rid)
                            .into_iter()
                            .map(|s| s.to_string())
                            .collect();
                        let mut best_idf = 1.0f64;
                        let mut connecting_entity = String::new();
                        for entity in &mem_entities {
                            if expanded_map.contains_key(entity) {
                                let idf = entity_idfs
                                    .get(&entity.to_lowercase())
                                    .copied()
                                    .unwrap_or(1.0);
                                if connecting_entity.is_empty() || idf > best_idf {
                                    best_idf = idf;
                                    connecting_entity = entity.clone();
                                }
                            }
                        }
                        let consolidation_factor = {
                            let cache = self.scoring_cache.read();
                            cache
                                .get(&result.rid)
                                .map(|r| {
                                    if r.consolidation_status == "consolidated" {
                                        0.5
                                    } else {
                                        1.0
                                    }
                                })
                                .unwrap_or(1.0)
                        };
                        // MULTIPLICATIVE, not additive. This was
                        // `result.score += boost` with boost capped at 0.25 —
                        // on composites that typically run 0.2-0.6, a flat
                        // +0.25 could nearly double a weak score, so a graph
                        // edge promoted records relevance had ranked below
                        // them. Same wall as importance, freshness and the
                        // graph composite; this was the last additive site.
                        //
                        // The quality modifiers are kept and now shape the
                        // EVIDENCE rather than the magnitude: proximity,
                        // discounted by the connecting entity's inverse
                        // document frequency (a hub entity is weak evidence)
                        // and by the consolidation penalty, scaled by the
                        // expansion strength. GRAPH_SCALE alone sets the
                        // ceiling: ~+3.5% at defaults (ln 1.30 x 0.13).
                        let evidence = ((base_boost / MAX_BOOST_PER_MEMORY)
                            * prox
                            * best_idf
                            * consolidation_factor)
                            .clamp(0.0, 1.0);
                        result.scores.graph_proximity = prox;
                        result.score *= scoring::graph_mult(evidence);
                        if !connecting_entity.is_empty() {
                            result
                                .why_retrieved
                                .push(format!("graph-connected via {connecting_entity}"));
                        }
                    }
                }

                let max_graph_only = ((MAX_GRAPH_FRACTION * top_k as f64).ceil() as usize).max(1);
                let all_entity_names: Vec<&str> =
                    expanded.iter().map(|(n, _, _)| n.as_str()).collect();
                let graph_rids = gi.memories_for_entities(&all_entity_names);
                let existing_rids: std::collections::HashSet<&str> =
                    scored.iter().map(|r| r.rid.as_str()).collect();

                let preselect_pool = max_graph_only * 5;
                let new_rids: Vec<String> = {
                    let cache = self.scoring_cache.read();
                    let mut candidates: Vec<(String, f64)> = graph_rids
                        .into_iter()
                        .filter(|r| !existing_rids.contains(r.as_str()))
                        .filter_map(|r| {
                            let row = cache.get(r.as_str())?;
                            // Graph-only admission used to skip domain,
                            // source and certainty; one predicate now.
                            if !passes_recall_filters(
                                &r,
                                row,
                                include_consolidated,
                                memory_type,
                                time_window,
                                namespace,
                                domain,
                                source,
                                certainty_min,
                                None,
                            ) {
                                return None;
                            }
                            let prox = gi.graph_proximity(&r, &expanded_map);
                            let rank = row.importance * (0.3 + 0.7 * prox);
                            Some((r, rank))
                        })
                        .collect();
                    // Fix (k): this rank feeds a truncation, and on uniform
                    // corpora it is a massive tie — the total order (quantized
                    // rank desc, rid asc) is what keeps the admitted subset
                    // identical across opens (the audit rule from fixes (e)/(f),
                    // applied to every remaining rank-and-take site at once).
                    candidates.sort_by(|a, b| {
                        crate::engine::lexical::quantize_score(b.1)
                            .total_cmp(&crate::engine::lexical::quantize_score(a.1))
                            .then_with(|| a.0.cmp(&b.0))
                    });
                    candidates
                        .into_iter()
                        .take(preselect_pool)
                        .map(|(rid, _)| rid)
                        .collect()
                };
                graph_expansion_count = new_rids.len();

                if !new_rids.is_empty() {
                    let new_rid_refs: Vec<&str> = new_rids.iter().map(|r| r.as_str()).collect();
                    let emb_map = self.fetch_embeddings_by_rids(&new_rid_refs)?;

                    let cache = self.scoring_cache.read();
                    for rid in &new_rids {
                        if let (Some(row), Some(emb_blob)) =
                            (cache.get(rid.as_str()), emb_map.get(rid.as_str()))
                        {
                            let mem_embedding = crate::serde_helpers::deserialize_f32(emb_blob);
                            let sim_score = crate::consolidate::cosine_similarity(
                                query_embedding,
                                &mem_embedding,
                            ) as f64;
                            let decay = scoring::ranking_decay(row.importance, row.created_at, ts);
                            let age = ts - row.created_at;
                            let recency = scoring::recency_score(age);
                            let prox = gi.graph_proximity(rid, &expanded_map);
                            let composite = scoring::adaptive_graph_composite_score(
                                sim_score,
                                decay,
                                recency,
                                row.importance,
                                row.valence,
                                prox,
                                query_sentiment,
                                &learned_weights,
                            );
                            let mut why =
                                scoring::build_why(sim_score, recency, decay, row.valence);

                            let mem_entities: Vec<String> = gi
                                .entities_for_memory(rid)
                                .into_iter()
                                .map(|s| s.to_string())
                                .collect();
                            for entity in &mem_entities {
                                if expanded_map.contains_key(entity) {
                                    why.push(format!("graph-connected via {entity}"));
                                    break;
                                }
                            }

                            let contributions = scoring::adaptive_graph_contributions(
                                sim_score,
                                decay,
                                recency,
                                row.importance,
                                prox,
                                &learned_weights,
                            );
                            let valence_multiplier =
                                scoring::query_valence_boost(row.valence, query_sentiment);

                            scored.push(RecallResult {
                                rid: rid.clone(),
                                memory_type: row.memory_type.clone(),
                                text: String::new(),
                                created_at: row.created_at,
                                importance: row.importance,
                                valence: row.valence,
                                score: composite,
                                scores: ScoreBreakdown {
                                    similarity: sim_score,
                                    decay,
                                    recency,
                                    importance: row.importance,
                                    graph_proximity: prox,
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
                    }
                }
            }
        }
        let graph_ms = t_graph.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 3.4: status-led eligibility filter (mirrors recall()) ──
        // Profiled variant applies the policy default (no per-call
        // include_superseded override) so timings stay representative
        // of the real read path.
        if self.status_read_policy() && !scored.is_empty() {
            let pool_rids: Vec<&str> = scored.iter().map(|r| r.rid.as_str()).collect();
            let superseded = self.superseded_rids_among(&pool_rids)?;
            if !superseded.is_empty() {
                scored.retain(|r| !superseded.contains(&r.rid));
            }
        }

        // ── Phase 3.5: Keyword slot reservation (mirrors recall()) ──
        apply_lane_agreement(&mut scored, &win_by_rid);
        crate::engine::lexical::apply_keyword_reserve(&mut scored, &lex_by_rid, top_k);

        // Lifecycle eligibility is rechecked after all profiled lanes merge,
        // matching the default recall path's final admission boundary.
        {
            let cache = self.scoring_cache.read();
            scored.retain(|result| {
                cache
                    .get(&result.rid)
                    .is_some_and(synthesis_lifecycle_allows)
            });
            apply_synthesis_representation_preference(&mut scored, &cache, query_text);
        }

        // ── Phase 4: MMR diversity selection ──
        let t_sort = Instant::now();
        scored.sort_by(crate::engine::lexical::rank_cmp);
        // Quotas run AFTER the sort, not before: the sort re-orders the
        // whole pool by score, so a quota pass placed above it was silently
        // undone by the very next statement. The function was correct and
        // correctly called, and had no effect whatsoever.
        apply_lane_quotas(&mut scored, &win_by_rid, top_k);

        // Novelty selection REPLACES MMR rather than composing with it.
        // They are two diversity mechanisms with different notions of
        // redundancy (embedding vs lexical); running both in sequence would
        // make each one's weight mean something different depending on what
        // the other did first, which is unsweepable.
        let novelty_w = crate::base::tuning::tuning().novelty_weight;
        let min_pool_for_mmr = top_k.saturating_mul(3).max(20);
        // Set-level re-selection.
        //
        // Needs TEXT, and step 5 below hydrates text only for the final
        // top_k — by design, so a 100-candidate recall does not fetch 100
        // bodies. The first version of this ran here against the
        // pre-hydration structs, where every `text` is String::new(); every
        // novelty score was therefore 0, every candidate tied, and the
        // greedy reproduced score order EXACTLY. It looked like a working
        // feature with a weight that did nothing.
        //
        // So when novelty is enabled we hydrate the SELECTION POOL early —
        // capped, and only on this path, so the default costs nothing.
        if std::env::var("YANTRIKDB_DEBUG_NOVELTY").is_ok() {
            eprintln!(
                "[novelty-gate] w={:.3} scored={} top_k={} enters={}",
                novelty_w,
                scored.len(),
                top_k,
                novelty_w > 0.0 && scored.len() > top_k
            );
        }
        if novelty_w > 0.0 && scored.len() > top_k {
            let pool = scored
                .len()
                .min(top_k.saturating_mul(10).max(min_pool_for_mmr));
            scored.truncate(pool);
            let pool_rids: Vec<&str> = scored.iter().map(|r| r.rid.as_str()).collect();
            let pool_text = self.fetch_text_metadata_by_rids(&pool_rids)?;
            for r in &mut scored {
                if let Some(tm) = pool_text.get(&r.rid) {
                    r.text = tm.text.clone();
                }
            }
            // Diagnostic, off unless asked for: distinguishes "pool too
            // small to select from" and "text never arrived" from "policy
            // ran and chose this". Without it the three are indistinguishable
            // from the outside, which cost a full debugging cycle.
            if std::env::var("YANTRIKDB_DEBUG_NOVELTY").is_ok() {
                let hydrated = scored.iter().filter(|r| !r.text.is_empty()).count();
                let before: Vec<String> =
                    scored.iter().take(top_k).map(|r| r.rid.clone()).collect();
                apply_novelty_selection(&mut scored, top_k);
                let after: Vec<String> = scored.iter().take(top_k).map(|r| r.rid.clone()).collect();
                eprintln!(
                    "[novelty] w={:.3} pool={} hydrated={} top_k={} reordered={}",
                    novelty_w,
                    pool,
                    hydrated,
                    top_k,
                    before != after
                );
            } else {
                apply_novelty_selection(&mut scored, top_k);
            }
            scored.truncate(top_k);
        }

        if novelty_w <= 0.0 && scored.len() > top_k && scored.len() >= min_pool_for_mmr {
            let pool_size = scored.len().min(top_k.saturating_mul(10));
            scored.truncate(pool_size);

            let pool_rids: Vec<&str> = scored.iter().map(|r| r.rid.as_str()).collect();
            let emb_map = self.fetch_embeddings_by_rids(&pool_rids)?;

            let pool_embeddings: Vec<Option<Vec<f32>>> = scored
                .iter()
                .map(|r| {
                    emb_map
                        .get(r.rid.as_str())
                        .map(|blob| crate::serde_helpers::deserialize_f32(blob))
                })
                .collect();

            // λ = 0.9 (2026-08-05): see recall_inner's MMR block for the
            // production-clone measurement behind the raise from 0.7.
            // Wired to tuning: YANTRIKDB_MMR_LAMBDA (default 0.9, clamped 0..=1
            // at parse). The hardcoded const made every λ sweep a silent no-op
            // while the fingerprint stamped the env value as if it governed.
            let lambda: f64 = crate::base::tuning::tuning().mmr_lambda;
            const SIM_THRESHOLD: f64 = 0.98;

            let mut selected: Vec<usize> = Vec::with_capacity(top_k);
            let mut selected_embeddings: Vec<&[f32]> = Vec::with_capacity(top_k);

            if !scored.is_empty() {
                selected.push(0);
                if let Some(Some(ref emb)) = pool_embeddings.first() {
                    selected_embeddings.push(emb);
                }
            }

            for _ in 1..top_k {
                let mut best_idx = None;
                let mut best_mmr = f64::NEG_INFINITY;

                for (idx, result) in scored.iter().enumerate() {
                    if selected.contains(&idx) {
                        continue;
                    }

                    let relevance = result.score;
                    let max_sim = if let Some(Some(ref cand_emb)) = pool_embeddings.get(idx) {
                        selected_embeddings
                            .iter()
                            .map(|sel_emb| {
                                crate::consolidate::cosine_similarity(cand_emb, sel_emb) as f64
                            })
                            .fold(0.0f64, f64::max)
                    } else {
                        0.0
                    };

                    if max_sim > SIM_THRESHOLD {
                        continue;
                    }

                    let mmr = lambda * relevance - (1.0 - lambda) * max_sim;
                    if mmr > best_mmr {
                        best_mmr = mmr;
                        best_idx = Some(idx);
                    }
                }

                match best_idx {
                    Some(idx) => {
                        selected.push(idx);
                        if let Some(Some(ref emb)) = pool_embeddings.get(idx) {
                            selected_embeddings.push(emb);
                        }
                    }
                    None => break,
                }
            }

            let mut diverse_results = Vec::with_capacity(selected.len());
            for i in selected {
                diverse_results.push(scored[i].clone());
            }
            scored = diverse_results;
        } else {
            scored.truncate(top_k);
        }

        let sort_truncate_ms = t_sort.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 5: Hydrate final top_k with text + metadata ──
        let t_fetch = Instant::now();
        {
            let final_rids: Vec<&str> = scored.iter().map(|r| r.rid.as_str()).collect();
            let text_map = self.fetch_text_metadata_by_rids(&final_rids)?;
            // Chunk lookup BEFORE the mutable hydration loop — final_rids
            // borrows `scored`, and using it after `&mut scored` is the
            // E0502 that only clippy --all-features compiles (this whole
            // function is feature-gated; same trap as 2026-05-14).
            let chunked = self.rids_with_chunks(&final_rids);
            for result in &mut scored {
                if let Some(tm) = text_map.get(result.rid.as_str()) {
                    result.text = tm.text.clone();
                    result.metadata = serde_json::from_str(&tm.metadata)
                        .unwrap_or(serde_json::Value::Object(Default::default()));
                }
            }
            // Snippet spans (mirrors recall() Step 5.5).
            win_by_rid.retain(|rid, _| chunked.contains(rid));
            crate::engine::snippet::stamp_best_spans(
                &mut scored,
                &win_by_rid,
                query_text,
                self.snippet_window(),
            );
        }
        let fetch_ms = t_fetch.elapsed().as_secs_f64() * 1000.0;

        // v0.10 Item 3 seqlock recheck (sol r4/r5) — Acquire-fenced validate
        // before reinforcement, so a discarded (incoherent) result leaves no
        // spaced-repetition effect.
        if !self.correction_epoch_validate(epoch0) {
            return Ok(None);
        }

        // ── Phase 6: Reinforce ──
        // Best-effort, same as the unprofiled twin: a read that found its
        // results must return them even when the bookkeeping UPDATE fails.
        let t_reinforce = Instant::now();
        if !skip_reinforce {
            for r in &scored {
                if let Err(e) = self.reinforce(&r.rid) {
                    tracing::warn!(rid = %r.rid, error = %e, "reinforce failed; recall unaffected");
                }
            }
        }
        let reinforce_ms = t_reinforce.elapsed().as_secs_f64() * 1000.0;

        let total_ms = t_start.elapsed().as_secs_f64() * 1000.0;

        Ok(Some(RecallProfiledResult {
            results: scored,
            timings: RecallTimings {
                vec_search_ms,
                cache_score_ms: cache_score_ms + fallback_ms + fts_ms,
                fetch_ms,
                scoring_ms: 0.0,
                graph_ms,
                reinforce_ms,
                sort_truncate_ms,
                total_ms,
                candidate_count,
                graph_expansion_count,
            },
        }))
    }

    /// Fetch only text and metadata for a set of RIDs (post-scoring hydration).
    /// v0.10 Item 1 — which of `rids` are currently superseded, i.e.
    /// have a SELECTED active inbound `supersedes` edge (edges are
    /// stored NEW→OLD, so `target_rid` is the superseded record).
    /// One batched IN-clause query against `idx_record_links_target_sel`.
    pub(crate) fn superseded_rids_among(
        &self,
        rids: &[&str],
    ) -> Result<std::collections::HashSet<String>> {
        if rids.is_empty() {
            return Ok(std::collections::HashSet::new());
        }
        let placeholders: String = (0..rids.len())
            .map(|i| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(",");
        let sql = format!(
            "SELECT DISTINCT target_rid FROM record_links \
             WHERE link_type = 'supersedes' \
             AND status = 'active' AND selection_state = 'selected' \
             AND target_rid IN ({placeholders})"
        );
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        for r in rids {
            param_values.push(Box::new(r.to_string()));
        }
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params_ref.as_slice(), |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<std::collections::HashSet<_>, _>>()?;
        Ok(rows)
    }

    pub(crate) fn fetch_text_metadata_by_rids(
        &self,
        rids: &[&str],
    ) -> Result<HashMap<String, TextMetadataRow>> {
        if rids.is_empty() {
            return Ok(HashMap::new());
        }
        let placeholders: String = (0..rids.len())
            .map(|i| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(",");
        let sql =
            format!("SELECT rid, type, text, metadata FROM memories WHERE rid IN ({placeholders})");
        let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
        for r in rids {
            param_values.push(Box::new(r.to_string()));
        }
        let params_ref: Vec<&dyn rusqlite::types::ToSql> =
            param_values.iter().map(|p| p.as_ref()).collect();

        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params_ref.as_slice(), |row| {
                Ok((
                    row.get::<_, String>("rid")?,
                    row.get::<_, String>("text")?,
                    row.get::<_, String>("metadata")?,
                ))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        drop(stmt);
        drop(conn);

        let mut map = HashMap::new();
        for (rid, stored_text, stored_meta) in rows {
            let text = self.decrypt_text(&stored_text)?;
            let metadata = self.decrypt_text(&stored_meta)?;
            map.insert(
                rid.clone(),
                TextMetadataRow {
                    rid,
                    text,
                    metadata,
                },
            );
        }
        Ok(map)
    }

    /// Batch-fetch embeddings for a set of RIDs (for graph-only candidate scoring).
    ///
    /// **Issue #41 brainstorm-4 §5 — routes through the
    /// `DurableEmbeddingStore` module boundary.** Recall is the hot
    /// user-facing query path and is the file the §5 audit test
    /// guards; direct `SELECT ... embedding FROM memories` here
    /// would bypass the generation-stamp surface and silently
    /// return stale-vector-space bytes during a reembed. Going
    /// through the store gives every entry an
    /// `EmbeddingWithGeneration` tag — discarded here for the
    /// graph-only scoring path which tolerates lag, surfaced where
    /// callers need to discriminate.
    pub(crate) fn fetch_embeddings_by_rids(
        &self,
        rids: &[&str],
    ) -> Result<HashMap<String, Vec<u8>>> {
        let store = super::durable_embeddings::DurableEmbeddingStore::new(self);
        let stamped = store.read_embeddings_for_rids(rids)?;
        Ok(stamped
            .into_iter()
            .map(|(rid, entry)| (rid, entry.bytes))
            .collect())
    }

    /// #149 phase 2 — the ELIGIBLE UNIVERSE for a valid-time-bounded
    /// recall, fetched from the indexed v48 columns
    /// (`idx_memories_event_time`), BEFORE any relevance ranking runs.
    ///
    /// Semantics (the reviewer-confirmed contract, verbatim):
    /// - rows with NULL `event_time_min` are EXCLUDED whenever either
    ///   bound is set (unknown-when is not in-window);
    /// - inclusive interval overlap against `[event_time_min,
    ///   event_time_max]`: `after` alone ⇒ `max >= after`; `before`
    ///   alone ⇒ `min <= before`; both ⇒ the intervals overlap.
    ///   Boundary equality is included (>=/<=). A row whose max is NULL
    ///   is a point event at its min (`COALESCE(max, min)`).
    ///
    /// The result is consumed as an allow-SET passed into
    /// `passes_recall_filters` (membership, never a textual
    /// `RID IN (...)` list) plus the direct-scoring universe lane in
    /// `recall_inner` — never as a post-filter over a bounded
    /// similarity pool.
    fn event_time_eligible_rids(
        &self,
        namespace: Option<&str>,
        event_after: Option<f64>,
        event_before: Option<f64>,
    ) -> Result<std::collections::HashSet<String>> {
        let mut sql = String::from("SELECT rid FROM memories WHERE event_time_min IS NOT NULL");
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        if let Some(ns) = namespace {
            sql.push_str(&format!(" AND namespace = ?{}", params.len() + 1));
            params.push(Box::new(ns.to_string()));
        }
        if let Some(after) = event_after {
            sql.push_str(&format!(
                " AND COALESCE(event_time_max, event_time_min) >= ?{}",
                params.len() + 1
            ));
            params.push(Box::new(after));
        }
        if let Some(before) = event_before {
            sql.push_str(&format!(" AND event_time_min <= ?{}", params.len() + 1));
            params.push(Box::new(before));
        }
        let conn = self.read_conn();
        let mut stmt = conn.prepare(&sql)?;
        let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        let rids = stmt
            .query_map(param_refs.as_slice(), |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<std::collections::HashSet<String>, _>>()?;
        Ok(rids)
    }

    /// Reinforce a memory on access — increase half_life, update last_access,
    /// and increment access_count.
    fn reinforce(&self, rid: &str) -> Result<()> {
        let ts = now();

        // Read half_life from cache (eliminates SELECT query)
        let current_half_life = {
            let cache = self.scoring_cache.read();
            cache.get(rid).map(|r| r.half_life)
        };
        let new_half_life = match current_half_life {
            Some(hl) => (hl * 1.2_f64).min(31536000.0),
            None => 604800.0, // fallback if not in cache
        };

        {
            // The WRITE connection, deliberately. This is an UPDATE; running
            // it on a read-pool connection raced the writer for the WAL
            // write lock — a busy writer meant a 5s BUSY stall and then a
            // FAILED recall, from a bookkeeping side-effect (residual race
            // F3 in the 2026-08-15 catalog). The write mutex serializes
            // instead of erroring, and both call sites are additionally
            // best-effort now: reinforcement must never cost the caller
            // their results.
            let conn = self.conn();
            conn.execute(
                "UPDATE memories SET last_access = ?1, half_life = ?2, \
                 access_count = access_count + 1 WHERE rid = ?3",
                params![ts, new_half_life, rid],
            )?;
        } // drop conn before taking write lock on scoring_cache

        // Update cache with new values
        {
            let mut cache = self.scoring_cache.write();
            if let Some(row) = cache.get_mut(rid) {
                row.last_access = ts;
                row.half_life = new_half_life;
                row.access_count += 1;
            }
        }

        self.log_op(
            "reinforce",
            Some(rid),
            &serde_json::json!({
                "rid": rid,
                "last_access": ts,
                "half_life": new_half_life,
                "local_only": true,
            }),
            None,
        )?;

        Ok(())
    }
}

#[cfg(test)]
mod synthesis_representation_tests {
    use super::*;

    #[test]
    fn event_ordering_query_selects_atomic_contributed_view() {
        assert_eq!(
            synthesis_representation_intent(Some(
                "Can you list the order in which I brought up these project aspects?"
            )),
            Some(SynthesisRepresentationIntent {
                axis: Some("contributed"),
                granularity: Some(SynthesisGranularityIntent::Atomic),
            })
        );
    }

    #[test]
    fn summary_and_question_queries_select_their_distinct_views() {
        assert_eq!(
            synthesis_representation_intent(Some("Summarize what I asked about the project")),
            Some(SynthesisRepresentationIntent {
                axis: Some("asked"),
                granularity: Some(SynthesisGranularityIntent::Rollup),
            })
        );
        assert_eq!(
            synthesis_representation_intent(Some("Who told me about the launch?")),
            Some(SynthesisRepresentationIntent {
                axis: Some("who_said"),
                granularity: None,
            })
        );
        assert_eq!(
            synthesis_representation_intent(Some("Give me an overview of my decisions")),
            Some(SynthesisRepresentationIntent {
                axis: None,
                granularity: Some(SynthesisGranularityIntent::Rollup),
            })
        );
    }

    #[test]
    fn mixed_list_and_summary_request_is_a_neutral_granularity_conflict() {
        assert_eq!(
            synthesis_representation_intent(Some("Summarize these items as a list")),
            Some(SynthesisRepresentationIntent {
                axis: None,
                granularity: Some(SynthesisGranularityIntent::Conflict),
            })
        );
    }

    #[test]
    fn classifier_uses_boundaries_and_leaves_neutral_queries_alone() {
        assert_eq!(
            synthesis_representation_intent(Some("Show project status")),
            None
        );
        assert_eq!(
            synthesis_representation_intent(Some("The reorder buffer")),
            None
        );
        assert_eq!(synthesis_representation_intent(Some("basket status")), None);
        assert_eq!(
            synthesis_representation_intent(Some("What details did I share?")),
            None
        );
        assert_eq!(
            synthesis_representation_intent(Some("Which examples were relevant?")),
            None
        );
        assert_eq!(
            synthesis_representation_intent(Some("What steps did I take?")),
            None
        );
        assert_eq!(synthesis_representation_intent(None), None);
    }
}

#[cfg(test)]
mod novelty_selection_tests {
    use super::*;

    fn r(rid: &str, score: f64, text: &str) -> crate::types::RecallResult {
        crate::types::RecallResult {
            rid: rid.into(),
            memory_type: "fact".into(),
            text: text.into(),
            created_at: 0.0,
            importance: 0.5,
            valence: 0.0,
            score,
            // Built explicitly rather than via a derived Default: a default
            // `valence_multiplier` of 0.0 would silently annihilate scores,
            // so ScoreBreakdown deliberately has no Default to reach for.
            scores: crate::types::ScoreBreakdown {
                similarity: score,
                decay: 1.0,
                recency: 1.0,
                importance: 0.5,
                graph_proximity: 0.0,
                contributions: crate::types::ScoreContributions {
                    similarity: score,
                    decay: 0.0,
                    recency: 0.0,
                    importance: 0.0,
                    graph_proximity: 0.0,
                },
                valence_multiplier: 1.0,
            },
            why_retrieved: vec![],
            metadata: serde_json::Value::Null,
            namespace: "default".into(),
            certainty: 1.0,
            domain: String::new(),
            source: String::new(),
            emotional_state: None,
            current_status: Default::default(),
            superseded_by: None,
            disputed_with: vec![],
            aged_last_verified: None,
            best_span: None,
        }
    }

    /// Three near-duplicates outscore the one chunk covering the other half
    /// of the story. This is the event_ordering failure in miniature.
    fn redundant_pool() -> Vec<crate::types::RecallResult> {
        vec![
            r("a1", 0.90, "deployment pipeline rollout staging cluster"),
            r(
                "a2",
                0.88,
                "deployment pipeline rollout staging cluster again",
            ),
            r(
                "a3",
                0.86,
                "deployment pipeline rollout staging cluster once more",
            ),
            r(
                "b1",
                0.60,
                "invoice reconciliation quarterly finance ledger",
            ),
        ]
    }

    /// THE CONNECTIVITY ASSERTION. Four parameters once reported "inert"
    /// across a whole sweep because they were never wired to this file, and
    /// the sweep could not tell an ineffective knob from a disconnected one.
    /// A value that SHOULD change the output must change it, or the feature
    /// is not shipped.
    #[test]
    fn full_novelty_weight_changes_the_selection() {
        let mut v = redundant_pool();
        apply_novelty_selection_w(&mut v, 2, 1.0);
        assert_eq!(
            v[1].rid,
            "b1",
            "pure set-cover must take the uncovered topic second, got {:?}",
            v.iter().map(|x| x.rid.as_str()).collect::<Vec<_>>()
        );
    }

    /// The default path must be byte-identical to today's behaviour, so
    /// shipping this OFF is genuinely a no-op.
    #[test]
    fn zero_weight_is_exactly_score_order() {
        let mut v = redundant_pool();
        let before: Vec<String> = v.iter().map(|x| x.rid.clone()).collect();
        apply_novelty_selection_w(&mut v, 2, 0.0);
        let after: Vec<String> = v.iter().map(|x| x.rid.clone()).collect();
        assert_eq!(before, after);
    }

    /// Selection REORDERS, it never drops: the unpicked tail is retained in
    /// its original relative order. A selection stage that silently shrinks
    /// the result set would break every caller that pages through recall.
    #[test]
    fn nothing_is_lost_and_the_tail_keeps_its_order() {
        let mut v = redundant_pool();
        apply_novelty_selection_w(&mut v, 2, 1.0);
        assert_eq!(v.len(), 4);
        let mut rids: Vec<&str> = v.iter().map(|x| x.rid.as_str()).collect();
        rids.sort_unstable();
        assert_eq!(rids, vec!["a1", "a2", "a3", "b1"]);
        let tail: Vec<&str> = v[2..].iter().map(|x| x.rid.as_str()).collect();
        let mut sorted_tail = tail.clone();
        sorted_tail.sort_unstable();
        assert_eq!(tail, sorted_tail, "tail must stay in original score order");
    }

    /// REGRESSION, and the reason this feature shipped inert once.
    ///
    /// Candidates carry `text: String::new()` until step 5 hydrates the
    /// final top_k. Run against un-hydrated candidates, every novelty score
    /// is 0, every value ties, and the greedy reproduces score order exactly
    /// — a weight that changes nothing while looking wired. The other tests
    /// in this module could not catch it because they all build results WITH
    /// text. This one pins the degenerate behaviour so that the caller's
    /// obligation to hydrate first is visible from the test file.
    #[test]
    fn empty_text_degenerates_to_score_order_so_callers_must_hydrate_first() {
        let mut v = redundant_pool();
        for x in v.iter_mut() {
            x.text = String::new();
        }
        let before: Vec<String> = v.iter().map(|x| x.rid.clone()).collect();
        apply_novelty_selection_w(&mut v, 2, 1.0);
        let after: Vec<String> = v.iter().map(|x| x.rid.clone()).collect();
        assert_eq!(
            before, after,
            "un-hydrated candidates must be a documented no-op, not a silent \
             partial effect — the caller hydrates the pool before selecting"
        );
    }

    /// The first pick is always the top-scoring record regardless of weight:
    /// nothing has been covered yet, so every candidate's novelty is 1.0 and
    /// score breaks the tie. Guards against a diversity policy that answers
    /// a specific question with an irrelevant-but-unusual record.
    #[test]
    fn best_scoring_record_still_leads() {
        for w in [0.25, 0.5, 0.9, 1.0] {
            let mut v = redundant_pool();
            apply_novelty_selection_w(&mut v, 3, w);
            assert_eq!(v[0].rid, "a1", "w={w}");
        }
    }
}
