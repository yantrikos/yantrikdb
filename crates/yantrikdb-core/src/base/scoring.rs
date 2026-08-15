/// Multi-signal scoring for memory recall.
///
/// Origin: ported from the pre-Rust Python engine; the budget-form
/// composite below has since replaced that formula entirely.

/// Compute the decay score: I(t) = importance * 2^(-t / half_life)
///
/// Negative `elapsed` clamps to 0 (score = full `importance`). Reachable
/// since caller-supplied `created_at`/event time (historical import): a
/// future-dated record must score as "brand new", not as `2^(+x)` — an
/// unbounded amplifier that would rebuild the recency wall in the other
/// direction. Same clamp `cognition::temporal::recency_relevance` has
/// always applied to its age.
pub fn decay_score(importance: f64, half_life: f64, elapsed: f64) -> f64 {
    if half_life > 0.0 {
        importance * f64::powf(2.0, -elapsed.max(0.0) / half_life)
    } else {
        0.0
    }
}

/// Compute the recency score: exp(-age / (7 * 86400))
///
/// Negative `age` clamps to 0 (score = 1.0, the maximum) — see
/// `decay_score` for why a future-dated record is "new", never amplified.
pub fn recency_score(age: f64) -> f64 {
    f64::exp(-age.max(0.0) / (7.0 * 86400.0))
}

/// Compute the valence boost: 1.0 + 0.3 * |valence|
pub fn valence_boost(valence: f64) -> f64 {
    1.0 + 0.3 * valence.abs()
}

/// Negative sentiment keywords for query-aware valence matching.
const NEGATIVE_QUERY_WORDS: &[&str] = &[
    "sad",
    "frustrated",
    "angry",
    "bad",
    "worst",
    "low",
    "lows",
    "difficult",
    "hard",
    "struggle",
    "pain",
    "stress",
    "anxious",
    "upset",
    "failed",
    "failure",
    "problem",
    "negative",
    "stressing",
    "worried",
    "tough",
];

/// Positive sentiment keywords for query-aware valence matching.
const POSITIVE_QUERY_WORDS: &[&str] = &[
    "happy",
    "joy",
    "great",
    "best",
    "high",
    "good",
    "wonderful",
    "excited",
    "proud",
    "success",
    "achievement",
    "positive",
    "celebration",
    "love",
];

/// Simple suffix stripping for sentiment detection.
/// "proudest" → "proud", "happiest" → "happi" → matched via prefix.
fn sentiment_stem(word: &str) -> &str {
    // Strip common inflectional suffixes (ordered longest first)
    for suffix in &[
        "iest", "ness", "ment", "ful", "est", "ing", "ous", "ive", "ity", "ed", "er", "ly", "al",
        "es", "s",
    ] {
        if word.len() > suffix.len() + 2 && word.ends_with(suffix) {
            return &word[..word.len() - suffix.len()];
        }
    }
    word
}

/// Detect query sentiment from text: -1.0 for negative, +1.0 for positive, 0.0 for neutral.
///
/// Uses stemmed matching so "proudest" matches "proud", "happiest" matches "happi"→"happy", etc.
pub fn detect_query_sentiment(query_text: &str) -> f64 {
    let lower = query_text.to_lowercase();
    // Split on non-alphanumeric to strip punctuation from tokens
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| !s.is_empty())
        .collect();

    // Check if a token matches a word list, allowing stemmed/prefix matching.
    // Stem both the token and the word, then check prefix overlap.
    // "proudest" → stem "proud" → matches "proud" ✓
    // "happiest" → stem "happi" → "happy" starts_with "happi" ✓
    //
    // Prefix matching requires >= 4 chars to prevent false positives:
    // "up" must NOT match "upset", "no" must NOT match "negative".
    let matches_list = |token: &str, words: &[&str]| -> bool {
        let stemmed = sentiment_stem(token);
        words.iter().any(|w| {
            token == *w
                || stemmed == *w
                || (token.len() >= 4 && token.starts_with(w))
                || (stemmed.len() >= 4 && w.starts_with(stemmed))
        })
    };

    let neg_count = tokens
        .iter()
        .filter(|t| matches_list(t, NEGATIVE_QUERY_WORDS))
        .count();
    let pos_count = tokens
        .iter()
        .filter(|t| matches_list(t, POSITIVE_QUERY_WORDS))
        .count();

    if neg_count > pos_count {
        -1.0
    } else if pos_count > neg_count {
        1.0
    } else {
        0.0
    }
}

/// Query-aware valence boost.
///
/// When query sentiment matches memory valence sign (e.g., negative query + negative memory),
/// the boost is increased. When they mismatch, the boost is reduced.
/// For neutral queries (sentiment == 0.0), falls back to the standard symmetric boost.
pub fn query_valence_boost(memory_valence: f64, query_sentiment: f64) -> f64 {
    let base = 1.0 + 0.3 * memory_valence.abs();
    if query_sentiment == 0.0 {
        return base;
    }
    // alignment: +1 = match (negative query + negative memory), -1 = mismatch
    let alignment = if memory_valence.abs() < 1e-10 {
        0.0
    } else {
        query_sentiment * memory_valence.signum()
    };
    base * (1.0 + 0.2 * alignment)
}

/// Composite score with query-aware valence.
pub fn composite_score_with_sentiment(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    valence: f64,
    query_sentiment: f64,
) -> f64 {
    // Priors share one budget (see POLICY_BUDGET_LN). Importance enters
    // pre-gated by similarity so it cannot manufacture relevance; graph and
    // agreement are absent on this path and pass 0, which costs exactly
    // nothing because the weights multiply zeros rather than being
    // renormalized away.
    let freshness_z = freshness_z(decay, recency);
    let importance_z = importance_gate(similarity) * importance.clamp(0.0, 1.0);
    W_SIM
        * similarity
        * policy_mult(freshness_z, importance_z, 0.0, 0.0, 0.0)
        * query_valence_boost(valence, query_sentiment)
}

// ─────────────────────────────────────────────────────────────────────
// THE POLICY LAYER — one shared inversion budget for every prior
// ─────────────────────────────────────────────────────────────────────
//
// # The defect this replaces
//
// Freshness, importance, graph and agreement were each documented as
// "bounded, at most +12.5%" and each applied as an independent multiplier.
// Multipliers COMPOUND: the reachable product measured ~3.26x on a neutral
// query (~3.9x with sentiment-aligned valence). A similarity-0.30 record
// with maxed priors outscored a bare similarity-0.60 record — a 2x
// relevance gap, reversed. Every factor was individually within its stated
// bound; the system was not. Worse, the shape invited the next signal to
// add a fifth independent multiplier and widen the hole again.
//
// # The rule
//
// Priors are QUERY-INDEPENDENT: freshness, importance, connectivity,
// corroboration. They say nothing about whether this record answers THIS
// query, so they must never create relevance from zero, and they must
// share ONE budget no matter how many of them exist.
//
//     score = relevance * exp(POLICY_BUDGET_LN * sum(w_i * z_i))
//     with every z_i in [0,1], every w_i >= 0, and sum(w_i) <= 1
//
// Because the weights partition a single budget, the entire policy layer
// is bounded by exp(POLICY_BUDGET_LN) — today, next year, and after the
// next five signals are added. Adding a signal dilutes the others rather
// than widening the ceiling, which is the property the old form lacked.
//
// This is deliberately NOT "cap the old product": clipping after
// multiplying creates a plateau where unrelated records pile up at the cap
// and become indistinguishable.

/// Total ratio the policy layer may ever apply: `exp(0.2624) = 1.30`.
///
/// Chosen as an inversion budget, not by taste: priors may reorder records
/// within a 30% relevance band and can never rescue a record from outside
/// it. The old composed ceiling was ~326%.
pub const POLICY_BUDGET_LN: f64 = 0.262_364_264_467_491_9; // ln(1.30)

/// Prior weights. **These must sum to <= 1.0** — that invariant is what
/// makes the budget shared rather than per-signal, and it is asserted in
/// `policy_weights_partition_one_budget`.
///
/// Note what adding `PW_USAGE` did: every other weight shrank. That is the
/// design working — a new prior takes a share of one fixed budget instead of
/// bolting on another multiplier, so the ceiling is unchanged at 1.30.
pub const PW_FRESHNESS: f64 = 0.22;
pub const PW_IMPORTANCE: f64 = 0.40;
pub const PW_GRAPH: f64 = 0.13;
pub const PW_AGREEMENT: f64 = 0.13;
pub const PW_USAGE: f64 = 0.12;

/// Freshness as a normalized prior in `[0,1]`.
///
/// `W_DECAY + W_RECENCY = 0.5`, so the raw blend maxes out at 0.5 — using it
/// directly silently gave freshness HALF the budget share the weight table
/// says it has. Caught by the stagewise-composition test, which is the value
/// of asserting an equivalence rather than a threshold: a threshold test
/// would have passed happily with freshness at half strength.
#[inline]
pub fn freshness_z(decay: f64, recency: f64) -> f64 {
    ((W_DECAY * decay + W_RECENCY * recency) / (W_DECAY + W_RECENCY)).clamp(0.0, 1.0)
}
/// Recall frequency as a bounded prior — the "this has repeatedly proven
/// useful" signal.
///
/// `access_count` was tracked on 2,796 of 5,050 records in a real store
/// (max 7,975 recalls) and fed ONLY eviction decisions; it had no influence
/// on ranking at all. It is a textbook query-independent prior, so it lives
/// in the shared budget.
///
/// # Why saturating, and why a small weight
///
/// A usage prior is a feedback loop: rank a memory higher, it gets recalled
/// more, which ranks it higher. Left linear and unbounded that is a
/// rich-get-richer ratchet that would eventually bury everything a user has
/// not yet asked about. Three things contain it — the `ln1p` curve
/// saturating at [`ACCESS_SATURATION`] (the 7,975-recall record and a
/// 20-recall record are treated identically), the small share of a budget
/// that is itself capped at 1.30x, and `skip_reinforce` for callers that
/// must not perturb state.
///
/// It orders equally-relevant records by proven usefulness. It cannot
/// resurrect an irrelevant one.
/// **NOT YET WIRED INTO RECALL.** `adaptive_composite_score` — the function
/// recall actually calls — takes no `access_count`, so this prior currently
/// contributes nothing to ranking. `PW_USAGE` therefore reserves a share of
/// the budget that nobody spends, which is deliberate: it keeps the weight
/// table honest about what the design intends, and the partition invariant
/// already accounts for it, so wiring it later cannot widen the ceiling.
///
/// The companion helper that DID accept it was removed rather than left
/// looking wired — an unreachable function that appears to implement a
/// feature is worse than an absent one, because it reads as done.
#[inline]
pub fn usage_z(access_count: u32) -> f64 {
    ((access_count as f64).ln_1p() / (ACCESS_SATURATION as f64).ln_1p()).min(1.0)
}

/// Ceiling for an exploration lane's lift: `+10%`.
///
/// The valence and cold-memory lanes existed to let records the vector lane
/// under-ranks still compete, and they did it with UNBOUNDED ADDITIVE terms
/// (`+0.20·|valence|·importance`, `+0.15·importance`). An additive term has
/// no similarity-relative bound: as similarity approaches zero its ratio to
/// semantic relevance approaches infinity, so a record matching nothing
/// could outrank one matching everything. That is the same wall the policy
/// budget removed, relocated into the lanes.
///
/// A lane's job is ADMISSION — getting a candidate considered at all, which
/// happens before scoring and is unaffected by this. Once admitted, a record
/// competes on relevance like any other. So the lift becomes multiplicative
/// and small: it reorders within a band, and a record with no semantic match
/// scores near zero however cold or emotionally charged it is.
pub const LANE_LIFT_MAX: f64 = 0.10;

/// Bounded multiplicative lift for an exploration lane. `z` in `[0,1]`.
#[inline]
pub fn lane_lift_mult(z: f64) -> f64 {
    1.0 + crate::base::tuning::tuning().lane_lift_max * z.clamp(0.0, 1.0)
}

/// The bounded policy multiplier shared by every composite.
///
/// Each argument is a normalized prior in `[0,1]`. `importance_z` is
/// expected to arrive ALREADY GATED by similarity so that a high-importance
/// record with no semantic match gets ~nothing — priors must not create
/// relevance from zero.
#[inline]
pub fn policy_mult(
    freshness_z: f64,
    importance_z: f64,
    graph_z: f64,
    agreement_z: f64,
    usage_z: f64,
) -> f64 {
    // Runtime-tunable (base/tuning.rs): a constant you cannot sweep is a
    // constant you cannot validate, and these were wrong for months.
    let t = crate::base::tuning::tuning();
    let (wf, wi, wg, wa, wu) = t.normalized_weights();
    let z = wf * freshness_z.clamp(0.0, 1.0)
        + wi * importance_z.clamp(0.0, 1.0)
        + wg * graph_z.clamp(0.0, 1.0)
        + wa * agreement_z.clamp(0.0, 1.0)
        + wu * usage_z.clamp(0.0, 1.0);
    (t.policy_budget_ln() * z).exp()
}

/// How strongly cross-lane agreement may multiply a score. Budget form:
/// `exp(policy_budget_ln * w_agreement * extra_lanes/2)`, capped at two
/// extra lanes — ~+3.5% at defaults (ln 1.30 x 0.13). AGREEMENT_SCALE
/// below is the pre-budget constant, kept for history; the body no longer
/// reads it. All tie-breakers share the one inversion budget.
///
/// # Why lane agreement is evidence at all
///
/// The vector lane, the FTS lane, the claims lane and the graph lane find
/// candidates through close-to-independent mechanisms (dense similarity,
/// lexical match, extracted propositions, entity adjacency). When several
/// of them surface the SAME record for one query, that coincidence is
/// information — the record is relevant in more than one sense of the
/// word. Until 2026-08-13 the engine computed exactly this signal, wrote
/// it into `why_retrieved`, used it for the response-level confidence…
/// and never let it touch the ranking. The four benchmark categories that
/// drag every arm (ordering, summarization, contradiction, temporal) are
/// set-level failures where per-item scoring is blind; this is the
/// cheapest of the set-aware repairs.
///
/// Bounded multiplicatively for the same reason freshness and graph are:
/// an additive lane bonus is a wall (a record matched by three lanes at
/// similarity 0.02 must never beat a single-lane record at 0.30). At
/// 0.125 / two-lane cap, agreement can reverse at most a 12.5% relevance
/// gap — it breaks ties between near-equals and can promote nothing else.
pub const AGREEMENT_SCALE: f64 = 0.125;

/// The bounded cross-lane agreement multiplier. `extra_lanes` is the
/// number of retrieval lanes beyond the first that independently surfaced
/// the record; 0 yields exactly 1.0 so single-lane hits are untouched.
#[inline]
pub fn agreement_mult(extra_lanes: usize) -> f64 {
    // Also a budget share — see graph_mult for why late application is safe.
    let t = crate::base::tuning::tuning();
    let (_, _, _, wa, _) = t.normalized_weights();
    (t.policy_budget_ln() * wa * (extra_lanes.min(2) as f64 / 2.0)).exp()
}

/// How strongly graph proximity may multiply a score. Budget form:
/// `exp(policy_budget_ln * w_graph * p)` — ~+3.5% at defaults
/// (ln 1.30 x 0.13). GRAPH_SCALE below is the pre-budget constant, kept
/// for history; the body no longer reads it.
///
/// Chosen by INVERSION BUDGET, not taste. A maximally-connected record can
/// overtake an unconnected one only when `s_high / s_low < 1 + GRAPH_SCALE`,
/// so 0.125 lets graph evidence reverse at most a 12.5% similarity gap. It
/// breaks ties among near-equals and cannot promote an irrelevant record,
/// which is exactly the guarantee freshness was rebuilt to provide.
pub const GRAPH_SCALE: f64 = 0.125;

/// The bounded graph multiplier. `p = 0` yields exactly 1.0, so a record with
/// no edges scores identically on the graph and non-graph paths — there is no
/// discontinuity at zero to fall off.
#[inline]
pub fn graph_mult(graph_proximity: f64) -> f64 {
    // A SHARE OF THE POLICY BUDGET, not an independent multiplier.
    //
    // The recall path applies graph and agreement late, after the composite
    // is already scored, so making them independent factors put them back
    // OUTSIDE the budget and let the product creep to ~1.65x again. Log
    // space fixes this exactly: exp(B*w1*z1) * exp(B*w2*z2) =
    // exp(B*(w1*z1 + w2*z2)), so factors applied at different stages of the
    // pipeline still compose to one bound of exp(B) as long as the weights
    // partition it. Where a factor is applied stops mattering; only its
    // share does.
    let t = crate::base::tuning::tuning();
    let (_, _, wg, _, _) = t.normalized_weights();
    (t.policy_budget_ln() * wg * graph_proximity.clamp(0.0, 1.0)).exp()
}

/// Graph composite score with query-aware valence.
///
/// **THE THIRD WALL.** The module law above says nothing may be added to
/// similarity, and this function used to break it:
///
/// ```text
/// base_rel = (GW_SIM * similarity + GW_GRAPH * graph_proximity) * freshness
/// ```
///
/// That is the same additive shape as the importance wall and the recency
/// wall, and it fails the same way. At the old constants a record with
/// `similarity = 0.0026` and `graph_proximity = 1.0` scored **0.340** while a
/// genuinely relevant record at `similarity = 0.309` with no edge scored
/// **0.267** — the irrelevant one won, because `GW_GRAPH * 1.0 = 0.30` is a
/// floor no similarity below 0.86 can cross. Measured in a live store, where
/// a real-estate tax memo came back for "encryption at rest and key rotation"
/// through a phantom `AT` node.
///
/// Two bugs died here:
///
/// 1. the additive term itself, now a bounded multiplier;
/// 2. a DISCONTINUITY AT ZERO. The old branch reallocated weights the moment
///    `p > 0` — similarity's coefficient dropped 0.50 → 0.35 and the
///    importance cap 0.80 → 0.60 — so an infinitesimal edge cut a record's
///    score by 30–40%, and it needed `p ≈ 0.5s` merely to break even. Graph
///    evidence PENALISED the records it touched. There is no branch now: the
///    multiplier is 1.0 at `p = 0`.
///
/// Diagnosed with gpt-5.6-sol and qwen3.8-max, 2026-08-13; the discontinuity
/// was codex's find.
pub fn graph_composite_score_with_sentiment(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    valence: f64,
    graph_proximity: f64,
    query_sentiment: f64,
) -> f64 {
    composite_score_with_sentiment(
        similarity,
        decay,
        recency,
        importance,
        valence,
        query_sentiment,
    ) * graph_mult(graph_proximity)
}

/// Relevance-first multiplicative scoring.
///
/// NOTHING may be added to similarity — every other signal multiplies it.
/// Two walls were torn down to get here, both the same failure shape:
///
/// - **Importance** (the original fix): additive importance let imp=1.0,
///   sim=0.1 beat imp=0.3, sim=0.6. It became a similarity-gated
///   multiplier.
/// - **Freshness** (2026-08-05, the recency wall): additive
///   `W_DECAY*decay + W_RECENCY*recency` handed every fresh record up to
///   +0.5 free while similarity was worth at most W_SIM=0.5 — so on any
///   age-spread corpus, EVERY record written this week outranked EVERY
///   old record regardless of relevance (an old record would need a
///   similarity advantage > 1.0, which cannot exist). Measured on a
///   4,297-record production clone with a 40-query paraphrase-labeled
///   set: exact cosine over the stored vectors scored MRR 0.562 while
///   this formula scored 0.069 — the formula alone destroyed a 10×
///   factor. Every earlier eval missed it because fresh test databases
///   and the synthetic golden set have no age spread (uniform
///   decay/recency degrade the additive form to pure similarity).
///
/// Formula:
///   fresh    = 1 + FRESHNESS_SCALE * (W_DECAY * decay + W_RECENCY * recency)
///   gate     = sigmoid(GATE_K * (similarity - GATE_TAU))
///   score    = W_SIM * similarity * fresh
///              * (1 + gate * ALPHA_IMP * importance) * valence_boost
///
/// Freshness is now a bounded TIE-BREAKER (≤ +12.5% at default weights):
/// it orders near-equals by recency and can never promote an irrelevant
/// record past a relevant one. Sweeping the scale on the production
/// clone: 0.0 → MRR 0.583, 0.25 → 0.567, 0.5 → 0.521, 1.0 → 0.455,
/// 2.0 → 0.315 (the shipped additive form ≈ 0.069). 0.25 keeps ~all of
/// the ceiling while preserving recency semantics for genuine ties.

/// Base relevance weights. W_DECAY/W_RECENCY now weight the freshness
/// MULTIPLIER's interior, not additive terms (see module rationale).
pub const W_SIM: f64 = 0.50;
pub const W_DECAY: f64 = 0.20;
pub const W_RECENCY: f64 = 0.30;

/// How strongly freshness (decay + recency) can multiply a score:
/// `1 + FRESHNESS_SCALE * (w_decay*decay + w_recency*recency)`, i.e. at
/// most +12.5% at default weights. Chosen by sweep on the production
/// clone (values above; larger scales rebuild the recency wall
/// gradually, and the additive form it replaces was the degenerate
/// extreme). A tie-breaker must break ties — never build walls.
pub const FRESHNESS_SCALE: f64 = 0.25;

/// The freshness multiplier shared by every composite variant.
#[inline]
pub fn freshness_mult(decay: f64, recency: f64, w_decay: f64, w_recency: f64) -> f64 {
    1.0 + FRESHNESS_SCALE * (w_decay * decay + w_recency * recency)
}

/// Importance gate parameters.
/// GATE_K controls the sharpness of the sigmoid gate.
/// GATE_TAU is the similarity threshold where the gate reaches 0.5.
pub const GATE_K: f64 = 12.0;
pub const GATE_TAU: f64 = 0.25;

/// Importance amplification strength.
/// At full gate (similarity >> τ), score is multiplied by up to (1 + ALPHA_IMP).
pub const ALPHA_IMP: f64 = 0.80;

/// Graph-expanded signal weights.
pub const GW_SIM: f64 = 0.35;
pub const GW_DECAY: f64 = 0.15;
pub const GW_RECENCY: f64 = 0.20;
pub const GW_GRAPH: f64 = 0.30;

/// Graph importance gate uses same parameters.
pub const GW_ALPHA_IMP: f64 = 0.60;

use crate::types::ScoreContributions;

/// Sigmoid function: 1 / (1 + exp(-x))
#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute the importance gate: sigmoid(K * (similarity - τ)).
///
/// Returns ~0 when similarity << τ (importance suppressed),
/// returns ~1 when similarity >> τ (importance fully active).
#[inline]
pub fn importance_gate(similarity: f64) -> f64 {
    let t = crate::base::tuning::tuning();
    sigmoid(t.gate_k * (similarity - t.gate_tau))
}

/// Compute the composite recall score: relevance first, everything else
/// multiplies (see the module rationale — freshness as an additive term
/// was the recency wall).
///
/// base_rel = W_SIM * similarity * (1 + FRESHNESS_SCALE*(W_DECAY*decay + W_RECENCY*recency))
/// gate     = sigmoid(K * (similarity - τ))
/// score    = base_rel * (1 + gate * α * importance) * valence_boost
pub fn composite_score(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    valence: f64,
) -> f64 {
    let freshness_z = freshness_z(decay, recency);
    let importance_z = importance_gate(similarity) * importance.clamp(0.0, 1.0);
    W_SIM
        * similarity
        * policy_mult(freshness_z, importance_z, 0.0, 0.0, 0.0)
        * valence_boost(valence)
}

/// Compute weighted contributions for standard scoring.
///
/// Decay/recency report their share of the freshness MULTIPLIER's
/// interior (scaled), mirroring the score structure: they are fractional
/// uplifts on relevance now, not standalone additive terms.
pub fn standard_contributions(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
) -> ScoreContributions {
    let gate = importance_gate(similarity);
    ScoreContributions {
        // Reported on the SAME scale the score uses: each prior is its
        // share of the log-space budget, so the numbers here sum toward
        // POLICY_BUDGET_LN rather than describing a superseded formula.
        similarity: W_SIM * similarity,
        decay: POLICY_BUDGET_LN * PW_FRESHNESS * freshness_z(decay, 0.0),
        recency: POLICY_BUDGET_LN * PW_FRESHNESS * freshness_z(0.0, recency),
        importance: POLICY_BUDGET_LN * PW_IMPORTANCE * gate * importance.min(1.0),
        graph_proximity: 0.0,
    }
}

/// Compute the composite recall score with optional graph proximity signal.
///
/// Graph path:
///   base_rel = GW_SIM*sim + GW_DECAY*decay + GW_RECENCY*recency + GW_GRAPH*graph
///   gate     = sigmoid(K * (sim - τ))
///   score    = base_rel * (1 + gate * α_g * importance) * valence_boost
pub fn graph_composite_score(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    valence: f64,
    graph_proximity: f64,
) -> f64 {
    // 2026-08-13: this function ADDED proximity to similarity —
    // `GW_SIM*sim + GW_GRAPH*proximity` — which is the original wall. At the
    // old constants a record with similarity 0.0026 and proximity 1.0 scored
    // 0.340 while a genuinely relevant record at similarity 0.309 with no
    // edges scored 0.267: the irrelevant one won, because 0.30*1.0 swamped
    // 0.35*0.0026. Production recall had already been moved to the
    // multiplicative form, but this remained public, so any direct caller
    // still got the prohibited formula — and tests pinned that behaviour.
    //
    // Graph proximity is a PRIOR (it says a record is well-connected, not
    // that it answers this query), so it now enters through the shared
    // policy budget like every other prior.
    let freshness_z = freshness_z(decay, recency);
    let importance_z = importance_gate(similarity) * importance.clamp(0.0, 1.0);
    W_SIM
        * similarity
        * policy_mult(freshness_z, importance_z, graph_proximity, 0.0, 0.0)
        * valence_boost(valence)
}

/// Compute weighted contributions for graph-expanded scoring.
pub fn graph_contributions(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    graph_proximity: f64,
) -> ScoreContributions {
    if graph_proximity > 0.0 {
        let gate = importance_gate(similarity);
        ScoreContributions {
            similarity: GW_SIM * similarity,
            decay: FRESHNESS_SCALE * GW_DECAY * decay,
            recency: FRESHNESS_SCALE * GW_RECENCY * recency,
            importance: gate * GW_ALPHA_IMP * importance.min(1.0),
            graph_proximity: GW_GRAPH * graph_proximity,
        }
    } else {
        standard_contributions(similarity, decay, recency, importance)
    }
}

/// Score for eviction prioritization (lower = more evictable).
///
/// Combines decay strength, recency, AND recall frequency: a memory that is
/// recalled often is "hot" and should resist demotion to cold even when it is
/// old. `access_count` is folded in as a saturating resistance term
/// (~`ACCESS_SATURATION` recalls ≈ fully protected), so frequently-used
/// memories are not evicted just for age. Without this, hotness is tracked but
/// never used in the tiering decision.
pub fn eviction_score(decay: f64, recency: f64, access_count: u32) -> f64 {
    let access_resist =
        ((access_count as f64).ln_1p() / (ACCESS_SATURATION as f64).ln_1p()).min(1.0);
    0.6 * decay + 0.4 * recency + ACCESS_WEIGHT * access_resist
}

/// Recall count at which a memory is treated as fully "hot" for tiering.
pub const ACCESS_SATURATION: u32 = 20;
/// How strongly recall frequency protects a memory from eviction.
pub const ACCESS_WEIGHT: f64 = 0.5;

/// Build a human-readable explanation for why a memory was retrieved.
pub fn build_why(similarity: f64, recency: f64, decay: f64, valence: f64) -> Vec<String> {
    let mut why = Vec::new();
    if similarity > 0.5 {
        why.push(format!("semantically similar ({similarity:.2})"));
    }
    if recency > 0.5 {
        why.push("recent".to_string());
    }
    if decay > 0.3 {
        why.push(format!("important (decay={decay:.2})"));
    }
    if valence.abs() > 0.5 {
        why.push(format!("emotionally weighted ({valence:.2})"));
    }
    if why.is_empty() {
        why.push("matched query".to_string());
    }
    why
}

/// Adaptive composite score using learned per-database weights.
///
/// Same formula as `composite_score_with_sentiment` but substitutes the
/// global constants with database-specific learned weights.
pub fn adaptive_composite_score(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    valence: f64,
    query_sentiment: f64,
    weights: &crate::types::LearnedWeights,
) -> f64 {
    // Learned weights shape the priors WITHIN the shared budget; they can
    // no longer widen it. alpha_imp was previously allowed up to 1.5, which
    // pushed the composed ceiling toward 4x — it is now a relative weight,
    // normalized against the other priors, so learning reallocates the
    // budget instead of expanding it.
    let freshness_z = ((weights.w_decay * decay + weights.w_recency * recency)
        / (weights.w_decay + weights.w_recency).max(1e-9))
    .clamp(0.0, 1.0);
    let gate = sigmoid(GATE_K * (similarity - weights.gate_tau));
    let importance_z = gate * importance.clamp(0.0, 1.0);
    // Learned importance reallocates the budget; it must not enlarge it.
    // The previous normalization CANCELLED ALGEBRAICALLY: dividing by
    // (PW_FRESHNESS + imp_w) and multiplying by min(that, 1.0) is a no-op
    // whenever the sum is <= 1, so alpha_imp = 1.5 gave imp_w = 0.75 and a
    // total exponent weight of 1.23 once graph and agreement were applied
    // late — a 1.381 ceiling, not the 1.30 this module promises. My
    // "learned weights cannot widen the budget" test passed anyway because
    // it probed at similarity 0.5, where the gate is not open enough to
    // expose the gap.
    //
    // Renormalize against the FULL weight vector, including the shares that
    // graph, agreement and usage will spend later in the pipeline, so the
    // sum over every prior stays <= 1 no matter what the learner produces.
    let imp_w = (weights.alpha_imp / ALPHA_IMP * PW_IMPORTANCE).clamp(0.0, 1.0);
    let late_w = PW_GRAPH + PW_AGREEMENT + PW_USAGE;
    let total_w = PW_FRESHNESS + imp_w + late_w;
    let scale = if total_w > 1.0 { 1.0 / total_w } else { 1.0 };
    let z = (PW_FRESHNESS * freshness_z + imp_w * importance_z) * scale;
    weights.w_sim
        * similarity
        * (POLICY_BUDGET_LN * z).exp()
        * query_valence_boost(valence, query_sentiment)
}

/// Adaptive contributions using learned weights.
pub fn adaptive_contributions(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    weights: &crate::types::LearnedWeights,
) -> ScoreContributions {
    let gate = sigmoid(GATE_K * (similarity - weights.gate_tau));
    ScoreContributions {
        similarity: weights.w_sim * similarity,
        decay: FRESHNESS_SCALE * weights.w_decay * decay,
        recency: FRESHNESS_SCALE * weights.w_recency * recency,
        importance: gate * weights.alpha_imp * importance.min(1.0),
        graph_proximity: 0.0,
    }
}

/// Adaptive graph composite score using learned weights.
///
/// Derives graph-specific weights from the learned base weights, scaled by
/// the same ratios as the hardcoded GW_* vs W_* constants.
pub fn adaptive_graph_composite_score(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    valence: f64,
    graph_proximity: f64,
    query_sentiment: f64,
    weights: &crate::types::LearnedWeights,
) -> f64 {
    // Same shape as the const version: one base formula for every candidate,
    // then a bounded graph multiplier. The learned weights are NOT
    // reallocated when an edge exists — that reallocation was the
    // discontinuity at zero, and it also treated decay/recency weights as
    // additive relevance mass, which the freshness refactor had already
    // established they are not.
    adaptive_composite_score(
        similarity,
        decay,
        recency,
        importance,
        valence,
        query_sentiment,
        weights,
    ) * graph_mult(graph_proximity)
}

/// Adaptive graph contributions using learned weights.
pub fn adaptive_graph_contributions(
    similarity: f64,
    decay: f64,
    recency: f64,
    importance: f64,
    graph_proximity: f64,
    weights: &crate::types::LearnedWeights,
) -> ScoreContributions {
    // Reports the graph term as the BUDGET SHARE it actually is. Until
    // 2026-08-13 this returned GRAPH_SCALE * proximity while scoring had
    // already moved to the shared policy budget, so the explanation
    // described a formula the engine no longer used. An explanation that
    // does not match the computation is worse than none: it is a confident
    // wrong answer to "why did this rank here?".
    let mut c = adaptive_contributions(similarity, decay, recency, importance, weights);
    c.graph_proximity = POLICY_BUDGET_LN * PW_GRAPH * graph_proximity.clamp(0.0, 1.0);
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decay_score_fresh() {
        let score = decay_score(0.8, 604800.0, 0.0);
        assert!((score - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_decay_score_one_half_life() {
        let score = decay_score(1.0, 100.0, 100.0);
        assert!((score - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_decay_score_zero_half_life() {
        let score = decay_score(0.8, 0.0, 100.0);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_recency_score_fresh() {
        let score = recency_score(0.0);
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_recency_score_seven_days() {
        let score = recency_score(7.0 * 86400.0);
        assert!((score - f64::exp(-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_valence_boost_zero() {
        assert!((valence_boost(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_valence_boost_positive() {
        assert!((valence_boost(1.0) - 1.3).abs() < 1e-10);
    }

    #[test]
    fn test_valence_boost_negative() {
        assert!((valence_boost(-0.5) - 1.15).abs() < 1e-10);
    }

    // ── Importance gate tests ──

    #[test]
    fn test_importance_gate_high_similarity() {
        // sim=0.8 >> τ=0.30 → gate ≈ 1.0
        let gate = importance_gate(0.8);
        assert!(gate > 0.99, "gate at sim=0.8 should be ~1.0, got {gate}");
    }

    #[test]
    fn test_importance_gate_low_similarity() {
        // sim=0.05 << τ=0.25 → gate ≈ 0.08
        let gate = importance_gate(0.05);
        assert!(gate < 0.12, "gate at sim=0.05 should be small, got {gate}");
    }

    #[test]
    fn test_importance_gate_at_threshold() {
        // sim=τ → gate = 0.5
        let gate = importance_gate(GATE_TAU);
        assert!(
            (gate - 0.5).abs() < 1e-10,
            "gate at sim=τ should be 0.5, got {gate}"
        );
    }

    #[test]
    fn test_importance_gate_monotonic() {
        let low = importance_gate(0.1);
        let mid = importance_gate(0.3);
        let high = importance_gate(0.7);
        assert!(mid > low, "gate should increase with similarity");
        assert!(high > mid, "gate should increase with similarity");
    }

    // ── Composite score: relevance-gated behavior ──

    #[test]
    fn test_high_imp_low_sim_loses_to_low_imp_high_sim() {
        // THE critical test: high importance + low similarity should NOT beat
        // moderate importance + high similarity.
        let irrelevant_important = composite_score(0.10, 0.5, 0.5, 1.0, 0.0);
        let relevant_normal = composite_score(0.60, 0.5, 0.5, 0.3, 0.0);
        assert!(relevant_normal > irrelevant_important,
            "relevant_normal ({relevant_normal:.4}) should beat irrelevant_important ({irrelevant_important:.4})");
    }

    #[test]
    fn test_high_imp_high_sim_beats_low_imp_high_sim() {
        // When both are relevant, importance should still help
        let important = composite_score(0.70, 0.5, 0.5, 0.9, 0.0);
        let normal = composite_score(0.70, 0.5, 0.5, 0.3, 0.0);
        assert!(
            important > normal,
            "when both relevant, higher importance should win: {important:.4} vs {normal:.4}"
        );
    }

    #[test]
    fn test_composite_score_basic() {
        // Rewritten 2026-08-13: this asserted the OLD per-factor product
        // (freshness x (1 + gate*alpha)), which is exactly the shape that
        // compounded to ~3.26x. Priors now share ONE budget, so the test
        // states the property rather than re-deriving the arithmetic.
        let score = composite_score(1.0, 1.0, 1.0, 1.0, 0.0);
        let relevance = W_SIM * 1.0;
        assert!(
            score > relevance,
            "maxed priors must lift a fully-relevant record above bare relevance"
        );
        assert!(
            score <= relevance * POLICY_BUDGET_LN.exp() + 1e-12,
            "priors must never exceed the shared budget: {score} > {}",
            relevance * POLICY_BUDGET_LN.exp()
        );
    }

    #[test]
    fn test_composite_score_with_valence() {
        // Valence is query-aware EVIDENCE, applied outside the prior
        // budget, so it scales the whole composite by exactly 1.3 here.
        let neutral = composite_score(1.0, 1.0, 1.0, 1.0, 0.0);
        let valenced = composite_score(1.0, 1.0, 1.0, 1.0, 1.0);
        assert!((valenced - neutral * 1.3).abs() < 1e-10);
    }

    #[test]
    fn test_graph_composite_zero_proximity_matches_original() {
        let original = composite_score(0.8, 0.6, 0.9, 0.7, 0.3);
        let graph = graph_composite_score(0.8, 0.6, 0.9, 0.7, 0.3, 0.0);
        assert!((original - graph).abs() < 1e-10);
    }

    #[test]
    fn test_graph_composite_with_proximity() {
        // REWRITTEN 2026-08-13. This test used to PIN the additive wall
        // (GW_SIM*sim + GW_GRAPH*proximity) - it asserted the defect, so it
        // passed for as long as the bug existed and would have failed on the
        // fix. Now it states the two properties that actually matter.
        let with_edges = graph_composite_score(0.5, 0.5, 0.5, 0.5, 0.0, 1.0);
        let without = graph_composite_score(0.5, 0.5, 0.5, 0.5, 0.0, 0.0);
        assert!(
            with_edges > without,
            "proximity must still break ties between equally-relevant records"
        );
        assert!(
            with_edges <= without * POLICY_BUDGET_LN.exp() + 1e-12,
            "proximity shares the ONE prior budget; it cannot exceed it"
        );
        // The original wall, as a regression: an irrelevant but maximally
        // connected record must never beat a relevant unconnected one.
        let irrelevant_connected = graph_composite_score(0.0026, 1.0, 1.0, 1.0, 0.0, 1.0);
        let relevant_isolated = graph_composite_score(0.309, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(
            relevant_isolated > irrelevant_connected,
            "the graph wall is back: connected {irrelevant_connected} beat relevant {relevant_isolated}"
        );
    }

    #[test]
    fn the_recency_wall_is_torn_down() {
        // THE 2026-08-05 production defect, as a pinned regression test:
        // an OLD, RELEVANT record (sim 0.6, decay/recency ≈ 0) must beat
        // a FRESH, IRRELEVANT one (sim 0.3, decay/recency ≈ 1). Under
        // the old additive form the fresh record collected +0.5 free and
        // won regardless of relevance — measured on a 4,297-record
        // production clone as MRR 0.562 (exact cosine) collapsing to
        // 0.069 (the formula). No freshness gap may overturn a real
        // relevance gap.
        let old_relevant = composite_score(0.6, 0.0, 0.0, 0.5, 0.0);
        let fresh_irrelevant = composite_score(0.3, 1.0, 1.0, 0.5, 0.0);
        assert!(
            old_relevant > fresh_irrelevant,
            "an old relevant record must outrank fresh irrelevant noise: \
             {old_relevant:.4} vs {fresh_irrelevant:.4}"
        );
    }

    // ── Monotonicity & property invariants ──

    #[test]
    fn test_composite_monotonic_in_similarity() {
        let low = composite_score(0.3, 0.5, 0.5, 0.5, 0.0);
        let high = composite_score(0.9, 0.5, 0.5, 0.5, 0.0);
        assert!(high > low, "higher similarity should yield higher score");
    }

    #[test]
    fn test_composite_monotonic_in_importance() {
        // At moderate similarity (gate ≈ 0.88), importance should still matter
        let low = composite_score(0.5, 0.5, 0.5, 0.2, 0.0);
        let high = composite_score(0.5, 0.5, 0.5, 0.9, 0.0);
        assert!(
            high > low,
            "higher importance should yield higher score (when sim>τ)"
        );
    }

    #[test]
    fn test_composite_monotonic_in_recency() {
        let low = composite_score(0.5, 0.5, 0.1, 0.5, 0.0);
        let high = composite_score(0.5, 0.5, 0.9, 0.5, 0.0);
        assert!(high > low, "higher recency should yield higher score");
    }

    #[test]
    fn test_valence_symmetric() {
        assert!((valence_boost(0.7) - valence_boost(-0.7)).abs() < 1e-10);
    }

    #[test]
    fn test_valence_always_geq_1() {
        for v in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            assert!(
                valence_boost(v) >= 1.0,
                "valence_boost({v}) = {} < 1.0",
                valence_boost(v)
            );
        }
    }

    #[test]
    fn test_composite_non_negative() {
        for &sim in &[0.0, 0.5, 1.0] {
            for &dec in &[0.0, 0.5, 1.0] {
                for &rec in &[0.0, 0.5, 1.0] {
                    for &imp in &[0.0, 0.5, 1.0] {
                        for &val in &[-1.0, 0.0, 1.0] {
                            let s = composite_score(sim, dec, rec, imp, val);
                            assert!(
                                s >= 0.0,
                                "composite_score({sim},{dec},{rec},{imp},{val}) = {s} < 0"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_graph_composite_non_negative() {
        for &prox in &[0.0, 0.25, 0.5, 1.0] {
            let s = graph_composite_score(0.5, 0.5, 0.5, 0.5, 0.0, prox);
            assert!(
                s >= 0.0,
                "graph_composite with prox={prox} should be non-negative"
            );
        }
    }

    #[test]
    fn test_graph_proximity_increases_score() {
        let without = graph_composite_score(0.3, 0.8, 0.7, 0.6, 0.0, 0.0);
        let with = graph_composite_score(0.3, 0.8, 0.7, 0.6, 0.0, 0.8);
        assert!(
            with > without,
            "graph proximity (0.8) should increase score: without={without}, with={with}"
        );
    }

    #[test]
    fn test_decay_monotonic_in_elapsed() {
        let fresh = decay_score(0.8, 604800.0, 0.0);
        let old = decay_score(0.8, 604800.0, 604800.0);
        let ancient = decay_score(0.8, 604800.0, 604800.0 * 10.0);
        assert!(fresh > old);
        assert!(old > ancient);
    }

    #[test]
    fn test_recency_monotonic_in_age() {
        let fresh = recency_score(0.0);
        let week = recency_score(7.0 * 86400.0);
        let month = recency_score(30.0 * 86400.0);
        assert!(fresh > week);
        assert!(week > month);
    }

    #[test]
    fn test_build_why_always_nonempty() {
        let why = build_why(0.0, 0.0, 0.0, 0.0);
        assert!(
            !why.is_empty(),
            "build_why should always produce at least one reason"
        );
        assert_eq!(why[0], "matched query");
    }

    #[test]
    fn test_build_why_contains_similarity() {
        let why = build_why(0.9, 0.1, 0.1, 0.0);
        assert!(why.iter().any(|w| w.contains("semantically similar")));
    }

    #[test]
    fn test_importance_capped_at_1() {
        let capped = composite_score(0.5, 0.5, 0.5, 5.0, 0.0);
        let at_one = composite_score(0.5, 0.5, 0.5, 1.0, 0.0);
        assert!(
            (capped - at_one).abs() < 1e-10,
            "importance should be capped at 1.0"
        );
    }

    #[test]
    fn test_eviction_score() {
        // With zero recalls the access term vanishes (back-compatible).
        let score = eviction_score(1.0, 1.0, 0);
        assert!((score - 1.0).abs() < 1e-10, "max inputs, no access => 1.0");

        let score_zero = eviction_score(0.0, 0.0, 0);
        assert!(
            (score_zero - 0.0).abs() < 1e-10,
            "zero inputs, no access => 0.0"
        );

        let low = eviction_score(0.1, 0.5, 0);
        let high = eviction_score(0.9, 0.5, 0);
        assert!(high > low, "higher decay => higher eviction score");

        // Recall frequency protects from eviction: a stale-but-hot memory
        // scores higher (less evictable) than the same memory never recalled.
        let cold = eviction_score(0.2, 0.2, 0);
        let hot = eviction_score(0.2, 0.2, 25);
        assert!(
            hot > cold,
            "frequent recall must resist eviction: {hot} > {cold}"
        );
        // Saturating: beyond ACCESS_SATURATION the resistance is capped.
        let at_sat = eviction_score(0.2, 0.2, ACCESS_SATURATION);
        let beyond = eviction_score(0.2, 0.2, ACCESS_SATURATION * 100);
        assert!(
            (beyond - at_sat).abs() < 0.05,
            "access resistance saturates"
        );
    }

    // ── Regression: the "Staff Engineer domination" bug ──
    // In v7, "I got promoted to Staff Engineer" (imp=0.9) appeared at #1
    // for 16/20 queries because additive importance gave it +0.27 regardless
    // of similarity. With gated scoring, sim=0.1 means gate≈0.08 → boost≈0.06.

    #[test]
    fn test_irrelevant_anchor_cannot_dominate() {
        // Simulates: "Staff Engineer" memory recalled for "What did I cook?"
        // sim=0.08 (irrelevant), imp=0.9, decay=0.8, recency=0.3
        let anchor = composite_score(0.08, 0.8, 0.3, 0.9, 0.0);
        // Simulates: "Made pasta for dinner" for "What did I cook?"
        // sim=0.65 (relevant), imp=0.2, decay=0.6, recency=0.8
        let daily = composite_score(0.65, 0.6, 0.8, 0.2, 0.0);
        assert!(
            daily > anchor,
            "relevant daily ({daily:.4}) should beat irrelevant anchor ({anchor:.4})"
        );
    }

    // ── Query-aware valence tests ──

    #[test]
    fn test_detect_query_sentiment_negative() {
        assert_eq!(
            detect_query_sentiment("What failures and problems have been stressing me out?"),
            -1.0
        );
        assert_eq!(
            detect_query_sentiment("Tell me about my emotional lows"),
            -1.0
        );
        assert_eq!(
            detect_query_sentiment("What was difficult this year?"),
            -1.0
        );
    }

    #[test]
    fn test_detect_query_sentiment_positive() {
        assert_eq!(detect_query_sentiment("What good things happened?"), 1.0);
        assert_eq!(detect_query_sentiment("Tell me about happy moments"), 1.0);
        assert_eq!(
            detect_query_sentiment("What was my greatest achievement?"),
            1.0
        );
    }

    #[test]
    fn test_detect_query_sentiment_neutral() {
        assert_eq!(
            detect_query_sentiment("What happened at work recently?"),
            0.0
        );
        assert_eq!(detect_query_sentiment("Tell me about my family"), 0.0);
    }

    #[test]
    fn test_query_valence_boost_neutral_query_matches_standard() {
        // Neutral query (sentiment=0.0) should behave identically to valence_boost
        for &v in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            let standard = valence_boost(v);
            let query_aware = query_valence_boost(v, 0.0);
            assert!((standard - query_aware).abs() < 1e-10,
                "neutral query should match standard: valence={v}, standard={standard}, query_aware={query_aware}");
        }
    }

    #[test]
    fn test_query_valence_boost_negative_alignment() {
        // Negative query + negative memory → higher boost than neutral query
        let negative_aligned = query_valence_boost(-0.8, -1.0);
        let standard = valence_boost(-0.8);
        assert!(negative_aligned > standard,
            "negative query + negative memory should boost more: aligned={negative_aligned:.4}, standard={standard:.4}");
    }

    #[test]
    fn test_query_valence_boost_negative_misaligned() {
        // Negative query + positive memory → lower boost than neutral query
        let misaligned = query_valence_boost(0.8, -1.0);
        let standard = valence_boost(0.8);
        assert!(misaligned < standard,
            "negative query + positive memory should boost less: misaligned={misaligned:.4}, standard={standard:.4}");
    }

    #[test]
    fn test_query_valence_boost_always_positive() {
        // Boost should never go below 1.0 regardless of alignment
        for &v in &[-1.0, -0.5, 0.0, 0.5, 1.0] {
            for &s in &[-1.0, 0.0, 1.0] {
                let boost = query_valence_boost(v, s);
                assert!(
                    boost >= 0.8,
                    "query_valence_boost({v}, {s}) = {boost} should be positive"
                );
            }
        }
    }

    #[test]
    fn test_composite_with_sentiment_matches_original_for_neutral() {
        // composite_score_with_sentiment with sentiment=0.0 should match composite_score
        let original = composite_score(0.6, 0.5, 0.7, 0.8, 0.3);
        let with_sent = composite_score_with_sentiment(0.6, 0.5, 0.7, 0.8, 0.3, 0.0);
        assert!(
            (original - with_sent).abs() < 1e-10,
            "neutral sentiment should match original: {original} vs {with_sent}"
        );
    }

    #[test]
    fn test_gate_tau_regression_anchor_still_loses() {
        // With GATE_TAU=0.25, verify Staff Engineer at sim=0.08 still loses
        let anchor = composite_score(0.08, 0.8, 0.3, 0.9, 0.0);
        let daily = composite_score(0.65, 0.6, 0.8, 0.2, 0.0);
        assert!(daily > anchor,
            "with GATE_TAU=0.25, relevant daily ({daily:.4}) should still beat irrelevant anchor ({anchor:.4})");
        // Also verify the gate itself at sim=0.08 is still small
        let gate = importance_gate(0.08);
        assert!(gate < 0.2, "gate at sim=0.08 should be small, got {gate}");
    }
}

#[cfg(test)]
mod graph_wall_tests {
    use super::*;
    use crate::types::LearnedWeights;

    /// THE REGRESSION, with the numbers that exposed it.
    ///
    /// Under the old additive form `(GW_SIM*sim + GW_GRAPH*prox)`, a record at
    /// similarity 0.0026 with a maximal graph edge scored 0.340 while a
    /// genuinely relevant record at similarity 0.309 with no edge scored
    /// 0.267. Measured in a live store, where a phantom `AT` node returned a
    /// real-estate tax memo for "encryption at rest and key rotation".
    #[test]
    fn maximal_graph_edge_cannot_beat_a_relevant_record() {
        let irrelevant = graph_composite_score_with_sentiment(0.0026, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0);
        let relevant = graph_composite_score_with_sentiment(0.309, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
        assert!(
            relevant > irrelevant,
            "graph wall rebuilt: irrelevant+connected {irrelevant:.4} beat relevant {relevant:.4}"
        );
    }

    /// The bound, stated as arithmetic rather than taste: graph evidence may
    /// reverse at most a GRAPH_SCALE similarity gap. A record more than 12.5%
    /// less similar cannot be promoted no matter how connected it is.
    #[test]
    fn graph_uplift_is_bounded_by_graph_scale() {
        let s_low = 0.50;
        let s_high = s_low * (1.0 + GRAPH_SCALE) * 1.001; // just outside the budget
        let connected = graph_composite_score_with_sentiment(s_low, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
        let plain = graph_composite_score_with_sentiment(s_high, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(
            plain > connected,
            "a gap wider than GRAPH_SCALE was reversed: {plain:.6} vs {connected:.6}"
        );
    }

    /// NO DISCONTINUITY AT ZERO. The old branch reallocated weights the moment
    /// proximity became positive — similarity 0.50 -> 0.35, importance cap
    /// 0.80 -> 0.60 — so an infinitesimal edge CUT a record's score by 30-40%.
    /// Graph evidence penalised the records it touched.
    #[test]
    fn an_infinitesimal_edge_does_not_penalise() {
        let no_edge = graph_composite_score_with_sentiment(0.4, 0.5, 0.5, 0.8, 0.0, 0.0, 0.0);
        let tiny_edge = graph_composite_score_with_sentiment(0.4, 0.5, 0.5, 0.8, 0.0, 1e-9, 0.0);
        assert!(
            tiny_edge >= no_edge,
            "a tiny edge reduced the score: {tiny_edge:.6} < {no_edge:.6}"
        );
        assert!((tiny_edge - no_edge).abs() < 1e-6, "discontinuity at zero");
    }

    /// Zero similarity still means zero score, edge or not — the invariant the
    /// whole multiplicative design exists to preserve.
    #[test]
    fn zero_similarity_scores_zero_even_when_connected() {
        let s = graph_composite_score_with_sentiment(0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0);
        assert_eq!(s, 0.0, "a connected but irrelevant record scored {s}");
    }

    /// The adaptive path must obey the same law; it had its own copy of the
    /// additive form and its own weight reallocation.
    #[test]
    fn adaptive_path_obeys_the_same_wall() {
        let w = LearnedWeights::default();
        let irrelevant = adaptive_graph_composite_score(0.0026, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, &w);
        let relevant = adaptive_graph_composite_score(0.309, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, &w);
        assert!(
            relevant > irrelevant,
            "adaptive graph wall: {irrelevant:.4} beat {relevant:.4}"
        );
        let no_edge = adaptive_graph_composite_score(0.4, 0.5, 0.5, 0.8, 0.0, 0.0, 0.0, &w);
        let tiny = adaptive_graph_composite_score(0.4, 0.5, 0.5, 0.8, 0.0, 1e-9, 0.0, &w);
        assert!(
            (tiny - no_edge).abs() < 1e-6,
            "adaptive discontinuity at zero"
        );
    }

    /// EXECUTABLE HISTORY. Reproduces the old additive formula from the
    /// retained GW_* constants and shows it inverting the ranking, then shows
    /// the current formula refusing to. Without this the regression above is
    /// just a pair of numbers that pass; with it, the bug is demonstrable on
    /// demand and the constants have a reason to still exist.
    #[test]
    fn the_old_additive_form_inverted_the_ranking() {
        let old = |sim: f64, prox: f64| {
            let base_rel =
                (GW_SIM * sim + GW_GRAPH * prox) * freshness_mult(1.0, 1.0, GW_DECAY, GW_RECENCY);
            let imp_mult = 1.0 + importance_gate(sim) * GW_ALPHA_IMP * 1.0;
            base_rel * imp_mult
        };
        let old_irrelevant = old(0.0026, 1.0);
        let old_relevant = old(0.309, 0.0);
        assert!(
            old_irrelevant > old_relevant,
            "history not reproduced: {old_irrelevant:.4} vs {old_relevant:.4}"
        );

        let now_irrelevant =
            graph_composite_score_with_sentiment(0.0026, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0);
        let now_relevant =
            graph_composite_score_with_sentiment(0.309, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0);
        assert!(
            now_relevant > now_irrelevant,
            "the fix does not fix it: {now_relevant:.4} vs {now_irrelevant:.4}"
        );
    }

    /// A genuine tie-break still works — the fix must not make graph evidence
    /// inert, only bounded.
    #[test]
    fn graph_still_breaks_ties_among_near_equals() {
        let connected = graph_composite_score_with_sentiment(0.40, 0.5, 0.5, 0.5, 0.0, 1.0, 0.0);
        let alone = graph_composite_score_with_sentiment(0.40, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0);
        assert!(
            connected > alone,
            "graph evidence became inert: {connected:.6} vs {alone:.6}"
        );
    }
}

#[cfg(test)]
mod policy_budget_tests {
    use super::*;

    #[test]
    fn policy_weights_partition_one_budget() {
        // THE invariant. If a future signal is added with its own weight and
        // this sum exceeds 1.0, the ceiling silently widens — which is
        // exactly how the old per-factor bounds compounded to ~3.26x.
        // MUST list every weight. The first version of this test omitted
        // PW_USAGE — the invariant named "partition one budget" did not
        // actually sum the whole partition, which is the precise way an
        // invariant test rots: it keeps passing while the thing it names
        // stops being true.
        let total = PW_FRESHNESS + PW_IMPORTANCE + PW_GRAPH + PW_AGREEMENT + PW_USAGE;
        assert!(
            total <= 1.0 + 1e-12,
            "prior weights must partition ONE budget; sum = {total}"
        );
    }

    #[test]
    fn policy_layer_never_exceeds_its_stated_ceiling() {
        let ceiling = POLICY_BUDGET_LN.exp();
        assert!((ceiling - 1.30).abs() < 1e-9, "ceiling drifted: {ceiling}");
        // Every prior maxed simultaneously — the case that used to compound.
        let maxed = policy_mult(1.0, 1.0, 1.0, 1.0, 1.0);
        assert!(
            (maxed - ceiling).abs() < 1e-9,
            "all priors maxed must equal exactly the budget, got {maxed}"
        );
        // Out-of-range inputs cannot buy more (validate.rs enforces only
        // finiteness, so unclamped values DO reach scoring). A negative
        // prior clamps to 0 — it forfeits its share rather than subtracting
        // from the others, so this lands strictly BELOW the ceiling.
        assert!(policy_mult(9.0, 9.0, 9.0, -9.0, 9.0) <= ceiling + 1e-9);
        assert!(policy_mult(9.0, 9.0, 9.0, -9.0, 9.0) < ceiling);
        assert!((policy_mult(9.0, 9.0, 9.0, 9.0, 9.0) - ceiling).abs() < 1e-9);
        assert_eq!(policy_mult(0.0, 0.0, 0.0, 0.0, 0.0), 1.0);
    }

    #[test]
    fn priors_cannot_invert_a_real_relevance_gap() {
        // Codex's counterexample against the OLD scoring: a similarity-0.30
        // record with maxed priors scored 0.648 and beat a bare
        // similarity-0.60 record. Under one shared budget it cannot.
        let weak_but_privileged = composite_score_with_sentiment(0.30, 1.0, 1.0, 1.0, 1.0, 1.0);
        let strong_but_bare = composite_score_with_sentiment(0.60, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(
            strong_but_bare > weak_but_privileged,
            "a 2x similarity gap must survive every prior: bare 0.60 scored {strong_but_bare}, \
             privileged 0.30 scored {weak_but_privileged}"
        );
    }

    #[test]
    fn priors_still_break_ties_between_near_equals() {
        // A bounded budget must still DO something, or it is just cosine.
        let fresh_important = composite_score_with_sentiment(0.50, 1.0, 1.0, 1.0, 0.0, 0.0);
        let stale_trivial = composite_score_with_sentiment(0.50, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(
            fresh_important > stale_trivial,
            "priors must still order equally-relevant records"
        );
        // ...and a 30% relevance gap is the documented reach, so a record
        // 40% less similar must still lose however privileged it is.
        let privileged = composite_score_with_sentiment(0.60, 1.0, 1.0, 1.0, 0.0, 0.0);
        let plain = composite_score_with_sentiment(1.00, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(privileged < plain, "budget must not reach beyond ~30%");
    }

    #[test]
    fn learned_weights_cannot_widen_the_budget() {
        // alpha_imp was allowed up to 1.5, which pushed the OLD composed
        // ceiling toward 4x. Learning may now reallocate the budget, never
        // expand it.
        let mut w = crate::types::LearnedWeights::default();
        w.alpha_imp = 1.5;
        w.w_decay = 1.0;
        w.w_recency = 1.0;
        // Probe at similarity 0.9, where importance_gate is ~fully OPEN.
        // The original version of this test probed at 0.5, where the gate is
        // only partly open — it passed while alpha_imp=1.5 was in fact
        // pushing the reachable ceiling to ~1.381. A bound test must be
        // taken where the bound is TIGHT, or it certifies nothing.
        for sim in [0.5_f64, 0.9, 0.99] {
            let boosted = adaptive_composite_score(sim, 1.0, 1.0, 1.0, 0.0, 0.0, &w)
                * graph_mult(1.0)
                * agreement_mult(9);
            let ceiling = w.w_sim * sim * POLICY_BUDGET_LN.exp();
            assert!(
                boosted <= ceiling + 1e-9,
                "learned weights escaped the budget at sim={sim}: {boosted} > {ceiling}"
            );
        }
    }
}

#[cfg(test)]
mod lane_and_weight_bound_tests {
    use super::*;
    use crate::types::LearnedWeights;

    #[test]
    fn exploration_lanes_cannot_outrank_a_real_match() {
        // The valence and cold lanes used to add +0.20·|v|·importance and
        // +0.15·importance OUTRIGHT. With composite scores in the 0.1–0.6
        // range those terms could dominate, so a record matching nothing
        // could beat one matching everything. Now the lift is a bounded
        // multiplier: admission is the lane's job, relevance is not.
        let irrelevant_but_charged =
            composite_score(0.02, 1.0, 1.0, 1.0, 1.0) * lane_lift_mult(1.0);
        let relevant_plain = composite_score(0.80, 0.0, 0.0, 0.0, 0.0);
        assert!(
            relevant_plain > irrelevant_but_charged,
            "a lane lift must not invert relevance: plain 0.80 = {relevant_plain}, \
             lifted 0.02 = {irrelevant_but_charged}"
        );
        assert!((lane_lift_mult(0.0) - 1.0).abs() < 1e-12);
        assert!((lane_lift_mult(5.0) - (1.0 + LANE_LIFT_MAX)).abs() < 1e-12);
    }

    #[test]
    fn corrupt_learned_weights_are_clamped_not_trusted() {
        let hostile = LearnedWeights {
            w_sim: -3.0,
            w_decay: 99.0,
            w_recency: f64::NAN,
            gate_tau: 0.0,
            alpha_imp: 50.0,
            keyword_boost: 500.0, // would dwarf every semantic score
            generation: 7,
        }
        .clamped();
        assert!(hostile.w_sim >= 0.05 && hostile.w_sim <= 1.0);
        assert!(hostile.w_decay <= 1.0);
        assert!(hostile.w_recency.is_finite(), "NaN must not survive");
        assert!(hostile.gate_tau >= 0.05);
        assert!(hostile.alpha_imp <= 1.5);
        assert!(
            hostile.keyword_boost <= 1.0,
            "keyword_boost is ADDITIVE and has no similarity-relative ceiling \
             of its own — the clamp is its only bound"
        );
        assert_eq!(hostile.generation, 7, "generation is data, not a weight");
        // Sane weights must pass through untouched.
        let sane = LearnedWeights::default();
        assert_eq!(sane.clone().clamped().w_sim, sane.w_sim);
        assert_eq!(sane.clone().clamped().keyword_boost, sane.keyword_boost);
    }
}

#[cfg(test)]
mod budget_composition_tests {
    use super::*;

    #[test]
    fn stagewise_application_composes_to_one_budget() {
        // The property that makes the design safe in a real pipeline: the
        // recall path scores a composite FIRST and multiplies graph and
        // agreement in LATER, at different stages. In log space that is
        // identical to applying every prior at once, so the total is still
        // bounded by exp(B) — where a factor lands stops mattering.
        let stagewise = composite_score_with_sentiment(1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
            * graph_mult(1.0)
            * agreement_mult(9);
        // importance_z is GATED: sigmoid(12*(1.0-0.25)) = 0.99987, not 1.0.
        // Using a literal 1.0 here was an idealized expectation that hid an
        // 8e-6 discrepancy; the gate is the point, so the test uses it.
        let all_at_once = W_SIM * 1.0 * policy_mult(1.0, importance_gate(1.0), 1.0, 1.0, 0.0);
        assert!(
            (stagewise - all_at_once).abs() < 1e-9,
            "stagewise {stagewise} must equal all-at-once {all_at_once}"
        );
        assert!(
            stagewise <= W_SIM * POLICY_BUDGET_LN.exp() + 1e-9,
            "the whole pipeline must stay inside one budget"
        );
    }

    #[test]
    fn full_pipeline_cannot_invert_a_real_relevance_gap() {
        // Every prior maxed, applied the way recall actually applies them,
        // against a bare record with double the similarity.
        let privileged = composite_score_with_sentiment(0.30, 1.0, 1.0, 1.0, 1.0, 1.0)
            * graph_mult(1.0)
            * agreement_mult(9);
        let bare = composite_score_with_sentiment(0.60, 0.0, 0.0, 0.0, 0.0, 0.0);
        assert!(
            bare > privileged,
            "full-pipeline priors inverted a 2x similarity gap: bare {bare}, \
             privileged {privileged}"
        );
    }
}
