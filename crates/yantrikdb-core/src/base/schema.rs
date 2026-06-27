pub const SCHEMA_VERSION: i32 = 32;

pub const SCHEMA_SQL: &str = "
-- Memory records: the source of truth
CREATE TABLE IF NOT EXISTS memories (
    rid TEXT PRIMARY KEY,                -- UUIDv7, stable across devices
    type TEXT NOT NULL DEFAULT 'episodic', -- episodic | semantic | procedural | emotional
    text TEXT NOT NULL,                  -- raw memory content
    embedding BLOB,                     -- vector embedding (float32 array)

    -- Temporal
    created_at REAL NOT NULL,           -- unix timestamp (float for sub-second)
    updated_at REAL NOT NULL,

    -- Decay parameters (stored, not continuously updated)
    importance REAL NOT NULL DEFAULT 0.5,  -- base importance I0 [0, 1]
    half_life REAL NOT NULL DEFAULT 604800.0, -- seconds (default: 7 days)
    last_access REAL NOT NULL,            -- unix timestamp of last recall/reinforce
    access_count INTEGER NOT NULL DEFAULT 0, -- number of times retrieved via recall
    valence REAL NOT NULL DEFAULT 0.0,    -- emotional weight [-1, 1]

    -- Consolidation tracking
    consolidated_into TEXT,              -- rid of the semantic memory this was merged into
    consolidation_status TEXT DEFAULT 'active', -- active | consolidated | tombstoned

    -- Storage tier
    storage_tier TEXT NOT NULL DEFAULT 'hot', -- hot | cold

    -- Metadata
    metadata TEXT DEFAULT '{}',          -- JSON blob for extensibility

    -- Namespace for memory isolation
    namespace TEXT NOT NULL DEFAULT 'default',

    -- Cognitive dimensions (V10)
    certainty REAL NOT NULL DEFAULT 0.8,     -- confidence in accuracy [0, 1]
    domain TEXT NOT NULL DEFAULT 'general',   -- topic domain (work, health, family, finance, etc.)
    source TEXT NOT NULL DEFAULT 'user',      -- origin (user, system, document, inference)
    emotional_state TEXT,                     -- rich emotion label (joy, sadness, anger, fear, etc.)

    -- Session & temporal (V13)
    session_id TEXT,                          -- FK to sessions.session_id (nullable)
    due_at REAL,                              -- unix timestamp for upcoming() queries
    temporal_kind TEXT,                        -- deadline | reminder | event | follow_up

    -- v25 (RFC issue #9): cluster-replication determinism columns
    tombstone_reason TEXT,                     -- caller-supplied reason for tombstone_with_rid (NULL for live rows)
    created_at_unix_micros INTEGER NOT NULL DEFAULT 0, -- caller-supplied i64 micros, materialized at leader for byte-deterministic follower replay
    embedding_model TEXT,                      -- engine-deterministic-surface version pin (e.g. 'bge-base-en-v1.5'); RFC 013 may swap for richer type

    -- v26 (RFC 026 / issue #29): conflict-aware-write provenance metadata.
    -- Foundation for issue #30 WriteResolution API. Columns capture the
    -- epistemic operation chosen at write time so conflict resolution +
    -- paper adoption analysis are first-class queries, not JSON-blob crawls.
    -- All NULL on pre-v26 rows.
    prior_rid TEXT,                            -- supersedes/updates/merges target rid; NULL for append_as_new
    resolution_kind TEXT,                      -- 'append' | 'update' | 'merge' | 'supersede' | 'dismiss'
    dismissal_reason TEXT,                     -- non-empty when resolution_kind='dismiss'; audit trail
    confidence_at_write REAL,                  -- substrate's conflict-confidence at write time, [0.0, 1.0]

    -- v27 (issue #41): staging columns for db.reembed() operation.
    -- During Encoding phase, new embeddings are written here under the new
    -- embedder. During Swap phase, an atomic transaction moves these into
    -- the active `embedding` + `embedding_model` columns. Verifying phase
    -- nulls them out. On non-reembed rows, both are NULL and cost is zero.
    -- Pre-existing recall paths NEVER read these columns; only the reembed
    -- machinery does. See the brainstorm comment chain on #41 for why a
    -- two-column staging approach is required (recall re-reads active
    -- `embedding` post-HNSW; in-place mutation would dim-mismatch concurrent
    -- recalls).
    embedding_new BLOB,                        -- pending new embedding bytes; NULL except during active reembed
    embedding_new_model TEXT,                  -- name of embedder that produced embedding_new; NULL when no pending reembed

    -- v28 (issue #41 brainstorm-4 section 6): durable per-row generation stamp.
    -- Records which SearchState generation this row embedding column
    -- was encoded under. NULL on pre-v28 rows (treated as generation 0 by
    -- the post-swap materializer). Reembed Phase-2 swap transaction
    -- writes the new generation here atomically with promoting
    -- embedding_new into embedding. The materializer scan for rows under
    -- a stale generation uses idx_memories_embedding_generation.
    embedding_generation INTEGER,

    -- v32 (structural query / list_records): indexed generated columns that
    -- extract JSON metadata fields for typed enumeration without scanning the
    -- opaque metadata blob. VIRTUAL = computed on read; the secondary index
    -- materializes the value. The json_valid guard makes encrypted-metadata
    -- rows (ciphertext, not valid JSON) resolve to NULL instead of erroring
    -- the insert/index build. NULL when the key is absent.
    kind TEXT GENERATED ALWAYS AS (
        CASE WHEN json_valid(metadata) THEN json_extract(metadata, '$.kind') END
    ) VIRTUAL,
    drive_id TEXT GENERATED ALWAYS AS (
        CASE WHEN json_valid(metadata) THEN json_extract(metadata, '$.drive_id') END
    ) VIRTUAL
);
-- v28 index for the post-swap materializer scan of stale-generation rows.
CREATE INDEX IF NOT EXISTS idx_memories_embedding_generation
    ON memories(embedding_generation);
-- v26 partial indexes for the resolution/supersession query patterns the
-- WriteResolution API will exercise. Partial so they cost nothing on the
-- (initially) overwhelming majority of rows that are append-with-no-conflict.
CREATE INDEX IF NOT EXISTS idx_memories_prior_rid ON memories(prior_rid) WHERE prior_rid IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_resolution_kind ON memories(resolution_kind) WHERE resolution_kind IS NOT NULL;

-- Session tracking (V13)
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    client_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    started_at REAL NOT NULL,
    ended_at REAL,
    summary TEXT,
    avg_valence REAL,
    memory_count INTEGER NOT NULL DEFAULT 0,
    topics TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    hlc BLOB,
    origin_actor TEXT
);

-- ──────────────────────────────────────────────────────────────────
-- RFC 007 Phase 0: Meta-Cognitive Primitives — reasoning substrate.
-- Five layers: evidence (claims), propositions, variables + state
-- assertions, rule edges, scenarios. Every primitive operates on a
-- specific layer; conflating layers is how memory systems produce
-- confidently-wrong outputs.
-- ──────────────────────────────────────────────────────────────────

-- Layer 2 — Propositions: canonical identity for an abstract
-- (subject, relation, object) triple within a namespace. Evidence
-- rows in `claims` reference one proposition. Aggregation (support,
-- oppose, diversity) is computed at the proposition level.
CREATE TABLE IF NOT EXISTS propositions (
    proposition_id TEXT PRIMARY KEY,            -- UUIDv7
    src            TEXT NOT NULL,
    rel_type       TEXT NOT NULL,
    dst            TEXT NOT NULL,
    namespace      TEXT NOT NULL DEFAULT 'default',
    created_at     REAL NOT NULL,
    UNIQUE(src, rel_type, dst, namespace)
);
CREATE INDEX IF NOT EXISTS idx_propositions_src ON propositions(src);
CREATE INDEX IF NOT EXISTS idx_propositions_dst ON propositions(dst);
CREATE INDEX IF NOT EXISTS idx_propositions_rel ON propositions(rel_type);

-- Layer 3a — Variables: typed world-or-agent states that can be
-- observed or intervened on. Variables are what scenarios target;
-- they are distinct from propositions (which are abstract statements)
-- and from state_assertions (which are specific observations).
CREATE TABLE IF NOT EXISTS variables (
    variable_id    TEXT PRIMARY KEY,            -- UUIDv7
    name           TEXT NOT NULL,                 -- e.g. \"alice.sleep_quality\"
    namespace      TEXT NOT NULL DEFAULT 'default',
    value_space    TEXT NOT NULL,                 -- JSON: {type, values|range|unit}
    scope          TEXT NOT NULL,                 -- generic|individual|instance
    context_dims   TEXT NOT NULL DEFAULT '[]',    -- JSON array
    manipulable    INTEGER NOT NULL DEFAULT 0,    -- 0 = non-actionable
    actionability  TEXT,                          -- world_action|information_action|NULL
    created_at     REAL NOT NULL,
    UNIQUE(name, namespace)
);
CREATE INDEX IF NOT EXISTS idx_variables_ns ON variables(namespace);
CREATE INDEX IF NOT EXISTS idx_variables_scope ON variables(scope);

-- Layer 3b — State assertions: observations of a variable's value at
-- a point in time, optionally context-qualified.
CREATE TABLE IF NOT EXISTS state_assertions (
    state_id          TEXT PRIMARY KEY,          -- UUIDv7
    variable_id       TEXT NOT NULL REFERENCES variables(variable_id),
    value             TEXT NOT NULL,              -- JSON from variable's value_space
    valid_from        REAL NOT NULL,
    valid_to          REAL,                       -- NULL = still valid
    context_values    TEXT NOT NULL DEFAULT '{}', -- JSON
    confidence_band   TEXT NOT NULL DEFAULT 'medium',
    source            TEXT NOT NULL,
    source_memory_rid TEXT,
    namespace         TEXT NOT NULL,
    created_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_state_var ON state_assertions(variable_id);
CREATE INDEX IF NOT EXISTS idx_state_valid ON state_assertions(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_state_ns ON state_assertions(namespace);

-- Layer 4 — Rule edges: typed causal-or-structural edges between
-- variables. Whitelist enforced at schema level. Rule edges are
-- themselves first-class claims: `source_evidence_rids` tracks the
-- evidence supporting the rule's existence, and meta-contradictions
-- on rules resolve via the same polarity/aggregation logic as any
-- other proposition. Rules are NOT authoritative by fiat.
CREATE TABLE IF NOT EXISTS rule_edges (
    rule_id              TEXT PRIMARY KEY,       -- UUIDv7
    parent_variable_id   TEXT NOT NULL REFERENCES variables(variable_id),
    child_variable_id    TEXT NOT NULL REFERENCES variables(variable_id),
    edge_type            TEXT NOT NULL CHECK (edge_type IN
                           ('causal_promotes', 'causal_inhibits', 'requires')),
    direction_confidence TEXT NOT NULL,           -- low|medium|high
    lag_min_seconds      REAL,
    lag_max_seconds      REAL,
    persistence          TEXT NOT NULL,           -- instantaneous|transient|cumulative|permanent
    scope                TEXT NOT NULL,           -- generic|context_specific
    context_qualifier    TEXT,                    -- JSON; NULL for generic rules
    source               TEXT NOT NULL,
    source_evidence_rids TEXT NOT NULL DEFAULT '[]',  -- JSON array
    namespace            TEXT NOT NULL,
    tombstoned           INTEGER NOT NULL DEFAULT 0,
    created_at           REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rule_parent ON rule_edges(parent_variable_id);
CREATE INDEX IF NOT EXISTS idx_rule_child ON rule_edges(child_variable_id);
CREATE INDEX IF NOT EXISTS idx_rule_type ON rule_edges(edge_type);

-- Layer 5 — Scenario specs: saved assumption sets. Scenario execution
-- itself is request-scoped and in-memory — this table only persists
-- assumption lists so the same what-if can be re-run later.
-- DO NOT store derived results here; always recompute from current base.
CREATE TABLE IF NOT EXISTS scenario_specs (
    spec_id        TEXT PRIMARY KEY,              -- UUIDv7
    name           TEXT NOT NULL,
    namespace      TEXT NOT NULL,
    assumptions    TEXT NOT NULL,                 -- JSON array of overrides
    created_by     TEXT,
    engine_version TEXT,
    created_at     REAL NOT NULL,
    UNIQUE(name, namespace)
);
CREATE INDEX IF NOT EXISTS idx_scenario_ns ON scenario_specs(namespace);

-- ──────────────────────────────────────────────────────────────────
-- End RFC 007 Phase 0 tables. Claims table below gets a proposition_id FK.
-- ──────────────────────────────────────────────────────────────────

-- ──────────────────────────────────────────────────────────────────
-- RFC 008 Phase 1: Warrant Flow — the control stack foundations.
-- Scalar confidence is dead. These tables implement the 13-dim mobility
-- calculus that replaces it, plus the actor-profile layer that calibrates
-- every epistemic actor (sources, extractors, moves, agents, self-modes),
-- plus the compression-artifact layer with reversible loss accounting.
--
-- Architecture doc: Saga notes §§ 10-12 on Epic 35.
-- ──────────────────────────────────────────────────────────────────

-- Mobility state: the 13-dim vector M(c|ρ) keyed by (proposition, regime).
-- NOT a confidence score. Represents how the claim's warrant is moving
-- through its epistemic neighborhood. All components are optional because
-- they are materialized at different tiers (write/read/background) — see
-- the `tier_*_fresh` columns for which components are currently authoritative.
-- snapshot_ts lets background consolidation produce derived facts without
-- overwriting writes that happened while the job was running.
CREATE TABLE IF NOT EXISTS mobility_state (
    proposition_id      TEXT NOT NULL REFERENCES propositions(proposition_id),
    regime              TEXT NOT NULL DEFAULT 'default',
    snapshot_ts         REAL NOT NULL,
    -- 13-dim mobility components (all nullable, filled per tier)
    support_mass            REAL,  -- σ: sum of weighted support from evidence
    attack_mass             REAL,  -- α: sum of weighted attacks
    source_diversity        REAL,  -- δ: entropy-ish over source families
    effective_independence  REAL,  -- ι: dependence-discounted support
    temporal_coherence      REAL,  -- τ: polarity persistence across time
    transportability        REAL,  -- γ: cross-regime stability
    mutability              REAL,  -- μ: ease of revision under plausible evidence
    load_bearingness        REAL,  -- λ: downstream dependency weight
    modality_consilience    REAL,  -- χ: cross-modal independent corroboration
    self_gen_local          REAL,  -- ψ_l: fraction of immediate support self-generated
    self_gen_ancestral      REAL,  -- ψ_a: fraction of ancestry self-generated
    contamination_risk      REAL,  -- κ: shared-pipeline / dependency-collapse risk
    novelty_isolation       REAL,  -- ν: isolation from established graph neighborhoods
    -- Tier freshness flags — bit semantics TBD, using TEXT for now for legibility
    tier_write_components   TEXT NOT NULL DEFAULT '[]',  -- JSON array of component names
    tier_read_components    TEXT NOT NULL DEFAULT '[]',
    tier_bg_components      TEXT NOT NULL DEFAULT '[]',
    -- M3 additions (V21): reproducible-state discipline for write-tier recompute.
    -- content_hash is a sha256 over (formula_version || sorted claim_ids ||
    -- sorted per-dim lineage elements || polarity flags). If the hash of the
    -- current live claim set matches, the recompute is a no-op (idempotent).
    -- formula_version lets us retire stale rows when the math changes.
    -- state_status tracks liveness of the row itself: 'fresh' after recompute,
    -- 'recomputing' while async, 'failed' on error, 'stale_formula' when the
    -- row was written under an older formula version.
    formula_version         INTEGER NOT NULL DEFAULT 1,
    content_hash            TEXT NOT NULL DEFAULT '',
    live_claim_count        INTEGER NOT NULL DEFAULT 0,
    state_status            TEXT NOT NULL DEFAULT 'stale_formula'
        CHECK (state_status IN ('fresh', 'recomputing', 'failed', 'stale_formula')),
    computed_at             INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (proposition_id, regime, snapshot_ts)
);
CREATE INDEX IF NOT EXISTS idx_mobility_prop ON mobility_state(proposition_id);
CREATE INDEX IF NOT EXISTS idx_mobility_regime ON mobility_state(regime);
CREATE INDEX IF NOT EXISTS idx_mobility_status ON mobility_state(state_status);

-- RFC 008 Phase 1 M4 (V22): Contest state Γ(c). The contest operator ⋈
-- produces a compact, reproducible summary of the SHAPE of contest across
-- the live claim set — grounded diagnostic features only, NOT speculative
-- contradiction semantics. Per M4 locked spec (Saga note 16), we store
-- only features that are (a) reliably inferable from claim metadata,
-- (b) cheap to compute, and (c) tied to a concrete downstream consumer.
--
-- Current-state overwrite semantics (no timeline). Own derivation_version
-- and content_hash — contest logic evolves independently of mobility.
CREATE TABLE IF NOT EXISTS contest_state (
    proposition_id  TEXT NOT NULL REFERENCES propositions(proposition_id),
    regime          TEXT NOT NULL DEFAULT 'default',
    -- Polarity aggregates (same leave-one-out ⊕ math as mobility_state,
    -- same snapshot because recomputed inside the same lock scope).
    support_mass                   REAL NOT NULL DEFAULT 0.0,
    attack_mass                    REAL NOT NULL DEFAULT 0.0,
    support_effective_independence REAL NOT NULL DEFAULT 0.0,  -- Σ ω_k over supports
    attack_effective_independence  REAL NOT NULL DEFAULT 0.0,  -- Σ ω_k over attacks
    support_distinct_source_count  INTEGER NOT NULL DEFAULT 0,
    attack_distinct_source_count   INTEGER NOT NULL DEFAULT 0,
    -- Grounded contest diagnostics — strict gates, bounded computation
    same_source_opposite_polarity_count              INTEGER NOT NULL DEFAULT 0,
    same_artifact_extractor_polarity_conflict_count  INTEGER NOT NULL DEFAULT 0,
    temporal_overlap_conflict_count                  INTEGER NOT NULL DEFAULT 0,
    temporal_separable_opposition_count              INTEGER NOT NULL DEFAULT 0,
    referent_schema_heterogeneity_count              INTEGER NOT NULL DEFAULT 0,
    -- Heuristic flags bitset:
    --   Bit 0 DUPLICATION_RISK: support_mass > 2.0 AND support_effective_independence < 2.0
    --   Bit 1 SAME_SOURCE_CONFLICT: same_source_opposite_polarity_count > 0
    --   Bit 2 REFERENT_HETEROGENEITY_PRESENT: referent_schema_heterogeneity_count > 0
    --   Bit 3 SAME_ARTIFACT_EXTRACTOR_CONFLICT: same_artifact_extractor_polarity_conflict_count > 0
    --   Bit 4 PRESENT_TENSE_CONFLICT: temporal_overlap_conflict_count > 0
    heuristic_flags INTEGER NOT NULL DEFAULT 0,
    -- Reproducibility
    derivation_version INTEGER NOT NULL DEFAULT 1,
    content_hash       TEXT NOT NULL DEFAULT '',
    live_claim_count   INTEGER NOT NULL DEFAULT 0,
    state_status       TEXT NOT NULL DEFAULT 'stale_formula'
        CHECK (state_status IN ('fresh', 'recomputing', 'failed', 'stale_formula')),
    computed_at        INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (proposition_id, regime)
);
CREATE INDEX IF NOT EXISTS idx_contest_flags ON contest_state(heuristic_flags);
CREATE INDEX IF NOT EXISTS idx_contest_status ON contest_state(state_status);

-- ──────────────────────────────────────────────────────────────────
-- RFC 008 M5b (V23): Cognitive moves — the spine of reasoning.
--
-- Per M5a locked spec (Saga note 19): move_events is an append-only log
-- of reasoning transformations. Inputs/outputs/side-effects are stored
-- in normalized edge tables for indexed lookup. Corrections are first-
-- class events; originals are never mutated for semantic correction.
-- Adversarial instances are staged (candidate/confirmed/rejected) with
-- governance enforced at the API layer.
--
-- move_type is intentionally unconstrained at DB level — a soft
-- registry (move_type_registry) holds the canonical vocabulary but
-- does NOT reject unknown types. Observability lifecycle is enforced
-- by CHECK constraints since those values are definitional.
-- ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS move_events (
    move_id                       TEXT PRIMARY KEY,
    move_type                     TEXT NOT NULL,
    operator_version              TEXT NOT NULL,
    actor_id                      TEXT NOT NULL,
    context_regime                TEXT NOT NULL DEFAULT 'default',
    observability                 TEXT NOT NULL
        CHECK (observability IN ('observed', 'self_reported', 'inferred')),
    inference_confidence          REAL,
    inference_basis_json          TEXT,
    dependencies_json             TEXT NOT NULL DEFAULT '[]',
    cost_tokens                   INTEGER,
    cost_latency_ms               INTEGER,
    cost_memory_reads             INTEGER,
    yield_json                    TEXT NOT NULL DEFAULT '{}',
    posthoc_outcome               TEXT
        CHECK (posthoc_outcome IN ('corroborated', 'retracted', 'harmful_side_effect') OR posthoc_outcome IS NULL),
    posthoc_recorded_at           REAL,
    expected_evaluation_horizon_ms INTEGER,
    mobility_state_hash_at_move   TEXT,
    contest_state_hash_at_move    TEXT,
    created_at                    REAL NOT NULL,
    hlc                           BLOB NOT NULL,
    origin_actor                  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_move_type_time ON move_events(move_type, created_at);
CREATE INDEX IF NOT EXISTS idx_move_actor_time ON move_events(actor_id, created_at);
CREATE INDEX IF NOT EXISTS idx_move_regime_time ON move_events(context_regime, created_at);

CREATE TABLE IF NOT EXISTS move_input_edge (
    move_id    TEXT NOT NULL REFERENCES move_events(move_id),
    claim_id   TEXT NOT NULL,
    input_role TEXT NOT NULL DEFAULT 'input',
    ordinal    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (move_id, claim_id, input_role)
);
CREATE INDEX IF NOT EXISTS idx_move_input_claim ON move_input_edge(claim_id);

CREATE TABLE IF NOT EXISTS move_output_edge (
    move_id     TEXT NOT NULL REFERENCES move_events(move_id),
    claim_id    TEXT NOT NULL,
    output_role TEXT NOT NULL DEFAULT 'output',
    ordinal     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (move_id, claim_id, output_role)
);
CREATE INDEX IF NOT EXISTS idx_move_output_claim ON move_output_edge(claim_id);

CREATE TABLE IF NOT EXISTS move_side_effect_edge (
    move_id     TEXT NOT NULL REFERENCES move_events(move_id),
    claim_id    TEXT NOT NULL,
    effect_kind TEXT NOT NULL,
    PRIMARY KEY (move_id, claim_id, effect_kind)
);
CREATE INDEX IF NOT EXISTS idx_move_side_effect_claim ON move_side_effect_edge(claim_id);

CREATE TABLE IF NOT EXISTS move_correction_event (
    correction_id              TEXT PRIMARY KEY,
    original_move_id           TEXT NOT NULL REFERENCES move_events(move_id),
    corrected_move_type        TEXT,
    corrected_operator_version TEXT,
    corrected_context_regime   TEXT,
    correction_reason          TEXT NOT NULL,
    corrected_by_actor_id      TEXT NOT NULL,
    corrected_at               REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_correction_original ON move_correction_event(original_move_id);

CREATE TABLE IF NOT EXISTS move_adversarial_instance (
    instance_id        TEXT PRIMARY KEY,
    move_id            TEXT NOT NULL REFERENCES move_events(move_id),
    status             TEXT NOT NULL
        CHECK (status IN ('candidate', 'confirmed', 'rejected')),
    discovered_via     TEXT NOT NULL
        CHECK (discovered_via IN ('contradiction', 'retraction', 'calibration_signal', 'human_audit')),
    traced_root_cause  TEXT,
    generalized_lesson TEXT,
    lesson_scope_json  TEXT,
    curation_actor_id  TEXT,
    discovered_at      REAL NOT NULL,
    created_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_adv_move ON move_adversarial_instance(move_id);
CREATE INDEX IF NOT EXISTS idx_adv_status ON move_adversarial_instance(status);
CREATE INDEX IF NOT EXISTS idx_adv_discovered_via ON move_adversarial_instance(discovered_via);

CREATE TABLE IF NOT EXISTS move_type_registry (
    move_type                             TEXT PRIMARY KEY,
    status                                TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('proposed', 'active', 'deprecated')),
    description                           TEXT,
    introduced_at                         REAL NOT NULL,
    deprecated_at                         REAL,
    default_expected_evaluation_horizon_ms INTEGER
);

CREATE TABLE IF NOT EXISTS inference_basis_registry (
    basis_type  TEXT PRIMARY KEY,
    description TEXT,
    status      TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('proposed', 'active', 'deprecated'))
);

CREATE TABLE IF NOT EXISTS move_composition_rule (
    rule_id                TEXT PRIMARY KEY,
    left_move_type         TEXT NOT NULL,
    right_move_type        TEXT NOT NULL,
    left_operator_version  TEXT,
    right_operator_version TEXT,
    context_regime         TEXT,
    rule_kind              TEXT NOT NULL
        CHECK (rule_kind IN ('commutative', 'non_commutative', 'idempotent',
                             'precondition_violation', 'approx_identity')),
    precondition_json      TEXT,
    evidence_basis_json    TEXT,
    provenance             TEXT NOT NULL
        CHECK (provenance IN ('empirical', 'user_declared', 'inferred')),
    confidence             REAL NOT NULL DEFAULT 0.5,
    created_at             REAL NOT NULL,
    superseded_at          REAL
);
CREATE INDEX IF NOT EXISTS idx_comp_rule_types ON move_composition_rule(left_move_type, right_move_type);
CREATE INDEX IF NOT EXISTS idx_comp_rule_regime ON move_composition_rule(context_regime);

CREATE TABLE IF NOT EXISTS move_type_profile (
    move_type                       TEXT NOT NULL,
    operator_version                TEXT NOT NULL,
    context_regime                  TEXT NOT NULL,
    uses_count                      INTEGER NOT NULL DEFAULT 0,
    resolved_count                  INTEGER NOT NULL DEFAULT 0,
    corroborated_count              INTEGER NOT NULL DEFAULT 0,
    retracted_count                 INTEGER NOT NULL DEFAULT 0,
    harmful_side_effect_count       INTEGER NOT NULL DEFAULT 0,
    contradiction_introduction_rate REAL,
    avg_mobility_shift              REAL,
    predictive_gain_avg             REAL,
    calibration_gain_avg            REAL,
    last_updated                    REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (move_type, operator_version, context_regime)
);

-- Actor profile: calibration record for any epistemic actor — external
-- sources, extractors, summarizers, internal cognitive moves, other
-- agents, or specific self-modes. Regime-indexed because reliability is
-- local (an extractor may be precise in legal text and noisy in medical).
-- Updated by the closed-loop calibration job from downstream outcomes.
CREATE TABLE IF NOT EXISTS actor_profile (
    actor_id                 TEXT NOT NULL,
    actor_type               TEXT NOT NULL,
    -- Allowed actor_type values:
    --   'source'         — external data source
    --   'extractor'      — parser/NER/claim-extraction pipeline
    --   'summarizer'     — compression/consolidation operator
    --   'cognitive_move' — reasoning transform (analogy, decomposition, ...)
    --   'self_mode'      — agent's own reasoning mode
    --   'agent'          — peer agent in a federation
    regime                   TEXT NOT NULL DEFAULT 'default',
    -- Performance signature (not a single trust score)
    corroboration_rate       REAL,  -- fraction of claims later corroborated
    contradiction_hazard     REAL,  -- fraction later contradicted
    independence_contribution REAL, -- avg independence of claims from this actor
    latency_p50_ms           REAL,
    latency_p99_ms           REAL,
    repairability            REAL,  -- likelihood failures are recoverable
    bias_signature           TEXT,  -- JSON: structured bias metadata
    value_alignment_risk     REAL,  -- for meta-actors
    -- Update tracking
    last_updated             REAL NOT NULL,
    update_count             INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (actor_id, regime),
    CHECK (actor_type IN ('source', 'extractor', 'summarizer',
                          'cognitive_move', 'self_mode', 'agent'))
);
CREATE INDEX IF NOT EXISTS idx_actor_type ON actor_profile(actor_type);
CREATE INDEX IF NOT EXISTS idx_actor_updated ON actor_profile(last_updated);

-- Compression artifact: a summary/consolidation of some source span, with
-- REVERSIBLE LOSS ACCOUNTING. For month-scale minds, compression is forced
-- and silent compression is silent insanity. Each artifact tracks:
--   - what raw strata it covers (so queries can fall back on demand)
--   - what operator produced it (for re-run)
--   - what is known to be lost vs preserved
--   - compression_drift_score: divergence in downstream decisions between
--     using the artifact vs raw strata (computed against replay samples)
-- If drift exceeds threshold, artifact is demoted (status='demoted') and
-- queries fall back to raw strata until it's rebuilt.
CREATE TABLE IF NOT EXISTS compression_artifact (
    artifact_id              TEXT PRIMARY KEY,
    source_span_json         TEXT NOT NULL,     -- JSON: {rids, propositions, time_range, ...}
    abstraction_operator     TEXT NOT NULL,     -- which operator produced this
    operator_version         TEXT,
    known_omissions          TEXT NOT NULL DEFAULT '[]',  -- JSON list
    uncertainty_distortion   REAL,              -- estimated per-dim distortion (L2 of M deltas)
    dependency_impact        REAL,              -- how many downstream propositions rely on it
    reversibility_pointer    TEXT NOT NULL,     -- pointer to raw strata for fallback
    compression_drift_score  REAL NOT NULL DEFAULT 0.0,  -- computed by BG job
    status                   TEXT NOT NULL DEFAULT 'active',
    -- Allowed status values: 'active' | 'demoted' | 'expired' | 'rebuilding'
    namespace                TEXT NOT NULL,
    created_at               REAL NOT NULL,
    last_drift_check_at      REAL,
    CHECK (status IN ('active', 'demoted', 'expired', 'rebuilding'))
);
CREATE INDEX IF NOT EXISTS idx_compression_ns ON compression_artifact(namespace);
CREATE INDEX IF NOT EXISTS idx_compression_status ON compression_artifact(status);

-- ──────────────────────────────────────────────────────────────────
-- End RFC 008 Phase 1 tables. Write-time mobility signals on claims below.
-- ──────────────────────────────────────────────────────────────────

-- Claims: first-class semantic relationship ledger (RFC 006 Phase 5)
-- Each claim records a structured (subject, relation, object) triple.
-- The legacy 'edges' name is preserved as a read-only VIEW for backward compat.
CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,           -- UUIDv7
    src TEXT NOT NULL,                   -- entity name or memory rid
    dst TEXT NOT NULL,                   -- entity name or memory rid
    rel_type TEXT NOT NULL,              -- relationship type (e.g., \"ceo_of\", \"works_at\")
    weight REAL NOT NULL DEFAULT 1.0,    -- relationship strength [0, 1]
    created_at REAL NOT NULL,
    tombstoned INTEGER NOT NULL DEFAULT 0,
    -- RFC 006 claim qualifiers
    polarity INTEGER NOT NULL DEFAULT 1,           -- 1=positive, -1=negative, 0=unknown
    modality TEXT NOT NULL DEFAULT 'asserted',      -- asserted|reported|hypothetical|denied|quoted
    valid_from REAL,                                -- world-validity start (nullable)
    valid_to REAL,                                  -- world-validity end (null=present)
    extractor TEXT NOT NULL DEFAULT 'manual',       -- manual|structured_ingest|heuristic_v1|agent_llm
    extractor_version TEXT,
    confidence_band TEXT NOT NULL DEFAULT 'medium', -- low|medium|high
    source_memory_rid TEXT,                         -- provenance: which memory spawned this claim
    span_start INTEGER,                             -- byte offset in source memory text
    span_end INTEGER,
    namespace TEXT NOT NULL DEFAULT 'default',

    -- RFC 007 Phase 0: canonical proposition FK. Populated on insert (or by
    -- V18→V19 backfill for existing rows). Propositions are the canonical
    -- identity for (src, rel_type, dst, namespace) tuples across all evidence.
    proposition_id TEXT REFERENCES propositions(proposition_id),

    -- RFC 008 Phase 1: write-time mobility signals. These are the components
    -- of the mobility state M(c|ρ) that can be computed in <10ms on ingest
    -- without a graph walk. The full 13-dim state is aggregated at the
    -- proposition level in `mobility_state`; these are the per-claim inputs.
    regime_tag       TEXT NOT NULL DEFAULT 'default',
    self_generated   INTEGER NOT NULL DEFAULT 0,  -- ψ_l contribution: did this claim come from self-reasoning?
    source_lineage   TEXT NOT NULL DEFAULT '[]',   -- JSON: pipeline chain (source, extractor, summarizer, ...)
    modality_signal  TEXT NOT NULL DEFAULT 'text', -- contribution to χ: 'text'|'image'|'numeric'|'audio'|'code'|'telemetry'

    -- RFC 006: multiple sources can make conflicting claims about the same (src, rel, dst).
    -- Uniqueness is scoped to (src, dst, rel, extractor, polarity, namespace) so
    -- e.g. witness A can claim \"X did Y\" while witness B claims \"X did NOT do Y\" and
    -- both rows coexist. This enables polarity contradiction detection.
    UNIQUE(src, dst, rel_type, extractor, polarity, namespace)
);
CREATE INDEX IF NOT EXISTS idx_claims_proposition ON claims(proposition_id);

-- Backward-compatible VIEW: all code reading FROM edges continues to work.
CREATE VIEW IF NOT EXISTS edges AS
    SELECT claim_id AS edge_id, src, dst, rel_type, weight, created_at, tombstoned,
           polarity, modality, valid_from, valid_to, extractor, extractor_version,
           confidence_band, source_memory_rid, span_start, span_end, namespace
    FROM claims;

-- Entity aliases for alias-aware conflict detection (RFC 006 Layer B)
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'default',
    source TEXT NOT NULL DEFAULT 'explicit',  -- explicit|auto_suggested|approved
    created_at REAL NOT NULL,
    PRIMARY KEY (alias, namespace)
);
CREATE INDEX IF NOT EXISTS idx_alias_canonical ON entity_aliases(canonical_name, namespace);

-- Relation conflict policies (RFC 006 Phase 3)
-- Per-relation rules that govern how the conflict scanner treats claims.
CREATE TABLE IF NOT EXISTS relation_policies (
    relation_type TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT '*',           -- '*' = global default
    uniqueness_scope TEXT NOT NULL DEFAULT '[\"dst\"]', -- JSON: which fields define uniqueness
    overlap_allowed INTEGER NOT NULL DEFAULT 0,    -- 1 if multiple dst values are normal
    temporal_required INTEGER NOT NULL DEFAULT 0,  -- 1 if conflict needs temporal overlap
    missing_time_severity TEXT NOT NULL DEFAULT 'medium', -- low|medium|high
    qualifier_exceptions TEXT,                     -- JSON: e.g. [\"qualifier=co\", \"qualifier=interim\"]
    PRIMARY KEY (relation_type, namespace)
);

-- Entities extracted from memories
CREATE TABLE IF NOT EXISTS entities (
    name TEXT PRIMARY KEY,               -- normalized entity name
    entity_type TEXT DEFAULT 'unknown',  -- person | place | thing | concept | etc.
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    metadata TEXT DEFAULT '{}'
);

-- Append-only operation log (CRDT replication)
CREATE TABLE IF NOT EXISTS oplog (
    op_id TEXT PRIMARY KEY,              -- UUIDv7
    op_type TEXT NOT NULL,               -- record | relate | consolidate | decay | forget | update
    timestamp REAL NOT NULL,             -- when the operation occurred
    target_rid TEXT,                     -- primary memory affected
    payload TEXT NOT NULL DEFAULT '{}',  -- JSON: full operation details
    actor_id TEXT DEFAULT 'local',       -- device/agent identifier
    hlc BLOB,                           -- hybrid logical clock timestamp (16 bytes)
    embedding_hash BLOB,                -- BLAKE3 hash of embedding (if applicable)
    origin_actor TEXT NOT NULL DEFAULT 'local', -- which device originally created this op
    applied INTEGER NOT NULL DEFAULT 1, -- 1 = materialized locally, 0 = pending
    embedding BLOB,                     -- v24: full embedding bytes for ingest replay (NULL for non-record ops)

    -- v27 (issue #41): name of the embedder that produced the embedding
    -- bytes above. Used by the materializer drain post-reembed-swap to
    -- detect ops queued under the old embedder and re-encode them from
    -- text. Pre-v27 ops have NULL here; materializer treats NULL as the
    -- trust-embedding-as-is fallback for back-compat. Reembed Queue-mode
    -- writes set this column on log_op_pending.
    embedding_model TEXT,

    -- v27 (issue #41, brainstorm-2 correction): per-generation application
    -- tracking. Boolean `applied` above is ambiguous during reembed
    -- (applied to OLD generation index? new? logical DB only?), so the
    -- post-swap materializer cannot trust it. This column carries the
    -- index generation the op was applied to. NULL means
    -- never-applied-to-any-generation; the v27 materializer treats
    -- NULL OR (applied_generation < current_generation) as needs-replay.
    -- Boolean `applied` is kept as a derived hint for back-compat but is
    -- no longer the truth.
    applied_generation INTEGER
);
-- v27 index for the post-swap materializer query. Without this index,
-- post-swap drain is a full oplog scan on every materializer wakeup.
CREATE INDEX IF NOT EXISTS idx_oplog_applied_generation
    ON oplog(applied_generation, op_id);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- v27 (issue #41): durable audit log of db.reembed() phase transitions.
-- Authoritative source for crash recovery and observability. The
-- in-memory on_phase_complete callback in ReembedOptions is best-effort
-- only; this table is what the engine reads on open() to decide
-- whether/how to resume an interrupted reembed. One row per (generation,
-- phase) pair; phase strings are 'Probing' | 'Encoding' | 'Rebuilding'
-- | 'Swapping' | 'Verifying' | 'Aborted' | 'Completed'.
CREATE TABLE IF NOT EXISTS reembed_events (
    generation   INTEGER NOT NULL,
    phase        TEXT NOT NULL,
    timestamp    REAL NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_reembed_events_generation ON reembed_events(generation);

-- v29 (2026-05-20 postmortem): replication apply audit log.
-- Replication-apply paths INSERT a row here so audit queries
-- distinguish memories received via replication from true orphans.
CREATE TABLE IF NOT EXISTS replication_apply_log (
    rid           TEXT PRIMARY KEY,
    op_type       TEXT NOT NULL,
    source_actor  TEXT NOT NULL,
    applied_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_replication_apply_log_source_actor
    ON replication_apply_log(source_actor, applied_at);

-- v30 (issue #47): record_revisions — append-only audit log of correct()
-- calls. The new correct() mutates memories in place (preserves rid +
-- created_at) and writes the prior state here. revision_num starts at 1
-- and increments per-rid. reason is required (engine-enforced).
CREATE TABLE IF NOT EXISTS record_revisions (
    revision_id       TEXT PRIMARY KEY,
    rid               TEXT NOT NULL,
    revision_num      INTEGER NOT NULL,
    prior_text        TEXT NOT NULL,
    prior_metadata    TEXT NOT NULL,
    prior_importance  REAL NOT NULL,
    prior_valence     REAL NOT NULL,
    reason            TEXT NOT NULL,
    applied_at        REAL NOT NULL,
    hlc               BLOB NOT NULL,
    origin_actor      TEXT NOT NULL,
    UNIQUE(rid, revision_num)
);
CREATE INDEX IF NOT EXISTS idx_record_revisions_rid
    ON record_revisions(rid, revision_num);

-- v31 (issue #48): record_links — first-class record-to-record links.
-- Distinct from the entity graph (`claims`): rid-specific semantics, a
-- small closed set of link_types, and a forget()-aware lifecycle status.
-- Atomic with the write via record_with_links(). See
-- docs/record_link_model_rfc.md. Column order/naming deliberately mirror
-- a future typed graph_edges row so this can fold into a unified graph
-- later without a semantic rewrite.
CREATE TABLE IF NOT EXISTS record_links (
    link_id        TEXT PRIMARY KEY,
    source_rid     TEXT NOT NULL,
    target_rid     TEXT NOT NULL,
    link_type      TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'active',
    created_at     REAL NOT NULL,
    hlc            BLOB NOT NULL,
    origin_actor   TEXT NOT NULL,
    UNIQUE(source_rid, target_rid, link_type)
);
CREATE INDEX IF NOT EXISTS idx_record_links_source
    ON record_links(source_rid, link_type, status);
CREATE INDEX IF NOT EXISTS idx_record_links_target
    ON record_links(target_rid, link_type, status);

-- Peer tracking for delta sync
CREATE TABLE IF NOT EXISTS sync_peers (
    peer_actor TEXT PRIMARY KEY,
    last_synced_hlc BLOB NOT NULL,
    last_synced_op_id TEXT NOT NULL,
    last_sync_time REAL NOT NULL
);

-- Consolidation membership (set-union CRDT)
CREATE TABLE IF NOT EXISTS consolidation_members (
    consolidation_rid TEXT NOT NULL,     -- the consolidated memory
    source_rid TEXT NOT NULL,            -- original memory
    hlc BLOB NOT NULL,                  -- when this consolidation happened
    actor_id TEXT NOT NULL,             -- which device did it
    PRIMARY KEY (consolidation_rid, source_rid)
);

-- Conflict tracking (first-class data)
CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id TEXT PRIMARY KEY,           -- UUIDv7
    conflict_type TEXT NOT NULL,            -- identity_fact | preference | temporal | consolidation | minor
    priority TEXT NOT NULL DEFAULT 'medium',-- low | medium | high | critical
    status TEXT NOT NULL DEFAULT 'open',    -- open | resolved | dismissed
    memory_a TEXT NOT NULL,                 -- rid of first conflicting memory
    memory_b TEXT NOT NULL,                 -- rid of second conflicting memory
    entity TEXT,                            -- entity name (nullable)
    rel_type TEXT,                          -- relationship type in conflict (nullable)
    detected_at REAL NOT NULL,
    detected_by TEXT NOT NULL,              -- actor_id that detected it
    detection_reason TEXT NOT NULL,
    resolved_at REAL,
    resolved_by TEXT,
    strategy TEXT,                          -- keep_a | keep_b | keep_both | merge | correct
    winner_rid TEXT,
    resolution_note TEXT,
    hlc BLOB NOT NULL,
    origin_actor TEXT NOT NULL
);

-- Persisted triggers with lifecycle tracking
CREATE TABLE IF NOT EXISTS trigger_log (
    trigger_id TEXT PRIMARY KEY,
    trigger_type TEXT NOT NULL,
    urgency REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    reason TEXT NOT NULL,
    suggested_action TEXT NOT NULL,
    source_rids TEXT NOT NULL DEFAULT '[]',
    context TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    delivered_at REAL,
    acknowledged_at REAL,
    acted_at REAL,
    expires_at REAL,
    cooldown_key TEXT,
    hlc BLOB NOT NULL,
    origin_actor TEXT NOT NULL
);

-- Detected patterns across memories
CREATE TABLE IF NOT EXISTS patterns (
    pattern_id TEXT PRIMARY KEY,
    pattern_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    confidence REAL NOT NULL,
    description TEXT NOT NULL,
    evidence_rids TEXT NOT NULL DEFAULT '[]',
    entity_names TEXT NOT NULL DEFAULT '[]',
    context TEXT NOT NULL DEFAULT '{}',
    first_seen REAL NOT NULL,
    last_confirmed REAL NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    hlc BLOB NOT NULL,
    origin_actor TEXT NOT NULL
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_consolidation ON memories(consolidation_status);
CREATE INDEX IF NOT EXISTS idx_memories_storage_tier ON memories(storage_tier);
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
CREATE INDEX IF NOT EXISTS idx_memories_access_count ON memories(access_count);
-- v32 (structural query / list_records): secondary indexes over the generated
-- metadata columns, so list-by-kind / FK-by-drive_id are O(log n) index walks
-- instead of full-table json_extract scans.
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_drive_id ON memories(drive_id);
CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
CREATE INDEX IF NOT EXISTS idx_memories_emotional_state ON memories(emotional_state);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(namespace, session_id);
CREATE INDEX IF NOT EXISTS idx_memories_due_at ON memories(namespace, due_at) WHERE due_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_last_access ON memories(last_access);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_one_active ON sessions(namespace, client_id) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_sessions_client_started ON sessions(namespace, client_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_claims_src ON claims(src);
CREATE INDEX IF NOT EXISTS idx_claims_dst ON claims(dst);
CREATE INDEX IF NOT EXISTS idx_claims_rel ON claims(rel_type);
CREATE INDEX IF NOT EXISTS idx_oplog_timestamp ON oplog(timestamp);
CREATE INDEX IF NOT EXISTS idx_oplog_target ON oplog(target_rid);
CREATE INDEX IF NOT EXISTS idx_oplog_hlc ON oplog(hlc);
CREATE INDEX IF NOT EXISTS idx_oplog_actor ON oplog(origin_actor);
-- v25 (RFC issue #9): cluster-replication determinism column indexes
CREATE INDEX IF NOT EXISTS idx_memories_created_at_micros ON memories(created_at_unix_micros);
CREATE INDEX IF NOT EXISTS idx_memories_embedding_model ON memories(embedding_model) WHERE embedding_model IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_consolidation_source ON consolidation_members(source_rid);
CREATE INDEX IF NOT EXISTS idx_conflicts_status ON conflicts(status);
CREATE INDEX IF NOT EXISTS idx_conflicts_type ON conflicts(conflict_type);
CREATE INDEX IF NOT EXISTS idx_conflicts_priority ON conflicts(priority);
CREATE INDEX IF NOT EXISTS idx_conflicts_entity ON conflicts(entity);
CREATE INDEX IF NOT EXISTS idx_conflicts_memory_a ON conflicts(memory_a);
CREATE INDEX IF NOT EXISTS idx_conflicts_memory_b ON conflicts(memory_b);
CREATE INDEX IF NOT EXISTS idx_trigger_log_status ON trigger_log(status);
CREATE INDEX IF NOT EXISTS idx_trigger_log_type ON trigger_log(trigger_type);
CREATE INDEX IF NOT EXISTS idx_trigger_log_created ON trigger_log(created_at);
CREATE INDEX IF NOT EXISTS idx_trigger_log_cooldown ON trigger_log(cooldown_key);
CREATE INDEX IF NOT EXISTS idx_trigger_log_urgency ON trigger_log(urgency DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_status ON patterns(status);
CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON patterns(confidence DESC);

-- Memory-entity join table for graph-augmented recall
CREATE TABLE IF NOT EXISTS memory_entities (
    memory_rid TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    PRIMARY KEY (memory_rid, entity_name)
);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_name);
CREATE INDEX IF NOT EXISTS idx_memory_entities_rid ON memory_entities(memory_rid);

-- FTS5 for full-text search on memories
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text, content=memories, content_rowid=rowid);

-- Auto-sync triggers for FTS5
CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_delete BEFORE DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE OF text ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
END;

-- Normalized join tables for trigger/pattern JSON arrays
CREATE TABLE IF NOT EXISTS trigger_source_rids (
    trigger_id TEXT NOT NULL,
    rid TEXT NOT NULL,
    PRIMARY KEY (trigger_id, rid)
);
CREATE INDEX IF NOT EXISTS idx_trigger_source_rids_rid ON trigger_source_rids(rid);

CREATE TABLE IF NOT EXISTS pattern_evidence (
    pattern_id TEXT NOT NULL,
    rid TEXT NOT NULL,
    PRIMARY KEY (pattern_id, rid)
);
CREATE INDEX IF NOT EXISTS idx_pattern_evidence_rid ON pattern_evidence(rid);

CREATE TABLE IF NOT EXISTS pattern_entities (
    pattern_id TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    PRIMARY KEY (pattern_id, entity_name)
);
CREATE INDEX IF NOT EXISTS idx_pattern_entities_entity ON pattern_entities(entity_name);

-- Substitution categories for conflict detection (V14)
CREATE TABLE IF NOT EXISTS substitution_categories (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    conflict_mode TEXT NOT NULL DEFAULT 'exclusive',
    status TEXT NOT NULL DEFAULT 'active',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    hlc BLOB NOT NULL,
    origin_actor TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS substitution_members (
    id TEXT PRIMARY KEY,
    category_id TEXT NOT NULL REFERENCES substitution_categories(id),
    token_normalized TEXT NOT NULL,
    token_display TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    source TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    context_hint TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    hlc BLOB NOT NULL,
    origin_actor TEXT NOT NULL,
    UNIQUE(category_id, token_normalized)
);
CREATE INDEX IF NOT EXISTS idx_sub_members_token ON substitution_members(token_normalized);
CREATE INDEX IF NOT EXISTS idx_sub_members_category ON substitution_members(category_id);
CREATE INDEX IF NOT EXISTS idx_sub_members_source_status ON substitution_members(source, status);
CREATE INDEX IF NOT EXISTS idx_sub_categories_name ON substitution_categories(name);

-- Recall feedback for adaptive learning (V10)
CREATE TABLE IF NOT EXISTS recall_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT,
    query_embedding BLOB,
    rid TEXT NOT NULL,
    feedback TEXT NOT NULL,              -- 'relevant' | 'irrelevant'
    score_at_retrieval REAL,
    rank_at_retrieval INTEGER,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON recall_feedback(created_at);

-- Learned scoring weights (singleton row, V10)
CREATE TABLE IF NOT EXISTS learned_weights (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    w_sim REAL NOT NULL DEFAULT 0.50,
    w_decay REAL NOT NULL DEFAULT 0.20,
    w_recency REAL NOT NULL DEFAULT 0.30,
    gate_tau REAL NOT NULL DEFAULT 0.25,
    alpha_imp REAL NOT NULL DEFAULT 0.80,
    keyword_boost REAL NOT NULL DEFAULT 0.31,
    updated_at REAL,
    feedback_count INTEGER DEFAULT 0,
    generation INTEGER DEFAULT 0
);
INSERT OR IGNORE INTO learned_weights (id) VALUES (1);

-- Per-namespace importance distribution, for write-time importance
-- calibration (task 31). An EWMA of the raw importance writers request,
-- used to detect saturation (everything-marked-critical) and deflate
-- further high marks so the scale keeps headroom and 1.0 stays rare.
CREATE TABLE IF NOT EXISTS namespace_importance_stats (
    namespace TEXT PRIMARY KEY,
    ewma REAL NOT NULL,
    count INTEGER NOT NULL,
    updated_at REAL NOT NULL
);

-- Durable, auditable timeline of skill outcomes (task 28). The skill
-- registry keeps rolling aggregate counts; this records each individual
-- outcome event so effectiveness change is auditable and countable
-- (the skill_outcomes_recorded > 0 visibility the audit flagged).
CREATE TABLE IF NOT EXISTS skill_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dedup_key TEXT NOT NULL,
    outcome TEXT NOT NULL,              -- accepted | succeeded | failed | rejected
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_skill_outcomes_key ON skill_outcomes(dedup_key);

-- Conversation working-memory ring buffer (v0.9.0). Cheap, verbatim,
-- bounded FIFO of raw both-sides turns per namespace — short-term context,
-- distinct from semantic memory: NOT embedded and NOT kept forever (pruned to
-- the last N per namespace on insert). content is encrypted like memory text.
CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namespace TEXT NOT NULL,
    role TEXT NOT NULL,                 -- 'user' | 'assistant' | ...
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conversation_turns_ns ON conversation_turns(namespace, id);

-- Recall demand log (v0.9.0): cheap O(1)-per-recall aggregate of what gets
-- queried and how well it's answered, keyed by the NORMALIZED query. Lets the
-- substrate surface its own knowledge gaps — frequently-asked queries that
-- return little/nothing (high count, low avg top score). Bounded by distinct
-- query cardinality, not total recalls.
CREATE TABLE IF NOT EXISTS recall_demand (
    query_norm TEXT PRIMARY KEY,        -- normalized query (cluster key)
    sample_text TEXT NOT NULL,          -- a recent raw form, for display
    count INTEGER NOT NULL,             -- times asked
    sum_top_score REAL NOT NULL,        -- Σ best-hit score (avg = /count)
    sum_results INTEGER NOT NULL,       -- Σ result counts (avg = /count)
    last_seen REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_recall_demand_count ON recall_demand(count);

-- Task / chore store (v0.9.0): a minimal, GENERAL operational task primitive
-- — flat tasks with status, priority, and an optional parent for subtasks —
-- so an agent can maintain its chores in the same substrate as its memory,
-- cheaply (not embedded). Deliberately NOT a project/epic PM hierarchy; that
-- opinionated structure stays a convention on top. title encrypted like
-- memory text.
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    title TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',     -- open | in_progress | done | cancelled
    priority TEXT NOT NULL DEFAULT 'medium', -- low | medium | high | critical
    parent_id TEXT,                          -- optional subtask parent
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tasks_ns_status ON tasks(namespace, status);

-- Personality traits (V11)
CREATE TABLE IF NOT EXISTS personality_traits (
    trait_name TEXT PRIMARY KEY,
    score REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.0,
    sample_count INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0.0
);
INSERT OR IGNORE INTO personality_traits (trait_name, score, confidence, sample_count, updated_at)
    VALUES ('warmth', 0.5, 0.0, 0, 0.0),
           ('depth', 0.5, 0.0, 0, 0.0),
           ('energy', 0.5, 0.0, 0, 0.0),
           ('attentiveness', 0.5, 0.0, 0, 0.0);

-- Cognitive State Graph: Nodes (V12)
CREATE TABLE IF NOT EXISTS cognitive_nodes (
    node_id INTEGER PRIMARY KEY,            -- compact NodeId (4-bit kind + 28-bit seq)
    kind TEXT NOT NULL,                      -- node kind string (entity, belief, goal, etc.)
    label TEXT NOT NULL,                     -- human-readable label
    -- Universal cognitive attributes
    confidence REAL NOT NULL DEFAULT 0.5,
    activation REAL NOT NULL DEFAULT 0.0,
    salience REAL NOT NULL DEFAULT 0.5,
    persistence REAL NOT NULL DEFAULT 0.5,
    valence REAL NOT NULL DEFAULT 0.0,
    urgency REAL NOT NULL DEFAULT 0.0,
    novelty REAL NOT NULL DEFAULT 1.0,
    volatility REAL NOT NULL DEFAULT 0.1,
    provenance TEXT NOT NULL DEFAULT 'observed',
    evidence_count INTEGER NOT NULL DEFAULT 1,
    last_updated_ms INTEGER NOT NULL,
    -- Kind-specific payload (JSON)
    payload TEXT NOT NULL DEFAULT '{}',
    -- Metadata (JSON)
    metadata TEXT NOT NULL DEFAULT '{}',
    -- Lifecycle
    created_at REAL NOT NULL,
    tombstoned INTEGER NOT NULL DEFAULT 0,
    -- Replication
    hlc BLOB,
    origin_actor TEXT
);
CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_kind ON cognitive_nodes(kind);
CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_activation ON cognitive_nodes(activation);
CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_urgency ON cognitive_nodes(urgency);

-- Cognitive State Graph: Edges (V12)
CREATE TABLE IF NOT EXISTS cognitive_edges (
    src_id INTEGER NOT NULL,                 -- source NodeId
    dst_id INTEGER NOT NULL,                 -- destination NodeId
    kind TEXT NOT NULL,                      -- edge kind string (supports, contradicts, etc.)
    weight REAL NOT NULL DEFAULT 0.5,        -- edge weight [-1.0, 1.0]
    confidence REAL NOT NULL DEFAULT 0.5,
    observation_count INTEGER NOT NULL DEFAULT 1,
    created_at_ms INTEGER NOT NULL,
    last_confirmed_ms INTEGER NOT NULL,
    tombstoned INTEGER NOT NULL DEFAULT 0,
    hlc BLOB,
    origin_actor TEXT,
    PRIMARY KEY (src_id, dst_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_cognitive_edges_dst ON cognitive_edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_cognitive_edges_kind ON cognitive_edges(kind);

-- High-water marks for NodeId allocator (V12)
CREATE TABLE IF NOT EXISTS cognitive_node_hwm (
    kind TEXT PRIMARY KEY,                   -- node kind string
    high_water_mark INTEGER NOT NULL DEFAULT 0
);
";

/// SQL to migrate from schema V1 to V2.
pub const MIGRATE_V1_TO_V2: &str = "
ALTER TABLE oplog ADD COLUMN hlc BLOB;
ALTER TABLE oplog ADD COLUMN embedding_hash BLOB;
ALTER TABLE oplog ADD COLUMN origin_actor TEXT NOT NULL DEFAULT 'local';
ALTER TABLE oplog ADD COLUMN applied INTEGER NOT NULL DEFAULT 1;

CREATE INDEX IF NOT EXISTS idx_oplog_hlc ON oplog(hlc);
CREATE INDEX IF NOT EXISTS idx_oplog_actor ON oplog(origin_actor);

CREATE TABLE IF NOT EXISTS sync_peers (
    peer_actor TEXT PRIMARY KEY,
    last_synced_hlc BLOB NOT NULL,
    last_synced_op_id TEXT NOT NULL,
    last_sync_time REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS consolidation_members (
    consolidation_rid TEXT NOT NULL,
    source_rid TEXT NOT NULL,
    hlc BLOB NOT NULL,
    actor_id TEXT NOT NULL,
    PRIMARY KEY (consolidation_rid, source_rid)
);
CREATE INDEX IF NOT EXISTS idx_consolidation_source ON consolidation_members(source_rid);
";

/// SQL to migrate from schema V2 to V3.
pub const MIGRATE_V2_TO_V3: &str = "
CREATE TABLE IF NOT EXISTS conflicts (
    conflict_id TEXT PRIMARY KEY,
    conflict_type TEXT NOT NULL,
    priority TEXT NOT NULL DEFAULT 'medium',
    status TEXT NOT NULL DEFAULT 'open',
    memory_a TEXT NOT NULL,
    memory_b TEXT NOT NULL,
    entity TEXT,
    rel_type TEXT,
    detected_at REAL NOT NULL,
    detected_by TEXT NOT NULL,
    detection_reason TEXT NOT NULL,
    resolved_at REAL,
    resolved_by TEXT,
    strategy TEXT,
    winner_rid TEXT,
    resolution_note TEXT,
    hlc BLOB NOT NULL,
    origin_actor TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_conflicts_status ON conflicts(status);
CREATE INDEX IF NOT EXISTS idx_conflicts_type ON conflicts(conflict_type);
CREATE INDEX IF NOT EXISTS idx_conflicts_priority ON conflicts(priority);
CREATE INDEX IF NOT EXISTS idx_conflicts_entity ON conflicts(entity);
CREATE INDEX IF NOT EXISTS idx_conflicts_memory_a ON conflicts(memory_a);
CREATE INDEX IF NOT EXISTS idx_conflicts_memory_b ON conflicts(memory_b);
";

/// SQL to migrate from schema V3 to V4.
pub const MIGRATE_V3_TO_V4: &str = "
CREATE TABLE IF NOT EXISTS trigger_log (
    trigger_id TEXT PRIMARY KEY,
    trigger_type TEXT NOT NULL,
    urgency REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    reason TEXT NOT NULL,
    suggested_action TEXT NOT NULL,
    source_rids TEXT NOT NULL DEFAULT '[]',
    context TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    delivered_at REAL,
    acknowledged_at REAL,
    acted_at REAL,
    expires_at REAL,
    cooldown_key TEXT,
    hlc BLOB NOT NULL,
    origin_actor TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patterns (
    pattern_id TEXT PRIMARY KEY,
    pattern_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    confidence REAL NOT NULL,
    description TEXT NOT NULL,
    evidence_rids TEXT NOT NULL DEFAULT '[]',
    entity_names TEXT NOT NULL DEFAULT '[]',
    context TEXT NOT NULL DEFAULT '{}',
    first_seen REAL NOT NULL,
    last_confirmed REAL NOT NULL,
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    hlc BLOB NOT NULL,
    origin_actor TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trigger_log_status ON trigger_log(status);
CREATE INDEX IF NOT EXISTS idx_trigger_log_type ON trigger_log(trigger_type);
CREATE INDEX IF NOT EXISTS idx_trigger_log_created ON trigger_log(created_at);
CREATE INDEX IF NOT EXISTS idx_trigger_log_cooldown ON trigger_log(cooldown_key);
CREATE INDEX IF NOT EXISTS idx_trigger_log_urgency ON trigger_log(urgency DESC);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_status ON patterns(status);
CREATE INDEX IF NOT EXISTS idx_patterns_confidence ON patterns(confidence DESC);
";

/// SQL to migrate from schema V4 to V5.
pub const MIGRATE_V4_TO_V5: &str = "
CREATE TABLE IF NOT EXISTS memory_entities (
    memory_rid TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    PRIMARY KEY (memory_rid, entity_name)
);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_name);
CREATE INDEX IF NOT EXISTS idx_memory_entities_rid ON memory_entities(memory_rid);
";

/// SQL to migrate from schema V5 to V6.
pub const MIGRATE_V5_TO_V6: &str = "
ALTER TABLE memories ADD COLUMN storage_tier TEXT NOT NULL DEFAULT 'hot';
CREATE INDEX IF NOT EXISTS idx_memories_storage_tier ON memories(storage_tier);
";

/// SQL to migrate from schema V6 to V7.
pub const MIGRATE_V6_TO_V7: &str = "
-- FTS5 for full-text search on memories
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(text, content=memories, content_rowid=rowid);

-- Populate FTS5 from existing data
INSERT INTO memories_fts(memories_fts) VALUES('rebuild');

-- Auto-sync triggers for FTS5
CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_delete BEFORE DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE OF text ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
END;

-- Normalized join tables
CREATE TABLE IF NOT EXISTS trigger_source_rids (
    trigger_id TEXT NOT NULL,
    rid TEXT NOT NULL,
    PRIMARY KEY (trigger_id, rid)
);
CREATE INDEX IF NOT EXISTS idx_trigger_source_rids_rid ON trigger_source_rids(rid);

CREATE TABLE IF NOT EXISTS pattern_evidence (
    pattern_id TEXT NOT NULL,
    rid TEXT NOT NULL,
    PRIMARY KEY (pattern_id, rid)
);
CREATE INDEX IF NOT EXISTS idx_pattern_evidence_rid ON pattern_evidence(rid);

CREATE TABLE IF NOT EXISTS pattern_entities (
    pattern_id TEXT NOT NULL,
    entity_name TEXT NOT NULL,
    PRIMARY KEY (pattern_id, entity_name)
);
CREATE INDEX IF NOT EXISTS idx_pattern_entities_entity ON pattern_entities(entity_name);

-- Backfill join tables from JSON columns
INSERT OR IGNORE INTO trigger_source_rids (trigger_id, rid)
    SELECT trigger_id, json_each.value FROM trigger_log, json_each(source_rids)
    WHERE source_rids IS NOT NULL AND source_rids != '[]';

INSERT OR IGNORE INTO pattern_evidence (pattern_id, rid)
    SELECT pattern_id, json_each.value FROM patterns, json_each(evidence_rids)
    WHERE evidence_rids IS NOT NULL AND evidence_rids != '[]';

INSERT OR IGNORE INTO pattern_entities (pattern_id, entity_name)
    SELECT pattern_id, json_each.value FROM patterns, json_each(entity_names)
    WHERE entity_names IS NOT NULL AND entity_names != '[]';
";

/// SQL to migrate from schema V7 to V8.
pub const MIGRATE_V7_TO_V8: &str = "
ALTER TABLE memories ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default';
CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace);
";

/// SQL to migrate from schema V8 to V9.
pub const MIGRATE_V8_TO_V9: &str = "
ALTER TABLE memories ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_memories_access_count ON memories(access_count);
";

/// SQL to migrate from schema V9 to V10.
pub const MIGRATE_V9_TO_V10: &str = "
-- New cognitive dimension columns
ALTER TABLE memories ADD COLUMN certainty REAL NOT NULL DEFAULT 0.8;
ALTER TABLE memories ADD COLUMN domain TEXT NOT NULL DEFAULT 'general';
ALTER TABLE memories ADD COLUMN source TEXT NOT NULL DEFAULT 'user';
ALTER TABLE memories ADD COLUMN emotional_state TEXT;
CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);
CREATE INDEX IF NOT EXISTS idx_memories_source ON memories(source);
CREATE INDEX IF NOT EXISTS idx_memories_emotional_state ON memories(emotional_state);

-- Recall feedback for adaptive learning
CREATE TABLE IF NOT EXISTS recall_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text TEXT,
    query_embedding BLOB,
    rid TEXT NOT NULL,
    feedback TEXT NOT NULL,
    score_at_retrieval REAL,
    rank_at_retrieval INTEGER,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_created ON recall_feedback(created_at);

-- Learned scoring weights (singleton)
CREATE TABLE IF NOT EXISTS learned_weights (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    w_sim REAL NOT NULL DEFAULT 0.50,
    w_decay REAL NOT NULL DEFAULT 0.20,
    w_recency REAL NOT NULL DEFAULT 0.30,
    gate_tau REAL NOT NULL DEFAULT 0.25,
    alpha_imp REAL NOT NULL DEFAULT 0.80,
    keyword_boost REAL NOT NULL DEFAULT 0.31,
    updated_at REAL,
    feedback_count INTEGER DEFAULT 0,
    generation INTEGER DEFAULT 0
);
INSERT OR IGNORE INTO learned_weights (id) VALUES (1);
";

/// SQL to migrate from schema V10 to V11.
pub const MIGRATE_V10_TO_V11: &str = "
-- Personality traits derived from memory signals
CREATE TABLE IF NOT EXISTS personality_traits (
    trait_name TEXT PRIMARY KEY,
    score REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.0,
    sample_count INTEGER NOT NULL DEFAULT 0,
    updated_at REAL NOT NULL DEFAULT 0.0
);
INSERT OR IGNORE INTO personality_traits (trait_name, score, confidence, sample_count, updated_at)
    VALUES ('warmth', 0.5, 0.0, 0, 0.0),
           ('depth', 0.5, 0.0, 0, 0.0),
           ('energy', 0.5, 0.0, 0, 0.0),
           ('attentiveness', 0.5, 0.0, 0, 0.0);
";

/// SQL to migrate from schema V11 to V12.
pub const MIGRATE_V11_TO_V12: &str = "
-- Cognitive State Graph: Nodes
CREATE TABLE IF NOT EXISTS cognitive_nodes (
    node_id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    activation REAL NOT NULL DEFAULT 0.0,
    salience REAL NOT NULL DEFAULT 0.5,
    persistence REAL NOT NULL DEFAULT 0.5,
    valence REAL NOT NULL DEFAULT 0.0,
    urgency REAL NOT NULL DEFAULT 0.0,
    novelty REAL NOT NULL DEFAULT 1.0,
    volatility REAL NOT NULL DEFAULT 0.1,
    provenance TEXT NOT NULL DEFAULT 'observed',
    evidence_count INTEGER NOT NULL DEFAULT 1,
    last_updated_ms INTEGER NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    tombstoned INTEGER NOT NULL DEFAULT 0,
    hlc BLOB,
    origin_actor TEXT
);
CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_kind ON cognitive_nodes(kind);
CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_activation ON cognitive_nodes(activation);
CREATE INDEX IF NOT EXISTS idx_cognitive_nodes_urgency ON cognitive_nodes(urgency);

-- Cognitive State Graph: Edges
CREATE TABLE IF NOT EXISTS cognitive_edges (
    src_id INTEGER NOT NULL,
    dst_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.5,
    confidence REAL NOT NULL DEFAULT 0.5,
    observation_count INTEGER NOT NULL DEFAULT 1,
    created_at_ms INTEGER NOT NULL,
    last_confirmed_ms INTEGER NOT NULL,
    tombstoned INTEGER NOT NULL DEFAULT 0,
    hlc BLOB,
    origin_actor TEXT,
    PRIMARY KEY (src_id, dst_id, kind)
);
CREATE INDEX IF NOT EXISTS idx_cognitive_edges_dst ON cognitive_edges(dst_id);
CREATE INDEX IF NOT EXISTS idx_cognitive_edges_kind ON cognitive_edges(kind);

-- High-water marks for NodeId allocator
CREATE TABLE IF NOT EXISTS cognitive_node_hwm (
    kind TEXT PRIMARY KEY,
    high_water_mark INTEGER NOT NULL DEFAULT 0
);
";

/// SQL to migrate from schema V12 to V13.
pub const MIGRATE_V12_TO_V13: &str = "
-- Session tracking
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL DEFAULT 'default',
    client_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    started_at REAL NOT NULL,
    ended_at REAL,
    summary TEXT,
    avg_valence REAL,
    memory_count INTEGER NOT NULL DEFAULT 0,
    topics TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    hlc BLOB,
    origin_actor TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_one_active
    ON sessions(namespace, client_id) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_sessions_client_started
    ON sessions(namespace, client_id, started_at DESC);

-- Memories: session & temporal columns
ALTER TABLE memories ADD COLUMN session_id TEXT;
ALTER TABLE memories ADD COLUMN due_at REAL;
ALTER TABLE memories ADD COLUMN temporal_kind TEXT;
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(namespace, session_id);
CREATE INDEX IF NOT EXISTS idx_memories_due_at ON memories(namespace, due_at)
    WHERE due_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_last_access ON memories(last_access);
";

/// SQL to migrate from schema V13 to V14.
pub const MIGRATE_V13_TO_V14: &str = "
-- Substitution categories for feedback-driven conflict learning
CREATE TABLE IF NOT EXISTS substitution_categories (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    conflict_mode TEXT NOT NULL DEFAULT 'exclusive',
    status TEXT NOT NULL DEFAULT 'active',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    hlc BLOB NOT NULL,
    origin_actor TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS substitution_members (
    id TEXT PRIMARY KEY,
    category_id TEXT NOT NULL REFERENCES substitution_categories(id),
    token_normalized TEXT NOT NULL,
    token_display TEXT NOT NULL,
    confidence REAL NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
    source TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    context_hint TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    hlc BLOB NOT NULL,
    origin_actor TEXT NOT NULL,
    UNIQUE(category_id, token_normalized)
);
CREATE INDEX IF NOT EXISTS idx_sub_members_token ON substitution_members(token_normalized);
CREATE INDEX IF NOT EXISTS idx_sub_members_category ON substitution_members(category_id);
CREATE INDEX IF NOT EXISTS idx_sub_members_source_status ON substitution_members(source, status);
CREATE INDEX IF NOT EXISTS idx_sub_categories_name ON substitution_categories(name);
";

/// SQL to migrate from schema V14 to V15 (RFC 006 Phase 1).
///
/// Extends edges with claim-like qualifier columns for scoped conflict
/// detection: polarity, modality, valid_from/to, extractor, confidence_band,
/// source provenance, and namespace. Also adds entity_aliases for
/// alias-aware entity linking.
pub const MIGRATE_V14_TO_V15: &str = "
-- RFC 006 Phase 1: extend edges into claim-like records
ALTER TABLE edges ADD COLUMN polarity INTEGER NOT NULL DEFAULT 1;
ALTER TABLE edges ADD COLUMN modality TEXT NOT NULL DEFAULT 'asserted';
ALTER TABLE edges ADD COLUMN valid_from REAL;
ALTER TABLE edges ADD COLUMN valid_to REAL;
ALTER TABLE edges ADD COLUMN extractor TEXT NOT NULL DEFAULT 'manual';
ALTER TABLE edges ADD COLUMN extractor_version TEXT;
ALTER TABLE edges ADD COLUMN confidence_band TEXT NOT NULL DEFAULT 'medium';
ALTER TABLE edges ADD COLUMN source_memory_rid TEXT;
ALTER TABLE edges ADD COLUMN span_start INTEGER;
ALTER TABLE edges ADD COLUMN span_end INTEGER;
ALTER TABLE edges ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default';

-- Entity aliases for alias-aware conflict detection (RFC 006 Layer B)
CREATE TABLE IF NOT EXISTS entity_aliases (
    alias TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT 'default',
    source TEXT NOT NULL DEFAULT 'explicit',
    created_at REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (alias, namespace)
);
CREATE INDEX IF NOT EXISTS idx_alias_canonical ON entity_aliases(canonical_name, namespace);
";

/// SQL to migrate from schema V15 to V16 (RFC 006 Phase 3).
pub const MIGRATE_V15_TO_V16: &str = "
-- Relation conflict policies
CREATE TABLE IF NOT EXISTS relation_policies (
    relation_type TEXT NOT NULL,
    namespace TEXT NOT NULL DEFAULT '*',
    uniqueness_scope TEXT NOT NULL DEFAULT '[\"dst\"]',
    overlap_allowed INTEGER NOT NULL DEFAULT 0,
    temporal_required INTEGER NOT NULL DEFAULT 0,
    missing_time_severity TEXT NOT NULL DEFAULT 'medium',
    qualifier_exceptions TEXT,
    PRIMARY KEY (relation_type, namespace)
);

-- Seed starter policies for RFC 006 whitelist relations
INSERT OR IGNORE INTO relation_policies (relation_type, namespace, overlap_allowed, temporal_required, missing_time_severity)
VALUES
    ('ceo_of',            '*', 0, 1, 'medium'),
    ('cto_of',            '*', 0, 1, 'medium'),
    ('cfo_of',            '*', 0, 1, 'medium'),
    ('founded',           '*', 1, 0, 'low'),
    ('leads',             '*', 0, 1, 'medium'),
    ('works_at',          '*', 1, 0, 'low'),
    ('born_in',           '*', 0, 0, 'high'),
    ('headquartered_in',  '*', 0, 0, 'high'),
    ('married_to',        '*', 0, 1, 'medium'),
    ('acquired',          '*', 0, 0, 'high'),
    ('subsidiary_of',     '*', 0, 0, 'high'),
    ('speaks',            '*', 1, 0, 'low');
";

/// SQL to migrate from schema V16 to V17 (RFC 006 Phase 5).
/// Renames `edges` table to `claims` and creates `edges` as a read-only VIEW.
pub const MIGRATE_V16_TO_V17: &str = "
-- Rename edges → claims (atomic, preserves all data + indexes)
ALTER TABLE edges RENAME TO claims;
-- Rename primary key column
ALTER TABLE claims RENAME COLUMN edge_id TO claim_id;
-- Create backward-compat VIEW so all SELECT FROM edges queries still work
CREATE VIEW IF NOT EXISTS edges AS
    SELECT claim_id AS edge_id, src, dst, rel_type, weight, created_at, tombstoned,
           polarity, modality, valid_from, valid_to, extractor, extractor_version,
           confidence_band, source_memory_rid, span_start, span_end, namespace
    FROM claims;
";

/// SQL to migrate from schema V17 to V18 (RFC 006 Phase 6).
///
/// The V17 UNIQUE constraint on (src, dst, rel_type) caused ingest_claim() to
/// overwrite a previous source's claim whenever another source asserted the
/// same (src, dst, rel_type) — destroying the polarity contradiction cases
/// RFC 006 is designed to detect.
///
/// V18 widens the constraint to (src, dst, rel_type, extractor, polarity,
/// namespace). Now two sources can make contradictory claims about the same
/// fact and both rows survive, enabling proper multi-witness investigation.
///
/// Migration strategy (SQLite can't ALTER UNIQUE):
///   1. Drop the edges VIEW (depends on claims table)
///   2. Create claims_new with new constraint
///   3. Copy data — deduplicate where old UNIQUE would have rejected
///   4. Drop old claims, rename claims_new → claims
///   5. Recreate indexes + edges VIEW
pub const MIGRATE_V17_TO_V18: &str = "
DROP VIEW IF EXISTS edges;

CREATE TABLE claims_new (
    claim_id TEXT PRIMARY KEY,
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    rel_type TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    tombstoned INTEGER NOT NULL DEFAULT 0,
    polarity INTEGER NOT NULL DEFAULT 1,
    modality TEXT NOT NULL DEFAULT 'asserted',
    valid_from REAL,
    valid_to REAL,
    extractor TEXT NOT NULL DEFAULT 'manual',
    extractor_version TEXT,
    confidence_band TEXT NOT NULL DEFAULT 'medium',
    source_memory_rid TEXT,
    span_start INTEGER,
    span_end INTEGER,
    namespace TEXT NOT NULL DEFAULT 'default',
    UNIQUE(src, dst, rel_type, extractor, polarity, namespace)
);

INSERT INTO claims_new
    SELECT claim_id, src, dst, rel_type, weight, created_at, tombstoned,
           polarity, modality, valid_from, valid_to, extractor, extractor_version,
           confidence_band, source_memory_rid, span_start, span_end, namespace
    FROM claims;

DROP TABLE claims;
ALTER TABLE claims_new RENAME TO claims;

CREATE INDEX IF NOT EXISTS idx_claims_src ON claims(src);
CREATE INDEX IF NOT EXISTS idx_claims_dst ON claims(dst);
CREATE INDEX IF NOT EXISTS idx_claims_rel ON claims(rel_type);

CREATE VIEW IF NOT EXISTS edges AS
    SELECT claim_id AS edge_id, src, dst, rel_type, weight, created_at, tombstoned,
           polarity, modality, valid_from, valid_to, extractor, extractor_version,
           confidence_band, source_memory_rid, span_start, span_end, namespace
    FROM claims;
";

/// SQL to migrate from schema V18 to V19 (RFC 007 Phase 0).
///
/// Adds the five-layer reasoning substrate on top of RFC 006 claims:
///   - propositions: canonical identity for (src, rel_type, dst, namespace) triples
///   - variables: typed world/agent states with value_space + manipulability
///   - state_assertions: observations of variable values at a point in time
///   - rule_edges: whitelisted causal/structural edges between variables
///   - scenario_specs: saved assumption sets (NOT derived state)
///
/// Claims table gains `proposition_id` column. Backfill: every unique
/// (src, rel_type, dst, namespace) from claims becomes a proposition row,
/// and claims.proposition_id is populated for all existing rows.
///
/// No data loss. Fresh installs run SCHEMA_SQL which already has the new tables.
/// Variables, state_assertions, rule_edges, scenario_specs are empty after
/// migration — manual curation, NOT auto-created from propositions.
pub const MIGRATE_V18_TO_V19: &str = "
CREATE TABLE IF NOT EXISTS propositions (
    proposition_id TEXT PRIMARY KEY,
    src            TEXT NOT NULL,
    rel_type       TEXT NOT NULL,
    dst            TEXT NOT NULL,
    namespace      TEXT NOT NULL DEFAULT 'default',
    created_at     REAL NOT NULL,
    UNIQUE(src, rel_type, dst, namespace)
);
CREATE INDEX IF NOT EXISTS idx_propositions_src ON propositions(src);
CREATE INDEX IF NOT EXISTS idx_propositions_dst ON propositions(dst);
CREATE INDEX IF NOT EXISTS idx_propositions_rel ON propositions(rel_type);

CREATE TABLE IF NOT EXISTS variables (
    variable_id    TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    namespace      TEXT NOT NULL DEFAULT 'default',
    value_space    TEXT NOT NULL,
    scope          TEXT NOT NULL,
    context_dims   TEXT NOT NULL DEFAULT '[]',
    manipulable    INTEGER NOT NULL DEFAULT 0,
    actionability  TEXT,
    created_at     REAL NOT NULL,
    UNIQUE(name, namespace)
);
CREATE INDEX IF NOT EXISTS idx_variables_ns ON variables(namespace);
CREATE INDEX IF NOT EXISTS idx_variables_scope ON variables(scope);

CREATE TABLE IF NOT EXISTS state_assertions (
    state_id          TEXT PRIMARY KEY,
    variable_id       TEXT NOT NULL REFERENCES variables(variable_id),
    value             TEXT NOT NULL,
    valid_from        REAL NOT NULL,
    valid_to          REAL,
    context_values    TEXT NOT NULL DEFAULT '{}',
    confidence_band   TEXT NOT NULL DEFAULT 'medium',
    source            TEXT NOT NULL,
    source_memory_rid TEXT,
    namespace         TEXT NOT NULL,
    created_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_state_var ON state_assertions(variable_id);
CREATE INDEX IF NOT EXISTS idx_state_valid ON state_assertions(valid_from, valid_to);
CREATE INDEX IF NOT EXISTS idx_state_ns ON state_assertions(namespace);

CREATE TABLE IF NOT EXISTS rule_edges (
    rule_id              TEXT PRIMARY KEY,
    parent_variable_id   TEXT NOT NULL REFERENCES variables(variable_id),
    child_variable_id    TEXT NOT NULL REFERENCES variables(variable_id),
    edge_type            TEXT NOT NULL CHECK (edge_type IN
                           ('causal_promotes', 'causal_inhibits', 'requires')),
    direction_confidence TEXT NOT NULL,
    lag_min_seconds      REAL,
    lag_max_seconds      REAL,
    persistence          TEXT NOT NULL,
    scope                TEXT NOT NULL,
    context_qualifier    TEXT,
    source               TEXT NOT NULL,
    source_evidence_rids TEXT NOT NULL DEFAULT '[]',
    namespace            TEXT NOT NULL,
    tombstoned           INTEGER NOT NULL DEFAULT 0,
    created_at           REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rule_parent ON rule_edges(parent_variable_id);
CREATE INDEX IF NOT EXISTS idx_rule_child ON rule_edges(child_variable_id);
CREATE INDEX IF NOT EXISTS idx_rule_type ON rule_edges(edge_type);

CREATE TABLE IF NOT EXISTS scenario_specs (
    spec_id        TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    namespace      TEXT NOT NULL,
    assumptions    TEXT NOT NULL,
    created_by     TEXT,
    engine_version TEXT,
    created_at     REAL NOT NULL,
    UNIQUE(name, namespace)
);
CREATE INDEX IF NOT EXISTS idx_scenario_ns ON scenario_specs(namespace);

-- Add proposition_id to claims. SQLite can't ADD COLUMN with REFERENCES, so
-- we add a plain TEXT column here and rely on application-level referential
-- integrity. Fresh installs via SCHEMA_SQL get the full FK constraint.
ALTER TABLE claims ADD COLUMN proposition_id TEXT;
CREATE INDEX IF NOT EXISTS idx_claims_proposition ON claims(proposition_id);

-- Backfill: one proposition per unique (src, rel_type, dst, namespace) from
-- non-tombstoned claims. Uses lower(hex(randomblob(16))) for id generation —
-- not UUIDv7-sortable, but acceptable for a one-time migration. New claims
-- going forward will get Rust-generated UUIDv7 proposition_ids.
INSERT OR IGNORE INTO propositions (proposition_id, src, rel_type, dst, namespace, created_at)
SELECT
    lower(hex(randomblob(16))) AS proposition_id,
    src,
    rel_type,
    dst,
    namespace,
    strftime('%s','now') * 1.0 AS created_at
FROM claims
WHERE tombstoned = 0
GROUP BY src, rel_type, dst, namespace;

-- Populate claims.proposition_id from the new propositions table.
UPDATE claims
SET proposition_id = (
    SELECT p.proposition_id
    FROM propositions p
    WHERE p.src = claims.src
      AND p.rel_type = claims.rel_type
      AND p.dst = claims.dst
      AND p.namespace = claims.namespace
)
WHERE proposition_id IS NULL AND tombstoned = 0;
";

/// SQL to migrate from schema V19 to V20 (RFC 008 Phase 1 — Warrant Flow foundations).
///
/// Adds the three control-stack tables that start replacing scalar confidence
/// with the mobility calculus:
///   - mobility_state: 13-dim vector M(c|ρ) keyed by (proposition, regime, snapshot)
///   - actor_profile: regime-indexed calibration for any epistemic actor
///   - compression_artifact: summaries with reversible loss accounting
///
/// Also adds four write-time mobility signal columns to the claims table.
/// These are populated on every future claim insert; existing claims get
/// sensible defaults (regime='default', self_generated=0, lineage=[], modality='text').
/// Backfilling accurate values for historical rows is a separate background job
/// and not attempted in the migration path.
///
/// No data loss. mobility_state starts empty; it is populated incrementally
/// as Phase 1 algorithm components come online.
pub const MIGRATE_V19_TO_V20: &str = "
CREATE TABLE IF NOT EXISTS mobility_state (
    proposition_id          TEXT NOT NULL REFERENCES propositions(proposition_id),
    regime                  TEXT NOT NULL DEFAULT 'default',
    snapshot_ts             REAL NOT NULL,
    support_mass            REAL,
    attack_mass             REAL,
    source_diversity        REAL,
    effective_independence  REAL,
    temporal_coherence      REAL,
    transportability        REAL,
    mutability              REAL,
    load_bearingness        REAL,
    modality_consilience    REAL,
    self_gen_local          REAL,
    self_gen_ancestral      REAL,
    contamination_risk      REAL,
    novelty_isolation       REAL,
    tier_write_components   TEXT NOT NULL DEFAULT '[]',
    tier_read_components    TEXT NOT NULL DEFAULT '[]',
    tier_bg_components      TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (proposition_id, regime, snapshot_ts)
);
CREATE INDEX IF NOT EXISTS idx_mobility_prop ON mobility_state(proposition_id);
CREATE INDEX IF NOT EXISTS idx_mobility_regime ON mobility_state(regime);

CREATE TABLE IF NOT EXISTS actor_profile (
    actor_id                 TEXT NOT NULL,
    actor_type               TEXT NOT NULL,
    regime                   TEXT NOT NULL DEFAULT 'default',
    corroboration_rate       REAL,
    contradiction_hazard     REAL,
    independence_contribution REAL,
    latency_p50_ms           REAL,
    latency_p99_ms           REAL,
    repairability            REAL,
    bias_signature           TEXT,
    value_alignment_risk     REAL,
    last_updated             REAL NOT NULL,
    update_count             INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (actor_id, regime),
    CHECK (actor_type IN ('source', 'extractor', 'summarizer',
                          'cognitive_move', 'self_mode', 'agent'))
);
CREATE INDEX IF NOT EXISTS idx_actor_type ON actor_profile(actor_type);
CREATE INDEX IF NOT EXISTS idx_actor_updated ON actor_profile(last_updated);

CREATE TABLE IF NOT EXISTS compression_artifact (
    artifact_id              TEXT PRIMARY KEY,
    source_span_json         TEXT NOT NULL,
    abstraction_operator     TEXT NOT NULL,
    operator_version         TEXT,
    known_omissions          TEXT NOT NULL DEFAULT '[]',
    uncertainty_distortion   REAL,
    dependency_impact        REAL,
    reversibility_pointer    TEXT NOT NULL,
    compression_drift_score  REAL NOT NULL DEFAULT 0.0,
    status                   TEXT NOT NULL DEFAULT 'active',
    namespace                TEXT NOT NULL,
    created_at               REAL NOT NULL,
    last_drift_check_at      REAL,
    CHECK (status IN ('active', 'demoted', 'expired', 'rebuilding'))
);
CREATE INDEX IF NOT EXISTS idx_compression_ns ON compression_artifact(namespace);
CREATE INDEX IF NOT EXISTS idx_compression_status ON compression_artifact(status);

-- Add write-time mobility signal columns to claims. SQLite can't add columns
-- with arbitrary CHECK constraints via ALTER; we add plain-typed columns and
-- rely on application-level validation for modality_signal values.
ALTER TABLE claims ADD COLUMN regime_tag      TEXT NOT NULL DEFAULT 'default';
ALTER TABLE claims ADD COLUMN self_generated  INTEGER NOT NULL DEFAULT 0;
ALTER TABLE claims ADD COLUMN source_lineage  TEXT NOT NULL DEFAULT '[]';
ALTER TABLE claims ADD COLUMN modality_signal TEXT NOT NULL DEFAULT 'text';
";

// RFC 008 M3: reproducible-state discipline for write-tier mobility recompute.
// Adds content_hash (sha256 of normalized input set), formula_version, live_claim_count,
// state_status, and computed_at to mobility_state. Existing rows are marked
// stale_formula so the next access or the background reconciler recomputes them
// under the M3 locked formula (leave-one-out symmetric Jaccard).
//
// SQLite ALTER TABLE ADD COLUMN requires either NOT NULL + DEFAULT or nullable.
// We use DEFAULT for all five to backfill existing rows.
pub const MIGRATE_V20_TO_V21: &str = "
ALTER TABLE mobility_state ADD COLUMN formula_version  INTEGER NOT NULL DEFAULT 1;
ALTER TABLE mobility_state ADD COLUMN content_hash     TEXT NOT NULL DEFAULT '';
ALTER TABLE mobility_state ADD COLUMN live_claim_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE mobility_state ADD COLUMN state_status     TEXT NOT NULL DEFAULT 'stale_formula';
ALTER TABLE mobility_state ADD COLUMN computed_at      INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_mobility_status ON mobility_state(state_status);
";

// RFC 008 M4: contest_state for Γ(c) grounded diagnostics. Reliable features
// only — no speculative semantic typing, no pair list storage. One row per
// (proposition_id, regime), current-state overwrite. See Saga note 16 for
// the locked spec and why each field earned its place.
// RFC 008 M5b: cognitive moves substrate. Seven authoritative tables + two
// soft registries + two derived tables. Per locked spec, Saga note 19.
// Append-only event log, normalized edges, corrections as events, staged
// adversarial memory. Registries are seeded by application code at
// bootstrap via seed_move_type_registry() / seed_inference_basis_registry().
pub const MIGRATE_V22_TO_V23: &str = "
CREATE TABLE IF NOT EXISTS move_events (
    move_id                       TEXT PRIMARY KEY,
    move_type                     TEXT NOT NULL,
    operator_version              TEXT NOT NULL,
    actor_id                      TEXT NOT NULL,
    context_regime                TEXT NOT NULL DEFAULT 'default',
    observability                 TEXT NOT NULL
        CHECK (observability IN ('observed', 'self_reported', 'inferred')),
    inference_confidence          REAL,
    inference_basis_json          TEXT,
    dependencies_json             TEXT NOT NULL DEFAULT '[]',
    cost_tokens                   INTEGER,
    cost_latency_ms               INTEGER,
    cost_memory_reads             INTEGER,
    yield_json                    TEXT NOT NULL DEFAULT '{}',
    posthoc_outcome               TEXT
        CHECK (posthoc_outcome IN ('corroborated', 'retracted', 'harmful_side_effect') OR posthoc_outcome IS NULL),
    posthoc_recorded_at           REAL,
    expected_evaluation_horizon_ms INTEGER,
    mobility_state_hash_at_move   TEXT,
    contest_state_hash_at_move    TEXT,
    created_at                    REAL NOT NULL,
    hlc                           BLOB NOT NULL,
    origin_actor                  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_move_type_time ON move_events(move_type, created_at);
CREATE INDEX IF NOT EXISTS idx_move_actor_time ON move_events(actor_id, created_at);
CREATE INDEX IF NOT EXISTS idx_move_regime_time ON move_events(context_regime, created_at);
CREATE TABLE IF NOT EXISTS move_input_edge (
    move_id    TEXT NOT NULL REFERENCES move_events(move_id),
    claim_id   TEXT NOT NULL,
    input_role TEXT NOT NULL DEFAULT 'input',
    ordinal    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (move_id, claim_id, input_role)
);
CREATE INDEX IF NOT EXISTS idx_move_input_claim ON move_input_edge(claim_id);
CREATE TABLE IF NOT EXISTS move_output_edge (
    move_id     TEXT NOT NULL REFERENCES move_events(move_id),
    claim_id    TEXT NOT NULL,
    output_role TEXT NOT NULL DEFAULT 'output',
    ordinal     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (move_id, claim_id, output_role)
);
CREATE INDEX IF NOT EXISTS idx_move_output_claim ON move_output_edge(claim_id);
CREATE TABLE IF NOT EXISTS move_side_effect_edge (
    move_id     TEXT NOT NULL REFERENCES move_events(move_id),
    claim_id    TEXT NOT NULL,
    effect_kind TEXT NOT NULL,
    PRIMARY KEY (move_id, claim_id, effect_kind)
);
CREATE INDEX IF NOT EXISTS idx_move_side_effect_claim ON move_side_effect_edge(claim_id);
CREATE TABLE IF NOT EXISTS move_correction_event (
    correction_id              TEXT PRIMARY KEY,
    original_move_id           TEXT NOT NULL REFERENCES move_events(move_id),
    corrected_move_type        TEXT,
    corrected_operator_version TEXT,
    corrected_context_regime   TEXT,
    correction_reason          TEXT NOT NULL,
    corrected_by_actor_id      TEXT NOT NULL,
    corrected_at               REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_correction_original ON move_correction_event(original_move_id);
CREATE TABLE IF NOT EXISTS move_adversarial_instance (
    instance_id        TEXT PRIMARY KEY,
    move_id            TEXT NOT NULL REFERENCES move_events(move_id),
    status             TEXT NOT NULL
        CHECK (status IN ('candidate', 'confirmed', 'rejected')),
    discovered_via     TEXT NOT NULL
        CHECK (discovered_via IN ('contradiction', 'retraction', 'calibration_signal', 'human_audit')),
    traced_root_cause  TEXT,
    generalized_lesson TEXT,
    lesson_scope_json  TEXT,
    curation_actor_id  TEXT,
    discovered_at      REAL NOT NULL,
    created_at         REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_adv_move ON move_adversarial_instance(move_id);
CREATE INDEX IF NOT EXISTS idx_adv_status ON move_adversarial_instance(status);
CREATE INDEX IF NOT EXISTS idx_adv_discovered_via ON move_adversarial_instance(discovered_via);
CREATE TABLE IF NOT EXISTS move_type_registry (
    move_type                             TEXT PRIMARY KEY,
    status                                TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('proposed', 'active', 'deprecated')),
    description                           TEXT,
    introduced_at                         REAL NOT NULL,
    deprecated_at                         REAL,
    default_expected_evaluation_horizon_ms INTEGER
);
CREATE TABLE IF NOT EXISTS inference_basis_registry (
    basis_type  TEXT PRIMARY KEY,
    description TEXT,
    status      TEXT NOT NULL DEFAULT 'active'
        CHECK (status IN ('proposed', 'active', 'deprecated'))
);
CREATE TABLE IF NOT EXISTS move_composition_rule (
    rule_id                TEXT PRIMARY KEY,
    left_move_type         TEXT NOT NULL,
    right_move_type        TEXT NOT NULL,
    left_operator_version  TEXT,
    right_operator_version TEXT,
    context_regime         TEXT,
    rule_kind              TEXT NOT NULL
        CHECK (rule_kind IN ('commutative', 'non_commutative', 'idempotent',
                             'precondition_violation', 'approx_identity')),
    precondition_json      TEXT,
    evidence_basis_json    TEXT,
    provenance             TEXT NOT NULL
        CHECK (provenance IN ('empirical', 'user_declared', 'inferred')),
    confidence             REAL NOT NULL DEFAULT 0.5,
    created_at             REAL NOT NULL,
    superseded_at          REAL
);
CREATE INDEX IF NOT EXISTS idx_comp_rule_types ON move_composition_rule(left_move_type, right_move_type);
CREATE INDEX IF NOT EXISTS idx_comp_rule_regime ON move_composition_rule(context_regime);
CREATE TABLE IF NOT EXISTS move_type_profile (
    move_type                       TEXT NOT NULL,
    operator_version                TEXT NOT NULL,
    context_regime                  TEXT NOT NULL,
    uses_count                      INTEGER NOT NULL DEFAULT 0,
    resolved_count                  INTEGER NOT NULL DEFAULT 0,
    corroborated_count              INTEGER NOT NULL DEFAULT 0,
    retracted_count                 INTEGER NOT NULL DEFAULT 0,
    harmful_side_effect_count       INTEGER NOT NULL DEFAULT 0,
    contradiction_introduction_rate REAL,
    avg_mobility_shift              REAL,
    predictive_gain_avg             REAL,
    calibration_gain_avg            REAL,
    last_updated                    REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (move_type, operator_version, context_regime)
);
";

pub const MIGRATE_V21_TO_V22: &str = "
CREATE TABLE IF NOT EXISTS contest_state (
    proposition_id  TEXT NOT NULL REFERENCES propositions(proposition_id),
    regime          TEXT NOT NULL DEFAULT 'default',
    support_mass                   REAL NOT NULL DEFAULT 0.0,
    attack_mass                    REAL NOT NULL DEFAULT 0.0,
    support_effective_independence REAL NOT NULL DEFAULT 0.0,
    attack_effective_independence  REAL NOT NULL DEFAULT 0.0,
    support_distinct_source_count  INTEGER NOT NULL DEFAULT 0,
    attack_distinct_source_count   INTEGER NOT NULL DEFAULT 0,
    same_source_opposite_polarity_count              INTEGER NOT NULL DEFAULT 0,
    same_artifact_extractor_polarity_conflict_count  INTEGER NOT NULL DEFAULT 0,
    temporal_overlap_conflict_count                  INTEGER NOT NULL DEFAULT 0,
    temporal_separable_opposition_count              INTEGER NOT NULL DEFAULT 0,
    referent_schema_heterogeneity_count              INTEGER NOT NULL DEFAULT 0,
    heuristic_flags    INTEGER NOT NULL DEFAULT 0,
    derivation_version INTEGER NOT NULL DEFAULT 1,
    content_hash       TEXT NOT NULL DEFAULT '',
    live_claim_count   INTEGER NOT NULL DEFAULT 0,
    state_status       TEXT NOT NULL DEFAULT 'stale_formula'
        CHECK (state_status IN ('fresh', 'recomputing', 'failed', 'stale_formula')),
    computed_at        INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (proposition_id, regime)
);
CREATE INDEX IF NOT EXISTS idx_contest_flags ON contest_state(heuristic_flags);
CREATE INDEX IF NOT EXISTS idx_contest_status ON contest_state(state_status);
";

/// v23 → v24: extend oplog with `embedding` BLOB column.
///
/// Engine v0.7.0 (decoupled write path RFC) needs the full embedding bytes
/// in the oplog so that pending ingest entries (applied=0) can be replayed
/// by background materializer workers without consulting the memories table.
/// Existing rows get NULL — only writes after upgrade carry the embedding.
///
/// `record()` itself stays log-after-apply for now (Phase 4 of the RFC flips
/// it to log-then-apply). This migration is a Phase 1 prerequisite.
pub const MIGRATE_V23_TO_V24: &str = "
ALTER TABLE oplog ADD COLUMN embedding BLOB;
CREATE INDEX IF NOT EXISTS idx_oplog_pending ON oplog(applied) WHERE applied = 0;
";

/// v24 → v25: cluster-replication determinism columns on memories
/// (issue yantrikos/yantrikdb#9).
///
/// Three additive columns on `memories`:
///   - tombstone_reason TEXT NULL — caller-supplied reason for
///     tombstone_with_rid; queryable for cluster incident debugging.
///   - created_at_unix_micros INTEGER NOT NULL — caller-supplied i64
///     micros, materialized at the leader so follower apply is
///     byte-deterministic. Backfill: cast (created_at * 1_000_000) as
///     INTEGER for existing rows.
///   - embedding_model TEXT NULL — engine-deterministic-surface version
///     pin. RFC 013 (HNSW lifecycle + embedder model migration) may
///     swap for a richer typed alias later.
///
/// Schema decisions locked via swarm exchange dd2e0439 / 2c465959 with
/// yantrikdb-server. Rationale:
///   - tombstone_reason on row (not audit-only): cluster incident debug
///     queries should not require a join.
///   - created_at_unix_micros as new column (not coercion of existing
///     created_at REAL): keeps the i64 contract end-to-end and avoids
///     touching back-compat scoring paths.
///   - embedding_model as String for v0.7.0; RFC 013 can introduce
///     a typed alias later behind the same column name.
pub const MIGRATE_V24_TO_V25: &str = "
ALTER TABLE memories ADD COLUMN tombstone_reason TEXT;
ALTER TABLE memories ADD COLUMN created_at_unix_micros INTEGER NOT NULL DEFAULT 0;
ALTER TABLE memories ADD COLUMN embedding_model TEXT;
UPDATE memories SET created_at_unix_micros = CAST(created_at * 1000000 AS INTEGER) WHERE created_at_unix_micros = 0;
CREATE INDEX IF NOT EXISTS idx_memories_created_at_micros ON memories(created_at_unix_micros);
CREATE INDEX IF NOT EXISTS idx_memories_embedding_model ON memories(embedding_model) WHERE embedding_model IS NOT NULL;
";

/// v25 → v26: conflict-aware-write provenance metadata (issue
/// yantrikos/yantrikdb#29, RFC 026 v0.8.x experiment).
///
/// Four additive columns on `memories`, all NULL on pre-v26 rows:
///   - prior_rid TEXT NULL — set by the WriteResolution API (issue #30)
///     when resolution is `update_existing` / `merge_with_existing` /
///     `supersede_chain`; references the prior memory's rid. NULL for
///     `append_as_new`.
///   - resolution_kind TEXT NULL — one of 'append' | 'update' | 'merge'
///     | 'supersede' | 'dismiss'. NULL on pre-v26 rows (no resolution
///     was recorded). Feeds H1 paper analytics: `SELECT COUNT(*) GROUP
///     BY resolution_kind WHERE arm='treatment'` is the headline query.
///   - dismissal_reason TEXT NULL — non-empty only when
///     resolution_kind='dismiss'. The WriteResolution API enforces
///     ≥16 chars at High confidence tier; ≥1 char at Medium. Audit
///     trail for paper anti-gaming.
///   - confidence_at_write REAL NULL — conflict detector confidence at
///     the moment of the write decision, [0.0, 1.0]. NULL on pre-v26
///     rows. Used for falsification gates (e.g. false-abandonment
///     rate = `count(dismiss) / count(High-tier opportunities)`).
///
/// Plus source-field enum normalization. The `source` column was free-
/// text TEXT NOT NULL DEFAULT 'user' through v25; v26 conceptually
/// constrains it to the enum {user, inference, document, system}.
/// Because the column is NOT NULL we can't tombstone non-conforming
/// rows; instead we coerce them to 'user' here and log the count of
/// affected rows once via a `source_normalization_log` row in the
/// `meta` table. Going forward, app-level validation in the
/// WriteResolution API (#30) rejects non-enum source values.
///
/// Two partial indexes for the resolution/supersession query patterns:
///   - idx_memories_prior_rid for "what memory supersedes this rid"
///   - idx_memories_resolution_kind for paper analytics group-by
///
/// All ALTER TABLE statements are additive and idempotent (the runner
/// in engine/mod.rs swallows "duplicate column name" / "already exists"
/// per the v0.7.3 replay-resilience fix). Safe to re-run.
pub const MIGRATE_V25_TO_V26: &str = "
ALTER TABLE memories ADD COLUMN prior_rid TEXT;
ALTER TABLE memories ADD COLUMN resolution_kind TEXT;
ALTER TABLE memories ADD COLUMN dismissal_reason TEXT;
ALTER TABLE memories ADD COLUMN confidence_at_write REAL;
INSERT OR REPLACE INTO meta (key, value)
    SELECT 'source_normalization_log_v26',
           'normalized ' || COUNT(*) || ' rows from non-enum source to user'
    FROM memories
    WHERE source NOT IN ('user', 'inference', 'document', 'system');
UPDATE memories SET source = 'user'
    WHERE source NOT IN ('user', 'inference', 'document', 'system');
CREATE INDEX IF NOT EXISTS idx_memories_prior_rid ON memories(prior_rid) WHERE prior_rid IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_memories_resolution_kind ON memories(resolution_kind) WHERE resolution_kind IS NOT NULL;
";

/// v26 → v27: foundation for `db.reembed()` operation (issue #41).
///
/// Three additive columns + one new audit-log table, all purely additive
/// and replay-safe per the v0.7.3 idempotent runner contract.
///
/// ## On `memories`
///
/// - `embedding_new BLOB NULL` — staging column for new-embedder bytes
///   written during the Encoding phase of `db.reembed()`. The active
///   `embedding` column stays untouched throughout Encoding/Rebuilding,
///   so concurrent recall traffic continues to score consistently against
///   old-dim vectors. At the Swap phase, an atomic SQL transaction copies
///   `embedding_new` → `embedding` (and clears the staging column) inside
///   the same critical section as the `ArcSwap<SearchState>::store`. The
///   Verifying phase ensures any remaining non-NULL staging values are
///   cleaned up (defensive against partial failures).
///
///   Why two columns and not in-place mutation: the recall path
///   (`fetch_embeddings_by_rids` in engine/recall.rs and ~13 sibling
///   call-sites) re-reads `memories.embedding` AFTER HNSW returns
///   candidate rids, to rescore via `serde_helpers::deserialize_f32`.
///   In-place `UPDATE memories.embedding = <new-dim bytes>` during
///   Encoding would cause concurrent recalls to deserialize new-dim
///   bytes and compare them against old-dim query vectors. Garbage
///   scores or panic on dim_mismatch. Verified by grep prior to
///   shipping this migration; see the brainstorm comment chain on #41.
///
/// - `embedding_new_model TEXT NULL` — name of the embedder that
///   produced `embedding_new`. Used by the Swap phase to populate
///   `embedding_model` correctly, and by the reembed crash-recovery
///   path on open() to verify the staged work matches the persisted
///   `meta.reembed_state` target embedder.
///
/// ## On `oplog`
///
/// - `embedding_model TEXT NULL` — name of the embedder active when
///   this oplog row's `embedding` BLOB (v24-added column) was produced.
///   Necessary for Queue-mode reembed: when concurrent record() calls
///   during reembed go through `log_op_pending`, they record the
///   embedder name in this column. After the SearchState swap, the
///   materializer drain compares each pending op's `embedding_model`
///   against the active SearchState's embedder name. If they differ,
///   the materializer re-encodes the op's text under the new embedder
///   (ignoring the now-stale `oplog.embedding` bytes). If they match,
///   the existing `oplog.embedding` is applied directly. Pre-v27 ops
///   have NULL `embedding_model`; the materializer treats NULL as
///   "trust embedding as-is" for backwards compatibility.
///
/// ## New table: `reembed_events`
///
/// Durable audit log of every phase transition during a `db.reembed()`
/// call. Authoritative source of truth for crash recovery — on open(),
/// the engine reads the latest non-Completed/non-Aborted event for the
/// current generation in `meta.reembed_state` and resumes from there.
///
/// The in-memory `on_phase_complete` callback in `ReembedOptions` is
/// best-effort notification only. If the process dies between callback
/// invocation and the next phase transition, the callback's side
/// effects might be lost — but `reembed_events` has the durable record.
///
/// One row per `(generation, phase)` pair. Phase strings match the
/// `ReembedPhase` enum string repr: 'Probing' | 'Encoding' |
/// 'Rebuilding' | 'Swapping' | 'Verifying' | 'Aborted' | 'Completed'.
/// `payload_json` carries phase-specific detail (memories_total,
/// memories_encoded, last_error, etc.).
pub const MIGRATE_V26_TO_V27: &str = "
ALTER TABLE memories ADD COLUMN embedding_new BLOB;
ALTER TABLE memories ADD COLUMN embedding_new_model TEXT;
ALTER TABLE oplog ADD COLUMN embedding_model TEXT;
ALTER TABLE oplog ADD COLUMN applied_generation INTEGER;
CREATE TABLE IF NOT EXISTS reembed_events (
    generation   INTEGER NOT NULL,
    phase        TEXT NOT NULL,
    timestamp    REAL NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_reembed_events_generation ON reembed_events(generation);
CREATE INDEX IF NOT EXISTS idx_oplog_applied_generation ON oplog(applied_generation, op_id);
";

/// **Issue #41 brainstorm-4 §6 — durable linearization point for
/// reembed crash recovery.**
///
/// Two surfaces locked together:
///
/// 1. `memories.embedding_generation INTEGER` — per-row stamp of
///    which SearchState generation the row's `embedding` column was
///    encoded under. NULL on pre-v28 rows (treated by the post-swap
///    materializer as "covered by generation 0 — the initial
///    pre-reembed generation"). Reembed Phase-2's swap transaction
///    writes the new generation here atomically with promoting
///    `embedding_new` into `embedding`.
///
/// 2. `meta.active_generation` row — durable record of the current
///    active SearchState generation. Initialized to '0' on v28
///    install / migration. Phase-2 swap atomically increments this
///    AND swaps the in-memory SearchState; if the engine crashes
///    after the SQL transaction commits but before the ArcSwap
///    store, `open()` reads `meta.active_generation` and rebuilds
///    the SearchState at the new generation from the v28 column —
///    no replay logic, just "the durable record says what it is."
///
/// 3. Index on `(embedding_generation)` — the post-swap materializer
///    scans for "rows under generation < active" to apply queued
///    re-encode ops; this needs index support to be O(N_changed)
///    not O(N_total) for large DBs.
///
/// Brainstorm-4 §6 motivation: without a single durable linearization
/// point, crash recovery needed multi-step replay logic that
/// re-derived the active generation from oplog scans + reembed_events
/// table joins. That logic is itself the bug surface (the
/// reembed_events table is best-effort; oplog scans miss tail writes
/// during the SQL transaction window). One column + one row + one
/// transaction is the clean answer.
///
/// Additive-only migration. Idempotent (the runner in engine/mod.rs
/// swallows "duplicate column name" / "already exists" / "duplicate
/// key" per the v0.7.3 replay-resilience fix). Safe to re-run.
pub const MIGRATE_V27_TO_V28: &str = "
ALTER TABLE memories ADD COLUMN embedding_generation INTEGER;
CREATE INDEX IF NOT EXISTS idx_memories_embedding_generation
    ON memories(embedding_generation);
INSERT OR IGNORE INTO meta (key, value) VALUES ('active_generation', '0');
";

/// **v0.7.19 — replication apply audit log.**
///
/// Surfaced by trader's default-DB postmortem (2026-05-20): 23k
/// memories rows with zero corresponding `record` / `record_with_rid`
/// oplog ops. Replication apply paths (`materialize_record`,
/// `materialize_consolidate`, `materialize_correct` in
/// `distributed/replication.rs`) INSERT into `memories` but do NOT
/// write a local oplog row (that would create a replication loop).
/// Without a sibling audit table, "received via replication" is
/// indistinguishable at SQL level from "true orphan from a write-path
/// bug" — both look like memories.rid NOT IN oplog.
///
/// `replication_apply_log` is that sibling. Each replication-apply
/// site writes a row at apply time. Audit queries then distinguish
/// three populations:
///
/// - **Locally originated**: `memories.rid` IN (SELECT target_rid FROM oplog WHERE origin_actor = self.actor_id AND op_type IN ('record', 'record_with_rid', 'consolidate', 'correct'))
/// - **Received via replication**: `memories.rid` IN (SELECT rid FROM replication_apply_log)
/// - **True orphan** (Backpressure-orphan or bug): in neither
///
/// `source_actor` records WHO sent the op so cross-cluster origin
/// can be traced. `applied_at` is when this node applied it (NOT the
/// originator's creation time — that's on `memories.created_at`).
///
/// Retroactive backfill (e.g. trader's 23k existing rows) is NOT in
/// scope for this migration. Operators with pre-existing replication
/// traffic can run a one-off SQL probe to seed it if desired:
/// `INSERT INTO replication_apply_log SELECT rid, 'record', origin_actor, updated_at FROM memories WHERE rid NOT IN (SELECT target_rid FROM oplog WHERE target_rid IS NOT NULL)`.
///
/// Additive-only migration. Idempotent (CREATE TABLE IF NOT EXISTS).
pub const MIGRATE_V28_TO_V29: &str = "
CREATE TABLE IF NOT EXISTS replication_apply_log (
    rid           TEXT PRIMARY KEY,
    op_type       TEXT NOT NULL,
    source_actor  TEXT NOT NULL,
    applied_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_replication_apply_log_source_actor
    ON replication_apply_log(source_actor, applied_at);
";

// **Issue #47 — `correct()` revision history (v0.7.20).**
//
// The previous `correct()` minted a new rid + tombstoned the original.
// That broke link integrity (every inbound reference dangled) and lost
// the audit trail (the only record of WHY a correction happened was a
// metadata.correction_note on the new row, easily missed). The v0.7.20
// `correct()` mutates in place, preserving rid + created_at, and writes
// a revision row capturing the prior state.
//
// `revision_num` is monotonically increasing per-rid starting at 1.
// `reason` is REQUIRED (engine validates non-empty); the audit trail is
// load-bearing.
//
// Embedding changes are NOT supported by the new `correct()` — HNSW
// doesn't support in-place update of an existing entry, and rebuilding
// the cold tier on every correction is too expensive. Embedding-level
// corrections still go through forget+record. The revision table
// therefore stores text + metadata + importance + valence only.
pub const MIGRATE_V29_TO_V30: &str = "
CREATE TABLE IF NOT EXISTS record_revisions (
    revision_id       TEXT PRIMARY KEY,
    rid               TEXT NOT NULL,
    revision_num      INTEGER NOT NULL,
    prior_text        TEXT NOT NULL,
    prior_metadata    TEXT NOT NULL,
    prior_importance  REAL NOT NULL,
    prior_valence     REAL NOT NULL,
    reason            TEXT NOT NULL,
    applied_at        REAL NOT NULL,
    hlc               BLOB NOT NULL,
    origin_actor      TEXT NOT NULL,
    UNIQUE(rid, revision_num)
);
CREATE INDEX IF NOT EXISTS idx_record_revisions_rid
    ON record_revisions(rid, revision_num);
";

// **Issue #48 — first-class record-to-record links (schema v31).**
//
// `record_links` is a dedicated table (NOT a reuse of `claims`) so
// record-to-record relations get rid-specific semantics without
// polluting the entity graph with phantom `unknown`-typed entity rows
// (see RFC §"Considered alternatives"). The migration only creates the
// table + indexes; the metadata.supersedes reification runs in Rust at
// migration time (it needs the engine HLC clock — pure SQL can't stamp
// a real HLC). See engine/mod.rs migration registration.
pub const MIGRATE_V30_TO_V31: &str = "
CREATE TABLE IF NOT EXISTS record_links (
    link_id        TEXT PRIMARY KEY,
    source_rid     TEXT NOT NULL,
    target_rid     TEXT NOT NULL,
    link_type      TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'active',
    created_at     REAL NOT NULL,
    hlc            BLOB NOT NULL,
    origin_actor   TEXT NOT NULL,
    UNIQUE(source_rid, target_rid, link_type)
);
CREATE INDEX IF NOT EXISTS idx_record_links_source
    ON record_links(source_rid, link_type, status);
CREATE INDEX IF NOT EXISTS idx_record_links_target
    ON record_links(target_rid, link_type, status);
";

// **Structural query — list_records (schema v32).**
//
// Adds indexed VIRTUAL generated columns over JSON metadata fields so typed
// enumeration ("newest N records of kind=X", "records where drive_id=D") is an
// O(log n) index walk instead of a full-table json_extract scan. VIRTUAL
// generated columns ARE addable via ALTER TABLE (unlike STORED); the index
// materializes the computed value on insert. The json_valid guard keeps
// encrypted-metadata rows (ciphertext, not JSON) from erroring the index
// build — they resolve to NULL. run_migration_idempotent swallows the
// duplicate-column / duplicate-index errors on forward-upgrade replays.
pub const MIGRATE_V31_TO_V32: &str = "
ALTER TABLE memories ADD COLUMN kind TEXT GENERATED ALWAYS AS (
    CASE WHEN json_valid(metadata) THEN json_extract(metadata, '$.kind') END
) VIRTUAL;
ALTER TABLE memories ADD COLUMN drive_id TEXT GENERATED ALWAYS AS (
    CASE WHEN json_valid(metadata) THEN json_extract(metadata, '$.drive_id') END
) VIRTUAL;
CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
CREATE INDEX IF NOT EXISTS idx_memories_drive_id ON memories(drive_id);
";
