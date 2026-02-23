pub const SCHEMA_VERSION: i32 = 1;

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
    valence REAL NOT NULL DEFAULT 0.0,    -- emotional weight [-1, 1]

    -- Consolidation tracking
    consolidated_into TEXT,              -- rid of the semantic memory this was merged into
    consolidation_status TEXT DEFAULT 'active', -- active | consolidated | tombstoned

    -- Metadata
    metadata TEXT DEFAULT '{}'           -- JSON blob for extensibility
);

-- Entity relationship graph
CREATE TABLE IF NOT EXISTS edges (
    edge_id TEXT PRIMARY KEY,            -- UUIDv7
    src TEXT NOT NULL,                   -- entity name or memory rid
    dst TEXT NOT NULL,                   -- entity name or memory rid
    rel_type TEXT NOT NULL,              -- relationship type (e.g., \"is_about\", \"related_to\")
    weight REAL NOT NULL DEFAULT 1.0,    -- relationship strength [0, 1]
    created_at REAL NOT NULL,
    tombstoned INTEGER NOT NULL DEFAULT 0,

    UNIQUE(src, dst, rel_type)
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

-- Append-only operation log (future CRDT replication)
CREATE TABLE IF NOT EXISTS oplog (
    op_id TEXT PRIMARY KEY,              -- UUIDv7
    op_type TEXT NOT NULL,               -- record | relate | consolidate | decay | forget | update
    timestamp REAL NOT NULL,             -- when the operation occurred
    target_rid TEXT,                     -- primary memory affected
    payload TEXT NOT NULL DEFAULT '{}',  -- JSON: full operation details
    actor_id TEXT DEFAULT 'local'        -- device/agent identifier
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_consolidation ON memories(consolidation_status);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(rel_type);
CREATE INDEX IF NOT EXISTS idx_oplog_timestamp ON oplog(timestamp);
CREATE INDEX IF NOT EXISTS idx_oplog_target ON oplog(target_rid);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
";
