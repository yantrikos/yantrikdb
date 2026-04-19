# YantrikDB — A Cognitive Memory Engine for Persistent AI Systems

> The memory engine for AI that actually knows you.

[![PyPI](https://img.shields.io/pypi/v/yantrikdb)](https://pypi.org/project/yantrikdb/)
[![Crates.io](https://img.shields.io/crates/v/yantrikdb)](https://crates.io/crates/yantrikdb)
[![License: AGPL-3.0](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)

## Get Started in 60 Seconds

### For AI agents (MCP — works with Claude, Cursor, Windsurf, Copilot)

```bash
pip install yantrikdb-mcp
```

Add to your MCP client config:

```json
{
  "mcpServers": {
    "yantrikdb": {
      "command": "yantrikdb-mcp"
    }
  }
}
```

That's it. The agent auto-recalls context, auto-remembers decisions, and auto-detects contradictions — no prompting needed. See [yantrikdb-mcp](https://github.com/yantrikos/yantrikdb-mcp) for full docs.

### As a Python library

```bash
pip install yantrikdb
```

```python
import yantrikdb
from sentence_transformers import SentenceTransformer

# Single file, no server, no config
db = yantrikdb.YantrikDB("memory.db", embedding_dim=384)
db.set_embedder(SentenceTransformer("all-MiniLM-L6-v2"))

# Record memories with importance, domain, and emotional valence
db.record("Alice is the engineering lead", importance=0.8, domain="people")
db.record("Project deadline is March 30", importance=0.9, domain="work")
db.record("User prefers dark mode", importance=0.6, domain="preference")

# Semantic recall — ranked by relevance, recency, importance, and graph proximity
results = db.recall("who leads the team?", top_k=3)
# → [{"text": "Alice is the engineering lead", "score": 1.0}, ...]

# Knowledge graph — entity relationships
db.relate("Alice", "Engineering", "leads")
db.get_edges("Alice")
# → [{"src": "Alice", "dst": "Engineering", "rel_type": "leads", "weight": 1.0}]

# Cognitive maintenance — consolidate, detect conflicts, mine patterns
db.think()
# → {"consolidation_count": 2, "conflicts_found": 0, "patterns_new": 1}

db.close()
```

### As a Rust crate

```toml
[dependencies]
yantrikdb = "0.4"
```

## The Problem

Current AI memory is:

> Store everything → Embed → Retrieve top-k → Inject into context → Hope it helps.

That's not memory. That's a search engine with extra steps.

Real memory is hierarchical, compressed, contextual, self-updating, emotionally weighted, time-aware, and predictive. YantrikDB is built for that.

## Why Not Existing Solutions?

| Solution | What it does | What it lacks |
|----------|-------------|---------------|
| **Vector DBs** (Pinecone, Weaviate) | Nearest-neighbor lookup | No decay, no causality, no self-organization |
| **Knowledge Graphs** (Neo4j) | Structured relations | Poor for fuzzy memory, not adaptive |
| **Memory Frameworks** (LangChain, Mem0) | Retrieval wrappers | Not a memory architecture — just middleware |
| **File-based** (CLAUDE.md, memory files) | Dump everything into context | O(n) token cost, no relevance filtering |

### Benchmark: Selective Recall vs. File-Based Memory

| Memories | File-Based | YantrikDB | Token Savings | Precision |
|----------|-----------|-----------|---------------|-----------|
| 100 | 1,770 tokens | 69 tokens | **96%** | 66% |
| 500 | 9,807 tokens | 72 tokens | **99.3%** | 77% |
| 1,000 | 19,988 tokens | 72 tokens | **99.6%** | 84% |
| 5,000 | 101,739 tokens | 53 tokens | **99.9%** | 88% |

At 500 memories, file-based exceeds 32K context windows. At 5,000, it doesn't fit in any context window — not even 200K. YantrikDB stays at ~70 tokens per query. Precision *improves* with more data — the opposite of context stuffing.

## Architecture

### Design Principles

- **Embedded, not client-server** — single file, no server process (like SQLite)
- **Local-first, sync-native** — works offline, syncs when connected
- **Cognitive operations, not SQL** — `record()`, `recall()`, `relate()`, not `SELECT`
- **Living system, not passive store** — does work between conversations
- **Thread-safe** — `Send + Sync` with internal Mutex/RwLock, safe for concurrent access

### Five Indexes, One Engine

```
┌──────────────────────────────────────────────────────┐
│                   YantrikDB Engine                    │
│                                                      │
│  ┌──────────┬──────────┬──────────┬──────────┐       │
│  │  Vector  │  Graph   │ Temporal │  Decay   │       │
│  │  (HNSW)  │(Entities)│ (Events) │  (Heap)  │       │
│  └──────────┴──────────┴──────────┴──────────┘       │
│  ┌──────────┐                                        │
│  │ Key-Value│  WAL + Replication Log (CRDT)          │
│  └──────────┘                                        │
└──────────────────────────────────────────────────────┘
```

1. **Vector Index (HNSW)** — semantic similarity search across memories
2. **Graph Index** — entity relationships, profile aggregation, bridge detection
3. **Temporal Index** — time-aware queries ("what happened Tuesday", "upcoming deadlines")
4. **Decay Heap** — importance scores that degrade over time, like human memory
5. **Key-Value Store** — fast facts, session state, scoring weights

### Memory Types (Tulving's Taxonomy)

| Type | What it stores | Example |
|------|---------------|---------|
| **Semantic** | Facts, knowledge | "User is a software engineer at Meta" |
| **Episodic** | Events with context | "Had a rough day at work on Feb 20" |
| **Procedural** | Strategies, what worked | "Deploy with blue-green, not rolling update" |

All memories carry **importance**, **valence** (emotional tone), **domain**, **source**, **certainty**, and **timestamps** — used in a multi-signal scoring function that goes far beyond cosine similarity.

## Key Capabilities

### Relevance-Conditioned Scoring

Not just vector similarity. Every recall combines:

- **Semantic similarity** (HNSW) — what's topically related
- **Temporal decay** — recent memories score higher
- **Importance weighting** — critical decisions beat trivia
- **Graph proximity** — entity relationships boost connected memories
- **Retrieval feedback** — learns from past recall quality

Weights are tuned automatically from usage patterns.

### Conflict Detection & Resolution

When memories contradict, YantrikDB doesn't guess — it creates a conflict segment:

```
"works at Google" (recorded Jan 15) vs. "works at Meta" (recorded Mar 1)
→ Conflict: identity_fact, priority: high, strategy: ask_user
```

Resolution is conversational: the AI asks naturally, not programmatically.

### Semantic Consolidation

After many conversations, memories pile up. `think()` runs:

1. **Consolidation** — merge similar memories, extract patterns
2. **Conflict scan** — find contradictions across the knowledge base
3. **Pattern mining** — cross-domain discovery ("work stress correlates with health entries")
4. **Trigger evaluation** — proactive insights worth surfacing

### Proactive Triggers

The engine generates triggers when it detects something worth reaching out about:

- Memory conflicts needing resolution
- Approaching deadlines (temporal awareness)
- Patterns detected across domains
- High-importance memories about to decay
- Goal tracking ("how's the marathon training?")

Every trigger is grounded in real memory data — not engagement farming.

### Multi-Device Sync (CRDT)

Local-first with append-only replication log:

- **CRDT merging** — graph edges, memories, and metadata merge without conflicts
- **Vector indexes rebuild locally** — raw memories sync, each device rebuilds HNSW
- **Forget propagation** — tombstones ensure forgotten memories stay forgotten
- **Conflict detection** — contradictions across devices are flagged for resolution

### Sessions & Temporal Awareness

```python
sid = db.session_start("default", "claude-code")
db.record("decided to use PostgreSQL")  # auto-linked to session
db.record("Alice suggested Redis for caching")
db.session_end(sid)
# → computes: memory_count, avg_valence, topics, duration

db.stale(days=14)    # high-importance memories not accessed recently
db.upcoming(days=7)  # memories with approaching deadlines
```

## Full API

| Operation | Methods |
|-----------|---------|
| **Core** | `record`, `record_batch`, `recall`, `recall_with_response`, `recall_refine`, `forget`, `correct` |
| **Knowledge Graph** | `relate`, `get_edges`, `search_entities`, `entity_profile`, `relationship_depth`, `link_memory_entity` |
| **Cognition** | `think`, `get_patterns`, `scan_conflicts`, `resolve_conflict`, `derive_personality` |
| **Triggers** | `get_pending_triggers`, `acknowledge_trigger`, `deliver_trigger`, `act_on_trigger`, `dismiss_trigger` |
| **Sessions** | `session_start`, `session_end`, `session_history`, `active_session`, `session_abandon_stale` |
| **Temporal** | `stale`, `upcoming` |
| **Procedural** | `record_procedural`, `surface_procedural`, `reinforce_procedural` |
| **Lifecycle** | `archive`, `hydrate`, `decay`, `evict`, `list_memories`, `stats` |
| **Sync** | `extract_ops_since`, `apply_ops`, `get_peer_watermark`, `set_peer_watermark` |
| **Maintenance** | `rebuild_vec_index`, `rebuild_graph_index`, `learned_weights` |

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Core language** | Rust | Memory safety, no GC, ideal for embedded engines |
| **Architecture** | Embedded (like SQLite) | No server overhead, sub-ms reads, single-tenant |
| **Bindings** | Python (PyO3), TypeScript | Agent/AI layer integration |
| **Storage** | Single file per user | Portable, backupable, no infrastructure |
| **Sync** | CRDTs + append-only log | Conflict-free for most operations, deterministic |
| **Thread safety** | Mutex/RwLock, Send+Sync | Safe concurrent access from multiple threads |
| **Query interface** | Cognitive operations API | Not SQL — designed for how agents think |

## Ecosystem

| Package | What | Install |
|---------|------|---------|
| [yantrikdb](https://crates.io/crates/yantrikdb) | Rust engine | `cargo add yantrikdb` |
| [yantrikdb](https://pypi.org/project/yantrikdb/) | Python bindings (PyO3) | `pip install yantrikdb` |
| [yantrikdb-mcp](https://pypi.org/project/yantrikdb-mcp/) | MCP server for AI agents | `pip install yantrikdb-mcp` |

## Roadmap

- [x] **V0** — Embedded engine, core memory model (record, recall, relate, consolidate, decay)
- [x] **V1** — Replication log, CRDT-based sync between devices
- [x] **V2** — Conflict resolution with human-in-the-loop
- [x] **V3** — Proactive cognition loop, pattern detection, trigger system
- [x] **V4** — Sessions, temporal awareness, cross-domain pattern mining, entity profiles
- [ ] **V5** — Multi-agent shared memory, federated learning across users

## Worked example: Wirecard (RFC 008 substrate — with honest limits)

For nearly a decade, Wirecard's filings and EY's audit attested to €1.9B in Philippine escrow accounts. In June 2020 both banks and the central bank formally denied the accounts existed.

When the `source_lineage` fields are hand-populated — EY as `[wirecard, ey]` to capture audit dependence on Wirecard-provided documents, BSP as `[bsp, bpi, bdo]` to capture restatement of the commercial banks — RFC 008's `⊕` discounts the dependent claims, and the contest operator's temporal split distinguishes present-tense contradictions from historical state changes. On this hand-populated data, the substrate produces useful annotations.

**Honest limits** (surfaced by Phase 2 empirical testing, Apr 2026):

- On naturalistic evidence where a real agent populates the fields, the substrate's gates don't reliably fire. Cases B and C of the Phase 2 eval need an extractor/canonicalizer (not yet built) to work; Case A exposed that `⊕` is mathematically incapable of flipping decisions at realistic N, regardless of coefficient tuning.
- **Current claim**: structured schema for evidence provenance/temporal/conflict annotation, useful for audit and inspection. The dependence-discount operator works on curated inputs but needs replacement before it can drive decisions.
- **Not a current claim**: "decision-improvement substrate for AGI-capable agents." That framing is withdrawn pending RFC 009.

See **[docs/showcase/wirecard.md](docs/showcase/wirecard.md)** for the full walkthrough including the Phase 2 negative result and the gold-state ablation that partitioned operator failure from extraction failure. Run the hand-populated demonstration directly:

```bash
cargo run --example showcase_wirecard
```

## Research & Publications

- **U.S. Patent Application 19/573,392** (March 2026): "Cognitive Memory Database System with Relevance-Conditioned Scoring and Autonomous Knowledge Management"
- **Zenodo:** [YantrikDB: A Cognitive Memory Engine for Persistent AI Systems](https://zenodo.org/records/14933693)

## Author

**Pranab Sarkar** — [ORCID](https://orcid.org/0009-0009-8683-1481) · [LinkedIn](https://www.linkedin.com/in/pranab-sarkar-b0511160/) · developer@pranab.co.in

## License

AGPL-3.0. See [LICENSE](LICENSE) for the full text.

The [MCP server](https://github.com/yantrikos/yantrikdb-mcp) is MIT-licensed — using the engine via the MCP server does not trigger AGPL obligations on your code.
