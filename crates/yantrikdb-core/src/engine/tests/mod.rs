use rusqlite::params;

use crate::hlc::HLCTimestamp;
use crate::types::*;

use super::YantrikDB;

mod helpers;
use self::helpers::*;

mod alias_migration;
mod backpressure_lifecycle;
mod basics;
mod bundled_embedder;
mod chunking;
mod cognition_gates;
mod consolidate_cluster;
mod corrections;
mod created_at;
mod dimensions;
mod encryption;
// 0.13.2: 'encrypted means encrypted', verified against the raw file
// bytes rather than against the engine's own claim.
#[cfg(feature = "bundled-embedder")]
mod encryption_canary;
mod event_time_columns;
mod event_time_recall;
// The explain suite drives the full text path (record_text/
// recall_text_explained), which needs the bundled embedder — slim
// builds (--no-default-features) have no `with_default`.
#[cfg(feature = "bundled-embedder")]
mod explain;
mod idempotency;
mod interactive_recall;
#[cfg(feature = "bundled-embedder")]
mod learned_templates;
mod learning_categories;
mod maintenance_debt;
mod metamorphic;
mod migrations;
mod mobility_contest;
mod moves;
mod recall_confidence;
mod recall_graph;
mod reembed_router;
#[cfg(feature = "bundled-embedder")]
mod reextract_claims;
#[cfg(feature = "bundled-embedder")]
mod reextract_entities;
mod replication_api;
mod ryw_visibility;
mod source_turn_columns;
#[cfg(feature = "bundled-embedder")]
mod stated_claims;
mod storage_fts;
mod trace_entities;
mod write_stats;
