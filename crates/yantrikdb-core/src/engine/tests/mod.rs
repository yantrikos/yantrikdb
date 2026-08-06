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
mod corrections;
mod dimensions;
mod encryption;
// The explain suite drives the full text path (record_text/
// recall_text_explained), which needs the bundled embedder — slim
// builds (--no-default-features) have no `with_default`.
#[cfg(feature = "bundled-embedder")]
mod explain;
mod idempotency;
mod interactive_recall;
mod learning_categories;
mod migrations;
mod mobility_contest;
mod moves;
mod recall_confidence;
mod recall_graph;
mod reembed_router;
mod replication_api;
mod ryw_visibility;
mod storage_fts;
mod trace_entities;
mod write_stats;
