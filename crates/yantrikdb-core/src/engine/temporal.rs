//! Engine-level temporal reasoning API.
//!
//! Wires the temporal reasoning primitives from `cognition::temporal` into
//! `YantrikDB` methods that load nodes/edges from the database and apply
//! temporal analysis.

use crate::error::Result;
use crate::state::{CognitiveEdgeKind, CognitiveNode, NodeId, NodeKind, NodePayload};
use crate::temporal::{
    self, BurstConfig, BurstResult, DeadlineUrgencyConfig, DetectedPeriod, DomainRecencyMap,
    EwmaTracker, LabeledEvent, MotifConfig, PeriodicityConfig, PeriodicityResult, RecencyConfig,
    SeasonalHistogram, TemporalEvent, TemporalMotif, TemporalOrder, TemporalRelevanceConfig,
    TimeInterval,
};

use super::{now, YantrikDB};

impl YantrikDB {
    // ── Periodicity ──

    /// Detect periodicities in a routine node's triggering history.
    ///
    /// Loads Episode nodes linked via Triggers edges to the routine,
    /// extracts their timestamps, and runs periodicity detection.
    pub fn detect_routine_periodicity(
        &self,
        routine_id: NodeId,
        config: &PeriodicityConfig,
    ) -> Result<PeriodicityResult> {
        let edges = self.load_cognitive_edges_to(routine_id)?;
        let trigger_sources: Vec<NodeId> = edges
            .iter()
            .filter(|e| e.kind == CognitiveEdgeKind::Triggers)
            .map(|e| e.src)
            .collect();

        let mut timestamps = Vec::new();
        for src_id in trigger_sources {
            if let Some(node) = self.load_cognitive_node(src_id)? {
                if let NodePayload::Episode(ep) = &node.payload {
                    timestamps.push(ep.occurred_at);
                } else {
                    // Use last_updated_ms as fallback timestamp
                    timestamps.push(node.attrs.last_updated_ms as f64 / 1000.0);
                }
            }
        }

        Ok(temporal::detect_periodicity(&timestamps, config))
    }

    /// Detect periodicities from all Episode nodes.
    ///
    /// Groups episodes by label pattern and runs periodicity detection
    /// on each group.
    pub fn detect_episode_periodicities(
        &self,
        config: &PeriodicityConfig,
    ) -> Result<Vec<(String, PeriodicityResult)>> {
        let episodes = self.load_cognitive_nodes_by_kind(NodeKind::Episode)?;

        // Group by summary prefix (first word as label)
        let mut groups: std::collections::HashMap<String, Vec<f64>> =
            std::collections::HashMap::new();
        for ep_node in &episodes {
            if let NodePayload::Episode(ep) = &ep_node.payload {
                let label = ep
                    .summary
                    .split_whitespace()
                    .next()
                    .unwrap_or("unknown")
                    .to_lowercase();
                groups.entry(label).or_default().push(ep.occurred_at);
            }
        }

        let mut results = Vec::new();
        for (label, timestamps) in groups {
            let result = temporal::detect_periodicity(&timestamps, config);
            if !result.detected.is_empty() {
                results.push((label, result));
            }
        }

        // Sort by strongest correlation
        results.sort_by(|a, b| {
            let best_a = a.1.detected.first().map(|d| d.correlation).unwrap_or(0.0);
            let best_b = b.1.detected.first().map(|d| d.correlation).unwrap_or(0.0);
            best_b
                .partial_cmp(&best_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    // ── Burst Detection ──

    /// Detect bursts in Episode node creation timestamps.
    pub fn detect_episode_bursts(&self, config: &BurstConfig) -> Result<BurstResult> {
        let episodes = self.load_cognitive_nodes_by_kind(NodeKind::Episode)?;
        let timestamps: Vec<f64> = episodes
            .iter()
            .filter_map(|n| {
                if let NodePayload::Episode(ep) = &n.payload {
                    Some(ep.occurred_at)
                } else {
                    None
                }
            })
            .collect();

        Ok(temporal::detect_bursts(&timestamps, config))
    }

    // ── Temporal Motifs ──

    /// Mine temporal motifs from Episode nodes.
    ///
    /// Extracts (label, timestamp) pairs from episodes and discovers
    /// recurring sequences.
    pub fn mine_episode_motifs(&self, config: &MotifConfig) -> Result<Vec<TemporalMotif>> {
        let episodes = self.load_cognitive_nodes_by_kind(NodeKind::Episode)?;
        let mut events: Vec<LabeledEvent> = episodes
            .iter()
            .filter_map(|n| {
                if let NodePayload::Episode(ep) = &n.payload {
                    Some(LabeledEvent {
                        label: ep
                            .summary
                            .split_whitespace()
                            .next()
                            .unwrap_or("unknown")
                            .to_lowercase(),
                        timestamp: ep.occurred_at,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(temporal::mine_temporal_motifs(&events, config))
    }

    // ── Topological Ordering ──

    /// Compute temporal order of Episode nodes using PrecedesTemporally edges.
    pub fn temporal_order_episodes(&self) -> Result<TemporalOrder> {
        let episodes = self.load_cognitive_nodes_by_kind(NodeKind::Episode)?;
        let temporal_edges =
            self.load_cognitive_edges_by_kind(CognitiveEdgeKind::PrecedesTemporally)?;
        let trigger_edges = self.load_cognitive_edges_by_kind(CognitiveEdgeKind::Triggers)?;

        // Build predecessor map from edges
        let mut predecessors: std::collections::HashMap<u32, Vec<NodeId>> =
            std::collections::HashMap::new();
        for edge in temporal_edges.iter().chain(trigger_edges.iter()) {
            predecessors
                .entry(edge.dst.to_raw())
                .or_default()
                .push(edge.src);
        }

        let events: Vec<TemporalEvent> = episodes
            .iter()
            .map(|n| {
                let ts = if let NodePayload::Episode(ep) = &n.payload {
                    ep.occurred_at
                } else {
                    n.attrs.last_updated_ms as f64 / 1000.0
                };
                TemporalEvent {
                    node_id: n.id,
                    timestamp: ts,
                    predecessors: predecessors
                        .get(&n.id.to_raw())
                        .cloned()
                        .unwrap_or_default(),
                }
            })
            .collect();

        Ok(temporal::topological_order(&events))
    }

    // ── Seasonal Analysis ──

    /// Build hour-of-day histogram from Episode timestamps.
    pub fn episode_hour_histogram(&self) -> Result<SeasonalHistogram> {
        let episodes = self.load_cognitive_nodes_by_kind(NodeKind::Episode)?;
        let mut hist = SeasonalHistogram::hour_of_day();

        for n in &episodes {
            if let NodePayload::Episode(ep) = &n.payload {
                let hour = temporal::hour_of_day_utc(ep.occurred_at);
                hist.add(hour);
            }
        }

        Ok(hist)
    }

    /// Build day-of-week histogram from Episode timestamps.
    pub fn episode_dow_histogram(&self) -> Result<SeasonalHistogram> {
        let episodes = self.load_cognitive_nodes_by_kind(NodeKind::Episode)?;
        let mut hist = SeasonalHistogram::day_of_week();

        for n in &episodes {
            if let NodePayload::Episode(ep) = &n.payload {
                let dow = temporal::day_of_week_utc(ep.occurred_at);
                hist.add(dow);
            }
        }

        Ok(hist)
    }

    // ── Temporal Relevance Scoring ──

    /// Score all nodes of a given kind by temporal relevance.
    ///
    /// Returns `(NodeId, relevance_score)` sorted by relevance descending.
    pub fn temporal_relevance_scores(
        &self,
        kind: NodeKind,
        config: &TemporalRelevanceConfig,
    ) -> Result<Vec<(NodeId, f64)>> {
        let nodes = self.load_cognitive_nodes_by_kind(kind)?;
        let ts = now();

        let mut scored: Vec<(NodeId, f64)> = nodes
            .iter()
            .map(|n| {
                let last_updated = n.attrs.last_updated_ms as f64 / 1000.0;

                // Extract deadline if applicable
                let deadline = match &n.payload {
                    NodePayload::Task(t) => t.deadline,
                    NodePayload::Goal(g) => g.deadline,
                    NodePayload::Opportunity(o) => Some(o.expires_at),
                    _ => None,
                };

                // Extract next_occurrence if routine
                let next_occ = match &n.payload {
                    NodePayload::Routine(r) => Some(r.next_occurrence(ts)),
                    _ => None,
                };

                let score = temporal::temporal_relevance_composite(
                    last_updated,
                    ts,
                    next_occ,
                    deadline,
                    config,
                );

                (n.id, score)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored)
    }

    // ── Deadline Dashboard ──

    /// Get all nodes approaching their deadlines, sorted by urgency.
    ///
    /// Returns `(NodeId, description, deadline, urgency)` for nodes
    /// with deadlines within `horizon_secs` of now.
    pub fn deadline_dashboard(&self, horizon_secs: f64) -> Result<Vec<(NodeId, String, f64, f64)>> {
        let ts = now();
        let config = DeadlineUrgencyConfig::default();
        let mut entries = Vec::new();

        // Tasks with deadlines
        let tasks = self.load_cognitive_nodes_by_kind(NodeKind::Task)?;
        for n in &tasks {
            if let NodePayload::Task(t) = &n.payload {
                if let Some(dl) = t.deadline {
                    if dl - ts < horizon_secs {
                        let urgency = temporal::deadline_urgency(dl, ts, &config);
                        entries.push((n.id, t.description.clone(), dl, urgency));
                    }
                }
            }
        }

        // Goals with deadlines
        let goals = self.load_cognitive_nodes_by_kind(NodeKind::Goal)?;
        for n in &goals {
            if let NodePayload::Goal(g) = &n.payload {
                if let Some(dl) = g.deadline {
                    if dl - ts < horizon_secs {
                        let urgency = temporal::deadline_urgency(dl, ts, &config);
                        entries.push((n.id, g.description.clone(), dl, urgency));
                    }
                }
            }
        }

        // Opportunities expiring soon
        let opps = self.load_cognitive_nodes_by_kind(NodeKind::Opportunity)?;
        for n in &opps {
            if let NodePayload::Opportunity(o) = &n.payload {
                if o.expires_at - ts < horizon_secs {
                    let urgency = temporal::deadline_urgency(o.expires_at, ts, &config);
                    entries.push((n.id, o.description.clone(), o.expires_at, urgency));
                }
            }
        }

        // Sort by urgency descending
        entries.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

        Ok(entries)
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::state::*;
    use crate::temporal::{
        BurstConfig, DeadlineUrgencyConfig, MotifConfig, PeriodicityConfig, TemporalRelevanceConfig,
    };

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    fn persist_episode(
        db: &YantrikDB,
        alloc: &mut NodeIdAllocator,
        summary: &str,
        occurred_at: f64,
    ) -> NodeId {
        let id = alloc.alloc(NodeKind::Episode);
        let node = CognitiveNode::new(
            id,
            summary.to_string(),
            NodePayload::Episode(EpisodePayload {
                memory_rid: format!("rid_{}", id.seq()),
                summary: summary.to_string(),
                occurred_at,
                participants: vec![],
            }),
        );
        db.persist_cognitive_node(&node).unwrap();
        id
    }

    #[test]
    fn test_episode_hour_histogram() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        // Create episodes at 9am UTC (32400s into day)
        for i in 0..10 {
            let ts = 86400.0 * i as f64 + 32400.0;
            persist_episode(&db, &mut alloc, "morning_check", ts);
        }
        // A few at 3pm UTC (54000s)
        for i in 0..3 {
            let ts = 86400.0 * i as f64 + 54000.0;
            persist_episode(&db, &mut alloc, "afternoon_review", ts);
        }
        db.persist_node_id_allocator(&alloc).unwrap();

        let hist = db.episode_hour_histogram().unwrap();
        let (peak_bin, peak_count) = hist.peak();
        assert_eq!(peak_bin, 9); // 9am UTC
        assert_eq!(peak_count, 10);
    }

    #[test]
    fn test_deadline_dashboard() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        // Task with deadline 1 hour from now
        let task_id = alloc.alloc(NodeKind::Task);
        let now_ts = crate::engine::now();
        let task = CognitiveNode::new(
            task_id,
            "Urgent task".to_string(),
            NodePayload::Task(TaskPayload {
                description: "Urgent task".to_string(),
                status: TaskStatus::InProgress,
                goal_id: None,
                deadline: Some(now_ts + 3600.0),
                priority: Priority::High,
                estimated_minutes: Some(30),
                prerequisites: vec![],
            }),
        );
        db.persist_cognitive_node(&task).unwrap();
        db.persist_node_id_allocator(&alloc).unwrap();

        let dashboard = db.deadline_dashboard(86400.0).unwrap(); // 1 day horizon
        assert_eq!(dashboard.len(), 1);
        assert_eq!(dashboard[0].0, task_id);
        assert!(dashboard[0].3 > 0.0); // Has some urgency
    }

    #[test]
    fn test_temporal_relevance_scoring() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        // Create two tasks — one recent, one old
        let recent_id = alloc.alloc(NodeKind::Task);
        let mut recent = CognitiveNode::new(
            recent_id,
            "Recent task".to_string(),
            NodePayload::Task(TaskPayload {
                description: "Recent task".to_string(),
                status: TaskStatus::InProgress,
                goal_id: None,
                deadline: None,
                priority: Priority::Medium,
                estimated_minutes: None,
                prerequisites: vec![],
            }),
        );
        // Recent — default last_updated_ms is now

        let old_id = alloc.alloc(NodeKind::Task);
        let mut old = CognitiveNode::new(
            old_id,
            "Old task".to_string(),
            NodePayload::Task(TaskPayload {
                description: "Old task".to_string(),
                status: TaskStatus::InProgress,
                goal_id: None,
                deadline: None,
                priority: Priority::Low,
                estimated_minutes: None,
                prerequisites: vec![],
            }),
        );
        old.attrs.last_updated_ms = 1000; // Very old

        db.persist_cognitive_node(&recent).unwrap();
        db.persist_cognitive_node(&old).unwrap();
        db.persist_node_id_allocator(&alloc).unwrap();

        let config = TemporalRelevanceConfig::default();
        let scores = db
            .temporal_relevance_scores(NodeKind::Task, &config)
            .unwrap();
        assert_eq!(scores.len(), 2);
        // Recent task should score higher
        assert_eq!(scores[0].0, recent_id);
        assert!(scores[0].1 > scores[1].1);
    }

    #[test]
    fn test_episode_motif_mining() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        // Create a repeating pattern: email → calendar → standup
        for day in 0..4 {
            let base = 1_700_000_000.0 + day as f64 * 86400.0;
            persist_episode(&db, &mut alloc, "email check", base + 100.0);
            persist_episode(&db, &mut alloc, "calendar review", base + 600.0);
            persist_episode(&db, &mut alloc, "standup meeting", base + 1200.0);
        }
        db.persist_node_id_allocator(&alloc).unwrap();

        let config = MotifConfig {
            max_gap_secs: 7200.0,
            min_length: 2,
            max_length: 3,
            min_occurrences: 3,
        };
        let motifs = db.mine_episode_motifs(&config).unwrap();
        // Should find the email→calendar motif
        let has_email_cal = motifs.iter().any(|m| {
            m.sequence.len() >= 2 && m.sequence[0] == "email" && m.sequence[1] == "calendar"
        });
        assert!(
            has_email_cal,
            "Should detect email→calendar motif: {:?}",
            motifs
        );
    }

    #[test]
    fn test_temporal_order_episodes() {
        let db = test_db();
        let mut alloc = NodeIdAllocator::new();

        let e1 = persist_episode(&db, &mut alloc, "first", 100.0);
        let e2 = persist_episode(&db, &mut alloc, "second", 200.0);
        let e3 = persist_episode(&db, &mut alloc, "third", 300.0);

        // Add temporal edges: e1 → e2 → e3
        db.persist_cognitive_edge(&CognitiveEdge {
            src: e1,
            dst: e2,
            kind: CognitiveEdgeKind::PrecedesTemporally,
            weight: 1.0,
            confidence: 1.0,
            observation_count: 1,
            created_at_ms: 100_000,
            last_confirmed_ms: 100_000,
        })
        .unwrap();
        db.persist_cognitive_edge(&CognitiveEdge {
            src: e2,
            dst: e3,
            kind: CognitiveEdgeKind::PrecedesTemporally,
            weight: 1.0,
            confidence: 1.0,
            observation_count: 1,
            created_at_ms: 200_000,
            last_confirmed_ms: 200_000,
        })
        .unwrap();
        db.persist_node_id_allocator(&alloc).unwrap();

        let order = db.temporal_order_episodes().unwrap();
        assert_eq!(order.ordered.len(), 3);
        assert_eq!(order.ordered[0], e1);
        assert_eq!(order.ordered[1], e2);
        assert_eq!(order.ordered[2], e3);
        assert!(order.cycles.is_empty());
    }
}
