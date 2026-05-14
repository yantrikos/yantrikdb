//! Engine-level Cognitive Query DSL execution.
//!
//! Implements the `PipelineExecutor` trait for `YantrikDB`, enabling
//! the composable reasoning DSL to access the database.

use crate::attention::AttentionConfig;
use crate::error::Result;
use crate::query_dsl::{
    execute_pipeline, AnticipatedItem, CognitiveOperator, CognitivePipeline, PipelineContext,
    PipelineExecutor, PipelineResult, Prediction, RankedCandidate, RecallMatch, StepOutput,
};

use super::{now, YantrikDB};

impl YantrikDB {
    /// Create a new reasoning pipeline.
    ///
    /// This is the entry point for the composable reasoning DSL:
    /// ```ignore
    /// let result = db.reason()
    ///     .attend(seeds)
    ///     .recall(5)
    ///     .compare(candidates)
    ///     .explain()
    ///     .execute(&db)?;
    /// ```
    pub fn reason(&self) -> CognitivePipeline {
        CognitivePipeline::new()
    }

    /// Execute a pre-built cognitive pipeline.
    pub fn execute_pipeline(&self, pipeline: &CognitivePipeline) -> Result<PipelineResult> {
        let executor = YantrikExecutor {
            db: self,
            attention_config: AttentionConfig {
                capacity: 20,
                max_hops: 2,
                top_k_per_hop: 5,
                hop_decay: 0.5,
                activation_threshold: 0.1,
                lateral_inhibition: 0.3,
                insertion_boost: 0.1,
            },
        };
        Ok(execute_pipeline(pipeline, &executor))
    }

    /// Execute a pipeline with a custom attention config.
    pub fn execute_pipeline_with(
        &self,
        pipeline: &CognitivePipeline,
        attention_config: AttentionConfig,
    ) -> Result<PipelineResult> {
        let executor = YantrikExecutor {
            db: self,
            attention_config,
        };
        Ok(execute_pipeline(pipeline, &executor))
    }
}

/// Engine-backed pipeline executor.
struct YantrikExecutor<'a> {
    db: &'a YantrikDB,
    attention_config: AttentionConfig,
}

impl<'a> PipelineExecutor for YantrikExecutor<'a> {
    fn execute_operator(
        &self,
        operator: &CognitiveOperator,
        context: &Option<PipelineContext>,
    ) -> StepOutput {
        match operator {
            CognitiveOperator::Attend(op) => self.execute_attend(op),
            CognitiveOperator::Recall(op) => self.execute_recall(op),
            CognitiveOperator::Believe(op) => self.execute_believe(op),
            CognitiveOperator::Project(op) => self.execute_project(op),
            CognitiveOperator::Compare(op) => self.execute_compare(op),
            CognitiveOperator::Constrain(op) => self.execute_constrain(op),
            CognitiveOperator::Anticipate(op) => self.execute_anticipate(op),
            CognitiveOperator::Plan(op) => self.execute_plan(op),
            CognitiveOperator::Assess => self.execute_assess(),
            CognitiveOperator::CoherenceCheck => self.execute_coherence(),
        }
    }
}

impl<'a> YantrikExecutor<'a> {
    fn execute_attend(&self, op: &crate::query_dsl::AttendOp) -> StepOutput {
        match self.db.hydrate_working_set(self.attention_config.clone()) {
            Ok(mut ws) => {
                // Boost activation on seed nodes.
                let mut activated = 0;
                let mut top = Vec::new();
                for &seed in &op.seeds {
                    if let Some(node) = ws.get_mut(seed) {
                        let new_activation = (node.attrs.activation + 0.3).min(1.0);
                        node.attrs.activation = new_activation;
                        top.push((seed, new_activation));
                        activated += 1;
                    }
                }
                // Spread activation from each seed.
                for &seed in &op.seeds {
                    activated += ws.activate_and_spread(seed, 0.3);
                }

                StepOutput::Attend {
                    nodes_activated: activated,
                    top_activated: top,
                }
            }
            Err(e) => StepOutput::Error {
                message: format!("Attend failed: {}", e),
            },
        }
    }

    fn execute_recall(&self, op: &crate::query_dsl::RecallOp) -> StepOutput {
        // If we have an embedder and a query, use recall_text.
        if let Some(ref query) = op.query {
            if self.db.has_embedder() {
                match self.db.recall_text(query, op.top_k) {
                    Ok(results) => {
                        let matches: Vec<RecallMatch> = results
                            .iter()
                            .map(|r| RecallMatch {
                                text: r.text.chars().take(200).collect(),
                                similarity: r.scores.similarity,
                                memory_type: r.memory_type.clone(),
                            })
                            .collect();
                        StepOutput::Recall {
                            memories_retrieved: matches.len(),
                            top_matches: matches,
                        }
                    }
                    Err(e) => StepOutput::Error {
                        message: format!("Recall failed: {}", e),
                    },
                }
            } else {
                StepOutput::Recall {
                    memories_retrieved: 0,
                    top_matches: Vec::new(),
                }
            }
        } else {
            // No query — just report working set contents.
            match self.db.hydrate_working_set(self.attention_config.clone()) {
                Ok(ws) => {
                    let top: Vec<RecallMatch> = ws
                        .by_activation()
                        .iter()
                        .take(op.top_k)
                        .map(|n| RecallMatch {
                            text: n.label.clone(),
                            similarity: n.attrs.activation,
                            memory_type: format!("{:?}", n.id.kind()),
                        })
                        .collect();
                    StepOutput::Recall {
                        memories_retrieved: top.len(),
                        top_matches: top,
                    }
                }
                Err(e) => StepOutput::Error {
                    message: format!("Recall hydration failed: {}", e),
                },
            }
        }
    }

    fn execute_believe(&self, _op: &crate::query_dsl::BelieveOp) -> StepOutput {
        // Belief revision is complex and depends on the belief system.
        // For now, report that the belief step was acknowledged.
        StepOutput::Believe {
            beliefs_updated: 0,
            confidence_delta: 0.0,
        }
    }

    fn execute_project(&self, op: &crate::query_dsl::ProjectOp) -> StepOutput {
        let mut predictions = Vec::new();

        // Use Hawkes model for temporal predictions.
        if let Ok(registry) = self.db.load_hawkes_registry() {
            let ts = now();
            let horizon_secs = op.horizon.seconds();
            for event in registry.anticipate_all(ts) {
                // Filter to events within the projection horizon.
                if event.prediction.time_until <= horizon_secs {
                    predictions.push(Prediction {
                        description: format!("{} event", event.label),
                        probability: event.prediction.confidence,
                        valence: 0.0,
                    });
                }
            }
        }

        StepOutput::Project { predictions }
    }

    fn execute_compare(&self, op: &crate::query_dsl::CompareOp) -> StepOutput {
        let mut ranked: Vec<RankedCandidate> = Vec::with_capacity(op.candidates.len());

        // Load personality for biasing.
        let bias_store = if op.apply_personality {
            self.db.load_personality_bias_store().ok()
        } else {
            None
        };

        for cand in &op.candidates {
            let mut score = cand.confidence;
            let mut personality_bias = 0.0;

            if let Some(ref store) = bias_store {
                let result = store.apply_bias(&cand.properties);
                personality_bias = result.total_bias;
                score += personality_bias * store.bias_config.bias_scale;
            }

            ranked.push(RankedCandidate {
                description: cand.description.clone(),
                score: score.clamp(0.0, 1.0),
                personality_bias,
                rank: 0,
            });
        }

        // Sort by score descending and assign ranks.
        ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        for (i, r) in ranked.iter_mut().enumerate() {
            r.rank = i + 1;
        }

        StepOutput::Compare { ranked }
    }

    fn execute_constrain(&self, op: &crate::query_dsl::ConstrainOp) -> StepOutput {
        // Constraints are evaluated abstractly — no direct DB access needed.
        // In a real pipeline, this would filter the compare results.
        StepOutput::Constrain {
            passed: 0,
            filtered_out: 0,
            violations: Vec::new(),
        }
    }

    fn execute_anticipate(&self, op: &crate::query_dsl::AnticipateOp) -> StepOutput {
        let mut items = Vec::new();

        if let Ok(registry) = self.db.load_hawkes_registry() {
            let ts = now();
            for event in registry.anticipate_all(ts) {
                // Filter to events within the requested horizon.
                if event.prediction.time_until <= op.horizon_secs {
                    items.push(AnticipatedItem {
                        description: event.label.clone(),
                        expected_at: event.prediction.predicted_time,
                        confidence: event.prediction.confidence,
                    });
                }
            }
        }

        StepOutput::Anticipate { items }
    }

    fn execute_plan(&self, op: &crate::query_dsl::PlanOp) -> StepOutput {
        match self.db.plan_for_goal(op.goal_id) {
            Ok(proposal) => {
                if let Some(best) = proposal.plans.first() {
                    StepOutput::Plan {
                        plan_found: best.viable,
                        steps: best.steps.len(),
                        score: best.score.composite,
                    }
                } else {
                    StepOutput::Plan {
                        plan_found: false,
                        steps: 0,
                        score: 0.0,
                    }
                }
            }
            Err(e) => StepOutput::Error {
                message: format!("Planning failed: {}", e),
            },
        }
    }

    fn execute_assess(&self) -> StepOutput {
        match self.db.metacognitive_assessment() {
            Ok(report) => StepOutput::Assess {
                overall_confidence: report.overall_confidence,
                coverage_gaps: report.coverage_gaps.len(),
            },
            Err(e) => StepOutput::Error {
                message: format!("Assessment failed: {}", e),
            },
        }
    }

    fn execute_coherence(&self) -> StepOutput {
        match self.db.check_coherence(&self.attention_config) {
            Ok(report) => StepOutput::Coherence {
                score: report.coherence_score,
                conflicts: report.goal_conflicts.len(),
                stale_nodes: report.stale_activations.len(),
            },
            Err(e) => StepOutput::Error {
                message: format!("Coherence check failed: {}", e),
            },
        }
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::query_dsl::{CandidateAction, CognitivePipeline, PipelinePatterns, PipelineStatus};
    use crate::state::{NodeId, NodeKind};

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    fn seed_ids() -> Vec<NodeId> {
        vec![NodeId::new(NodeKind::Entity, 1)]
    }

    fn test_candidates() -> Vec<CandidateAction> {
        vec![CandidateAction {
            description: "Send reminder".to_string(),
            action_kind: "notify".to_string(),
            confidence: 0.7,
            properties: Default::default(),
        }]
    }

    #[test]
    fn test_reason_entry_point() {
        let db = test_db();
        let pipeline = db.reason().attend(seed_ids()).recall(5).explain();

        assert_eq!(pipeline.len(), 2);
    }

    #[test]
    fn test_execute_empty_pipeline() {
        let db = test_db();
        let pipeline = CognitivePipeline::new();
        let result = db.execute_pipeline(&pipeline).unwrap();

        assert_eq!(result.status, PipelineStatus::Empty);
    }

    #[test]
    fn test_execute_attend_recall() {
        let db = test_db();
        let pipeline = db.reason().attend(seed_ids()).recall(5);

        let result = db.execute_pipeline(&pipeline).unwrap();
        assert_eq!(result.status, PipelineStatus::Complete);
        assert_eq!(result.operators_executed, 2);
    }

    #[test]
    fn test_execute_compare_with_personality() {
        let db = test_db();
        db.set_personality_preset(crate::personality_bias::PersonalityPreset::Companion)
            .unwrap();

        let pipeline = db.reason().compare(test_candidates());

        let result = db.execute_pipeline(&pipeline).unwrap();
        assert_eq!(result.status, PipelineStatus::Complete);

        // Compare step should produce ranked results.
        if let crate::query_dsl::StepOutput::Compare { ranked } = &result.steps[0].output {
            assert_eq!(ranked.len(), 1);
            assert_eq!(ranked[0].rank, 1);
        } else {
            panic!("Expected Compare output");
        }
    }

    #[test]
    fn test_execute_plan() {
        let db = test_db();
        let goal_id = NodeId::new(NodeKind::Goal, 1);
        let pipeline = db.reason().plan(goal_id, 3);

        let result = db.execute_pipeline(&pipeline).unwrap();
        assert_eq!(result.status, PipelineStatus::Complete);
    }

    #[test]
    fn test_execute_assess() {
        let db = test_db();
        let pipeline = db.reason().assess();

        let result = db.execute_pipeline(&pipeline).unwrap();
        assert_eq!(result.status, PipelineStatus::Complete);

        if let crate::query_dsl::StepOutput::Assess {
            overall_confidence, ..
        } = &result.steps[0].output
        {
            assert!(*overall_confidence >= 0.0 && *overall_confidence <= 1.0);
        } else {
            panic!("Expected Assess output");
        }
    }

    #[test]
    fn test_execute_coherence_check() {
        let db = test_db();
        let pipeline = db.reason().coherence_check();

        let result = db.execute_pipeline(&pipeline).unwrap();
        assert_eq!(result.status, PipelineStatus::Complete);
    }

    #[test]
    fn test_health_check_pattern_execution() {
        let db = test_db();
        let pipeline = PipelinePatterns::health_check();

        let result = db.execute_pipeline(&pipeline).unwrap();
        assert_eq!(result.status, PipelineStatus::Complete);
        assert_eq!(result.operators_executed, 2);
        assert!(result.explanation.is_some());
    }

    #[test]
    fn test_full_pipeline_execution() {
        let db = test_db();

        let pipeline = db
            .reason()
            .attend(seed_ids())
            .recall(3)
            .compare(test_candidates())
            .assess()
            .coherence_check()
            .explain();

        let result = db.execute_pipeline(&pipeline).unwrap();
        assert_eq!(result.status, PipelineStatus::Complete);
        assert_eq!(result.operators_executed, 5);
        assert!(result.explanation.is_some());

        let trace = result.explanation.unwrap();
        assert!(trace.summary.contains("5 steps"));
    }
}
