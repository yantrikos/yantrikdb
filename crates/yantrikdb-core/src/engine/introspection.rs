//! Engine-level self-awareness introspection API.
//!
//! Loads all subsystem states from the database and delegates to
//! the cognition layer for report generation.

use crate::error::Result;
use crate::introspection::{
    explain_discovery, learning_summary_text, milestones_in_range, what_have_you_learned,
    DiscoveryExplanation, IntrospectionInputs, IntrospectionReport, LearningMilestone,
};

use super::{now, YantrikDB};

impl YantrikDB {
    /// Generate a comprehensive introspection report.
    ///
    /// Loads all subsystem states (beliefs, skills, experiments, calibration,
    /// world model, extractor, observer) and aggregates them into a single
    /// report showing everything the system has learned.
    pub fn what_have_you_learned(&self) -> Result<IntrospectionReport> {
        let belief_store = self.load_belief_store()?;
        let event_buffer = self.load_event_buffer()?;
        let skill_registry = self.load_skill_registry()?;
        let experiment_registry = self.load_experiment_registry()?;
        let template_store = self.load_template_store()?;
        let learning_state = self.load_learning_state()?;
        let transition_model = self.load_transition_model()?;
        let ts = now();

        let inputs = IntrospectionInputs {
            belief_store: &belief_store,
            event_buffer: &event_buffer,
            skill_registry: &skill_registry,
            experiment_registry: &experiment_registry,
            template_store: &template_store,
            learning_state: &learning_state,
            transition_model: &transition_model,
            now: ts,
        };

        Ok(what_have_you_learned(&inputs))
    }

    /// Get a compact, human-readable learning summary string.
    pub fn learning_summary(&self) -> Result<String> {
        let report = self.what_have_you_learned()?;
        Ok(learning_summary_text(&report))
    }

    /// Explain a specific discovery by its dedup_key.
    pub fn explain_discovery(&self, dedup_key: &str) -> Result<Option<DiscoveryExplanation>> {
        let belief_store = self.load_belief_store()?;
        Ok(explain_discovery(&belief_store, dedup_key))
    }

    /// Get learning milestones within a time range.
    pub fn learning_milestones(&self, since: f64, until: f64) -> Result<Vec<LearningMilestone>> {
        let belief_store = self.load_belief_store()?;
        let event_buffer = self.load_event_buffer()?;
        let skill_registry = self.load_skill_registry()?;
        let experiment_registry = self.load_experiment_registry()?;
        let template_store = self.load_template_store()?;
        let learning_state = self.load_learning_state()?;
        let transition_model = self.load_transition_model()?;
        let ts = now();

        let inputs = IntrospectionInputs {
            belief_store: &belief_store,
            event_buffer: &event_buffer,
            skill_registry: &skill_registry,
            experiment_registry: &experiment_registry,
            template_store: &template_store,
            learning_state: &learning_state,
            transition_model: &transition_model,
            now: ts,
        };

        Ok(milestones_in_range(&inputs, since, until))
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use crate::engine::YantrikDB;
    use crate::flywheel::{
        AutonomousBelief, BeliefCategory, BeliefEvidence, BeliefStage, BeliefStore,
    };
    use crate::observer::EventKind;

    fn test_db() -> YantrikDB {
        YantrikDB::new(":memory:", 8).unwrap()
    }

    fn temporal_evidence() -> BeliefEvidence {
        BeliefEvidence::Temporal {
            event_kind: EventKind::AppOpened,
            peak_hour: 9,
            quiet_hours: vec![],
            distribution_skew: 0.8,
        }
    }

    fn preference_evidence(preferred: &str) -> BeliefEvidence {
        BeliefEvidence::Preference {
            preferred_value: preferred.to_string(),
            accept_rate: 0.8,
            reject_rate: 0.2,
            sample_size: 10,
        }
    }

    fn behavioral_evidence() -> BeliefEvidence {
        BeliefEvidence::Behavioral {
            from_app: 1,
            to_app: 2,
            transition_count: 5,
            avg_gap_ms: 2000.0,
        }
    }

    #[test]
    fn test_empty_introspection() {
        let db = test_db();
        let report = db.what_have_you_learned().unwrap();

        assert_eq!(report.total_observations, 0);
        assert_eq!(report.beliefs_formed, 0);
        assert_eq!(report.skills_discovered, 0);
        assert!(report.discoveries.is_empty());
    }

    #[test]
    fn test_learning_summary_empty() {
        let db = test_db();
        let summary = db.learning_summary().unwrap();
        assert!(summary.contains("No learning data"));
    }

    #[test]
    fn test_introspection_with_beliefs() {
        let db = test_db();

        let mut store = BeliefStore::new();

        let mut b = AutonomousBelief::new(
            "Morning person".to_string(),
            BeliefCategory::Temporal,
            "temporal:morning".to_string(),
            temporal_evidence(),
            86400.0 * 100.0,
        );
        b.confidence = 0.85;
        b.stage = BeliefStage::Established;
        b.confirming_observations = 12;
        b.contradicting_observations = 3;
        store.upsert(b);

        // Override counters after upsert to simulate history
        store.total_formed = 3;
        store.total_established = 1;

        db.save_belief_store(&store).unwrap();

        let report = db.what_have_you_learned().unwrap();
        assert_eq!(report.beliefs_formed, 3);
        assert_eq!(report.beliefs_established, 1);
        assert_eq!(report.beliefs_active, 1);

        assert!(!report.discoveries.is_empty());
        assert!(report.discoveries[0].description.contains("Morning"));
    }

    #[test]
    fn test_explain_discovery() {
        let db = test_db();

        let mut store = BeliefStore::new();
        let mut b = AutonomousBelief::new(
            "Likes dark mode".to_string(),
            BeliefCategory::Preference,
            "pref:dark_mode".to_string(),
            preference_evidence("dark"),
            86400.0 * 100.0,
        );
        b.confidence = 0.8;
        b.confirming_observations = 10;
        b.contradicting_observations = 2;
        b.stage = BeliefStage::Established;
        store.upsert(b);
        db.save_belief_store(&store).unwrap();

        let explanation = db.explain_discovery("pref:dark_mode").unwrap().unwrap();
        assert_eq!(explanation.confirming_observations, 10);
        assert_eq!(explanation.contradicting_observations, 2);
        assert!(explanation.confirmation_ratio > 0.8);

        assert!(db.explain_discovery("nonexistent").unwrap().is_none());
    }

    #[test]
    fn test_learning_summary_with_data() {
        let db = test_db();

        let mut store = BeliefStore::new();

        let mut b = AutonomousBelief::new(
            "Test".to_string(),
            BeliefCategory::Behavioral,
            "beh:test".to_string(),
            behavioral_evidence(),
            86400.0 * 100.0,
        );
        b.confidence = 0.75;
        b.stage = BeliefStage::Established;
        store.upsert(b);
        store.total_formed = 7; // Override after upsert
        db.save_belief_store(&store).unwrap();

        let summary = db.learning_summary().unwrap();
        assert!(summary.contains("7 beliefs formed"), "Summary: {}", summary);
    }

    #[test]
    fn test_learning_milestones() {
        let db = test_db();

        let mut store = BeliefStore::new();
        let mut b = AutonomousBelief::new(
            "Milestone belief".to_string(),
            BeliefCategory::Temporal,
            "temp:milestone".to_string(),
            temporal_evidence(),
            86400.0 * 100.0 + 50.0,
        );
        b.confidence = 0.9;
        b.stage = BeliefStage::Certain;
        b.last_updated = 86400.0 * 100.0 + 100.0;
        store.upsert(b);
        db.save_belief_store(&store).unwrap();

        let milestones = db
            .learning_milestones(86400.0 * 100.0, 86400.0 * 100.0 + 200.0)
            .unwrap();
        assert!(!milestones.is_empty());
    }
}
