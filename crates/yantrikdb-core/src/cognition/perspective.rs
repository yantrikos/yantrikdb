//! CK-5.7 — Perspective Engine.
//!
//! Context-dependent dynamic reasoning: perspectives don't change facts,
//! they change what matters. Same cognitive graph, different weights.
//! Models how humans reason differently depending on current role/mood/context.
//!
//! # Design principles
//! - Pure functions only — no DB access (engine layer handles persistence)
//! - Perspectives are stackable (base personality + domain context + mood)
//! - Salience overrides are multiplicative across stack layers
//! - Cognitive style blends across active perspectives
//! - Auto-detection of perspective shifts from event patterns

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::state::{CognitiveEdgeKind, NodeId, NodeKind};

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Core Types
// ══════════════════════════════════════════════════════════════════════════════

/// Unique identifier for a perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PerspectiveId(pub u64);

/// Category of perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PerspectiveType {
    /// Domain-specific: "financial-planning", "coding", "health".
    Domain,
    /// Emotional state: "stressed", "calm", "excited".
    Emotional,
    /// Temporal focus: "urgent-deadline", "long-term-planning".
    Temporal,
    /// Social context: "1:1-meeting", "group-discussion", "solo-work".
    Social,
    /// Task mode: "creative-brainstorm", "analytical-review", "execution".
    TaskMode,
}

/// Temporal focus mode within a cognitive style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemporalFocus {
    /// Focus on the next few minutes/hours.
    Immediate,
    /// Focus on the next few days/weeks.
    ShortTerm,
    /// Focus on months/years ahead.
    LongTerm,
    /// Looking back, learning from the past.
    Reflective,
}

/// How reasoning behaviour changes in a perspective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveStyle {
    /// 0.0 = exploit known paths, 1.0 = explore widely.
    pub exploration_vs_exploitation: f64,
    /// 0.0 = conservative, 1.0 = aggressive.
    pub risk_tolerance: f64,
    /// 0.0 = concrete/detailed, 1.0 = abstract/big-picture.
    pub abstraction_level: f64,
    /// How much to weight social context (0.0-1.0).
    pub social_weight: f64,
    /// Primary temporal orientation.
    pub temporal_focus: TemporalFocus,
}

impl Default for CognitiveStyle {
    fn default() -> Self {
        Self {
            exploration_vs_exploitation: 0.5,
            risk_tolerance: 0.5,
            abstraction_level: 0.5,
            social_weight: 0.5,
            temporal_focus: TemporalFocus::ShortTerm,
        }
    }
}

impl CognitiveStyle {
    /// Weighted blend of two styles.
    pub fn blend(&self, other: &Self, other_weight: f64) -> Self {
        let w = other_weight.clamp(0.0, 1.0);
        let self_w = 1.0 - w;

        Self {
            exploration_vs_exploitation: self.exploration_vs_exploitation * self_w
                + other.exploration_vs_exploitation * w,
            risk_tolerance: self.risk_tolerance * self_w + other.risk_tolerance * w,
            abstraction_level: self.abstraction_level * self_w + other.abstraction_level * w,
            social_weight: self.social_weight * self_w + other.social_weight * w,
            // Temporal focus: use the higher-weighted one.
            temporal_focus: if w > 0.5 {
                other.temporal_focus
            } else {
                self.temporal_focus
            },
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Salience & Edge Weight Modifiers
// ══════════════════════════════════════════════════════════════════════════════

/// What to target with a salience override.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SalienceTarget {
    /// A specific node.
    Node(NodeId),
    /// All nodes of a kind.
    Kind(NodeKind),
    /// All nodes in a domain.
    Domain(String),
    /// All nodes with a tag.
    Tag(String),
}

/// Boost or dampen a node's salience.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalienceOverride {
    /// What to target.
    pub target: SalienceTarget,
    /// Multiplier: >1.0 boosts, <1.0 dampens, 0.0 suppresses.
    pub multiplier: f64,
}

/// Adjust edge weights contextually.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeWeightModifier {
    /// Which edge kind to modify.
    pub edge_kind: CognitiveEdgeKind,
    /// Multiplier for edge weight.
    pub multiplier: f64,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Activation Conditions
// ══════════════════════════════════════════════════════════════════════════════

/// When a perspective should auto-activate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActivationCondition {
    /// Time-of-day window (24h format).
    TimeWindow { start_hour: u8, end_hour: u8 },
    /// Specific app/context is active.
    AppContext { app_id: String },
    /// A specific goal is active.
    GoalActive(NodeId),
    /// Stress level exceeds threshold.
    StressAbove { threshold: f64 },
    /// User explicitly requested this perspective.
    ExplicitRequest,
    /// A pattern was detected in recent events.
    PatternDetected { pattern: String },
}

impl ActivationCondition {
    /// Check if this condition is met given context.
    pub fn is_met(&self, ctx: &ActivationContext) -> bool {
        match self {
            Self::TimeWindow {
                start_hour,
                end_hour,
            } => {
                if start_hour <= end_hour {
                    ctx.hour >= *start_hour && ctx.hour < *end_hour
                } else {
                    // Wraps midnight: e.g., 22-06.
                    ctx.hour >= *start_hour || ctx.hour < *end_hour
                }
            }
            Self::AppContext { app_id } => ctx.active_apps.contains(app_id),
            Self::GoalActive(goal_id) => ctx.active_goals.contains(goal_id),
            Self::StressAbove { threshold } => ctx.stress_level >= *threshold,
            Self::ExplicitRequest => false, // only by explicit API call
            Self::PatternDetected { pattern } => ctx.detected_patterns.contains(pattern),
        }
    }
}

/// Context for evaluating activation conditions.
#[derive(Debug, Clone, Default)]
pub struct ActivationContext {
    pub hour: u8,
    pub active_apps: Vec<String>,
    pub active_goals: Vec<NodeId>,
    pub stress_level: f64,
    pub detected_patterns: Vec<String>,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Perspective Definition
// ══════════════════════════════════════════════════════════════════════════════

/// A reasoning context that reweights the cognitive graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Perspective {
    pub id: PerspectiveId,
    pub name: String,
    pub perspective_type: PerspectiveType,
    /// Node salience overrides.
    pub salience_overrides: Vec<SalienceOverride>,
    /// Edge weight modifiers.
    pub edge_modifiers: Vec<EdgeWeightModifier>,
    /// Goals relevant in this perspective (boosted).
    pub active_goals: Vec<NodeId>,
    /// Goals irrelevant in this perspective (dampened).
    pub suppressed_goals: Vec<NodeId>,
    /// How reasoning behaviour changes.
    pub cognitive_style: CognitiveStyle,
    /// When to auto-activate.
    pub activation_conditions: Vec<ActivationCondition>,
    /// When this perspective was created (unix ms).
    pub created_at: u64,
    /// How many times this perspective has been used.
    pub usage_count: u32,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  Perspective Stack
// ══════════════════════════════════════════════════════════════════════════════

/// The active perspective context (supports layering).
///
/// Bottom = base personality, top = most recent context override.
/// Salience multipliers are multiplicative across layers.
/// Cognitive styles blend with higher layers getting more weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerspectiveStack {
    /// Ordered stack of active perspective IDs.
    pub stack: Vec<PerspectiveId>,
    /// Cached merged cognitive style.
    pub resolved_style: CognitiveStyle,
}

impl PerspectiveStack {
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            resolved_style: CognitiveStyle::default(),
        }
    }

    pub fn depth(&self) -> usize {
        self.stack.len()
    }

    pub fn is_active(&self, id: PerspectiveId) -> bool {
        self.stack.contains(&id)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  Perspective Store
// ══════════════════════════════════════════════════════════════════════════════

/// All known perspectives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerspectiveStore {
    pub perspectives: Vec<Perspective>,
    /// IDs of perspectives auto-discovered from usage patterns.
    pub learned_perspectives: Vec<PerspectiveId>,
    /// Next available ID.
    next_id: u64,
}

impl PerspectiveStore {
    pub fn new() -> Self {
        Self {
            perspectives: Vec::new(),
            learned_perspectives: Vec::new(),
            next_id: 1,
        }
    }

    /// Allocate a new perspective ID.
    pub fn alloc_id(&mut self) -> PerspectiveId {
        let id = PerspectiveId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Add a perspective.
    pub fn insert(&mut self, perspective: Perspective) {
        self.perspectives.push(perspective);
    }

    /// Find a perspective by ID.
    pub fn get(&self, id: PerspectiveId) -> Option<&Perspective> {
        self.perspectives.iter().find(|p| p.id == id)
    }

    /// Find a perspective by name.
    pub fn find_by_name(&self, name: &str) -> Option<&Perspective> {
        self.perspectives.iter().find(|p| p.name == name)
    }

    /// Number of perspectives.
    pub fn len(&self) -> usize {
        self.perspectives.len()
    }

    pub fn is_empty(&self) -> bool {
        self.perspectives.is_empty()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 7  Core Operations
// ══════════════════════════════════════════════════════════════════════════════

/// Push a perspective onto the stack. Recomputes merged style.
pub fn activate_perspective(
    stack: &mut PerspectiveStack,
    id: PerspectiveId,
    store: &PerspectiveStore,
) {
    if stack.is_active(id) {
        return; // already active
    }
    stack.stack.push(id);
    stack.resolved_style = resolve_cognitive_style(stack, store);

    // Increment usage count.
    // (Store mutation happens at engine level, not here.)
}

/// Remove a perspective from the stack. Recomputes merged style.
pub fn deactivate_perspective(
    stack: &mut PerspectiveStack,
    id: PerspectiveId,
    store: &PerspectiveStore,
) {
    stack.stack.retain(|&pid| pid != id);
    stack.resolved_style = resolve_cognitive_style(stack, store);
}

/// Compute effective salience for a node given all active perspectives.
///
/// Salience multipliers are multiplicative across stack layers.
/// A node's base salience is multiplied by each matching override.
pub fn resolve_salience(
    stack: &PerspectiveStack,
    node_id: NodeId,
    node_kind: NodeKind,
    node_domain: Option<&str>,
    node_tags: &[String],
    base_salience: f64,
    store: &PerspectiveStore,
) -> f64 {
    let mut multiplier = 1.0;

    for &pid in &stack.stack {
        if let Some(perspective) = store.get(pid) {
            for ovr in &perspective.salience_overrides {
                let matches = match &ovr.target {
                    SalienceTarget::Node(n) => *n == node_id,
                    SalienceTarget::Kind(k) => *k == node_kind,
                    SalienceTarget::Domain(d) => node_domain.map(|nd| nd == d).unwrap_or(false),
                    SalienceTarget::Tag(t) => node_tags.contains(t),
                };
                if matches {
                    multiplier *= ovr.multiplier;
                }
            }

            // Boost active goals, dampen suppressed goals.
            if node_kind == NodeKind::Goal {
                if perspective.active_goals.contains(&node_id) {
                    multiplier *= 2.0;
                }
                if perspective.suppressed_goals.contains(&node_id) {
                    multiplier *= 0.1;
                }
            }
        }
    }

    (base_salience * multiplier).clamp(0.0, 10.0)
}

/// Compute effective edge weight given active perspectives.
pub fn resolve_edge_weight(
    stack: &PerspectiveStack,
    edge_kind: CognitiveEdgeKind,
    base_weight: f64,
    store: &PerspectiveStore,
) -> f64 {
    let mut multiplier = 1.0;

    for &pid in &stack.stack {
        if let Some(perspective) = store.get(pid) {
            for modifier in &perspective.edge_modifiers {
                if modifier.edge_kind == edge_kind {
                    multiplier *= modifier.multiplier;
                }
            }
        }
    }

    (base_weight * multiplier).clamp(-10.0, 10.0)
}

/// Merge cognitive styles from all active perspectives.
///
/// Uses weighted blending: higher stack positions (more recent)
/// get progressively more weight.
pub fn resolve_cognitive_style(
    stack: &PerspectiveStack,
    store: &PerspectiveStore,
) -> CognitiveStyle {
    if stack.stack.is_empty() {
        return CognitiveStyle::default();
    }

    let n = stack.stack.len();
    let mut result = CognitiveStyle::default();

    for (i, &pid) in stack.stack.iter().enumerate() {
        if let Some(perspective) = store.get(pid) {
            // Weight increases with stack position: higher = more recent = more weight.
            let weight = (i + 1) as f64 / n as f64;
            result = result.blend(&perspective.cognitive_style, weight);
        }
    }

    result
}

// ══════════════════════════════════════════════════════════════════════════════
// § 8  Perspective Shift Detection
// ══════════════════════════════════════════════════════════════════════════════

/// A suggested perspective transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerspectiveTransition {
    /// Perspective to activate.
    pub activate: PerspectiveId,
    /// Perspective to deactivate (if any).
    pub deactivate: Option<PerspectiveId>,
    /// Why we suggest this transition.
    pub reason: String,
    /// Confidence in this suggestion (0.0-1.0).
    pub confidence: f64,
}

/// Detect perspective shifts based on activation conditions and context.
pub fn detect_perspective_shift(
    ctx: &ActivationContext,
    stack: &PerspectiveStack,
    store: &PerspectiveStore,
) -> Vec<PerspectiveTransition> {
    let mut transitions = Vec::new();

    for perspective in &store.perspectives {
        let already_active = stack.is_active(perspective.id);

        // Check if any activation condition is met.
        let met_conditions: Vec<&ActivationCondition> = perspective
            .activation_conditions
            .iter()
            .filter(|c| c.is_met(ctx))
            .collect();

        if !met_conditions.is_empty() && !already_active {
            let confidence =
                met_conditions.len() as f64 / perspective.activation_conditions.len().max(1) as f64;

            transitions.push(PerspectiveTransition {
                activate: perspective.id,
                deactivate: None,
                reason: format!(
                    "Conditions met for '{}': {} of {} conditions satisfied.",
                    perspective.name,
                    met_conditions.len(),
                    perspective.activation_conditions.len()
                ),
                confidence,
            });
        }

        // Check if an active perspective should deactivate.
        if already_active
            && !perspective.activation_conditions.is_empty()
            && met_conditions.is_empty()
        {
            transitions.push(PerspectiveTransition {
                activate: perspective.id, // not really activating
                deactivate: Some(perspective.id),
                reason: format!(
                    "No activation conditions met for '{}' — consider deactivating.",
                    perspective.name
                ),
                confidence: 0.5,
            });
        }
    }

    // Sort by confidence descending.
    transitions.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    transitions
}

// ══════════════════════════════════════════════════════════════════════════════
// § 9  Perspective Conflict Detection
// ══════════════════════════════════════════════════════════════════════════════

/// A conflict between active perspectives.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerspectiveConflict {
    pub perspective_a: PerspectiveId,
    pub perspective_b: PerspectiveId,
    pub conflict_type: ConflictType,
    pub severity: f64,
    pub description: String,
}

/// Type of conflict between perspectives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictType {
    /// Opposing exploration/exploitation preferences.
    ExplorationConflict,
    /// Opposing risk tolerances.
    RiskConflict,
    /// Goal suppression conflict (one activates what the other suppresses).
    GoalConflict,
    /// Contradictory edge weight modifiers on the same edge kind.
    EdgeWeightConflict,
}

/// Check for conflicts among active perspectives.
pub fn perspective_conflict_check(
    stack: &PerspectiveStack,
    store: &PerspectiveStore,
) -> Vec<PerspectiveConflict> {
    let mut conflicts = Vec::new();

    let active: Vec<&Perspective> = stack.stack.iter().filter_map(|&id| store.get(id)).collect();

    for i in 0..active.len() {
        for j in (i + 1)..active.len() {
            let a = active[i];
            let b = active[j];

            // Exploration conflict.
            let explore_diff = (a.cognitive_style.exploration_vs_exploitation
                - b.cognitive_style.exploration_vs_exploitation)
                .abs();
            if explore_diff > 0.5 {
                conflicts.push(PerspectiveConflict {
                    perspective_a: a.id,
                    perspective_b: b.id,
                    conflict_type: ConflictType::ExplorationConflict,
                    severity: explore_diff,
                    description: format!(
                        "'{}' favors {} while '{}' favors {}.",
                        a.name,
                        if a.cognitive_style.exploration_vs_exploitation > 0.5 {
                            "exploration"
                        } else {
                            "exploitation"
                        },
                        b.name,
                        if b.cognitive_style.exploration_vs_exploitation > 0.5 {
                            "exploration"
                        } else {
                            "exploitation"
                        },
                    ),
                });
            }

            // Risk conflict.
            let risk_diff =
                (a.cognitive_style.risk_tolerance - b.cognitive_style.risk_tolerance).abs();
            if risk_diff > 0.5 {
                conflicts.push(PerspectiveConflict {
                    perspective_a: a.id,
                    perspective_b: b.id,
                    conflict_type: ConflictType::RiskConflict,
                    severity: risk_diff,
                    description: format!(
                        "'{}' is {} while '{}' is {}.",
                        a.name,
                        if a.cognitive_style.risk_tolerance > 0.5 {
                            "risk-tolerant"
                        } else {
                            "risk-averse"
                        },
                        b.name,
                        if b.cognitive_style.risk_tolerance > 0.5 {
                            "risk-tolerant"
                        } else {
                            "risk-averse"
                        },
                    ),
                });
            }

            // Goal conflict: one activates what the other suppresses.
            for &goal in &a.active_goals {
                if b.suppressed_goals.contains(&goal) {
                    conflicts.push(PerspectiveConflict {
                        perspective_a: a.id,
                        perspective_b: b.id,
                        conflict_type: ConflictType::GoalConflict,
                        severity: 0.8,
                        description: format!(
                            "'{}' activates a goal that '{}' suppresses.",
                            a.name, b.name
                        ),
                    });
                    break; // one is enough
                }
            }

            // Edge weight conflict: opposing modifiers on same edge kind.
            for mod_a in &a.edge_modifiers {
                for mod_b in &b.edge_modifiers {
                    if mod_a.edge_kind == mod_b.edge_kind {
                        let ratio = if mod_b.multiplier > 0.0 {
                            mod_a.multiplier / mod_b.multiplier
                        } else {
                            10.0
                        };
                        if ratio > 3.0 || ratio < 0.33 {
                            conflicts.push(PerspectiveConflict {
                                perspective_a: a.id,
                                perspective_b: b.id,
                                conflict_type: ConflictType::EdgeWeightConflict,
                                severity: (ratio.max(1.0 / ratio) - 1.0).min(1.0),
                                description: format!(
                                    "'{}' and '{}' have opposing weights for {:?} edges.",
                                    a.name, b.name, mod_a.edge_kind
                                ),
                            });
                            break;
                        }
                    }
                }
            }
        }
    }

    // Sort by severity.
    conflicts.sort_by(|a, b| {
        b.severity
            .partial_cmp(&a.severity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    conflicts
}

// ══════════════════════════════════════════════════════════════════════════════
// § 10  Preset Perspectives
// ══════════════════════════════════════════════════════════════════════════════

/// Create commonly used preset perspectives.
pub fn create_preset(name: &str, id: PerspectiveId, now_ms: u64) -> Option<Perspective> {
    match name {
        "creative" => Some(Perspective {
            id,
            name: "creative-brainstorm".to_string(),
            perspective_type: PerspectiveType::TaskMode,
            salience_overrides: vec![],
            edge_modifiers: vec![
                EdgeWeightModifier {
                    edge_kind: CognitiveEdgeKind::AssociatedWith,
                    multiplier: 2.0,
                },
                EdgeWeightModifier {
                    edge_kind: CognitiveEdgeKind::Contradicts,
                    multiplier: 0.3,
                },
                EdgeWeightModifier {
                    edge_kind: CognitiveEdgeKind::SimilarTo,
                    multiplier: 1.5,
                },
            ],
            active_goals: vec![],
            suppressed_goals: vec![],
            cognitive_style: CognitiveStyle {
                exploration_vs_exploitation: 0.8,
                risk_tolerance: 0.7,
                abstraction_level: 0.6,
                social_weight: 0.3,
                temporal_focus: TemporalFocus::LongTerm,
            },
            activation_conditions: vec![],
            created_at: now_ms,
            usage_count: 0,
        }),
        "deadline" => Some(Perspective {
            id,
            name: "deadline-crunch".to_string(),
            perspective_type: PerspectiveType::Temporal,
            salience_overrides: vec![SalienceOverride {
                target: SalienceTarget::Kind(NodeKind::Task),
                multiplier: 2.5,
            }],
            edge_modifiers: vec![
                EdgeWeightModifier {
                    edge_kind: CognitiveEdgeKind::Requires,
                    multiplier: 2.0,
                },
                EdgeWeightModifier {
                    edge_kind: CognitiveEdgeKind::AssociatedWith,
                    multiplier: 0.3,
                },
            ],
            active_goals: vec![],
            suppressed_goals: vec![],
            cognitive_style: CognitiveStyle {
                exploration_vs_exploitation: 0.1,
                risk_tolerance: 0.2,
                abstraction_level: 0.2,
                social_weight: 0.3,
                temporal_focus: TemporalFocus::Immediate,
            },
            activation_conditions: vec![],
            created_at: now_ms,
            usage_count: 0,
        }),
        "reflective" => Some(Perspective {
            id,
            name: "reflective-review".to_string(),
            perspective_type: PerspectiveType::TaskMode,
            salience_overrides: vec![
                SalienceOverride {
                    target: SalienceTarget::Kind(NodeKind::Episode),
                    multiplier: 2.0,
                },
                SalienceOverride {
                    target: SalienceTarget::Kind(NodeKind::Belief),
                    multiplier: 1.5,
                },
            ],
            edge_modifiers: vec![EdgeWeightModifier {
                edge_kind: CognitiveEdgeKind::Causes,
                multiplier: 1.5,
            }],
            active_goals: vec![],
            suppressed_goals: vec![],
            cognitive_style: CognitiveStyle {
                exploration_vs_exploitation: 0.4,
                risk_tolerance: 0.3,
                abstraction_level: 0.7,
                social_weight: 0.4,
                temporal_focus: TemporalFocus::Reflective,
            },
            activation_conditions: vec![],
            created_at: now_ms,
            usage_count: 0,
        }),
        _ => None,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 11  Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{NodeId, NodeKind};

    fn goal(seq: u32) -> NodeId {
        NodeId::new(NodeKind::Goal, seq)
    }

    fn belief(seq: u32) -> NodeId {
        NodeId::new(NodeKind::Belief, seq)
    }

    fn task(seq: u32) -> NodeId {
        NodeId::new(NodeKind::Task, seq)
    }

    fn make_store_with_presets() -> PerspectiveStore {
        let mut store = PerspectiveStore::new();
        let creative_id = store.alloc_id();
        let deadline_id = store.alloc_id();
        let reflective_id = store.alloc_id();

        store.insert(create_preset("creative", creative_id, 1000).unwrap());
        store.insert(create_preset("deadline", deadline_id, 1000).unwrap());
        store.insert(create_preset("reflective", reflective_id, 1000).unwrap());

        store
    }

    fn make_custom_perspective(
        id: PerspectiveId,
        name: &str,
        explore: f64,
        risk: f64,
    ) -> Perspective {
        Perspective {
            id,
            name: name.to_string(),
            perspective_type: PerspectiveType::Domain,
            salience_overrides: vec![],
            edge_modifiers: vec![],
            active_goals: vec![],
            suppressed_goals: vec![],
            cognitive_style: CognitiveStyle {
                exploration_vs_exploitation: explore,
                risk_tolerance: risk,
                ..CognitiveStyle::default()
            },
            activation_conditions: vec![],
            created_at: 1000,
            usage_count: 0,
        }
    }

    // ── Stack operations ────────────────────────────────────────────────

    #[test]
    fn test_activate_deactivate() {
        let store = make_store_with_presets();
        let mut stack = PerspectiveStack::new();

        activate_perspective(&mut stack, PerspectiveId(1), &store);
        assert_eq!(stack.depth(), 1);
        assert!(stack.is_active(PerspectiveId(1)));

        deactivate_perspective(&mut stack, PerspectiveId(1), &store);
        assert_eq!(stack.depth(), 0);
        assert!(!stack.is_active(PerspectiveId(1)));
    }

    #[test]
    fn test_no_double_activation() {
        let store = make_store_with_presets();
        let mut stack = PerspectiveStack::new();

        activate_perspective(&mut stack, PerspectiveId(1), &store);
        activate_perspective(&mut stack, PerspectiveId(1), &store);
        assert_eq!(stack.depth(), 1); // not doubled
    }

    #[test]
    fn test_multiple_perspectives() {
        let store = make_store_with_presets();
        let mut stack = PerspectiveStack::new();

        activate_perspective(&mut stack, PerspectiveId(1), &store); // creative
        activate_perspective(&mut stack, PerspectiveId(2), &store); // deadline
        assert_eq!(stack.depth(), 2);
    }

    // ── Salience resolution ─────────────────────────────────────────────

    #[test]
    fn test_salience_boost() {
        let store = make_store_with_presets();
        let mut stack = PerspectiveStack::new();

        // Deadline mode boosts Task nodes.
        activate_perspective(&mut stack, PerspectiveId(2), &store);

        let salience = resolve_salience(&stack, task(1), NodeKind::Task, None, &[], 0.5, &store);

        // Should be boosted above base.
        assert!(salience > 0.5);
    }

    #[test]
    fn test_salience_dampen_domain() {
        let mut store = PerspectiveStore::new();
        let id = store.alloc_id();
        let mut p = make_custom_perspective(id, "focus", 0.5, 0.5);
        p.salience_overrides.push(SalienceOverride {
            target: SalienceTarget::Domain("health".to_string()),
            multiplier: 0.2,
        });
        store.insert(p);

        let mut stack = PerspectiveStack::new();
        activate_perspective(&mut stack, id, &store);

        let salience = resolve_salience(
            &stack,
            belief(1),
            NodeKind::Belief,
            Some("health"),
            &[],
            1.0,
            &store,
        );

        assert!(salience < 0.5); // dampened
    }

    #[test]
    fn test_goal_boost_and_suppress() {
        let mut store = PerspectiveStore::new();
        let id = store.alloc_id();
        let mut p = make_custom_perspective(id, "focus", 0.5, 0.5);
        p.active_goals.push(goal(1));
        p.suppressed_goals.push(goal(2));
        store.insert(p);

        let mut stack = PerspectiveStack::new();
        activate_perspective(&mut stack, id, &store);

        let boosted = resolve_salience(&stack, goal(1), NodeKind::Goal, None, &[], 0.5, &store);
        let suppressed = resolve_salience(&stack, goal(2), NodeKind::Goal, None, &[], 0.5, &store);

        assert!(boosted > 0.5);
        assert!(suppressed < 0.5);
    }

    // ── Edge weight resolution ──────────────────────────────────────────

    #[test]
    fn test_edge_weight_creative_mode() {
        let store = make_store_with_presets();
        let mut stack = PerspectiveStack::new();
        activate_perspective(&mut stack, PerspectiveId(1), &store); // creative

        // AssociatedWith should be boosted in creative mode.
        let weight = resolve_edge_weight(&stack, CognitiveEdgeKind::AssociatedWith, 1.0, &store);
        assert!(weight > 1.0);

        // Contradicts should be dampened.
        let weight = resolve_edge_weight(&stack, CognitiveEdgeKind::Contradicts, 1.0, &store);
        assert!(weight < 1.0);
    }

    // ── Cognitive style resolution ──────────────────────────────────────

    #[test]
    fn test_style_blend() {
        let base = CognitiveStyle {
            exploration_vs_exploitation: 0.3,
            risk_tolerance: 0.2,
            ..CognitiveStyle::default()
        };
        let overlay = CognitiveStyle {
            exploration_vs_exploitation: 0.9,
            risk_tolerance: 0.8,
            ..CognitiveStyle::default()
        };

        let blended = base.blend(&overlay, 0.7);
        // Should be closer to overlay.
        assert!(blended.exploration_vs_exploitation > 0.6);
        assert!(blended.risk_tolerance > 0.5);
    }

    #[test]
    fn test_resolved_style_from_stack() {
        let store = make_store_with_presets();
        let mut stack = PerspectiveStack::new();

        // Creative mode: high exploration.
        activate_perspective(&mut stack, PerspectiveId(1), &store);
        assert!(stack.resolved_style.exploration_vs_exploitation > 0.6);

        // Add deadline mode: should pull exploration down.
        activate_perspective(&mut stack, PerspectiveId(2), &store);
        // Blended, but deadline is on top → more weight.
        assert!(stack.resolved_style.exploration_vs_exploitation < 0.6);
    }

    // ── Activation condition tests ──────────────────────────────────────

    #[test]
    fn test_time_window_condition() {
        let cond = ActivationCondition::TimeWindow {
            start_hour: 9,
            end_hour: 17,
        };

        let mut ctx = ActivationContext::default();
        ctx.hour = 12;
        assert!(cond.is_met(&ctx));

        ctx.hour = 20;
        assert!(!cond.is_met(&ctx));
    }

    #[test]
    fn test_time_window_midnight_wrap() {
        let cond = ActivationCondition::TimeWindow {
            start_hour: 22,
            end_hour: 6,
        };

        let mut ctx = ActivationContext::default();
        ctx.hour = 23;
        assert!(cond.is_met(&ctx));

        ctx.hour = 3;
        assert!(cond.is_met(&ctx));

        ctx.hour = 12;
        assert!(!cond.is_met(&ctx));
    }

    #[test]
    fn test_goal_active_condition() {
        let cond = ActivationCondition::GoalActive(goal(42));
        let mut ctx = ActivationContext::default();

        assert!(!cond.is_met(&ctx));
        ctx.active_goals.push(goal(42));
        assert!(cond.is_met(&ctx));
    }

    // ── Perspective shift detection ─────────────────────────────────────

    #[test]
    fn test_detect_perspective_shift() {
        let mut store = PerspectiveStore::new();
        let id = store.alloc_id();
        let mut p = make_custom_perspective(id, "morning-routine", 0.5, 0.5);
        p.activation_conditions
            .push(ActivationCondition::TimeWindow {
                start_hour: 6,
                end_hour: 10,
            });
        store.insert(p);

        let mut ctx = ActivationContext::default();
        ctx.hour = 8;

        let stack = PerspectiveStack::new();
        let transitions = detect_perspective_shift(&ctx, &stack, &store);

        assert_eq!(transitions.len(), 1);
        assert_eq!(transitions[0].activate, id);
    }

    // ── Conflict detection ──────────────────────────────────────────────

    #[test]
    fn test_exploration_risk_conflict() {
        let mut store = PerspectiveStore::new();
        let id_a = store.alloc_id();
        let id_b = store.alloc_id();
        store.insert(make_custom_perspective(id_a, "explore", 0.9, 0.9));
        store.insert(make_custom_perspective(id_b, "cautious", 0.1, 0.1));

        let mut stack = PerspectiveStack::new();
        activate_perspective(&mut stack, id_a, &store);
        activate_perspective(&mut stack, id_b, &store);

        let conflicts = perspective_conflict_check(&stack, &store);
        assert!(conflicts.len() >= 2); // exploration + risk conflicts
    }

    #[test]
    fn test_goal_conflict() {
        let mut store = PerspectiveStore::new();
        let id_a = store.alloc_id();
        let id_b = store.alloc_id();

        let mut pa = make_custom_perspective(id_a, "focus-a", 0.5, 0.5);
        pa.active_goals.push(goal(1));

        let mut pb = make_custom_perspective(id_b, "focus-b", 0.5, 0.5);
        pb.suppressed_goals.push(goal(1));

        store.insert(pa);
        store.insert(pb);

        let mut stack = PerspectiveStack::new();
        activate_perspective(&mut stack, id_a, &store);
        activate_perspective(&mut stack, id_b, &store);

        let conflicts = perspective_conflict_check(&stack, &store);
        let goal_conflicts: Vec<_> = conflicts
            .iter()
            .filter(|c| c.conflict_type == ConflictType::GoalConflict)
            .collect();
        assert!(!goal_conflicts.is_empty());
    }

    // ── Preset perspectives ─────────────────────────────────────────────

    #[test]
    fn test_preset_creation() {
        let p = create_preset("creative", PerspectiveId(1), 1000);
        assert!(p.is_some());
        let p = p.unwrap();
        assert!(p.cognitive_style.exploration_vs_exploitation > 0.7);

        let p = create_preset("deadline", PerspectiveId(2), 1000);
        assert!(p.is_some());
        let p = p.unwrap();
        assert!(p.cognitive_style.exploration_vs_exploitation < 0.3);

        assert!(create_preset("nonexistent", PerspectiveId(3), 1000).is_none());
    }

    #[test]
    fn test_edge_weight_conflict() {
        let mut store = PerspectiveStore::new();
        let id_a = store.alloc_id();
        let id_b = store.alloc_id();

        let mut pa = make_custom_perspective(id_a, "boost-assoc", 0.5, 0.5);
        pa.edge_modifiers.push(EdgeWeightModifier {
            edge_kind: CognitiveEdgeKind::AssociatedWith,
            multiplier: 5.0,
        });

        let mut pb = make_custom_perspective(id_b, "dampen-assoc", 0.5, 0.5);
        pb.edge_modifiers.push(EdgeWeightModifier {
            edge_kind: CognitiveEdgeKind::AssociatedWith,
            multiplier: 0.1,
        });

        store.insert(pa);
        store.insert(pb);

        let mut stack = PerspectiveStack::new();
        activate_perspective(&mut stack, id_a, &store);
        activate_perspective(&mut stack, id_b, &store);

        let conflicts = perspective_conflict_check(&stack, &store);
        let edge_conflicts: Vec<_> = conflicts
            .iter()
            .filter(|c| c.conflict_type == ConflictType::EdgeWeightConflict)
            .collect();
        assert!(!edge_conflicts.is_empty());
    }
}
