//! Cognitive Extractor Cascade — turns natural language into structured
//! cognitive state updates.
//!
//! # 4-Tier Architecture
//!
//! ```text
//! User utterance
//!     │
//!     ├─►[Tier 1] Deterministic Recognizers  (<0.5ms, O(n) scan)
//!     │   Pattern table → CognitiveUpdate templates with slots
//!     │
//!     ├─►[Tier 2] Template Matching           (<3ms)
//!     │   Learned extraction templates from prior successes
//!     │   Keyword overlap scoring → slot filling
//!     │
//!     ├─►[Tier 3] Context Reinterpretation    (<2ms)
//!     │   Working set + graph neighborhood → disambiguate
//!     │   "I still haven't done it" + active task → task update
//!     │
//!     └─►[Tier 4] LLM Fallback Descriptor    (deferred)
//!         Returns an LlmExtractionRequest for the caller to fulfill
//!         Successful LLM results feed back into Tier 2 templates
//! ```
//!
//! # Flywheel Effect
//!
//! Every successful Tier 4 (LLM) extraction creates a new Tier 2 template.
//! Over time, the system learns to extract without LLM assistance.
//!
//! # Privacy
//!
//! - Raw text is processed in-memory only
//! - Templates store keyword patterns, not user content
//! - Extraction results are structured cognitive updates

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::state::{GoalStatus, NeedCategory, NodeId, Priority, TaskStatus};

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Cognitive Update Types
// ══════════════════════════════════════════════════════════════════════════════

/// The operation to perform on the cognitive graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UpdateOp {
    // ── Creation operations ──
    /// Create a new belief node.
    CreateBelief {
        proposition: String,
        domain: String,
        initial_log_odds: f64,
    },
    /// Create a new goal node.
    CreateGoal {
        description: String,
        priority: Priority,
        deadline: Option<f64>,
        completion_criteria: String,
    },
    /// Create a new task node.
    CreateTask {
        description: String,
        priority: Priority,
        deadline: Option<f64>,
        estimated_minutes: Option<u32>,
    },
    /// Create a new entity node.
    CreateEntity { name: String, entity_type: String },
    /// Create a new need node.
    CreateNeed {
        description: String,
        category: NeedCategory,
        intensity: f64,
    },
    /// Set or update a preference.
    SetPreference {
        domain: String,
        preferred: String,
        dispreferred: Option<String>,
    },
    /// Create or update a routine.
    CreateRoutine {
        description: String,
        action_description: String,
    },

    // ── Update operations (require target NodeId) ──
    /// Update a belief's log-odds.
    UpdateBelief {
        evidence_weight: f64,
        source: String,
    },
    /// Update a task's status.
    UpdateTaskStatus { new_status: TaskStatus },
    /// Update a goal's status.
    UpdateGoalStatus { new_status: GoalStatus },
    /// Update a goal's progress.
    UpdateGoalProgress { progress: f64 },

    // ── Correction operations ──
    /// Negate a belief (user says it's wrong).
    CorrectBelief { corrected_proposition: String },
    /// Mark something as no longer relevant.
    Deprecate,

    // ── Relationship operations ──
    /// Create a relationship entity (e.g., "my sister").
    CreateRelationship {
        person_name: String,
        relationship: String,
    },

    // ── Emotional/state markers ──
    /// Record an emotional state signal.
    EmotionalMarker { emotion: String, intensity: f64 },
}

/// How the extraction was derived.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExtractorTier {
    /// Tier 1: deterministic pattern match.
    Rule,
    /// Tier 2: learned template similarity match.
    Template,
    /// Tier 3: context-based reinterpretation.
    Context,
    /// Tier 4: LLM extraction (deferred).
    Llm,
}

impl ExtractorTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Rule => "rule",
            Self::Template => "template",
            Self::Context => "context",
            Self::Llm => "llm",
        }
    }
}

/// A single extracted cognitive update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveUpdate {
    /// The operation to perform.
    pub op: UpdateOp,
    /// Optional target node (for update operations).
    pub target: Option<NodeId>,
    /// Confidence in this extraction [0.0, 1.0].
    pub confidence: f64,
    /// Which tier produced this extraction.
    pub tier: ExtractorTier,
    /// The pattern or template that matched (for debugging/learning).
    pub match_source: String,
    /// The substring of input that triggered this match.
    pub matched_text: String,
}

/// Response from the extraction cascade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResponse {
    /// Extracted updates, ordered by confidence descending.
    pub updates: Vec<CognitiveUpdate>,
    /// Whether the cascade recommends LLM escalation.
    pub escalation_needed: bool,
    /// LLM extraction request descriptor (if escalation needed).
    pub llm_request: Option<LlmExtractionRequest>,
    /// Which tiers produced results.
    pub tiers_used: Vec<ExtractorTier>,
    /// Total extraction time in microseconds.
    pub extraction_time_us: u64,
}

/// Descriptor for a deferred LLM extraction request.
///
/// The extractor doesn't call the LLM itself — it returns this
/// descriptor for the caller (engine/companion) to fulfill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmExtractionRequest {
    /// The original text to extract from.
    pub text: String,
    /// System prompt for the LLM.
    pub system_prompt: String,
    /// Context summary to include.
    pub context_summary: String,
    /// Maximum tokens for the response.
    pub max_tokens: u32,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Extraction Context
// ══════════════════════════════════════════════════════════════════════════════

/// Context provided to the extractor for disambiguation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractionContext {
    /// Active goal descriptions (for "I finished it" → which goal?).
    pub active_goals: Vec<(NodeId, String)>,
    /// Active task descriptions.
    pub active_tasks: Vec<(NodeId, String)>,
    /// Recently mentioned entity names.
    pub recent_entities: Vec<(NodeId, String)>,
    /// Known relationship names (e.g., "sister" → "Alice").
    pub known_relationships: HashMap<String, String>,
    /// Current hour (0-23) for time-related extraction.
    pub current_hour: u8,
    /// Current day of week (0=Mon, 6=Sun).
    pub current_day: u8,
}

/// Configuration for the extraction cascade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractorConfig {
    /// Minimum confidence to include an extraction in results.
    pub min_confidence: f64,
    /// Confidence threshold below which LLM escalation is recommended.
    pub escalation_threshold: f64,
    /// Maximum number of updates to return.
    pub max_updates: usize,
    /// Whether Tier 3 (context reinterpretation) is enabled.
    pub enable_context_tier: bool,
    /// Whether to generate LLM escalation requests.
    pub enable_llm_escalation: bool,
    /// Maximum templates to retain in the template store.
    pub max_templates: usize,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.3,
            escalation_threshold: 0.5,
            max_updates: 10,
            enable_context_tier: true,
            enable_llm_escalation: true,
            max_templates: 500,
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Tier 1 — Deterministic Pattern Recognizers
// ══════════════════════════════════════════════════════════════════════════════

/// A deterministic extraction pattern.
///
/// Each pattern has a set of trigger phrases and a template for
/// the cognitive update to produce.
struct PatternRule {
    /// Trigger phrases (case-insensitive). Any match activates this rule.
    triggers: &'static [&'static str],
    /// The update operation template.
    op_template: OpTemplate,
    /// Base confidence for this pattern.
    confidence: f64,
    /// Category name for debugging.
    category: &'static str,
}

/// Template for generating an UpdateOp from matched text.
#[derive(Debug, Clone, Copy)]
enum OpTemplate {
    /// "I need to...", "I should..." → CreateTask
    TaskFromSuffix,
    /// "remind me to..." → CreateTask with high priority
    ReminderFromSuffix,
    /// "don't let me forget..." → CreateTask with high priority
    DontForgetFromSuffix,
    /// "I want to...", "my goal is..." → CreateGoal
    GoalFromSuffix,
    /// "I like...", "I love..." → SetPreference (positive)
    PositivePreference,
    /// "I hate...", "I don't like...", "I dislike..." → SetPreference (negative)
    NegativePreference,
    /// "I prefer X over Y" → SetPreference with both
    ComparativePreference,
    /// "I always...", "I usually..." → CreateRoutine
    RoutineFromSuffix,
    /// "I'm stressed", "I'm frustrated" etc → EmotionalMarker
    EmotionNegative,
    /// "I'm excited", "I'm happy" etc → EmotionalMarker
    EmotionPositive,
    /// "my sister...", "my manager..." → CreateRelationship
    RelationshipFromSuffix,
    /// "no, that's wrong", "actually..." → CorrectBelief
    Correction,
    /// "I'm done with...", "I finished..." → UpdateTaskStatus(Completed)
    TaskCompleted,
    /// "I need information about...", "I need to know..." → CreateNeed (Informational)
    InformationalNeed,
    /// "tomorrow I...", "next week I..." → CreateTask with deadline hint
    FutureTask,
    /// "every morning...", "every Monday..." → CreateRoutine
    PeriodicRoutine,
}

/// The built-in pattern table. Ordered by specificity (most specific first).
static PATTERN_RULES: &[PatternRule] = &[
    // ── High-specificity: reminders and don't-forget ──
    PatternRule {
        triggers: &["remind me to ", "reminder to ", "remind me about "],
        op_template: OpTemplate::ReminderFromSuffix,
        confidence: 0.90,
        category: "reminder",
    },
    PatternRule {
        triggers: &[
            "don't let me forget ",
            "dont let me forget ",
            "don't forget to ",
            "dont forget to ",
        ],
        op_template: OpTemplate::DontForgetFromSuffix,
        confidence: 0.90,
        category: "dont_forget",
    },
    // ── Corrections ──
    PatternRule {
        triggers: &[
            "no, that's wrong",
            "no that's wrong",
            "that's not right",
            "actually, ",
            "i meant ",
            "no, i meant ",
        ],
        op_template: OpTemplate::Correction,
        confidence: 0.75,
        category: "correction",
    },
    // ── Task completion ──
    PatternRule {
        triggers: &[
            "i'm done with ",
            "i finished ",
            "i've finished ",
            "i completed ",
            "i've completed ",
            "done with ",
        ],
        op_template: OpTemplate::TaskCompleted,
        confidence: 0.85,
        category: "task_complete",
    },
    // ── Preferences (comparative) ──
    PatternRule {
        triggers: &["i prefer ", "i'd rather "],
        op_template: OpTemplate::ComparativePreference,
        confidence: 0.80,
        category: "preference_comparative",
    },
    // ── Preferences (positive) ──
    PatternRule {
        triggers: &["i like ", "i love ", "i enjoy ", "i appreciate "],
        op_template: OpTemplate::PositivePreference,
        confidence: 0.80,
        category: "preference_positive",
    },
    // ── Preferences (negative) ──
    PatternRule {
        triggers: &[
            "i hate ",
            "i don't like ",
            "i dont like ",
            "i dislike ",
            "i can't stand ",
            "i cant stand ",
        ],
        op_template: OpTemplate::NegativePreference,
        confidence: 0.80,
        category: "preference_negative",
    },
    // ── Periodic routines ──
    PatternRule {
        triggers: &[
            "every morning ",
            "every evening ",
            "every night ",
            "every monday ",
            "every tuesday ",
            "every wednesday ",
            "every thursday ",
            "every friday ",
            "every saturday ",
            "every sunday ",
            "every week ",
            "every day ",
        ],
        op_template: OpTemplate::PeriodicRoutine,
        confidence: 0.80,
        category: "periodic_routine",
    },
    // ── Routines ──
    PatternRule {
        triggers: &["i always ", "i usually ", "i typically ", "i normally "],
        op_template: OpTemplate::RoutineFromSuffix,
        confidence: 0.70,
        category: "routine",
    },
    // ── Goals ──
    PatternRule {
        triggers: &[
            "my goal is to ",
            "my goal is ",
            "i want to achieve ",
            "i aim to ",
        ],
        op_template: OpTemplate::GoalFromSuffix,
        confidence: 0.85,
        category: "goal",
    },
    // ── Future tasks ──
    PatternRule {
        triggers: &[
            "tomorrow i ",
            "tomorrow i'll ",
            "next week i ",
            "later today i ",
            "tonight i ",
            "this weekend i ",
        ],
        op_template: OpTemplate::FutureTask,
        confidence: 0.75,
        category: "future_task",
    },
    // ── Tasks ──
    // NOTE: "i need to" must come AFTER more specific "i need to know" patterns
    PatternRule {
        triggers: &[
            "i should ",
            "i have to ",
            "i must ",
            "i gotta ",
            "i've got to ",
            "i've gotta ",
        ],
        op_template: OpTemplate::TaskFromSuffix,
        confidence: 0.80,
        category: "task",
    },
    // ── Informational needs ──
    PatternRule {
        triggers: &[
            "i need to know ",
            "i need information about ",
            "i need to find out ",
            "i want to know ",
            "how do i ",
            "what is ",
            "where can i ",
        ],
        op_template: OpTemplate::InformationalNeed,
        confidence: 0.70,
        category: "info_need",
    },
    // ── Generic "i need to" task (after informational needs) ──
    PatternRule {
        triggers: &["i need to "],
        op_template: OpTemplate::TaskFromSuffix,
        confidence: 0.80,
        category: "task",
    },
    // ── Wants/intentions (lower confidence → goals) ──
    PatternRule {
        triggers: &["i want to ", "i'd like to ", "i wish i could "],
        op_template: OpTemplate::GoalFromSuffix,
        confidence: 0.65,
        category: "want_goal",
    },
    // ── Emotional markers (negative) ──
    PatternRule {
        triggers: &[
            "i'm stressed",
            "i'm frustrated",
            "i'm overwhelmed",
            "i'm anxious",
            "i'm worried",
            "i'm tired",
            "i'm exhausted",
            "i'm upset",
            "i'm angry",
            "i'm sad",
        ],
        op_template: OpTemplate::EmotionNegative,
        confidence: 0.85,
        category: "emotion_negative",
    },
    // ── Emotional markers (positive) ──
    PatternRule {
        triggers: &[
            "i'm excited",
            "i'm happy",
            "i'm thrilled",
            "i'm grateful",
            "i'm relieved",
            "i'm proud",
            "i'm motivated",
            "i'm energized",
        ],
        op_template: OpTemplate::EmotionPositive,
        confidence: 0.85,
        category: "emotion_positive",
    },
    // ── Relationships ──
    PatternRule {
        triggers: &[
            "my sister ",
            "my brother ",
            "my mom ",
            "my mother ",
            "my dad ",
            "my father ",
            "my wife ",
            "my husband ",
            "my partner ",
            "my boss ",
            "my manager ",
            "my friend ",
            "my colleague ",
            "my coworker ",
        ],
        op_template: OpTemplate::RelationshipFromSuffix,
        confidence: 0.75,
        category: "relationship",
    },
];

/// Run Tier 1: deterministic pattern matching.
///
/// Scans the input text (lowercased) against all trigger phrases.
/// Returns extracted updates ordered by position in text.
fn tier1_extract(text: &str, _config: &ExtractorConfig) -> Vec<CognitiveUpdate> {
    let lower = text.to_lowercase();
    let mut updates = Vec::new();

    for rule in PATTERN_RULES {
        for &trigger in rule.triggers {
            if let Some(pos) = lower.find(trigger) {
                let suffix = &text[pos + trigger.len()..];
                let suffix_trimmed = suffix.trim();
                if suffix_trimmed.is_empty()
                    && !matches!(
                        rule.op_template,
                        OpTemplate::EmotionNegative
                            | OpTemplate::EmotionPositive
                            | OpTemplate::Correction
                    )
                {
                    continue; // Need content after trigger
                }

                if let Some(op) = build_op_from_template(rule.op_template, trigger, suffix_trimmed)
                {
                    updates.push(CognitiveUpdate {
                        op,
                        target: None,
                        confidence: rule.confidence,
                        tier: ExtractorTier::Rule,
                        match_source: format!("rule:{}", rule.category),
                        matched_text: text[pos..].to_string(),
                    });
                    break; // One match per rule
                }
            }
        }
    }

    // Sort by confidence descending
    updates.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    updates
}

/// Build an UpdateOp from a template and the matched suffix text.
fn build_op_from_template(template: OpTemplate, trigger: &str, suffix: &str) -> Option<UpdateOp> {
    // Clean suffix: remove trailing punctuation
    let clean = suffix.trim_end_matches(|c: char| c == '.' || c == '!' || c == '?' || c == ',');
    if clean.is_empty()
        && !matches!(
            template,
            OpTemplate::EmotionNegative | OpTemplate::EmotionPositive | OpTemplate::Correction
        )
    {
        return None;
    }

    match template {
        OpTemplate::TaskFromSuffix => Some(UpdateOp::CreateTask {
            description: clean.to_string(),
            priority: Priority::Medium,
            deadline: None,
            estimated_minutes: None,
        }),

        OpTemplate::ReminderFromSuffix | OpTemplate::DontForgetFromSuffix => {
            Some(UpdateOp::CreateTask {
                description: clean.to_string(),
                priority: Priority::High,
                deadline: None,
                estimated_minutes: None,
            })
        }

        OpTemplate::GoalFromSuffix => Some(UpdateOp::CreateGoal {
            description: clean.to_string(),
            priority: Priority::Medium,
            deadline: None,
            completion_criteria: String::new(),
        }),

        OpTemplate::PositivePreference => Some(UpdateOp::SetPreference {
            domain: infer_preference_domain(clean),
            preferred: clean.to_string(),
            dispreferred: None,
        }),

        OpTemplate::NegativePreference => Some(UpdateOp::SetPreference {
            domain: infer_preference_domain(clean),
            preferred: String::new(),
            dispreferred: Some(clean.to_string()),
        }),

        OpTemplate::ComparativePreference => {
            // Try to split on " over ", " to ", " than "
            let (preferred, dispreferred) = split_comparative(clean);
            Some(UpdateOp::SetPreference {
                domain: infer_preference_domain(&preferred),
                preferred,
                dispreferred: if dispreferred.is_empty() {
                    None
                } else {
                    Some(dispreferred)
                },
            })
        }

        OpTemplate::RoutineFromSuffix => Some(UpdateOp::CreateRoutine {
            description: format!("User routine: {}", clean),
            action_description: clean.to_string(),
        }),

        OpTemplate::PeriodicRoutine => {
            let period_hint = trigger.trim();
            Some(UpdateOp::CreateRoutine {
                description: format!("{}{}", period_hint, clean),
                action_description: clean.to_string(),
            })
        }

        OpTemplate::EmotionNegative => {
            let emotion = extract_emotion_from_trigger(trigger);
            Some(UpdateOp::EmotionalMarker {
                emotion,
                intensity: 0.7,
            })
        }

        OpTemplate::EmotionPositive => {
            let emotion = extract_emotion_from_trigger(trigger);
            Some(UpdateOp::EmotionalMarker {
                emotion,
                intensity: 0.7,
            })
        }

        OpTemplate::RelationshipFromSuffix => {
            let relationship = trigger.trim_start_matches("my ").trim();
            Some(UpdateOp::CreateRelationship {
                person_name: first_name_from_suffix(clean),
                relationship: relationship.to_string(),
            })
        }

        OpTemplate::Correction => Some(UpdateOp::CorrectBelief {
            corrected_proposition: clean.to_string(),
        }),

        OpTemplate::TaskCompleted => Some(UpdateOp::UpdateTaskStatus {
            new_status: TaskStatus::Completed,
        }),

        OpTemplate::InformationalNeed => Some(UpdateOp::CreateNeed {
            description: clean.to_string(),
            category: NeedCategory::Informational,
            intensity: 0.5,
        }),

        OpTemplate::FutureTask => Some(UpdateOp::CreateTask {
            description: clean.to_string(),
            priority: Priority::Medium,
            deadline: None, // Caller should parse "tomorrow" etc.
            estimated_minutes: None,
        }),
    }
}

/// Infer the preference domain from the preference text.
fn infer_preference_domain(text: &str) -> String {
    let lower = text.to_lowercase();
    let domains = [
        (
            &[
                "food",
                "eat",
                "drink",
                "coffee",
                "tea",
                "restaurant",
                "cuisine",
            ][..],
            "food",
        ),
        (
            &["music", "song", "playlist", "album", "band", "artist"],
            "music",
        ),
        (
            &[
                "work",
                "meeting",
                "office",
                "project",
                "code",
                "coding",
                "programming",
                "deploy",
            ],
            "work",
        ),
        (
            &["exercise", "gym", "run", "walk", "sport", "workout"],
            "health",
        ),
        (
            &["movie", "film", "show", "series", "watch", "book", "read"],
            "entertainment",
        ),
        (
            &["morning", "evening", "night", "schedule", "routine", "time"],
            "schedule",
        ),
        (
            &["notification", "alert", "remind", "email", "message"],
            "communication",
        ),
    ];

    for (keywords, domain) in &domains {
        if keywords.iter().any(|kw| lower.contains(kw)) {
            return domain.to_string();
        }
    }
    "general".to_string()
}

/// Split "X over Y" or "X to Y" or "X than Y" style comparatives.
fn split_comparative(text: &str) -> (String, String) {
    let lower = text.to_lowercase();
    // Order matters: check multi-word separators first
    for sep in &[" rather than ", " instead of ", " over ", " than ", " to "] {
        if let Some(pos) = lower.find(sep) {
            let preferred = text[..pos].trim().to_string();
            let dispreferred = text[pos + sep.len()..].trim().to_string();
            return (preferred, dispreferred);
        }
    }
    (text.to_string(), String::new())
}

/// Extract the emotion word from a trigger phrase like "i'm stressed".
fn extract_emotion_from_trigger(trigger: &str) -> String {
    trigger
        .trim()
        .rsplit(' ')
        .next()
        .unwrap_or("unknown")
        .to_string()
}

/// Extract the first name-like word from suffix text.
fn first_name_from_suffix(text: &str) -> String {
    // Take first capitalized word, or first word if none capitalized
    let first_word = text.split_whitespace().next().unwrap_or("");
    first_word.to_string()
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Tier 2 — Template Matching
// ══════════════════════════════════════════════════════════════════════════════

/// A learned extraction template.
///
/// Created from successful extractions (especially Tier 4 LLM results).
/// Stored in the template store and used for keyword-based similarity matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionTemplate {
    /// Unique identifier.
    pub id: u64,
    /// Keywords extracted from the original text.
    pub keywords: Vec<String>,
    /// The operation that was produced.
    pub op_template: SerializableOpTemplate,
    /// How many times this template has been used.
    pub use_count: u32,
    /// Cumulative confidence from successful uses.
    pub total_confidence: f64,
    /// When this template was created.
    pub created_at: f64,
    /// When this template was last used.
    pub last_used_at: f64,
}

impl ExtractionTemplate {
    /// Average confidence across all uses.
    pub fn avg_confidence(&self) -> f64 {
        if self.use_count == 0 {
            0.5
        } else {
            self.total_confidence / self.use_count as f64
        }
    }
}

/// Serializable version of OpTemplate (for template persistence).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SerializableOpTemplate {
    CreateTask { priority: Priority },
    CreateGoal { priority: Priority },
    SetPreference { domain: String },
    CreateNeed { category: NeedCategory },
    CreateRoutine,
    EmotionalMarker { emotion: String, positive: bool },
    CreateRelationship { relationship: String },
    Correction,
    TaskCompleted,
    CreateBelief { domain: String },
    UpdateBelief { direction: String },
    CreateEntity { entity_type: String },
}

/// Template store — holds all learned extraction templates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TemplateStore {
    pub templates: Vec<ExtractionTemplate>,
    pub total_created: u64,
    pub total_matches: u64,
}

impl TemplateStore {
    pub fn new() -> Self {
        Self {
            templates: Vec::new(),
            total_created: 0,
            total_matches: 0,
        }
    }

    /// Add a new template from a successful extraction.
    pub fn learn_template(
        &mut self,
        text: &str,
        op_template: SerializableOpTemplate,
        now: f64,
        max_templates: usize,
    ) {
        let keywords = extract_keywords(text);
        if keywords.is_empty() {
            return;
        }

        // Check for existing similar template (>50% keyword overlap)
        for existing in &mut self.templates {
            if existing.op_template == op_template
                && keyword_overlap(&existing.keywords, &keywords) > 0.5
            {
                // Reinforce existing template
                existing.use_count += 1;
                existing.last_used_at = now;
                // Merge keywords
                for kw in &keywords {
                    if !existing.keywords.contains(kw) {
                        existing.keywords.push(kw.clone());
                    }
                }
                return;
            }
        }

        let id = self.total_created;
        self.total_created += 1;

        self.templates.push(ExtractionTemplate {
            id,
            keywords,
            op_template,
            use_count: 1,
            total_confidence: 0.7,
            created_at: now,
            last_used_at: now,
        });

        // Enforce max templates
        if self.templates.len() > max_templates {
            // Remove least-used template
            if let Some(min_idx) = self
                .templates
                .iter()
                .enumerate()
                .min_by_key(|(_, t)| t.use_count)
                .map(|(i, _)| i)
            {
                self.templates.swap_remove(min_idx);
            }
        }
    }

    /// Find the best matching template for the given text.
    pub fn find_match(&self, text: &str, min_overlap: f64) -> Option<(usize, f64)> {
        let keywords = extract_keywords(text);
        if keywords.is_empty() {
            return None;
        }

        let mut best_idx = None;
        let mut best_score = 0.0f64;

        for (idx, template) in self.templates.iter().enumerate() {
            let overlap = keyword_overlap(&template.keywords, &keywords);
            if overlap >= min_overlap {
                let score = overlap * template.avg_confidence();
                if score > best_score {
                    best_score = score;
                    best_idx = Some(idx);
                }
            }
        }

        best_idx.map(|idx| (idx, best_score))
    }
}

/// Extract meaningful keywords from text (stopword removal + lowering).
fn extract_keywords(text: &str) -> Vec<String> {
    static STOPWORDS: &[&str] = &[
        "i", "me", "my", "we", "our", "you", "your", "the", "a", "an", "is", "am", "are", "was",
        "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "can", "to", "of", "in", "for", "on", "with",
        "at", "by", "from", "up", "about", "into", "through", "during", "before", "after", "and",
        "but", "or", "nor", "not", "no", "so", "if", "then", "that", "this", "these", "those",
        "it", "its", "just", "also", "very", "really", "quite", "much", "don't", "dont", "doesn't",
        "doesnt", "didn't", "didnt", "i'm", "im", "i've", "ive", "i'll", "ill", "i'd", "id", "let",
        "get", "got", "going", "go", "come", "make",
    ];

    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| w.len() >= 3 && !STOPWORDS.contains(w))
        .map(|w| w.to_string())
        .collect()
}

/// Compute Jaccard-like keyword overlap [0.0, 1.0].
fn keyword_overlap(a: &[String], b: &[String]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let matches = b.iter().filter(|kw| a.contains(kw)).count();
    let union = a.len().max(b.len());
    matches as f64 / union as f64
}

/// Run Tier 2: template matching.
fn tier2_extract(text: &str, store: &TemplateStore) -> Vec<CognitiveUpdate> {
    let min_overlap = 0.3;
    let mut updates = Vec::new();

    if let Some((idx, score)) = store.find_match(text, min_overlap) {
        let template = &store.templates[idx];
        let suffix = text.trim();
        let confidence = score.min(0.85); // Cap template confidence below rules

        if let Some(op) = build_op_from_serializable_template(&template.op_template, suffix) {
            updates.push(CognitiveUpdate {
                op,
                target: None,
                confidence,
                tier: ExtractorTier::Template,
                match_source: format!("template:{}", template.id),
                matched_text: suffix.to_string(),
            });
        }
    }

    updates
}

/// Build an UpdateOp from a serializable template and input text.
fn build_op_from_serializable_template(
    template: &SerializableOpTemplate,
    text: &str,
) -> Option<UpdateOp> {
    let clean = text.trim_end_matches(|c: char| c == '.' || c == '!' || c == '?');

    match template {
        SerializableOpTemplate::CreateTask { priority } => Some(UpdateOp::CreateTask {
            description: clean.to_string(),
            priority: *priority,
            deadline: None,
            estimated_minutes: None,
        }),
        SerializableOpTemplate::CreateGoal { priority } => Some(UpdateOp::CreateGoal {
            description: clean.to_string(),
            priority: *priority,
            deadline: None,
            completion_criteria: String::new(),
        }),
        SerializableOpTemplate::SetPreference { domain } => Some(UpdateOp::SetPreference {
            domain: domain.clone(),
            preferred: clean.to_string(),
            dispreferred: None,
        }),
        SerializableOpTemplate::CreateNeed { category } => Some(UpdateOp::CreateNeed {
            description: clean.to_string(),
            category: *category,
            intensity: 0.5,
        }),
        SerializableOpTemplate::CreateRoutine => Some(UpdateOp::CreateRoutine {
            description: clean.to_string(),
            action_description: clean.to_string(),
        }),
        SerializableOpTemplate::EmotionalMarker { emotion, .. } => {
            Some(UpdateOp::EmotionalMarker {
                emotion: emotion.clone(),
                intensity: 0.6,
            })
        }
        SerializableOpTemplate::CreateRelationship { relationship } => {
            Some(UpdateOp::CreateRelationship {
                person_name: first_name_from_suffix(clean),
                relationship: relationship.clone(),
            })
        }
        SerializableOpTemplate::Correction => Some(UpdateOp::CorrectBelief {
            corrected_proposition: clean.to_string(),
        }),
        SerializableOpTemplate::TaskCompleted => Some(UpdateOp::UpdateTaskStatus {
            new_status: TaskStatus::Completed,
        }),
        SerializableOpTemplate::CreateBelief { domain } => Some(UpdateOp::CreateBelief {
            proposition: clean.to_string(),
            domain: domain.clone(),
            initial_log_odds: 1.0,
        }),
        SerializableOpTemplate::UpdateBelief { direction } => Some(UpdateOp::UpdateBelief {
            evidence_weight: if direction == "positive" { 1.0 } else { -1.0 },
            source: "template_match".to_string(),
        }),
        SerializableOpTemplate::CreateEntity { entity_type } => Some(UpdateOp::CreateEntity {
            name: first_name_from_suffix(clean),
            entity_type: entity_type.clone(),
        }),
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  Tier 3 — Context Reinterpretation
// ══════════════════════════════════════════════════════════════════════════════

/// Anaphora/deictic patterns that need context to resolve.
static CONTEXT_PATTERNS: &[(&str, ContextAction)] = &[
    ("i still haven't done it", ContextAction::TaskPersistence),
    ("i still haven't done that", ContextAction::TaskPersistence),
    ("i haven't done it yet", ContextAction::TaskPersistence),
    ("not done yet", ContextAction::TaskPersistence),
    ("still working on it", ContextAction::TaskInProgress),
    ("still working on that", ContextAction::TaskInProgress),
    ("i'm working on it", ContextAction::TaskInProgress),
    ("that was great", ContextAction::PositiveAboutRecent),
    ("that was good", ContextAction::PositiveAboutRecent),
    ("that was awesome", ContextAction::PositiveAboutRecent),
    ("that was terrible", ContextAction::NegativeAboutRecent),
    ("that was bad", ContextAction::NegativeAboutRecent),
    ("that was awful", ContextAction::NegativeAboutRecent),
    ("i did it", ContextAction::TaskCompleted),
    ("it's done", ContextAction::TaskCompleted),
    ("that's done", ContextAction::TaskCompleted),
    ("finished it", ContextAction::TaskCompleted),
    ("cancel it", ContextAction::TaskCancelled),
    ("cancel that", ContextAction::TaskCancelled),
    ("forget about it", ContextAction::TaskCancelled),
    ("never mind", ContextAction::TaskCancelled),
    ("nevermind", ContextAction::TaskCancelled),
];

#[derive(Debug, Clone, Copy)]
enum ContextAction {
    TaskPersistence,     // Acknowledge task still pending
    TaskInProgress,      // Mark task as in progress
    TaskCompleted,       // Mark most recent active task as completed
    TaskCancelled,       // Cancel most recent active task
    PositiveAboutRecent, // Positive preference about recent entity
    NegativeAboutRecent, // Negative preference about recent entity
}

/// Run Tier 3: context reinterpretation.
///
/// Resolves anaphoric references ("it", "that") using the working set.
fn tier3_extract(text: &str, context: &ExtractionContext) -> Vec<CognitiveUpdate> {
    let lower = text.to_lowercase();
    let mut updates = Vec::new();

    for &(pattern, action) in CONTEXT_PATTERNS {
        if !lower.contains(pattern) {
            continue;
        }

        match action {
            ContextAction::TaskPersistence => {
                // "I still haven't done it" → find most recent active task
                if let Some((task_id, desc)) = context.active_tasks.first() {
                    updates.push(CognitiveUpdate {
                        op: UpdateOp::UpdateBelief {
                            evidence_weight: 0.5,
                            source: format!("user_persistence:{}", desc),
                        },
                        target: Some(*task_id),
                        confidence: 0.65,
                        tier: ExtractorTier::Context,
                        match_source: format!("context:task_persistence:{}", desc),
                        matched_text: pattern.to_string(),
                    });
                }
            }

            ContextAction::TaskInProgress => {
                if let Some((task_id, desc)) = context.active_tasks.first() {
                    updates.push(CognitiveUpdate {
                        op: UpdateOp::UpdateTaskStatus {
                            new_status: TaskStatus::InProgress,
                        },
                        target: Some(*task_id),
                        confidence: 0.70,
                        tier: ExtractorTier::Context,
                        match_source: format!("context:task_in_progress:{}", desc),
                        matched_text: pattern.to_string(),
                    });
                }
            }

            ContextAction::TaskCompleted => {
                if let Some((task_id, desc)) = context.active_tasks.first() {
                    updates.push(CognitiveUpdate {
                        op: UpdateOp::UpdateTaskStatus {
                            new_status: TaskStatus::Completed,
                        },
                        target: Some(*task_id),
                        confidence: 0.75,
                        tier: ExtractorTier::Context,
                        match_source: format!("context:task_completed:{}", desc),
                        matched_text: pattern.to_string(),
                    });
                }
            }

            ContextAction::TaskCancelled => {
                if let Some((task_id, desc)) = context.active_tasks.first() {
                    updates.push(CognitiveUpdate {
                        op: UpdateOp::UpdateTaskStatus {
                            new_status: TaskStatus::Cancelled,
                        },
                        target: Some(*task_id),
                        confidence: 0.70,
                        tier: ExtractorTier::Context,
                        match_source: format!("context:task_cancelled:{}", desc),
                        matched_text: pattern.to_string(),
                    });
                }
            }

            ContextAction::PositiveAboutRecent => {
                if let Some((_entity_id, name)) = context.recent_entities.first() {
                    updates.push(CognitiveUpdate {
                        op: UpdateOp::SetPreference {
                            domain: "general".to_string(),
                            preferred: name.clone(),
                            dispreferred: None,
                        },
                        target: None,
                        confidence: 0.60,
                        tier: ExtractorTier::Context,
                        match_source: format!("context:positive_about:{}", name),
                        matched_text: pattern.to_string(),
                    });
                }
            }

            ContextAction::NegativeAboutRecent => {
                if let Some((_entity_id, name)) = context.recent_entities.first() {
                    updates.push(CognitiveUpdate {
                        op: UpdateOp::SetPreference {
                            domain: "general".to_string(),
                            preferred: String::new(),
                            dispreferred: Some(name.clone()),
                        },
                        target: None,
                        confidence: 0.60,
                        tier: ExtractorTier::Context,
                        match_source: format!("context:negative_about:{}", name),
                        matched_text: pattern.to_string(),
                    });
                }
            }
        }

        break; // One context match per invocation
    }

    updates
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  Tier 4 — LLM Fallback Descriptor
// ══════════════════════════════════════════════════════════════════════════════

/// Build an LLM extraction request descriptor.
///
/// The extractor does NOT call the LLM — it returns a descriptor
/// for the engine/companion layer to fulfill asynchronously.
fn tier4_build_request(text: &str, context: &ExtractionContext) -> LlmExtractionRequest {
    let mut context_lines = Vec::new();

    if !context.active_tasks.is_empty() {
        let tasks: Vec<String> = context
            .active_tasks
            .iter()
            .take(5)
            .map(|(_, desc)| format!("- Task: {}", desc))
            .collect();
        context_lines.push(format!("Active tasks:\n{}", tasks.join("\n")));
    }

    if !context.active_goals.is_empty() {
        let goals: Vec<String> = context
            .active_goals
            .iter()
            .take(3)
            .map(|(_, desc)| format!("- Goal: {}", desc))
            .collect();
        context_lines.push(format!("Active goals:\n{}", goals.join("\n")));
    }

    if !context.recent_entities.is_empty() {
        let entities: Vec<String> = context
            .recent_entities
            .iter()
            .take(5)
            .map(|(_, name)| name.clone())
            .collect();
        context_lines.push(format!("Recent entities: {}", entities.join(", ")));
    }

    let context_summary = if context_lines.is_empty() {
        "No additional context.".to_string()
    } else {
        context_lines.join("\n\n")
    };

    LlmExtractionRequest {
        text: text.to_string(),
        system_prompt: LLM_EXTRACTION_PROMPT.to_string(),
        context_summary,
        max_tokens: 300,
    }
}

/// System prompt for LLM extraction.
static LLM_EXTRACTION_PROMPT: &str = r#"Extract structured cognitive updates from the user's message.
Return a JSON array of objects, each with:
- "op": one of "create_task", "create_goal", "set_preference", "create_need", "create_routine", "emotional_marker", "create_relationship", "correct_belief", "task_completed", "create_entity", "create_belief"
- "description": what was extracted
- "priority": "low", "medium", "high", or "critical" (if applicable)
- "domain": preference domain (if applicable)
- "emotion": emotion name (if applicable)
- "confidence": your confidence 0.0-1.0

Example: [{"op":"create_task","description":"call the dentist","priority":"medium","confidence":0.9}]

Only extract what is clearly stated or strongly implied. Do not invent information."#;

// ══════════════════════════════════════════════════════════════════════════════
// § 7  Cascade Orchestrator
// ══════════════════════════════════════════════════════════════════════════════

/// Run the full extraction cascade.
///
/// Executes tiers 1-3 locally, and optionally generates a Tier 4
/// LLM request descriptor if confidence is below threshold.
pub fn extract(
    text: &str,
    context: &ExtractionContext,
    template_store: &TemplateStore,
    config: &ExtractorConfig,
) -> ExtractionResponse {
    let start = std::time::Instant::now();

    let mut all_updates: Vec<CognitiveUpdate> = Vec::new();
    let mut tiers_used: Vec<ExtractorTier> = Vec::new();

    // ── Tier 1: Deterministic patterns ──
    let tier1_results = tier1_extract(text, config);
    if !tier1_results.is_empty() {
        tiers_used.push(ExtractorTier::Rule);
        all_updates.extend(tier1_results);
    }

    // ── Tier 2: Template matching ──
    let tier2_results = tier2_extract(text, template_store);
    if !tier2_results.is_empty() {
        tiers_used.push(ExtractorTier::Template);
        // Only add template results if they don't duplicate Tier 1
        for update in tier2_results {
            if !has_similar_update(&all_updates, &update) {
                all_updates.push(update);
            }
        }
    }

    // ── Tier 3: Context reinterpretation ──
    if config.enable_context_tier {
        let tier3_results = tier3_extract(text, context);
        if !tier3_results.is_empty() {
            tiers_used.push(ExtractorTier::Context);
            for update in tier3_results {
                if !has_similar_update(&all_updates, &update) {
                    all_updates.push(update);
                }
            }
        }
    }

    // ── Filter by minimum confidence ──
    all_updates.retain(|u| u.confidence >= config.min_confidence);

    // ── Sort by confidence descending ──
    all_updates.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // ── Truncate to max updates ──
    all_updates.truncate(config.max_updates);

    // ── Tier 4: LLM escalation decision ──
    let max_confidence = all_updates
        .iter()
        .map(|u| u.confidence)
        .fold(0.0f64, f64::max);
    let escalation_needed = config.enable_llm_escalation
        && (all_updates.is_empty() || max_confidence < config.escalation_threshold);

    let llm_request = if escalation_needed {
        Some(tier4_build_request(text, context))
    } else {
        None
    };

    let extraction_time_us = start.elapsed().as_micros() as u64;

    ExtractionResponse {
        updates: all_updates,
        escalation_needed,
        llm_request,
        tiers_used,
        extraction_time_us,
    }
}

/// Check if an update is semantically similar to any existing update.
fn has_similar_update(existing: &[CognitiveUpdate], new: &CognitiveUpdate) -> bool {
    existing
        .iter()
        .any(|e| std::mem::discriminant(&e.op) == std::mem::discriminant(&new.op))
}

/// Integrate an LLM extraction response back into the system.
///
/// Parses the LLM's JSON response and converts to CognitiveUpdates.
/// Also learns templates from successful extractions (the flywheel).
pub fn integrate_llm_response(
    original_text: &str,
    llm_json: &str,
    template_store: &mut TemplateStore,
    max_templates: usize,
    now: f64,
) -> Vec<CognitiveUpdate> {
    let parsed: Vec<serde_json::Value> = match serde_json::from_str(llm_json) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut updates = Vec::new();

    for item in &parsed {
        let op_str = item.get("op").and_then(|v| v.as_str()).unwrap_or("");
        let desc = item
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let confidence = item
            .get("confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5);
        let priority_str = item
            .get("priority")
            .and_then(|v| v.as_str())
            .unwrap_or("medium");
        let domain = item
            .get("domain")
            .and_then(|v| v.as_str())
            .unwrap_or("general");
        let emotion = item
            .get("emotion")
            .and_then(|v| v.as_str())
            .unwrap_or("neutral");

        let priority = Priority::from_str(priority_str);

        let (op, serializable) = match op_str {
            "create_task" => (
                UpdateOp::CreateTask {
                    description: desc.to_string(),
                    priority,
                    deadline: None,
                    estimated_minutes: None,
                },
                Some(SerializableOpTemplate::CreateTask { priority }),
            ),
            "create_goal" => (
                UpdateOp::CreateGoal {
                    description: desc.to_string(),
                    priority,
                    deadline: None,
                    completion_criteria: String::new(),
                },
                Some(SerializableOpTemplate::CreateGoal { priority }),
            ),
            "set_preference" => (
                UpdateOp::SetPreference {
                    domain: domain.to_string(),
                    preferred: desc.to_string(),
                    dispreferred: None,
                },
                Some(SerializableOpTemplate::SetPreference {
                    domain: domain.to_string(),
                }),
            ),
            "create_need" => (
                UpdateOp::CreateNeed {
                    description: desc.to_string(),
                    category: NeedCategory::Informational,
                    intensity: 0.5,
                },
                Some(SerializableOpTemplate::CreateNeed {
                    category: NeedCategory::Informational,
                }),
            ),
            "create_routine" => (
                UpdateOp::CreateRoutine {
                    description: desc.to_string(),
                    action_description: desc.to_string(),
                },
                Some(SerializableOpTemplate::CreateRoutine),
            ),
            "emotional_marker" => (
                UpdateOp::EmotionalMarker {
                    emotion: emotion.to_string(),
                    intensity: 0.7,
                },
                Some(SerializableOpTemplate::EmotionalMarker {
                    emotion: emotion.to_string(),
                    positive: confidence > 0.5,
                }),
            ),
            "create_relationship" => (
                UpdateOp::CreateRelationship {
                    person_name: desc.to_string(),
                    relationship: domain.to_string(),
                },
                Some(SerializableOpTemplate::CreateRelationship {
                    relationship: domain.to_string(),
                }),
            ),
            "correct_belief" | "correction" => (
                UpdateOp::CorrectBelief {
                    corrected_proposition: desc.to_string(),
                },
                Some(SerializableOpTemplate::Correction),
            ),
            "task_completed" => (
                UpdateOp::UpdateTaskStatus {
                    new_status: TaskStatus::Completed,
                },
                Some(SerializableOpTemplate::TaskCompleted),
            ),
            "create_entity" => {
                let etype = item
                    .get("entity_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                (
                    UpdateOp::CreateEntity {
                        name: desc.to_string(),
                        entity_type: etype.to_string(),
                    },
                    Some(SerializableOpTemplate::CreateEntity {
                        entity_type: etype.to_string(),
                    }),
                )
            }
            "create_belief" => (
                UpdateOp::CreateBelief {
                    proposition: desc.to_string(),
                    domain: domain.to_string(),
                    initial_log_odds: 1.0,
                },
                Some(SerializableOpTemplate::CreateBelief {
                    domain: domain.to_string(),
                }),
            ),
            _ => continue,
        };

        updates.push(CognitiveUpdate {
            op,
            target: None,
            confidence,
            tier: ExtractorTier::Llm,
            match_source: format!("llm:{}", op_str),
            matched_text: desc.to_string(),
        });

        // ── FLYWHEEL: Learn template from LLM success ──
        if let Some(tmpl) = serializable {
            if confidence >= 0.6 {
                template_store.learn_template(original_text, tmpl, now, max_templates);
            }
        }
    }

    updates
}

// ══════════════════════════════════════════════════════════════════════════════
// § 8  Extraction Summary
// ══════════════════════════════════════════════════════════════════════════════

/// Summary statistics for the extractor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractorSummary {
    pub total_templates: usize,
    pub total_template_matches: u64,
    pub total_templates_created: u64,
    pub avg_template_confidence: f64,
}

/// Generate an extractor summary.
pub fn summarize_extractor(store: &TemplateStore) -> ExtractorSummary {
    let avg_conf = if store.templates.is_empty() {
        0.0
    } else {
        store
            .templates
            .iter()
            .map(|t| t.avg_confidence())
            .sum::<f64>()
            / store.templates.len() as f64
    };

    ExtractorSummary {
        total_templates: store.templates.len(),
        total_template_matches: store.total_matches,
        total_templates_created: store.total_created,
        avg_template_confidence: avg_conf,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 9  Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> ExtractorConfig {
        ExtractorConfig::default()
    }

    fn empty_context() -> ExtractionContext {
        ExtractionContext::default()
    }

    fn empty_store() -> TemplateStore {
        TemplateStore::new()
    }

    // ── Tier 1: Deterministic patterns ──

    #[test]
    fn test_tier1_task_extraction() {
        let updates = tier1_extract("I need to call the dentist tomorrow", &default_config());
        assert!(!updates.is_empty(), "Should extract task");
        match &updates[0].op {
            UpdateOp::CreateTask {
                description,
                priority,
                ..
            } => {
                assert!(description.contains("call the dentist"));
                assert_eq!(*priority, Priority::Medium);
            }
            other => panic!("Expected CreateTask, got {:?}", other),
        }
        assert_eq!(updates[0].tier, ExtractorTier::Rule);
    }

    #[test]
    fn test_tier1_reminder_extraction() {
        let updates = tier1_extract("Remind me to buy groceries", &default_config());
        assert!(!updates.is_empty());
        match &updates[0].op {
            UpdateOp::CreateTask {
                description,
                priority,
                ..
            } => {
                assert!(description.contains("buy groceries"));
                assert_eq!(*priority, Priority::High);
            }
            other => panic!("Expected high-priority CreateTask, got {:?}", other),
        }
    }

    #[test]
    fn test_tier1_goal_extraction() {
        let updates = tier1_extract("My goal is to learn Spanish this year", &default_config());
        assert!(!updates.is_empty());
        match &updates[0].op {
            UpdateOp::CreateGoal { description, .. } => {
                assert!(description.contains("learn Spanish"));
            }
            other => panic!("Expected CreateGoal, got {:?}", other),
        }
    }

    #[test]
    fn test_tier1_positive_preference() {
        let updates = tier1_extract("I love working with Rust", &default_config());
        assert!(!updates.is_empty());
        match &updates[0].op {
            UpdateOp::SetPreference {
                preferred, domain, ..
            } => {
                assert!(preferred.contains("working with Rust"));
                assert_eq!(domain, "work");
            }
            other => panic!("Expected SetPreference, got {:?}", other),
        }
    }

    #[test]
    fn test_tier1_negative_preference() {
        let updates = tier1_extract("I hate early morning meetings", &default_config());
        assert!(!updates.is_empty());
        match &updates[0].op {
            UpdateOp::SetPreference { dispreferred, .. } => {
                assert!(dispreferred
                    .as_ref()
                    .unwrap()
                    .contains("early morning meetings"));
            }
            other => panic!("Expected SetPreference with dispreferred, got {:?}", other),
        }
    }

    #[test]
    fn test_tier1_comparative_preference() {
        let updates = tier1_extract("I prefer tea over coffee", &default_config());
        assert!(!updates.is_empty());
        match &updates[0].op {
            UpdateOp::SetPreference {
                preferred,
                dispreferred,
                ..
            } => {
                assert_eq!(preferred, "tea");
                assert_eq!(dispreferred.as_deref(), Some("coffee"));
            }
            other => panic!("Expected comparative preference, got {:?}", other),
        }
    }

    #[test]
    fn test_tier1_emotion_negative() {
        let updates = tier1_extract("I'm stressed about the deadline", &default_config());
        assert!(!updates.is_empty());
        match &updates[0].op {
            UpdateOp::EmotionalMarker { emotion, .. } => {
                assert_eq!(emotion, "stressed");
            }
            other => panic!("Expected EmotionalMarker, got {:?}", other),
        }
    }

    #[test]
    fn test_tier1_emotion_positive() {
        let updates = tier1_extract("I'm excited about the trip!", &default_config());
        assert!(!updates.is_empty());
        match &updates[0].op {
            UpdateOp::EmotionalMarker { emotion, .. } => {
                assert_eq!(emotion, "excited");
            }
            other => panic!("Expected EmotionalMarker, got {:?}", other),
        }
    }

    #[test]
    fn test_tier1_relationship() {
        let updates = tier1_extract("My sister Alice is visiting next week", &default_config());
        assert!(!updates.is_empty());
        match &updates[0].op {
            UpdateOp::CreateRelationship {
                person_name,
                relationship,
            } => {
                assert_eq!(person_name, "Alice");
                assert!(relationship.contains("sister"));
            }
            other => panic!("Expected CreateRelationship, got {:?}", other),
        }
    }

    #[test]
    fn test_tier1_routine() {
        let updates = tier1_extract(
            "I always check email first thing in the morning",
            &default_config(),
        );
        assert!(!updates.is_empty());
        match &updates[0].op {
            UpdateOp::CreateRoutine {
                action_description, ..
            } => {
                assert!(action_description.contains("check email"));
            }
            other => panic!("Expected CreateRoutine, got {:?}", other),
        }
    }

    #[test]
    fn test_tier1_periodic_routine() {
        let updates = tier1_extract(
            "Every Monday I review my goals for the week",
            &default_config(),
        );
        assert!(!updates.is_empty());
        match &updates[0].op {
            UpdateOp::CreateRoutine { description, .. } => {
                assert!(description.contains("monday") || description.contains("Monday"));
            }
            other => panic!("Expected CreateRoutine, got {:?}", other),
        }
    }

    #[test]
    fn test_tier1_correction() {
        let updates = tier1_extract("Actually, I meant the blue one", &default_config());
        assert!(!updates.is_empty());
        match &updates[0].op {
            UpdateOp::CorrectBelief {
                corrected_proposition,
            } => {
                assert!(corrected_proposition.contains("the blue one"));
            }
            other => panic!("Expected CorrectBelief, got {:?}", other),
        }
    }

    #[test]
    fn test_tier1_task_completed() {
        let updates = tier1_extract("I finished the report", &default_config());
        assert!(!updates.is_empty());
        match &updates[0].op {
            UpdateOp::UpdateTaskStatus { new_status } => {
                assert_eq!(*new_status, TaskStatus::Completed);
            }
            other => panic!("Expected UpdateTaskStatus(Completed), got {:?}", other),
        }
    }

    #[test]
    fn test_tier1_informational_need() {
        let updates = tier1_extract("How do I set up Docker on Ubuntu?", &default_config());
        assert!(!updates.is_empty());
        match &updates[0].op {
            UpdateOp::CreateNeed { category, .. } => {
                assert_eq!(*category, NeedCategory::Informational);
            }
            other => panic!("Expected CreateNeed(Informational), got {:?}", other),
        }
    }

    #[test]
    fn test_tier1_dont_forget() {
        let updates = tier1_extract("Don't let me forget to water the plants", &default_config());
        assert!(!updates.is_empty());
        match &updates[0].op {
            UpdateOp::CreateTask {
                description,
                priority,
                ..
            } => {
                assert!(description.contains("water the plants"));
                assert_eq!(*priority, Priority::High);
            }
            other => panic!("Expected high-priority CreateTask, got {:?}", other),
        }
    }

    #[test]
    fn test_tier1_no_match() {
        let updates = tier1_extract("Hello, how are you?", &default_config());
        assert!(
            updates.is_empty(),
            "Should not extract from generic greeting"
        );
    }

    #[test]
    fn test_tier1_empty_suffix_ignored() {
        let updates = tier1_extract("I need to", &default_config());
        assert!(
            updates.is_empty(),
            "Empty suffix should not produce extraction"
        );
    }

    // ── Tier 2: Template matching ──

    #[test]
    fn test_tier2_template_learning() {
        let mut store = TemplateStore::new();
        store.learn_template(
            "schedule a meeting with the design team",
            SerializableOpTemplate::CreateTask {
                priority: Priority::Medium,
            },
            1000.0,
            100,
        );

        assert_eq!(store.templates.len(), 1);
        assert!(store.templates[0]
            .keywords
            .contains(&"schedule".to_string()));
        assert!(store.templates[0].keywords.contains(&"meeting".to_string()));
    }

    #[test]
    fn test_tier2_template_matching() {
        let mut store = TemplateStore::new();
        store.learn_template(
            "schedule a meeting with the design team",
            SerializableOpTemplate::CreateTask {
                priority: Priority::Medium,
            },
            1000.0,
            100,
        );

        let results = tier2_extract("schedule a meeting with the engineering team", &store);
        assert!(!results.is_empty(), "Should match on keyword overlap");
        assert_eq!(results[0].tier, ExtractorTier::Template);
    }

    #[test]
    fn test_tier2_template_reinforcement() {
        let mut store = TemplateStore::new();
        let tmpl = SerializableOpTemplate::CreateTask {
            priority: Priority::High,
        };

        store.learn_template(
            "deploy the application to staging environment",
            tmpl.clone(),
            1000.0,
            100,
        );
        store.learn_template(
            "deploy the application to production environment",
            tmpl,
            2000.0,
            100,
        );

        // Should merge into existing template (keyword overlap > 50%):
        // "deploy", "application", "environment" overlap; "staging"/"production" differ
        assert_eq!(store.templates.len(), 1, "Similar templates should merge");
        assert_eq!(store.templates[0].use_count, 2);
    }

    #[test]
    fn test_tier2_no_match_low_overlap() {
        let mut store = TemplateStore::new();
        store.learn_template(
            "organize my photo album from vacation",
            SerializableOpTemplate::CreateTask {
                priority: Priority::Low,
            },
            1000.0,
            100,
        );

        let results = tier2_extract("what's the weather today?", &store);
        assert!(results.is_empty(), "Low overlap should not match");
    }

    // ── Tier 3: Context reinterpretation ──

    #[test]
    fn test_tier3_task_completed_with_context() {
        let task_id = NodeId::new(super::super::state::NodeKind::Task, 1);
        let context = ExtractionContext {
            active_tasks: vec![(task_id, "call the dentist".to_string())],
            ..Default::default()
        };

        let results = tier3_extract("I did it!", &context);
        assert!(!results.is_empty());
        match &results[0].op {
            UpdateOp::UpdateTaskStatus { new_status } => {
                assert_eq!(*new_status, TaskStatus::Completed);
            }
            other => panic!("Expected task completion, got {:?}", other),
        }
        assert_eq!(results[0].target, Some(task_id));
    }

    #[test]
    fn test_tier3_task_cancelled_with_context() {
        let task_id = NodeId::new(super::super::state::NodeKind::Task, 1);
        let context = ExtractionContext {
            active_tasks: vec![(task_id, "buy flowers".to_string())],
            ..Default::default()
        };

        let results = tier3_extract("never mind", &context);
        assert!(!results.is_empty());
        match &results[0].op {
            UpdateOp::UpdateTaskStatus { new_status } => {
                assert_eq!(*new_status, TaskStatus::Cancelled);
            }
            other => panic!("Expected task cancellation, got {:?}", other),
        }
    }

    #[test]
    fn test_tier3_positive_about_entity() {
        let entity_id = NodeId::new(super::super::state::NodeKind::Entity, 1);
        let context = ExtractionContext {
            recent_entities: vec![(entity_id, "Sushi Palace".to_string())],
            ..Default::default()
        };

        let results = tier3_extract("That was great!", &context);
        assert!(!results.is_empty());
        match &results[0].op {
            UpdateOp::SetPreference { preferred, .. } => {
                assert_eq!(preferred, "Sushi Palace");
            }
            other => panic!("Expected positive preference, got {:?}", other),
        }
    }

    #[test]
    fn test_tier3_no_context_no_match() {
        let context = ExtractionContext::default();
        let results = tier3_extract("I did it!", &context);
        assert!(
            results.is_empty(),
            "No context → no match for anaphoric reference"
        );
    }

    // ── Full cascade ──

    #[test]
    fn test_cascade_full_pipeline() {
        let config = default_config();
        let context = empty_context();
        let store = empty_store();

        let response = extract(
            "I need to call the dentist tomorrow",
            &context,
            &store,
            &config,
        );
        assert!(!response.updates.is_empty());
        assert!(response.tiers_used.contains(&ExtractorTier::Rule));
        assert!(
            !response.escalation_needed,
            "High-confidence rule match should not escalate"
        );
    }

    #[test]
    fn test_cascade_escalation_on_unknown() {
        let config = default_config();
        let context = empty_context();
        let store = empty_store();

        let response = extract("The weather is nice today", &context, &store, &config);
        assert!(
            response.escalation_needed,
            "No extraction should trigger escalation"
        );
        assert!(response.llm_request.is_some());
    }

    #[test]
    fn test_cascade_no_duplicate_ops() {
        let mut store = TemplateStore::new();
        store.learn_template(
            "I need to call the doctor about the results",
            SerializableOpTemplate::CreateTask {
                priority: Priority::Medium,
            },
            1000.0,
            100,
        );

        let config = default_config();
        let context = empty_context();

        // "I need to call the dentist" should match Tier 1 rule AND Tier 2 template
        // But cascade should deduplicate
        let response = extract("I need to call the dentist", &context, &store, &config);
        let task_count = response
            .updates
            .iter()
            .filter(|u| matches!(u.op, UpdateOp::CreateTask { .. }))
            .count();
        assert_eq!(
            task_count, 1,
            "Should deduplicate same-op from different tiers"
        );
    }

    #[test]
    fn test_cascade_multiple_extractions() {
        let config = default_config();
        let context = empty_context();
        let store = empty_store();

        let response = extract(
            "I'm stressed and I need to finish the report. Remind me to call mom.",
            &context,
            &store,
            &config,
        );
        assert!(
            response.updates.len() >= 2,
            "Should extract multiple updates: {:?}",
            response.updates
        );
    }

    // ── LLM integration (flywheel) ──

    #[test]
    fn test_llm_response_parsing() {
        let mut store = TemplateStore::new();
        let llm_json = r#"[
            {"op":"create_task","description":"buy birthday cake","priority":"high","confidence":0.9},
            {"op":"set_preference","description":"chocolate cake","domain":"food","confidence":0.7}
        ]"#;

        let updates = integrate_llm_response(
            "I need to buy a birthday cake, chocolate please",
            llm_json,
            &mut store,
            100,
            1000.0,
        );

        assert_eq!(updates.len(), 2);
        assert_eq!(updates[0].tier, ExtractorTier::Llm);

        // Flywheel: templates should have been created
        assert!(
            !store.templates.is_empty(),
            "LLM success should create templates"
        );
    }

    #[test]
    fn test_llm_response_invalid_json() {
        let mut store = TemplateStore::new();
        let updates = integrate_llm_response("test", "not valid json", &mut store, 100, 1000.0);
        assert!(updates.is_empty(), "Invalid JSON should return empty");
    }

    #[test]
    fn test_llm_response_unknown_op() {
        let mut store = TemplateStore::new();
        let llm_json = r#"[{"op":"unknown_op","description":"test","confidence":0.5}]"#;
        let updates = integrate_llm_response("test", llm_json, &mut store, 100, 1000.0);
        assert!(updates.is_empty(), "Unknown op should be skipped");
    }

    // ── Helper functions ──

    #[test]
    fn test_extract_keywords() {
        let kw = extract_keywords("I need to schedule a meeting with the design team");
        assert!(kw.contains(&"schedule".to_string()));
        assert!(kw.contains(&"meeting".to_string()));
        assert!(kw.contains(&"design".to_string()));
        assert!(kw.contains(&"team".to_string()));
        assert!(!kw.contains(&"the".to_string())); // Stopword
        assert!(!kw.contains(&"to".to_string())); // Stopword
    }

    #[test]
    fn test_keyword_overlap() {
        let a = vec![
            "meeting".to_string(),
            "schedule".to_string(),
            "team".to_string(),
        ];
        let b = vec![
            "meeting".to_string(),
            "team".to_string(),
            "review".to_string(),
        ];
        let overlap = keyword_overlap(&a, &b);
        assert!(
            (overlap - 2.0 / 3.0).abs() < 0.01,
            "2/3 overlap: {}",
            overlap
        );
    }

    #[test]
    fn test_split_comparative() {
        let (a, b) = split_comparative("tea over coffee");
        assert_eq!(a, "tea");
        assert_eq!(b, "coffee");

        let (a, b) = split_comparative("walking rather than driving");
        assert_eq!(a, "walking");
        assert_eq!(b, "driving");

        let (a, b) = split_comparative("just chocolate");
        assert_eq!(a, "just chocolate");
        assert!(b.is_empty());
    }

    #[test]
    fn test_preference_domain_inference() {
        assert_eq!(infer_preference_domain("morning coffee routine"), "food");
        assert_eq!(infer_preference_domain("jazz music at night"), "music");
        assert_eq!(infer_preference_domain("coding in Rust"), "work");
        assert_eq!(infer_preference_domain("something random"), "general");
    }

    // ── Template store ──

    #[test]
    fn test_template_store_max_enforcement() {
        let mut store = TemplateStore::new();
        let max = 3;

        for i in 0..5 {
            store.learn_template(
                &format!("unique phrase number {} with keyword{}", i, i),
                SerializableOpTemplate::CreateTask {
                    priority: Priority::Medium,
                },
                i as f64 * 100.0,
                max,
            );
        }

        assert!(store.templates.len() <= max, "Should enforce max templates");
    }

    #[test]
    fn test_extractor_summary() {
        let mut store = TemplateStore::new();
        store.learn_template(
            "test template with keywords",
            SerializableOpTemplate::CreateTask {
                priority: Priority::Medium,
            },
            1000.0,
            100,
        );

        let summary = summarize_extractor(&store);
        assert_eq!(summary.total_templates, 1);
        assert!(summary.avg_template_confidence > 0.0);
    }
}
