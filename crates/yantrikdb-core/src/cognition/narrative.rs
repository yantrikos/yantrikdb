//! CK-5.3 — Episodic Narrative Memory.
//!
//! Chains Episode nodes into Narrative Arcs with chapter boundaries,
//! turning points, and resolution status. Provides autobiographical
//! continuity — "how have I grown?" and "what's my arc with person X?"
//!
//! # Design principles
//! - Pure functions only — no DB access (engine layer handles persistence)
//! - Narrative arcs emerge from episode patterns, not imposed
//! - Chapter boundaries detected from time gaps, sentiment shifts, topic changes
//! - Turning points identified from large sentiment changes or goal transitions
//! - Arcs can merge when they turn out to be the same story

use serde::{Deserialize, Serialize};

use crate::state::NodeId;

// ══════════════════════════════════════════════════════════════════════════════
// § 1  Core Types
// ══════════════════════════════════════════════════════════════════════════════

/// Unique identifier for a narrative arc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ArcId(pub u64);

/// The thematic category of a narrative arc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArcTheme {
    /// Learning a new skill or growing competence.
    Growth,
    /// Overcoming obstacles or difficulties.
    Challenge,
    /// Evolution of a relationship with a person.
    Relationship,
    /// Progress on a project or work initiative.
    Project,
    /// Building or breaking a recurring behavior.
    Habit,
    /// Exploring something new or making a discovery.
    Discovery,
    /// Dealing with loss or ending.
    Loss,
    /// Bouncing back from a setback.
    Recovery,
}

impl ArcTheme {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Growth => "growth",
            Self::Challenge => "challenge",
            Self::Relationship => "relationship",
            Self::Project => "project",
            Self::Habit => "habit",
            Self::Discovery => "discovery",
            Self::Loss => "loss",
            Self::Recovery => "recovery",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "growth" => Self::Growth,
            "challenge" => Self::Challenge,
            "relationship" => Self::Relationship,
            "project" => Self::Project,
            "habit" => Self::Habit,
            "discovery" => Self::Discovery,
            "loss" => Self::Loss,
            "recovery" => Self::Recovery,
            _ => Self::Project,
        }
    }
}

/// Lifecycle status of a narrative arc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ArcStatus {
    /// Just detected, too few episodes to be certain.
    Emerging,
    /// Actively accumulating episodes.
    Active,
    /// No new episodes for a while, may resume.
    Paused,
    /// Goal achieved or story naturally concluded.
    Resolved,
    /// User or system decided to stop tracking.
    Abandoned,
}

impl ArcStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Emerging => "emerging",
            Self::Active => "active",
            Self::Paused => "paused",
            Self::Resolved => "resolved",
            Self::Abandoned => "abandoned",
        }
    }
}

/// The type of a chapter within a narrative arc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChapterType {
    /// Initial context setting.
    Setup,
    /// Building tension or progress.
    Rising,
    /// Peak moment of the arc.
    Climax,
    /// Winding down after the peak.
    Falling,
    /// Final conclusion.
    Resolution,
    /// A pause or side-thread between main chapters.
    Interlude,
}

/// Direction of change at a turning point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DirectionChange {
    /// Things got better.
    Positive,
    /// Things got worse.
    Negative,
    /// Direction changed entirely (not better/worse, just different).
    Pivot,
    /// Intensity increased.
    Escalation,
    /// Intensity decreased.
    DeEscalation,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 2  Chapter & Turning Point
// ══════════════════════════════════════════════════════════════════════════════

/// A bounded segment within a narrative arc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chapter {
    /// Chapter title (auto-generated or user-labeled).
    pub title: String,
    /// Episode nodes in this chapter.
    pub episodes: Vec<NodeId>,
    /// Auto-generated chapter summary.
    pub summary: String,
    /// The narrative function of this chapter.
    pub chapter_type: ChapterType,
    /// Time span: (start_ms, end_ms).
    pub time_span: (u64, u64),
    /// Sentiment trajectory at each episode.
    pub sentiment_trajectory: Vec<f64>,
}

impl Chapter {
    /// Average sentiment of this chapter.
    pub fn avg_sentiment(&self) -> f64 {
        if self.sentiment_trajectory.is_empty() {
            return 0.0;
        }
        self.sentiment_trajectory.iter().sum::<f64>() / self.sentiment_trajectory.len() as f64
    }

    /// Duration in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        self.time_span.1.saturating_sub(self.time_span.0)
    }
}

/// A moment of significant change in an arc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurningPoint {
    /// The pivotal episode.
    pub episode_id: NodeId,
    /// What happened.
    pub description: String,
    /// Direction of change.
    pub direction_change: DirectionChange,
    /// How significant [0.0, 1.0].
    pub magnitude: f64,
    /// When it occurred (unix ms).
    pub timestamp_ms: u64,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 3  Narrative Arc
// ══════════════════════════════════════════════════════════════════════════════

/// A coherent story thread spanning multiple episodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeArc {
    /// Unique identifier.
    pub id: ArcId,
    /// Arc title (auto-generated or user-labeled).
    pub title: String,
    /// Thematic category.
    pub theme: ArcTheme,
    /// Ordered sequence of chapters.
    pub chapters: Vec<Chapter>,
    /// Entity nodes involved in this arc.
    pub participants: Vec<NodeId>,
    /// Knowledge domains touched.
    pub domains: Vec<String>,
    /// Current lifecycle status.
    pub status: ArcStatus,
    /// Running average sentiment [-1.0, 1.0].
    pub emotional_valence: f64,
    /// When this arc started (unix ms).
    pub started_at: u64,
    /// When this arc was last updated (unix ms).
    pub last_updated_at: u64,
    /// Turning points in this arc.
    pub turning_points: Vec<TurningPoint>,
}

impl NarrativeArc {
    /// Total number of episodes across all chapters.
    pub fn episode_count(&self) -> usize {
        self.chapters.iter().map(|c| c.episodes.len()).sum()
    }

    /// All episode ids across all chapters.
    pub fn all_episodes(&self) -> Vec<NodeId> {
        self.chapters
            .iter()
            .flat_map(|c| c.episodes.iter().copied())
            .collect()
    }

    /// Duration from first episode to last (milliseconds).
    pub fn duration_ms(&self) -> u64 {
        if self.chapters.is_empty() {
            return 0;
        }
        let start = self.chapters.first().map(|c| c.time_span.0).unwrap_or(0);
        let end = self.chapters.last().map(|c| c.time_span.1).unwrap_or(0);
        end.saturating_sub(start)
    }

    /// Whether this arc is still active (not resolved or abandoned).
    pub fn is_active(&self) -> bool {
        matches!(self.status, ArcStatus::Emerging | ArcStatus::Active)
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 4  Milestone & Timeline
// ══════════════════════════════════════════════════════════════════════════════

/// A significant life event that may span multiple arcs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Milestone {
    /// When it happened (unix ms).
    pub timestamp_ms: u64,
    /// What happened.
    pub description: String,
    /// What areas of life it affected.
    pub impact_domains: Vec<String>,
    /// Related arcs.
    pub related_arcs: Vec<ArcId>,
}

/// The user's complete life timeline — all narrative arcs and milestones.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutobiographicalTimeline {
    /// All narrative arcs (active and resolved).
    pub arcs: Vec<NarrativeArc>,
    /// Major life events.
    pub milestones: Vec<Milestone>,
    /// Next arc id.
    next_arc_id: u64,
}

impl Default for AutobiographicalTimeline {
    fn default() -> Self {
        Self {
            arcs: Vec::new(),
            milestones: Vec::new(),
            next_arc_id: 1,
        }
    }
}

impl AutobiographicalTimeline {
    /// Allocate a new arc id.
    pub fn alloc_arc_id(&mut self) -> ArcId {
        let id = ArcId(self.next_arc_id);
        self.next_arc_id += 1;
        id
    }

    /// Find an arc by id.
    pub fn find_arc(&self, id: ArcId) -> Option<&NarrativeArc> {
        self.arcs.iter().find(|a| a.id == id)
    }

    /// Find a mutable arc by id.
    pub fn find_arc_mut(&mut self, id: ArcId) -> Option<&mut NarrativeArc> {
        self.arcs.iter_mut().find(|a| a.id == id)
    }

    /// Get all currently active arcs.
    pub fn active_arcs(&self) -> Vec<&NarrativeArc> {
        self.arcs.iter().filter(|a| a.is_active()).collect()
    }

    /// Get all unresolved arcs (active + paused).
    pub fn unresolved_arcs(&self) -> Vec<&NarrativeArc> {
        self.arcs
            .iter()
            .filter(|a| !matches!(a.status, ArcStatus::Resolved | ArcStatus::Abandoned))
            .collect()
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 5  Episode Input
// ══════════════════════════════════════════════════════════════════════════════

/// An episode to be classified into a narrative arc.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeEpisode {
    /// The episode node id.
    pub episode_id: NodeId,
    /// Summary text.
    pub summary: String,
    /// Participants involved (entity node ids).
    pub participants: Vec<NodeId>,
    /// Knowledge domains touched.
    pub domains: Vec<String>,
    /// Sentiment/valence [-1.0, 1.0].
    pub sentiment: f64,
    /// When it occurred (unix ms).
    pub timestamp_ms: u64,
    /// Optional: related goal (helps classify theme).
    pub related_goal: Option<NodeId>,
}

// ══════════════════════════════════════════════════════════════════════════════
// § 6  Arc Assignment
// ══════════════════════════════════════════════════════════════════════════════

/// Classify an episode into an existing arc or create a new one.
///
/// Matching criteria (weighted):
/// - Participant overlap (0.40)
/// - Domain overlap (0.30)
/// - Recency (0.15)
/// - Sentiment continuity (0.15)
pub fn assign_to_arc(episode: &NarrativeEpisode, timeline: &mut AutobiographicalTimeline) -> ArcId {
    let mut best_arc: Option<(ArcId, f64)> = None;

    for arc in &timeline.arcs {
        if !arc.is_active() {
            continue;
        }

        let score = arc_match_score(episode, arc);
        if score > 0.3 {
            if best_arc.is_none() || score > best_arc.unwrap().1 {
                best_arc = Some((arc.id, score));
            }
        }
    }

    if let Some((arc_id, _)) = best_arc {
        // Add to existing arc.
        add_episode_to_arc(episode, arc_id, timeline);
        arc_id
    } else {
        // Create a new arc.
        create_arc_from_episode(episode, timeline)
    }
}

/// Score how well an episode matches an existing arc.
fn arc_match_score(episode: &NarrativeEpisode, arc: &NarrativeArc) -> f64 {
    // Participant overlap.
    let participant_overlap = if arc.participants.is_empty() || episode.participants.is_empty() {
        0.0
    } else {
        let shared = episode
            .participants
            .iter()
            .filter(|p| arc.participants.contains(p))
            .count();
        shared as f64 / episode.participants.len().max(1) as f64
    };

    // Domain overlap.
    let domain_overlap = if arc.domains.is_empty() || episode.domains.is_empty() {
        0.0
    } else {
        let shared = episode
            .domains
            .iter()
            .filter(|d| arc.domains.contains(d))
            .count();
        shared as f64 / episode.domains.len().max(1) as f64
    };

    // Recency: how recently was the arc updated?
    let age_ms = episode.timestamp_ms.saturating_sub(arc.last_updated_at);
    let age_days = age_ms as f64 / 86_400_000.0;
    let recency = (-age_days / 14.0).exp(); // 14-day half-life.

    // Sentiment continuity: penalize large sentiment jumps.
    let sentiment_diff = (episode.sentiment - arc.emotional_valence).abs();
    let sentiment_cont = 1.0 - (sentiment_diff / 2.0);

    0.40 * participant_overlap + 0.30 * domain_overlap + 0.15 * recency + 0.15 * sentiment_cont
}

/// Add an episode to an existing arc.
fn add_episode_to_arc(
    episode: &NarrativeEpisode,
    arc_id: ArcId,
    timeline: &mut AutobiographicalTimeline,
) {
    if let Some(arc) = timeline.find_arc_mut(arc_id) {
        // Check if we need a new chapter.
        let needs_new_chapter = if let Some(last_chapter) = arc.chapters.last() {
            detect_chapter_boundary_internal(last_chapter, episode)
        } else {
            true
        };

        if needs_new_chapter {
            let chapter_num = arc.chapters.len() + 1;
            arc.chapters.push(Chapter {
                title: format!("Chapter {}", chapter_num),
                episodes: vec![episode.episode_id],
                summary: episode.summary.clone(),
                chapter_type: infer_chapter_type(chapter_num, arc.status),
                time_span: (episode.timestamp_ms, episode.timestamp_ms),
                sentiment_trajectory: vec![episode.sentiment],
            });
        } else if let Some(chapter) = arc.chapters.last_mut() {
            chapter.episodes.push(episode.episode_id);
            chapter.time_span.1 = episode.timestamp_ms;
            chapter.sentiment_trajectory.push(episode.sentiment);
        }

        // Check for turning point.
        if let Some(tp) = detect_turning_point_internal(arc, episode) {
            arc.turning_points.push(tp);
        }

        // Update arc metadata.
        arc.last_updated_at = episode.timestamp_ms;
        // EMA update of emotional valence.
        arc.emotional_valence = 0.8 * arc.emotional_valence + 0.2 * episode.sentiment;

        // Add new participants/domains.
        for p in &episode.participants {
            if !arc.participants.contains(p) {
                arc.participants.push(*p);
            }
        }
        for d in &episode.domains {
            if !arc.domains.contains(d) {
                arc.domains.push(d.clone());
            }
        }

        // Promote from Emerging to Active after 3 episodes.
        if arc.status == ArcStatus::Emerging && arc.episode_count() >= 3 {
            arc.status = ArcStatus::Active;
        }
    }
}

/// Create a new arc from a single episode.
fn create_arc_from_episode(
    episode: &NarrativeEpisode,
    timeline: &mut AutobiographicalTimeline,
) -> ArcId {
    let arc_id = timeline.alloc_arc_id();
    let theme = infer_theme_from_episode(episode);

    let arc = NarrativeArc {
        id: arc_id,
        title: format!("{}: {}", theme.as_str(), truncate(&episode.summary, 40)),
        theme,
        chapters: vec![Chapter {
            title: "Chapter 1".to_string(),
            episodes: vec![episode.episode_id],
            summary: episode.summary.clone(),
            chapter_type: ChapterType::Setup,
            time_span: (episode.timestamp_ms, episode.timestamp_ms),
            sentiment_trajectory: vec![episode.sentiment],
        }],
        participants: episode.participants.clone(),
        domains: episode.domains.clone(),
        status: ArcStatus::Emerging,
        emotional_valence: episode.sentiment,
        started_at: episode.timestamp_ms,
        last_updated_at: episode.timestamp_ms,
        turning_points: Vec::new(),
    };

    timeline.arcs.push(arc);
    arc_id
}

// ══════════════════════════════════════════════════════════════════════════════
// § 7  Chapter Boundaries & Turning Points
// ══════════════════════════════════════════════════════════════════════════════

/// Detect when a new chapter should start.
///
/// Triggers: time gap > 48h, sentiment reversal (>0.6 change),
/// or the last chapter has > 10 episodes.
pub fn detect_chapter_boundary(arc: &NarrativeArc, episode: &NarrativeEpisode) -> bool {
    if let Some(last_chapter) = arc.chapters.last() {
        detect_chapter_boundary_internal(last_chapter, episode)
    } else {
        true
    }
}

fn detect_chapter_boundary_internal(last_chapter: &Chapter, episode: &NarrativeEpisode) -> bool {
    // Time gap > 48 hours.
    let time_gap = episode
        .timestamp_ms
        .saturating_sub(last_chapter.time_span.1);
    if time_gap > 48 * 3600 * 1000 {
        return true;
    }

    // Sentiment reversal: large change from chapter average.
    let avg = last_chapter.avg_sentiment();
    let diff = (episode.sentiment - avg).abs();
    if diff > 0.6 {
        return true;
    }

    // Chapter too long.
    if last_chapter.episodes.len() >= 10 {
        return true;
    }

    false
}

/// Detect if an episode represents a turning point in an arc.
///
/// Turning points: large sentiment change (>0.5 from arc average),
/// or a goal state transition (indicated by sentiment sign change).
pub fn detect_turning_point(
    arc: &NarrativeArc,
    episode: &NarrativeEpisode,
) -> Option<TurningPoint> {
    detect_turning_point_internal(arc, episode)
}

fn detect_turning_point_internal(
    arc: &NarrativeArc,
    episode: &NarrativeEpisode,
) -> Option<TurningPoint> {
    let sentiment_delta = episode.sentiment - arc.emotional_valence;
    let magnitude = sentiment_delta.abs();

    if magnitude < 0.4 {
        return None; // Not significant enough.
    }

    let direction = if sentiment_delta > 0.0 && episode.sentiment > 0.3 {
        DirectionChange::Positive
    } else if sentiment_delta < 0.0 && episode.sentiment < -0.3 {
        DirectionChange::Negative
    } else if magnitude > 0.7 {
        DirectionChange::Pivot
    } else if sentiment_delta > 0.0 {
        DirectionChange::DeEscalation
    } else {
        DirectionChange::Escalation
    };

    Some(TurningPoint {
        episode_id: episode.episode_id,
        description: format!(
            "Sentiment shifted {:.1} (from {:.1} to {:.1})",
            sentiment_delta, arc.emotional_valence, episode.sentiment,
        ),
        direction_change: direction,
        magnitude: magnitude.min(1.0),
        timestamp_ms: episode.timestamp_ms,
    })
}

// ══════════════════════════════════════════════════════════════════════════════
// § 8  Arc Resolution & Health
// ══════════════════════════════════════════════════════════════════════════════

/// Detect when an arc naturally concludes.
///
/// Resolution signals:
/// - High positive sentiment (>0.5) sustained over last 3 episodes
/// - No episodes for 30+ days (stalled → resolve)
/// - Explicit signal (related goal achieved)
pub fn detect_arc_resolution(arc: &NarrativeArc, now_ms: u64) -> bool {
    // Check sustained positive sentiment.
    let recent_sentiments: Vec<f64> = arc
        .chapters
        .iter()
        .flat_map(|c| c.sentiment_trajectory.iter())
        .copied()
        .rev()
        .take(3)
        .collect();

    if recent_sentiments.len() >= 3 && recent_sentiments.iter().all(|&s| s > 0.5) {
        return true;
    }

    // Stalled for 30+ days.
    let age = now_ms.saturating_sub(arc.last_updated_at);
    if age > 30 * 86_400_000 {
        return true;
    }

    false
}

/// Alert about arc health issues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArcAlert {
    /// Which arc has the issue.
    pub arc_id: ArcId,
    /// Arc title.
    pub arc_title: String,
    /// What's wrong.
    pub alert_type: ArcAlertType,
    /// Severity [0.0, 1.0].
    pub severity: f64,
    /// Human-readable description.
    pub description: String,
}

/// Types of arc health alerts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArcAlertType {
    /// No episodes in 14+ days.
    Stalled,
    /// Sentiment trending negative.
    TrendingNegative,
    /// Arc was abandoned with unresolved goals.
    AbandonedUnresolved,
}

/// Check all arcs for health issues.
pub fn arc_health_check(timeline: &AutobiographicalTimeline, now_ms: u64) -> Vec<ArcAlert> {
    let mut alerts = Vec::new();

    for arc in &timeline.arcs {
        if matches!(arc.status, ArcStatus::Resolved | ArcStatus::Abandoned) {
            continue;
        }

        // Stalled: no episodes in 14+ days.
        let age = now_ms.saturating_sub(arc.last_updated_at);
        if age > 14 * 86_400_000 {
            alerts.push(ArcAlert {
                arc_id: arc.id,
                arc_title: arc.title.clone(),
                alert_type: ArcAlertType::Stalled,
                severity: (age as f64 / (30.0 * 86_400_000.0)).min(1.0),
                description: format!(
                    "Arc '{}' has had no episodes for {:.0} days",
                    arc.title,
                    age as f64 / 86_400_000.0,
                ),
            });
        }

        // Trending negative: last 3 sentiment values all < -0.2.
        let recent: Vec<f64> = arc
            .chapters
            .iter()
            .flat_map(|c| c.sentiment_trajectory.iter())
            .copied()
            .rev()
            .take(3)
            .collect();
        if recent.len() >= 3 && recent.iter().all(|&s| s < -0.2) {
            let avg = recent.iter().sum::<f64>() / recent.len() as f64;
            alerts.push(ArcAlert {
                arc_id: arc.id,
                arc_title: arc.title.clone(),
                alert_type: ArcAlertType::TrendingNegative,
                severity: (-avg).min(1.0),
                description: format!(
                    "Arc '{}' sentiment trending negative (avg: {:.2})",
                    arc.title, avg,
                ),
            });
        }
    }

    alerts
}

// ══════════════════════════════════════════════════════════════════════════════
// § 9  Arc Merging
// ══════════════════════════════════════════════════════════════════════════════

/// Merge two arcs that turn out to be part of the same story.
pub fn merge_arcs(a: &NarrativeArc, b: &NarrativeArc) -> NarrativeArc {
    // Determine which started first.
    let (first, second) = if a.started_at <= b.started_at {
        (a, b)
    } else {
        (b, a)
    };

    // Combine chapters chronologically.
    let mut chapters = first.chapters.clone();
    chapters.extend(second.chapters.iter().cloned());
    chapters.sort_by_key(|c| c.time_span.0);

    // Combine participants and domains.
    let mut participants = first.participants.clone();
    for p in &second.participants {
        if !participants.contains(p) {
            participants.push(*p);
        }
    }

    let mut domains = first.domains.clone();
    for d in &second.domains {
        if !domains.contains(d) {
            domains.push(d.clone());
        }
    }

    // Combine turning points chronologically.
    let mut turning_points = first.turning_points.clone();
    turning_points.extend(second.turning_points.iter().cloned());
    turning_points.sort_by_key(|tp| tp.timestamp_ms);

    let now = second.last_updated_at.max(first.last_updated_at);

    NarrativeArc {
        id: first.id, // Keep the older id.
        title: format!("{} + {}", first.title, second.title),
        theme: first.theme, // Keep the first theme.
        chapters,
        participants,
        domains,
        status: if first.is_active() || second.is_active() {
            ArcStatus::Active
        } else {
            first.status
        },
        emotional_valence: (first.emotional_valence + second.emotional_valence) / 2.0,
        started_at: first.started_at,
        last_updated_at: now,
        turning_points,
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 10  Narrative Queries
// ══════════════════════════════════════════════════════════════════════════════

/// A query against the autobiographical timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NarrativeQuery {
    /// Find arcs involving a specific participant.
    ArcsByParticipant(NodeId),
    /// Find arcs in a specific domain.
    ArcsByDomain(String),
    /// Find arcs with a specific theme.
    ArcsByTheme(ArcTheme),
    /// Get all currently active arcs.
    ActiveArcs,
    /// Get all unresolved threads.
    UnresolvedThreads,
    /// Find arcs within a time range.
    TimeRange(u64, u64),
}

/// Result of a narrative query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NarrativeResult {
    /// Matching arcs.
    pub arcs: Vec<ArcId>,
    /// Total episodes across matching arcs.
    pub total_episodes: usize,
    /// Summary of the result.
    pub summary: String,
}

/// Query the timeline.
pub fn query_timeline(
    timeline: &AutobiographicalTimeline,
    query: &NarrativeQuery,
) -> NarrativeResult {
    let matching: Vec<&NarrativeArc> = match query {
        NarrativeQuery::ArcsByParticipant(pid) => timeline
            .arcs
            .iter()
            .filter(|a| a.participants.contains(pid))
            .collect(),
        NarrativeQuery::ArcsByDomain(domain) => timeline
            .arcs
            .iter()
            .filter(|a| a.domains.contains(domain))
            .collect(),
        NarrativeQuery::ArcsByTheme(theme) => {
            timeline.arcs.iter().filter(|a| a.theme == *theme).collect()
        }
        NarrativeQuery::ActiveArcs => timeline.arcs.iter().filter(|a| a.is_active()).collect(),
        NarrativeQuery::UnresolvedThreads => timeline
            .arcs
            .iter()
            .filter(|a| !matches!(a.status, ArcStatus::Resolved | ArcStatus::Abandoned))
            .collect(),
        NarrativeQuery::TimeRange(start, end) => timeline
            .arcs
            .iter()
            .filter(|a| a.started_at <= *end && a.last_updated_at >= *start)
            .collect(),
    };

    let total_episodes: usize = matching.iter().map(|a| a.episode_count()).sum();
    let arc_ids: Vec<ArcId> = matching.iter().map(|a| a.id).collect();
    let titles: Vec<&str> = matching.iter().map(|a| a.title.as_str()).collect();

    NarrativeResult {
        arcs: arc_ids,
        total_episodes,
        summary: if titles.is_empty() {
            "No matching arcs found".to_string()
        } else {
            format!("{} arcs: {}", titles.len(), titles.join(", "))
        },
    }
}

/// Generate a human-readable summary of an arc.
pub fn generate_arc_summary(arc: &NarrativeArc) -> String {
    let duration_days = arc.duration_ms() as f64 / 86_400_000.0;
    let episode_count = arc.episode_count();
    let chapter_count = arc.chapters.len();
    let tp_count = arc.turning_points.len();

    let sentiment_desc = if arc.emotional_valence > 0.3 {
        "positive"
    } else if arc.emotional_valence < -0.3 {
        "challenging"
    } else {
        "neutral"
    };

    format!(
        "'{}' ({}, {}): {} episodes across {} chapters over {:.0} days. \
         {} turning points. Overall tone: {} ({:.2}). Status: {}.",
        arc.title,
        arc.theme.as_str(),
        arc.status.as_str(),
        episode_count,
        chapter_count,
        duration_days,
        tp_count,
        sentiment_desc,
        arc.emotional_valence,
        arc.status.as_str(),
    )
}

// ══════════════════════════════════════════════════════════════════════════════
// § 11  Helpers
// ══════════════════════════════════════════════════════════════════════════════

/// Truncate a string to max length with ellipsis.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

/// Infer a theme from an episode's metadata.
fn infer_theme_from_episode(episode: &NarrativeEpisode) -> ArcTheme {
    // Simple heuristic based on domains.
    for domain in &episode.domains {
        let d = domain.to_lowercase();
        if d.contains("learn") || d.contains("skill") || d.contains("study") {
            return ArcTheme::Growth;
        }
        if d.contains("project") || d.contains("work") || d.contains("ship") {
            return ArcTheme::Project;
        }
        if d.contains("health") || d.contains("exercise") || d.contains("diet") {
            return ArcTheme::Habit;
        }
        if d.contains("friend") || d.contains("family") || d.contains("partner") {
            return ArcTheme::Relationship;
        }
    }

    // Fallback: use sentiment.
    if episode.sentiment < -0.5 {
        ArcTheme::Challenge
    } else if episode.sentiment > 0.5 {
        ArcTheme::Discovery
    } else {
        ArcTheme::Project
    }
}

/// Infer chapter type from position in the arc.
fn infer_chapter_type(chapter_num: usize, arc_status: ArcStatus) -> ChapterType {
    if chapter_num == 1 {
        ChapterType::Setup
    } else if arc_status == ArcStatus::Resolved {
        ChapterType::Resolution
    } else if chapter_num <= 3 {
        ChapterType::Rising
    } else {
        ChapterType::Rising // Default for ongoing arcs.
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// § 12  Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{NodeId, NodeKind};

    fn make_episode(
        seq: u32,
        summary: &str,
        participants: Vec<NodeId>,
        domains: Vec<&str>,
        sentiment: f64,
        ts: u64,
    ) -> NarrativeEpisode {
        NarrativeEpisode {
            episode_id: NodeId::new(NodeKind::Episode, seq),
            summary: summary.to_string(),
            participants,
            domains: domains.into_iter().map(|d| d.to_string()).collect(),
            sentiment,
            timestamp_ms: ts,
            related_goal: None,
        }
    }

    fn alice() -> NodeId {
        NodeId::new(NodeKind::Entity, 100)
    }

    fn bob() -> NodeId {
        NodeId::new(NodeKind::Entity, 101)
    }

    #[test]
    fn test_arc_assignment_creates_new_arc() {
        let mut timeline = AutobiographicalTimeline::default();
        let ep = make_episode(
            1,
            "Started learning Rust",
            vec![alice()],
            vec!["learning"],
            0.5,
            1000,
        );

        let arc_id = assign_to_arc(&ep, &mut timeline);
        assert_eq!(timeline.arcs.len(), 1);
        assert_eq!(timeline.find_arc(arc_id).unwrap().episode_count(), 1);
        assert_eq!(
            timeline.find_arc(arc_id).unwrap().status,
            ArcStatus::Emerging,
        );
    }

    #[test]
    fn test_arc_assignment_adds_to_existing() {
        let mut timeline = AutobiographicalTimeline::default();
        let now = 1_000_000;

        let ep1 = make_episode(
            1,
            "Started learning Rust",
            vec![alice()],
            vec!["learning"],
            0.5,
            now,
        );
        let ep2 = make_episode(
            2,
            "Read Rust book ch1",
            vec![alice()],
            vec!["learning"],
            0.6,
            now + 3600_000,
        );

        assign_to_arc(&ep1, &mut timeline);
        assign_to_arc(&ep2, &mut timeline);

        assert_eq!(timeline.arcs.len(), 1, "Should reuse existing arc");
        assert_eq!(timeline.arcs[0].episode_count(), 2);
    }

    #[test]
    fn test_arc_promotes_to_active() {
        let mut timeline = AutobiographicalTimeline::default();
        let now = 1_000_000;

        for i in 0..3 {
            let ep = make_episode(
                i + 1,
                &format!("Learning episode {}", i + 1),
                vec![alice()],
                vec!["learning"],
                0.5,
                now + i as u64 * 3600_000,
            );
            assign_to_arc(&ep, &mut timeline);
        }

        assert_eq!(timeline.arcs[0].status, ArcStatus::Active);
    }

    #[test]
    fn test_chapter_boundary_time_gap() {
        let mut timeline = AutobiographicalTimeline::default();
        let now = 1_000_000;

        let ep1 = make_episode(1, "Day 1", vec![alice()], vec!["project"], 0.5, now);
        assign_to_arc(&ep1, &mut timeline);

        // 3 days later → new chapter.
        let ep2 = make_episode(
            2,
            "Day 4",
            vec![alice()],
            vec!["project"],
            0.5,
            now + 3 * 86_400_000,
        );
        assign_to_arc(&ep2, &mut timeline);

        assert_eq!(
            timeline.arcs[0].chapters.len(),
            2,
            "Should create new chapter"
        );
    }

    #[test]
    fn test_chapter_boundary_sentiment_reversal() {
        let mut timeline = AutobiographicalTimeline::default();
        let now = 1_000_000;

        let ep1 = make_episode(1, "Great day", vec![alice()], vec!["work"], 0.8, now);
        assign_to_arc(&ep1, &mut timeline);

        // Big sentiment drop → new chapter.
        let ep2 = make_episode(
            2,
            "Terrible day",
            vec![alice()],
            vec!["work"],
            -0.5,
            now + 3600_000,
        );
        assign_to_arc(&ep2, &mut timeline);

        assert!(
            timeline.arcs[0].chapters.len() >= 2,
            "Sentiment reversal should create chapter"
        );
    }

    #[test]
    fn test_turning_point_detection() {
        let mut timeline = AutobiographicalTimeline::default();
        let now = 1_000_000;

        // Build up a baseline.
        for i in 0..3 {
            let ep = make_episode(
                i + 1,
                "Normal day",
                vec![alice()],
                vec!["work"],
                0.3,
                now + i as u64 * 3600_000,
            );
            assign_to_arc(&ep, &mut timeline);
        }

        // Big positive shift.
        let ep4 = make_episode(
            4,
            "Got promoted!",
            vec![alice()],
            vec!["work"],
            0.9,
            now + 4 * 3600_000,
        );
        assign_to_arc(&ep4, &mut timeline);

        assert!(
            !timeline.arcs[0].turning_points.is_empty(),
            "Should detect turning point on big sentiment shift"
        );
    }

    #[test]
    fn test_arc_resolution_positive_sentiment() {
        let arc = NarrativeArc {
            id: ArcId(1),
            title: "test".to_string(),
            theme: ArcTheme::Project,
            chapters: vec![Chapter {
                title: "ch1".to_string(),
                episodes: vec![],
                summary: String::new(),
                chapter_type: ChapterType::Resolution,
                time_span: (0, 0),
                sentiment_trajectory: vec![0.6, 0.7, 0.8],
            }],
            participants: vec![],
            domains: vec![],
            status: ArcStatus::Active,
            emotional_valence: 0.7,
            started_at: 0,
            last_updated_at: 1000,
            turning_points: vec![],
        };

        assert!(detect_arc_resolution(&arc, 2000));
    }

    #[test]
    fn test_arc_resolution_stalled() {
        let arc = NarrativeArc {
            id: ArcId(1),
            title: "test".to_string(),
            theme: ArcTheme::Project,
            chapters: vec![Chapter {
                title: "ch1".to_string(),
                episodes: vec![],
                summary: String::new(),
                chapter_type: ChapterType::Setup,
                time_span: (0, 0),
                sentiment_trajectory: vec![0.3],
            }],
            participants: vec![],
            domains: vec![],
            status: ArcStatus::Active,
            emotional_valence: 0.3,
            started_at: 0,
            last_updated_at: 0,
            turning_points: vec![],
        };

        let now = 31 * 86_400_000; // 31 days later.
        assert!(detect_arc_resolution(&arc, now));
    }

    #[test]
    fn test_timeline_queries() {
        let mut timeline = AutobiographicalTimeline::default();
        let now = 1_000_000;

        // Arc 1: Alice learning.
        let ep1 = make_episode(1, "Learn", vec![alice()], vec!["learning"], 0.5, now);
        assign_to_arc(&ep1, &mut timeline);

        // Arc 2: Bob project (different participant + domain).
        let ep2 = make_episode(2, "Code", vec![bob()], vec!["project"], 0.3, now);
        assign_to_arc(&ep2, &mut timeline);

        // Query by participant.
        let r1 = query_timeline(&timeline, &NarrativeQuery::ArcsByParticipant(alice()));
        assert_eq!(r1.arcs.len(), 1);

        // Query by domain.
        let r2 = query_timeline(
            &timeline,
            &NarrativeQuery::ArcsByDomain("project".to_string()),
        );
        assert_eq!(r2.arcs.len(), 1);

        // Query active arcs.
        let r3 = query_timeline(&timeline, &NarrativeQuery::ActiveArcs);
        assert_eq!(r3.arcs.len(), 2); // Both are emerging/active.
    }

    #[test]
    fn test_arc_merge() {
        let now = 1_000_000;
        let a = NarrativeArc {
            id: ArcId(1),
            title: "Arc A".to_string(),
            theme: ArcTheme::Project,
            chapters: vec![Chapter {
                title: "ch1".to_string(),
                episodes: vec![NodeId::new(NodeKind::Episode, 1)],
                summary: "start".to_string(),
                chapter_type: ChapterType::Setup,
                time_span: (now, now + 1000),
                sentiment_trajectory: vec![0.5],
            }],
            participants: vec![alice()],
            domains: vec!["work".to_string()],
            status: ArcStatus::Active,
            emotional_valence: 0.5,
            started_at: now,
            last_updated_at: now + 1000,
            turning_points: vec![],
        };

        let b = NarrativeArc {
            id: ArcId(2),
            title: "Arc B".to_string(),
            theme: ArcTheme::Project,
            chapters: vec![Chapter {
                title: "ch1".to_string(),
                episodes: vec![NodeId::new(NodeKind::Episode, 2)],
                summary: "continue".to_string(),
                chapter_type: ChapterType::Rising,
                time_span: (now + 2000, now + 3000),
                sentiment_trajectory: vec![0.6],
            }],
            participants: vec![alice(), bob()],
            domains: vec!["work".to_string(), "coding".to_string()],
            status: ArcStatus::Active,
            emotional_valence: 0.6,
            started_at: now + 2000,
            last_updated_at: now + 3000,
            turning_points: vec![],
        };

        let merged = merge_arcs(&a, &b);
        assert_eq!(merged.episode_count(), 2);
        assert_eq!(merged.participants.len(), 2); // alice + bob
        assert_eq!(merged.domains.len(), 2); // work + coding
        assert_eq!(merged.id, ArcId(1)); // Keeps older id.
    }

    #[test]
    fn test_arc_health_check_stalled() {
        let now = 100 * 86_400_000u64; // Day 100.
        let old_update = 80 * 86_400_000u64; // Last update day 80 (20 days ago).

        let timeline = AutobiographicalTimeline {
            arcs: vec![NarrativeArc {
                id: ArcId(1),
                title: "Stalled arc".to_string(),
                theme: ArcTheme::Project,
                chapters: vec![],
                participants: vec![],
                domains: vec![],
                status: ArcStatus::Active,
                emotional_valence: 0.0,
                started_at: 0,
                last_updated_at: old_update,
                turning_points: vec![],
            }],
            milestones: vec![],
            next_arc_id: 2,
        };

        let alerts = arc_health_check(&timeline, now);
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].alert_type, ArcAlertType::Stalled);
    }

    #[test]
    fn test_arc_health_check_trending_negative() {
        let now = 1_000_000;
        let timeline = AutobiographicalTimeline {
            arcs: vec![NarrativeArc {
                id: ArcId(1),
                title: "Sad arc".to_string(),
                theme: ArcTheme::Challenge,
                chapters: vec![Chapter {
                    title: "ch1".to_string(),
                    episodes: vec![],
                    summary: String::new(),
                    chapter_type: ChapterType::Falling,
                    time_span: (0, now),
                    sentiment_trajectory: vec![-0.3, -0.5, -0.7],
                }],
                participants: vec![],
                domains: vec![],
                status: ArcStatus::Active,
                emotional_valence: -0.5,
                started_at: 0,
                last_updated_at: now,
                turning_points: vec![],
            }],
            milestones: vec![],
            next_arc_id: 2,
        };

        let alerts = arc_health_check(&timeline, now);
        let negative = alerts
            .iter()
            .find(|a| a.alert_type == ArcAlertType::TrendingNegative);
        assert!(negative.is_some(), "Should detect negative trend");
    }

    #[test]
    fn test_generate_arc_summary() {
        let arc = NarrativeArc {
            id: ArcId(1),
            title: "Learning Rust".to_string(),
            theme: ArcTheme::Growth,
            chapters: vec![Chapter {
                title: "ch1".to_string(),
                episodes: vec![
                    NodeId::new(NodeKind::Episode, 1),
                    NodeId::new(NodeKind::Episode, 2),
                ],
                summary: String::new(),
                chapter_type: ChapterType::Setup,
                time_span: (0, 86_400_000),
                sentiment_trajectory: vec![0.5, 0.6],
            }],
            participants: vec![alice()],
            domains: vec!["rust".to_string()],
            status: ArcStatus::Active,
            emotional_valence: 0.55,
            started_at: 0,
            last_updated_at: 86_400_000,
            turning_points: vec![],
        };

        let summary = generate_arc_summary(&arc);
        assert!(summary.contains("Learning Rust"));
        assert!(summary.contains("growth"));
        assert!(summary.contains("2 episodes"));
        assert!(summary.contains("positive"));
    }

    #[test]
    fn test_theme_inference() {
        let ep = make_episode(1, "Studied math", vec![], vec!["learning"], 0.5, 0);
        assert_eq!(infer_theme_from_episode(&ep), ArcTheme::Growth);

        let ep2 = make_episode(2, "Family dinner", vec![], vec!["family"], 0.7, 0);
        assert_eq!(infer_theme_from_episode(&ep2), ArcTheme::Relationship);
    }

    #[test]
    fn test_different_participants_create_separate_arcs() {
        let mut timeline = AutobiographicalTimeline::default();
        let now = 1_000_000;

        let ep1 = make_episode(
            1,
            "Meeting with Alice",
            vec![alice()],
            vec!["work"],
            0.5,
            now,
        );
        let ep2 = make_episode(
            2,
            "Meeting with Bob",
            vec![bob()],
            vec!["social"],
            0.5,
            now + 1000,
        );

        assign_to_arc(&ep1, &mut timeline);
        assign_to_arc(&ep2, &mut timeline);

        // Different participants + different domains → separate arcs.
        assert_eq!(timeline.arcs.len(), 2);
    }

    #[test]
    fn test_arc_duration() {
        let arc = NarrativeArc {
            id: ArcId(1),
            title: "test".to_string(),
            theme: ArcTheme::Project,
            chapters: vec![
                Chapter {
                    title: "ch1".to_string(),
                    episodes: vec![],
                    summary: String::new(),
                    chapter_type: ChapterType::Setup,
                    time_span: (1000, 2000),
                    sentiment_trajectory: vec![],
                },
                Chapter {
                    title: "ch2".to_string(),
                    episodes: vec![],
                    summary: String::new(),
                    chapter_type: ChapterType::Rising,
                    time_span: (5000, 8000),
                    sentiment_trajectory: vec![],
                },
            ],
            participants: vec![],
            domains: vec![],
            status: ArcStatus::Active,
            emotional_valence: 0.0,
            started_at: 1000,
            last_updated_at: 8000,
            turning_points: vec![],
        };

        assert_eq!(arc.duration_ms(), 7000);
    }
}
