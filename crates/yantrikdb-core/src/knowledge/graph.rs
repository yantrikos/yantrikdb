//! Graph traversal utilities for entity-augmented recall.

use std::collections::{HashMap, HashSet, VecDeque};

use rusqlite::{params, Connection};

use crate::error::Result;

// ── Word-boundary entity matching ──

/// Tokenize text into lowercase words, splitting on non-alphanumeric chars.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

/// Check if an entity name appears as whole-word(s) in pre-tokenized text.
/// Single-word entities require exact token match.
/// Multi-word entities require contiguous token sequence match.
pub fn entity_matches_text(entity: &str, text_tokens: &[String]) -> bool {
    let entity_tokens = tokenize(entity);
    if entity_tokens.is_empty() {
        return false;
    }
    if entity_tokens.len() == 1 {
        text_tokens.iter().any(|t| t == &entity_tokens[0])
    } else {
        text_tokens
            .windows(entity_tokens.len())
            .any(|window| window.iter().zip(entity_tokens.iter()).all(|(w, e)| w == e))
    }
}

// ── Heuristic proper-noun extraction ──

/// English function/pronoun/auxiliary words that should be stripped from the
/// start or end of a capitalized chunk. A sentence-initial "The" or "Our" is
/// capitalized by position, not because it names an entity.
const ENTITY_STOPWORDS: &[&str] = &[
    "The", "A", "An", "I", "We", "You", "He", "She", "It", "They",
    "This", "That", "These", "Those", "My", "Your", "His", "Her",
    "Its", "Our", "Their", "But", "And", "Or", "So", "If", "When",
    "Where", "What", "Who", "Why", "How", "Is", "Are", "Was", "Were",
    "Be", "Been", "Being", "Have", "Has", "Had", "Do", "Does", "Did",
    "Of", "In", "On", "At", "To", "For", "With", "From", "By",
    "As", "Than", "Then", "Also", "Just", "Only", "Very", "Much",
];

/// Extract candidate proper-noun entities from free-form text using a
/// capitalized-chunk heuristic. Groups consecutive capitalized words into
/// multi-word entities ("Alice Chen", "San Francisco", "Acme Corp") and
/// strips leading/trailing English stopwords.
///
/// This is intentionally not a full NER — it captures the common case of
/// people, companies, places, and products well enough that conflict
/// detection can fire without requiring users to call `/v1/relate` for every
/// entity. Acronyms, lowercase entities, and ambiguous mentions still need
/// explicit `relate()` calls to enter the graph.
pub fn extract_heuristic_entities(text: &str) -> Vec<String> {
    let mut entities: Vec<String> = Vec::new();
    let mut chunk: Vec<String> = Vec::new();

    let flush = |chunk: &mut Vec<String>, out: &mut Vec<String>| {
        while !chunk.is_empty() && ENTITY_STOPWORDS.contains(&chunk[0].as_str()) {
            chunk.remove(0);
        }
        // Trailing-stopword strip skips single-character tokens so multi-word
        // entities like "Series A" or "Version B" keep their letter suffix
        // (A is a stopword but is also a valid version designator when trailing).
        while let Some(last) = chunk.last() {
            if ENTITY_STOPWORDS.contains(&last.as_str()) && last.chars().count() > 1 {
                chunk.pop();
            } else {
                break;
            }
        }
        if !chunk.is_empty() {
            let candidate = chunk.join(" ");
            let alpha_chars = candidate.chars().filter(|c| c.is_alphanumeric()).count();
            if alpha_chars >= 2 {
                out.push(candidate);
            }
        }
        chunk.clear();
    };

    for word in text
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| !s.is_empty())
    {
        let first = word.chars().next().unwrap();
        let starts_upper = first.is_uppercase();
        let is_all_caps = word.len() > 1 && word.chars().all(|c| !c.is_alphabetic() || c.is_uppercase());

        let joins_chunk = if chunk.is_empty() {
            // Open a new chunk only on capitalized or all-caps tokens.
            starts_upper || is_all_caps
        } else {
            // Continue an existing chunk on capitalized words or short letter-suffixes
            // (e.g., "Series A", "Version B").
            starts_upper
                || is_all_caps
                || (word.len() == 1 && first.is_ascii_uppercase())
        };

        if joins_chunk {
            chunk.push(word.to_string());
        } else {
            flush(&mut chunk, &mut entities);
        }
    }
    flush(&mut chunk, &mut entities);

    // Deduplicate while preserving first-appearance order.
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    entities.retain(|e| seen.insert(e.clone()));
    entities
}

// ── Heuristic relation extraction (RFC 006 Phase 1) ──

/// A candidate relation extracted from text by pattern matching.
#[derive(Debug, Clone)]
pub struct RelationCandidate {
    pub src: String,
    pub rel_type: String,
    pub dst: String,
    pub polarity: i32,       // 1=positive, -1=negative
    pub modality: String,    // asserted, reported, hypothetical, denied
    pub confidence_band: String, // low, medium, high
}

/// Relation patterns: keyword phrases that appear BETWEEN two entities
/// and indicate a specific relationship. Each pattern maps to a rel_type.
const RELATION_PATTERNS: &[(&[&str], &str)] = &[
    // Role-based (entity A <pattern> entity B → rel_type)
    (&["is the ceo of", "is ceo of", "serves as ceo of"], "ceo_of"),
    (&["is the cto of", "is cto of", "serves as cto of"], "cto_of"),
    (&["is the cfo of", "is cfo of", "serves as cfo of"], "cfo_of"),
    (&["is the founder of", "is founder of", "co-founded"], "founded"),
    (&["founded"], "founded"),
    (&["leads", "heads", "runs", "manages", "directs"], "leads"),
    (&["works at", "works for", "employed at", "employed by", "joined"], "works_at"),
    // Location/origin
    (&["was born in", "born in"], "born_in"),
    (&["is headquartered in", "headquartered in", "is based in", "based in", "located in"], "headquartered_in"),
    // Personal
    (&["is married to", "married to", "wed to"], "married_to"),
    // Corporate
    (&["acquired", "bought", "purchased", "took over"], "acquired"),
    (&["is a subsidiary of", "subsidiary of", "is owned by", "owned by"], "subsidiary_of"),
    // Language/skill
    (&["speaks", "is fluent in"], "speaks"),
    // Generic membership/part-of
    (&["is a member of", "member of", "belongs to", "part of"], "member_of"),
    (&["reports to"], "reports_to"),
];

/// Possessive/appositive reverse patterns: "ORG's CEO, PERSON" or "ORG's CEO PERSON"
const REVERSE_ROLE_PATTERNS: &[(&str, &str)] = &[
    ("ceo", "ceo_of"),
    ("cto", "cto_of"),
    ("cfo", "cfo_of"),
    ("founder", "founded"),
    ("president", "leads"),
    ("director", "leads"),
    ("head", "leads"),
];

/// Extract candidate relations from text using entities as anchors.
///
/// For each ordered pair of entities (A before B in text), examines the
/// text between them for relation-indicating keywords. Also checks for
/// negation cues in the window to set polarity, and tense cues to infer
/// past-tense (which callers can use for valid_to).
///
/// Returns high-precision, low-recall candidates — only emits when a
/// clear keyword pattern matches. Designed for the RFC 006 Phase 1
/// relation whitelist.
pub fn extract_heuristic_relations(text: &str, entities: &[String]) -> Vec<RelationCandidate> {
    if entities.len() < 2 {
        return vec![];
    }

    let text_lower = text.to_lowercase();
    let mut candidates: Vec<RelationCandidate> = Vec::new();

    // Find position of each entity in the text (case-insensitive)
    let mut entity_positions: Vec<(usize, &str)> = Vec::new();
    for entity in entities {
        let entity_lower = entity.to_lowercase();
        if let Some(pos) = text_lower.find(&entity_lower) {
            entity_positions.push((pos, entity.as_str()));
        }
    }
    entity_positions.sort_by_key(|(pos, _)| *pos);

    // For each adjacent pair, check the text between them
    for i in 0..entity_positions.len() {
        for j in (i + 1)..entity_positions.len() {
            let (pos_a, entity_a) = entity_positions[i];
            let (pos_b, entity_b) = entity_positions[j];

            // Skip pairs too far apart (likely different sentences)
            if pos_b - pos_a > 150 {
                continue;
            }

            let between_start = pos_a + entity_a.to_lowercase().len();
            let between_end = pos_b;
            if between_start >= between_end || between_end > text_lower.len() {
                continue;
            }

            let between = text_lower[between_start..between_end].trim();
            if between.is_empty() {
                continue;
            }

            // Check negation in the between-window, then strip negation
            // words so pattern matching still works on "is NOT the CEO of"
            let has_negation = NEGATION_CUES.iter().any(|cue| {
                between.split_whitespace().any(|w| w == *cue)
            });
            let polarity = if has_negation { -1 } else { 1 };
            let between_stripped: String = between
                .split_whitespace()
                .filter(|w| !NEGATION_CUES.contains(w))
                .collect::<Vec<_>>()
                .join(" ");

            // Check modality cues
            let modality = if MODALITY_CUES.iter().any(|cue| between.contains(cue)) {
                "reported"
            } else {
                "asserted"
            };

            // Match forward patterns: entity_a <pattern> entity_b
            // Uses between_stripped (negation removed) for matching.
            for (patterns, rel_type) in RELATION_PATTERNS {
                for pattern in *patterns {
                    if between_stripped.contains(pattern) {
                        candidates.push(RelationCandidate {
                            src: entity_a.to_string(),
                            rel_type: rel_type.to_string(),
                            dst: entity_b.to_string(),
                            polarity,
                            modality: modality.to_string(),
                            confidence_band: "medium".to_string(),
                        });
                        break; // one match per pattern group per pair
                    }
                }
            }

            // Check possessive/appositive reverse: "Acme's CEO, Alice" → ceo_of(Alice, Acme)
            for (role_keyword, rel_type) in REVERSE_ROLE_PATTERNS {
                let possessive = format!("'s {}", role_keyword);
                let possessive2 = format!("s {}", role_keyword);
                if between_stripped.contains(&possessive) || between_stripped.contains(&possessive2) {
                    // Reversed: entity_a is the org, entity_b is the person
                    candidates.push(RelationCandidate {
                        src: entity_b.to_string(), // person
                        rel_type: rel_type.to_string(),
                        dst: entity_a.to_string(), // org
                        polarity,
                        modality: modality.to_string(),
                        confidence_band: "medium".to_string(),
                    });
                    break;
                }
            }
        }
    }

    // Deduplicate: same (src, rel_type, dst) keeps highest confidence
    let mut seen = std::collections::HashSet::new();
    candidates.retain(|c| seen.insert((c.src.clone(), c.rel_type.clone(), c.dst.clone())));

    candidates
}

// ── Text feature analysis (Phase 0 audit data for RFC 006) ──

/// Cues that indicate a statement is negated. Window-scanned around pattern
/// matches to flag `polarity=negative` in v0.6.0. In v0.5.13 we only count
/// occurrences for audit telemetry.
const NEGATION_CUES: &[&str] = &[
    "not", "no", "never", "denied", "refuted", "isn't", "wasn't",
    "aren't", "weren't", "doesn't", "didn't", "disputes", "denies",
];

/// Cues that indicate a statement has temporal scope. Used to flag that a
/// memory would benefit from `valid_from` / `valid_to` qualifiers.
const TEMPORAL_CUES: &[&str] = &[
    "was", "were", "until", "before", "after", "since", "during",
    "former", "current", "currently", "previously", "recently",
    "now", "then", "later", "earlier", "ago", "yesterday", "tomorrow",
];

/// Cues that indicate modality (hypothetical, reported, quoted).
const MODALITY_CUES: &[&str] = &[
    "may", "might", "allegedly", "reportedly", "rumor", "rumored",
    "said", "claims", "according", "stated", "announced",
];

/// Compound-sentence separators that a v0.6.0 extractor should split on
/// before running patterns. Counting these at audit time tells us how many
/// real-world memories contain multiple claims per write.
const COMPOUND_MARKERS: &[&str] = &[
    "; ", ", then ", ", subsequently ", " but ", " however ", " although ",
];

/// Text features collected for extraction-audit telemetry (RFC 006 Phase 0).
/// Captures everything the v0.6.0 extractor would need to know without
/// changing any storage behavior — purely observational.
#[derive(Debug, Clone, Default)]
pub struct TextFeatures {
    pub char_length: usize,
    pub sentence_count: usize,
    pub entity_count: usize,
    pub negation_cue_count: usize,
    pub temporal_cue_count: usize,
    pub modality_cue_count: usize,
    pub has_compound_markers: bool,
    pub likely_assertion: bool,
}

/// Compute text features for extraction audit. Pure function, no I/O.
pub fn analyze_text_features(text: &str, extracted_entities: &[String]) -> TextFeatures {
    let lower = text.to_lowercase();
    let tokens: Vec<&str> = text
        .split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| !s.is_empty())
        .collect();
    let tokens_lower: Vec<String> = tokens.iter().map(|t| t.to_lowercase()).collect();

    let sentence_count = text
        .chars()
        .filter(|c| matches!(c, '.' | '!' | '?'))
        .count()
        .max(1);

    let negation_cue_count = tokens_lower
        .iter()
        .filter(|t| NEGATION_CUES.contains(&t.as_str()))
        .count();

    let temporal_cue_count = tokens_lower
        .iter()
        .filter(|t| TEMPORAL_CUES.contains(&t.as_str()))
        .count();

    let modality_cue_count = tokens_lower
        .iter()
        .filter(|t| MODALITY_CUES.contains(&t.as_str()))
        .count();

    let has_compound_markers = COMPOUND_MARKERS.iter().any(|m| lower.contains(m));

    // Rough "assertion?" signal: not a question, has at least 2 tokens, not
    // pure modality/rumor. Used to estimate what fraction of agent writes
    // the v0.6.0 extractor should try to process at all.
    let likely_assertion = !text.trim_end().ends_with('?')
        && tokens.len() >= 2
        && modality_cue_count == 0;

    TextFeatures {
        char_length: text.chars().count(),
        sentence_count,
        entity_count: extracted_entities.len(),
        negation_cue_count,
        temporal_cue_count,
        modality_cue_count,
        has_compound_markers,
        likely_assertion,
    }
}

// ── Entity type classification ──

/// Tech terms that should NOT be classified as person names even if title-cased/all-caps.
const TECH_BLOCKLIST: &[&str] = &[
    "faiss", "onnx", "scann", "redis", "kafka", "docker", "kubernetes", "react",
    "python", "rust", "java", "swift", "flutter", "pytorch", "tensorflow",
    "numpy", "pandas", "spark", "hadoop", "nginx", "postgres", "mysql",
    "sqlite", "graphql", "grpc", "oauth", "jwt", "html", "css",
    "api", "sdk", "ml", "ai", "gpu", "cpu", "ram", "ssd", "aws", "gcp",
    "claude", "openai", "anthropic", "gemini", "llama", "ollama",
];

/// Words that indicate the entity is NOT a person when used as first word.
const NON_PERSON_PREFIXES: &[&str] = &[
    "project", "team", "company", "group", "department", "org", "the",
    "operation", "task", "plan", "system", "service", "app", "tool",
    "code", "server", "client", "api", "db", "database", "agent",
    "model", "version", "release", "build", "deploy", "config",
];

/// Classify an entity name into a type: "person", "tech", or "unknown".
/// This is a name-only heuristic — prefer `classify_with_relationship()` when
/// relationship context is available.
pub fn classify_entity_type(name: &str) -> &'static str {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return "unknown";
    }
    let lower = trimmed.to_lowercase();

    // Check tech blocklist
    if TECH_BLOCKLIST.contains(&lower.as_str()) {
        return "tech";
    }

    // All-caps multi-char → tech (e.g., "FAISS", "ONNX")
    if trimmed.len() > 1 && trimmed.chars().all(|c| c.is_uppercase() || !c.is_alphabetic()) {
        return "tech";
    }

    // Multi-word title-case (e.g., "Priya Sharma", "Sarah Chen") → likely person
    // But NOT if the first word is a non-person prefix (e.g., "Project Athena", "Claude Code")
    if trimmed.contains(' ') {
        let words: Vec<&str> = trimmed.split_whitespace().collect();
        if words.len() == 2
            && words
                .iter()
                .all(|w| w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false))
        {
            let first_lower = words[0].to_lowercase();
            if NON_PERSON_PREFIXES.contains(&first_lower.as_str()) {
                return "unknown";
            }
            // Also reject if any word is in tech blocklist
            if words.iter().any(|w| TECH_BLOCKLIST.contains(&w.to_lowercase().as_str())) {
                return "tech";
            }
            return "person";
        }
    }

    // Single-word classification is unreliable (Bangalore, Flipkart, Arjun all
    // look the same). Return "unknown" and let relationship context decide.
    "unknown"
}

/// Relationship types that imply both src and dst are persons.
const PERSON_PERSON_RELS: &[&str] = &[
    "married_to", "mother_of", "father_of", "daughter_of", "son_of",
    "sister_of", "brother_of", "sibling_of", "parent_of", "child_of",
    "knows", "friends_with", "met", "dating", "engaged_to",
    "mentors", "mentored_by", "reports_to", "manages",
    "colleagues", "roommate", "neighbor",
    "called", "texted", "messaged", "date_night",
];

/// Relationship types where dst is a place.
const PLACE_DST_RELS: &[&str] = &[
    "lives_in", "born_in", "grew_up_in", "located_in", "based_in",
    "visited", "moved_to", "traveled_to", "from",
];

/// Relationship types where dst is an organization / institution.
const ORG_DST_RELS: &[&str] = &[
    "works_at", "works_for", "employed_at", "employed_by",
    "studied_at", "attended", "enrolled_in", "graduated_from",
    "member_of", "belongs_to", "founded",
];

/// Relationship types where dst is tech/tool (src is project or person).
const TECH_DST_RELS: &[&str] = &[
    "built_with", "uses", "depends_on", "integrates", "requires",
    "written_in", "coded_in", "implemented_with", "powered_by",
    "runs_on", "compiled_with",
];

/// Relationship types where dst is infrastructure.
const INFRA_DST_RELS: &[&str] = &[
    "deployed_on", "hosted_on", "deployed_to", "hosted_at",
    "runs_on_infra", "served_by",
];

/// Relationship types where src is a person and dst is a project/thing.
const PERSON_PROJECT_RELS: &[&str] = &[
    "works_on", "contributes_to", "maintains", "leads", "created",
    "built", "designed", "architected", "owns",
];

/// Relationship types where src is a project and dst is a project (dependency).
const PROJECT_PROJECT_RELS: &[&str] = &[
    "depends_on_project", "extends", "forks", "replaces",
    "supersedes", "derived_from",
];

/// Relationship types where dst is an event or activity.
const EVENT_DST_RELS: &[&str] = &[
    "attended_event", "participated_in", "scheduled_for",
    "presented_at", "spoke_at",
];

/// Relationship types where dst is a concept/topic.
const CONCEPT_DST_RELS: &[&str] = &[
    "interested_in", "studies", "researches", "specializes_in",
    "expert_in", "learning", "teaches",
];

/// Classify entity types using relationship semantics.
/// Returns (src_type, dst_type) — either may be "unknown" if not inferable.
pub fn classify_with_relationship(
    src: &str,
    dst: &str,
    rel_type: &str,
) -> (&'static str, &'static str) {
    let rel_lower = rel_type.to_lowercase();
    let rel = rel_lower.as_str();

    // Person-person relationships
    if PERSON_PERSON_RELS.contains(&rel) {
        return ("person", "person");
    }

    // Person → Place relationships
    if PLACE_DST_RELS.contains(&rel) {
        return ("person", "place");
    }

    // Person → Organization relationships
    if ORG_DST_RELS.contains(&rel) {
        return ("person", "organization");
    }

    // * → Tech/Tool relationships (src type from name heuristic)
    if TECH_DST_RELS.contains(&rel) {
        let src_type = classify_entity_type(src);
        return (if src_type == "unknown" { "project" } else { src_type }, "tech");
    }

    // * → Infrastructure relationships
    if INFRA_DST_RELS.contains(&rel) {
        let src_type = classify_entity_type(src);
        return (if src_type == "unknown" { "project" } else { src_type }, "infrastructure");
    }

    // Person → Project relationships
    if PERSON_PROJECT_RELS.contains(&rel) {
        return ("person", "project");
    }

    // Project → Project relationships
    if PROJECT_PROJECT_RELS.contains(&rel) {
        return ("project", "project");
    }

    // * → Event relationships
    if EVENT_DST_RELS.contains(&rel) {
        return (classify_entity_type(src), "event");
    }

    // Person → Concept/Topic relationships
    if CONCEPT_DST_RELS.contains(&rel) {
        return ("person", "concept");
    }

    // Fall back to name-based heuristics
    (classify_entity_type(src), classify_entity_type(dst))
}

/// Given a set of memory RIDs, find all entities those memories are linked to.
pub fn entities_for_memories(conn: &Connection, rids: &[&str]) -> Result<Vec<String>> {
    if rids.is_empty() {
        return Ok(vec![]);
    }
    let placeholders: String = (0..rids.len()).map(|i| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT DISTINCT entity_name FROM memory_entities WHERE memory_rid IN ({placeholders})"
    );
    let mut stmt = conn.prepare(&sql)?;
    let param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
        rids.iter().map(|r| Box::new(r.to_string()) as Box<dyn rusqlite::types::ToSql>).collect();
    let params_ref: Vec<&dyn rusqlite::types::ToSql> = param_values.iter().map(|p| p.as_ref()).collect();
    let entities = stmt
        .query_map(params_ref.as_slice(), |row| row.get(0))?
        .collect::<std::result::Result<Vec<String>, _>>()?;
    Ok(entities)
}

/// Given a set of entity names, find all memory RIDs connected to those entities.
pub fn memories_for_entities(conn: &Connection, entity_names: &[&str]) -> Result<HashSet<String>> {
    if entity_names.is_empty() {
        return Ok(HashSet::new());
    }
    let placeholders: String = (0..entity_names.len()).map(|i| format!("?{}", i + 1)).collect::<Vec<_>>().join(",");
    let sql = format!(
        "SELECT DISTINCT memory_rid FROM memory_entities WHERE entity_name IN ({placeholders})"
    );
    let mut stmt = conn.prepare(&sql)?;
    let param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
        entity_names.iter().map(|e| Box::new(e.to_string()) as Box<dyn rusqlite::types::ToSql>).collect();
    let params_ref: Vec<&dyn rusqlite::types::ToSql> = param_values.iter().map(|p| p.as_ref()).collect();
    let rids = stmt
        .query_map(params_ref.as_slice(), |row| row.get(0))?
        .collect::<std::result::Result<HashSet<String>, _>>()?;
    Ok(rids)
}

/// Expand entity set N hops via the edges table (BFS).
/// Returns (entity_name, hops_from_seed, cumulative_edge_weight).
/// Seeds are returned with hops=0 and weight=1.0.
pub fn expand_entities_nhop(
    conn: &Connection,
    seeds: &[&str],
    max_hops: u8,
    max_entities: usize,
) -> Result<Vec<(String, u8, f64)>> {
    let mut result: Vec<(String, u8, f64)> = Vec::new();
    let mut visited: HashMap<String, (u8, f64)> = HashMap::new();

    // Initialize with seeds
    for s in seeds {
        visited.insert(s.to_string(), (0, 1.0));
        result.push((s.to_string(), 0, 1.0));
    }

    let mut frontier: VecDeque<(String, u8, f64)> = seeds
        .iter()
        .map(|s| (s.to_string(), 0u8, 1.0f64))
        .collect();

    while let Some((entity, hops, weight)) = frontier.pop_front() {
        if hops >= max_hops || result.len() >= max_entities {
            break;
        }

        // Find neighbors via edges (both directions)
        let mut stmt = conn.prepare(
            "SELECT src, dst, weight FROM edges WHERE (src = ?1 OR dst = ?1) AND tombstoned = 0",
        )?;
        let neighbors: Vec<(String, f64)> = stmt
            .query_map(params![entity], |row| {
                let src: String = row.get(0)?;
                let dst: String = row.get(1)?;
                let w: f64 = row.get(2)?;
                let neighbor = if src == entity { dst } else { src };
                Ok((neighbor, w))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        for (neighbor, edge_weight) in neighbors {
            if visited.contains_key(&neighbor) {
                continue;
            }
            if result.len() >= max_entities {
                break;
            }
            let cumulative = weight * edge_weight;
            let next_hops = hops + 1;
            visited.insert(neighbor.clone(), (next_hops, cumulative));
            result.push((neighbor.clone(), next_hops, cumulative));
            if next_hops < max_hops {
                frontier.push_back((neighbor, next_hops, cumulative));
            }
        }
    }

    Ok(result)
}

/// Compute graph proximity score for a memory based on its entity connections.
/// Returns the maximum proximity across all entities the memory is linked to.
/// proximity = cumulative_weight / 2^hops  (steeper decay to stay discriminative)
/// Seeds (hops=0) → 1.0, 1-hop → 0.5, 2-hop → 0.25
pub fn graph_proximity(
    conn: &Connection,
    memory_rid: &str,
    expanded_entities: &HashMap<String, (u8, f64)>,
) -> Result<f64> {
    let mem_entities: Vec<String> = conn
        .prepare("SELECT entity_name FROM memory_entities WHERE memory_rid = ?1")?
        .query_map(params![memory_rid], |row| row.get(0))?
        .collect::<std::result::Result<Vec<_>, _>>()?;

    let mut max_proximity = 0.0f64;
    for entity in &mem_entities {
        if let Some(&(hops, weight)) = expanded_entities.get(entity) {
            let prox = weight / f64::powf(2.0, hops as f64);
            if prox > max_proximity {
                max_proximity = prox;
            }
        }
    }
    Ok(max_proximity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::YantrikDB;

    #[test]
    fn test_extract_heuristic_entities_basic_names() {
        let got = extract_heuristic_entities("Alice Chen is the CEO of Acme Corp");
        assert!(got.contains(&"Alice Chen".to_string()), "got: {:?}", got);
        assert!(got.contains(&"Acme Corp".to_string()), "got: {:?}", got);
        // CEO is all-caps standalone — should appear as an entity candidate.
        assert!(got.contains(&"CEO".to_string()), "got: {:?}", got);
    }

    #[test]
    fn test_extract_heuristic_entities_strips_sentence_start() {
        let got = extract_heuristic_entities("The database backend is PostgreSQL");
        assert_eq!(got, vec!["PostgreSQL".to_string()]);
    }

    #[test]
    fn test_extract_heuristic_entities_multi_word_place() {
        let got = extract_heuristic_entities("Acme is headquartered in San Francisco");
        assert!(got.contains(&"Acme".to_string()), "got: {:?}", got);
        assert!(got.contains(&"San Francisco".to_string()), "got: {:?}", got);
    }

    #[test]
    fn test_extract_heuristic_entities_single_letter_suffix() {
        let got = extract_heuristic_entities("Series A funding was 20 million dollars");
        assert!(got.contains(&"Series A".to_string()), "got: {:?}", got);
    }

    #[test]
    fn test_extract_heuristic_entities_dedupe() {
        let got = extract_heuristic_entities("Alice met Alice at the cafe");
        let alice_count = got.iter().filter(|e| *e == "Alice").count();
        assert_eq!(alice_count, 1);
    }

    #[test]
    fn test_extract_heuristic_entities_empty_on_lowercase() {
        let got = extract_heuristic_entities("the quick brown fox jumps over the lazy dog");
        assert!(got.is_empty(), "got: {:?}", got);
    }

    // ── Relation extraction tests ──

    #[test]
    fn test_extract_relations_ceo_of() {
        let entities = vec!["Alice Chen".to_string(), "Acme Corp".to_string()];
        let rels = extract_heuristic_relations("Alice Chen is the CEO of Acme Corp", &entities);
        assert_eq!(rels.len(), 1, "got: {:?}", rels);
        assert_eq!(rels[0].src, "Alice Chen");
        assert_eq!(rels[0].rel_type, "ceo_of");
        assert_eq!(rels[0].dst, "Acme Corp");
        assert_eq!(rels[0].polarity, 1);
    }

    #[test]
    fn test_extract_relations_works_at() {
        let entities = vec!["Bob".to_string(), "Google".to_string()];
        let rels = extract_heuristic_relations("Bob works at Google as an engineer", &entities);
        assert!(rels.iter().any(|r| r.rel_type == "works_at"), "got: {:?}", rels);
    }

    #[test]
    fn test_extract_relations_headquartered() {
        let entities = vec!["Acme".to_string(), "San Francisco".to_string()];
        let rels = extract_heuristic_relations("Acme is headquartered in San Francisco", &entities);
        assert!(rels.iter().any(|r| r.rel_type == "headquartered_in"), "got: {:?}", rels);
    }

    #[test]
    fn test_extract_relations_negation_detected() {
        let entities = vec!["Alice".to_string(), "Acme".to_string()];
        let rels = extract_heuristic_relations("Alice is not the CEO of Acme", &entities);
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].polarity, -1, "negation should set polarity to -1");
    }

    #[test]
    fn test_extract_relations_no_match_unrelated() {
        let entities = vec!["Alice".to_string(), "Bob".to_string()];
        let rels = extract_heuristic_relations("Alice and Bob went for coffee", &entities);
        assert!(rels.is_empty(), "should not extract relation from unrelated text, got: {:?}", rels);
    }

    #[test]
    fn test_extract_relations_multiple_pairs() {
        let entities = vec!["Alice".to_string(), "Acme".to_string(), "San Francisco".to_string()];
        let rels = extract_heuristic_relations(
            "Alice is the CEO of Acme which is headquartered in San Francisco",
            &entities,
        );
        assert!(rels.len() >= 2, "should find CEO + headquartered, got: {:?}", rels);
    }

    #[test]
    fn test_extract_relations_needs_two_entities() {
        let entities = vec!["Alice".to_string()];
        let rels = extract_heuristic_relations("Alice is the CEO", &entities);
        assert!(rels.is_empty(), "cannot extract relation with only one entity");
    }

    // ── Text feature analysis tests ──

    #[test]
    fn test_analyze_text_features_basic_assertion() {
        let entities = vec!["Alice Chen".to_string(), "Acme Corp".to_string()];
        let f = analyze_text_features("Alice Chen is the CEO of Acme Corp", &entities);
        assert_eq!(f.entity_count, 2);
        assert_eq!(f.negation_cue_count, 0);
        assert_eq!(f.modality_cue_count, 0);
        assert!(f.likely_assertion);
        assert!(!f.has_compound_markers);
    }

    #[test]
    fn test_analyze_text_features_negation() {
        let f = analyze_text_features("Alice is not the CEO of Acme", &[]);
        assert_eq!(f.negation_cue_count, 1);
    }

    #[test]
    fn test_analyze_text_features_temporal() {
        let f = analyze_text_features("Alice was previously the CEO before 2024", &[]);
        assert!(f.temporal_cue_count >= 2, "got: {}", f.temporal_cue_count);
    }

    #[test]
    fn test_analyze_text_features_modality_suppresses_assertion() {
        let f = analyze_text_features("Alice may become CEO allegedly", &[]);
        assert!(f.modality_cue_count >= 2);
        assert!(!f.likely_assertion);
    }

    #[test]
    fn test_analyze_text_features_compound() {
        let f = analyze_text_features("Alice was CEO until 2024; then Bob took over", &[]);
        assert!(f.has_compound_markers);
    }

    #[test]
    fn test_analyze_text_features_question_not_assertion() {
        let f = analyze_text_features("Who is the CEO of Acme?", &[]);
        assert!(!f.likely_assertion);
    }

    #[test]
    fn test_extract_heuristic_entities_distinct_people() {
        // Regression guard for the false-merge case that motivated this:
        // two sentences structurally similar but referring to different people.
        let a = extract_heuristic_entities("Alice Chen is the CEO of Acme Corp");
        let b = extract_heuristic_entities("Sarah Kim is the CTO of Acme Corp");
        let a_set: std::collections::HashSet<_> = a.iter().collect();
        let b_set: std::collections::HashSet<_> = b.iter().collect();
        // They share Acme Corp but differ on person name — disjointness on people.
        assert!(a_set.contains(&"Alice Chen".to_string()));
        assert!(b_set.contains(&"Sarah Kim".to_string()));
        assert!(!a_set.contains(&"Sarah Kim".to_string()));
        assert!(!b_set.contains(&"Alice Chen".to_string()));
    }

    fn setup_db() -> YantrikDB {
        let db = YantrikDB::new(":memory:", 4).unwrap();
        // Create entities and edges
        db.relate("Alice", "Bob", "knows", 1.0).unwrap();
        db.relate("Bob", "Charlie", "knows", 0.8).unwrap();
        db.relate("Alice", "ProjectX", "works_on", 1.0).unwrap();
        db.relate("Dave", "ProjectX", "works_on", 0.9).unwrap();

        // Record memories and link to entities
        let emb = vec![1.0f32, 0.0, 0.0, 0.0];
        let r1 = db.record("Alice discussed the plan", "episodic", 0.5, 0.0, 604800.0, &serde_json::json!({}), &emb, "default", 0.8, "general", "user", None).unwrap();
        let r2 = db.record("Bob reviewed the code", "episodic", 0.5, 0.0, 604800.0, &serde_json::json!({}), &emb, "default", 0.8, "general", "user", None).unwrap();
        let r3 = db.record("Charlie deployed to production", "episodic", 0.5, 0.0, 604800.0, &serde_json::json!({}), &emb, "default", 0.8, "general", "user", None).unwrap();

        db.link_memory_entity(&r1, "Alice").unwrap();
        db.link_memory_entity(&r1, "ProjectX").unwrap();
        db.link_memory_entity(&r2, "Bob").unwrap();
        db.link_memory_entity(&r3, "Charlie").unwrap();

        db
    }

    #[test]
    fn test_entities_for_memories() {
        let db = setup_db();
        // Get the first memory's rid
        let rid: String = db.conn().query_row(
            "SELECT rid FROM memories ORDER BY created_at LIMIT 1", [], |row| row.get(0),
        ).unwrap();

        let entities = entities_for_memories(&*db.conn(), &[&rid]).unwrap();
        assert!(entities.contains(&"Alice".to_string()));
        assert!(entities.contains(&"ProjectX".to_string()));
    }

    #[test]
    fn test_memories_for_entities() {
        let db = setup_db();
        let rids = memories_for_entities(&*db.conn(), &["Alice"]).unwrap();
        assert_eq!(rids.len(), 1); // Only the Alice memory is linked
    }

    #[test]
    fn test_expand_1hop() {
        let db = setup_db();
        let expanded = expand_entities_nhop(&*db.conn(), &["Alice"], 1, 30).unwrap();
        let names: HashSet<String> = expanded.iter().map(|(n, _, _)| n.clone()).collect();
        // Alice (seed) + Bob (knows) + ProjectX (works_on)
        assert!(names.contains("Alice"));
        assert!(names.contains("Bob"));
        assert!(names.contains("ProjectX"));
    }

    #[test]
    fn test_expand_2hop() {
        let db = setup_db();
        let expanded = expand_entities_nhop(&*db.conn(), &["Alice"], 2, 30).unwrap();
        let names: HashSet<String> = expanded.iter().map(|(n, _, _)| n.clone()).collect();
        // 2-hop from Alice: Alice->Bob->Charlie, Alice->ProjectX->Dave
        assert!(names.contains("Charlie"));
        assert!(names.contains("Dave"));
    }

    #[test]
    fn test_expand_budget_limit() {
        let db = setup_db();
        let expanded = expand_entities_nhop(&*db.conn(), &["Alice"], 2, 3).unwrap();
        assert!(expanded.len() <= 3);
    }

    #[test]
    fn test_no_tombstoned_edges() {
        let db = setup_db();
        // Tombstone the Alice->Bob edge
        db.conn().execute(
            "UPDATE edges SET tombstoned = 1 WHERE src = 'Alice' AND dst = 'Bob'",
            [],
        ).unwrap();
        let expanded = expand_entities_nhop(&*db.conn(), &["Alice"], 1, 30).unwrap();
        let names: HashSet<String> = expanded.iter().map(|(n, _, _)| n.clone()).collect();
        // Bob should NOT be reachable via tombstoned edge
        assert!(!names.contains("Bob"));
        // ProjectX should still be reachable
        assert!(names.contains("ProjectX"));
    }

    #[test]
    fn test_graph_proximity_score() {
        let db = setup_db();
        let rid: String = db.conn().query_row(
            "SELECT rid FROM memories ORDER BY created_at LIMIT 1", [], |row| row.get(0),
        ).unwrap();

        let mut expanded = HashMap::new();
        expanded.insert("Alice".to_string(), (0u8, 1.0f64));
        expanded.insert("ProjectX".to_string(), (1u8, 1.0f64));

        let prox = graph_proximity(&*db.conn(), &rid, &expanded).unwrap();
        // Alice is hops=0 → proximity = 1.0 / (0+1) = 1.0
        assert!((prox - 1.0).abs() < 1e-10);
    }

    // ── Word-boundary matching tests ──

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("What is Sarah working on?");
        assert_eq!(tokens, vec!["what", "is", "sarah", "working", "on"]);
    }

    #[test]
    fn test_tokenize_preserves_apostrophes() {
        let tokens = tokenize("daughter's school play");
        assert_eq!(tokens, vec!["daughter's", "school", "play"]);
    }

    #[test]
    fn test_entity_matches_single_word() {
        let tokens = tokenize("Sarah discussed the plan with Mike");
        assert!(entity_matches_text("Sarah", &tokens));
        assert!(entity_matches_text("Mike", &tokens));
        assert!(!entity_matches_text("Sara", &tokens)); // partial ≠ match
    }

    #[test]
    fn test_entity_matches_multi_word() {
        let tokens = tokenize("The data pipeline crashed during migration");
        assert!(entity_matches_text("data pipeline", &tokens));
        assert!(!entity_matches_text("data migration", &tokens)); // non-contiguous
    }

    #[test]
    fn test_entity_no_substring_false_positive() {
        let tokens = tokenize("The database was updated successfully");
        // "data" should NOT match inside "database"
        assert!(!entity_matches_text("data", &tokens));
    }

    #[test]
    fn test_entity_matches_case_insensitive() {
        let tokens = tokenize("We evaluated FAISS for vector search");
        assert!(entity_matches_text("FAISS", &tokens));
        assert!(entity_matches_text("faiss", &tokens));
    }

    // ── Entity type classification tests ──

    #[test]
    fn test_classify_name_only_ambiguous() {
        // Single-word title-case is now "unknown" without relationship context
        assert_eq!(classify_entity_type("Sarah"), "unknown");
        assert_eq!(classify_entity_type("Bangalore"), "unknown");
        assert_eq!(classify_entity_type("Flipkart"), "unknown");
    }

    #[test]
    fn test_classify_name_multi_word_person() {
        // Multi-word title-case full names are still "person"
        assert_eq!(classify_entity_type("Sarah Chen"), "person");
        assert_eq!(classify_entity_type("Priya Sharma"), "person");
    }

    #[test]
    fn test_classify_tech_blocklist() {
        assert_eq!(classify_entity_type("FAISS"), "tech");
        assert_eq!(classify_entity_type("ONNX"), "tech");
        assert_eq!(classify_entity_type("Redis"), "tech");
        assert_eq!(classify_entity_type("Python"), "tech");
    }

    #[test]
    fn test_classify_tech_allcaps() {
        assert_eq!(classify_entity_type("GPU"), "tech");
        assert_eq!(classify_entity_type("API"), "tech");
    }

    #[test]
    fn test_classify_unknown() {
        assert_eq!(classify_entity_type("recommendation engine"), "unknown");
        assert_eq!(classify_entity_type("data pipeline"), "unknown");
        assert_eq!(classify_entity_type("sleep patterns"), "unknown");
    }

    // ── Relationship-based classification tests ──

    #[test]
    fn test_classify_with_rel_person_person() {
        let (s, d) = classify_with_relationship("Arjun", "Priya", "married_to");
        assert_eq!(s, "person");
        assert_eq!(d, "person");
    }

    #[test]
    fn test_classify_with_rel_person_place() {
        let (s, d) = classify_with_relationship("Priya", "Bangalore", "lives_in");
        assert_eq!(s, "person");
        assert_eq!(d, "place");
    }

    #[test]
    fn test_classify_with_rel_person_org() {
        let (s, d) = classify_with_relationship("Priya", "Flipkart", "works_at");
        assert_eq!(s, "person");
        assert_eq!(d, "organization");
    }

    #[test]
    fn test_classify_with_rel_tech_dst() {
        // "uses" implies dst is tech; FAISS is tech by name heuristic
        let (s, d) = classify_with_relationship("FAISS", "data pipeline", "uses");
        assert_eq!(s, "tech");
        assert_eq!(d, "tech");
    }

    #[test]
    fn test_classify_with_rel_built_with() {
        // "built_with" → src defaults to "project" if unknown, dst is tech
        let (s, d) = classify_with_relationship("MyApp", "React", "built_with");
        assert_eq!(s, "project");
        assert_eq!(d, "tech");
    }

    #[test]
    fn test_classify_with_rel_deployed_on() {
        let (s, d) = classify_with_relationship("MyApp", "AWS", "deployed_on");
        assert_eq!(s, "project");
        assert_eq!(d, "infrastructure");
    }

    #[test]
    fn test_classify_with_rel_works_on() {
        let (s, d) = classify_with_relationship("Pranab", "YantrikDB", "works_on");
        assert_eq!(s, "person");
        assert_eq!(d, "project");
    }

    #[test]
    fn test_classify_with_rel_fallback() {
        // Truly unknown relationship → falls back to name heuristics
        let (s, d) = classify_with_relationship("FAISS", "data pipeline", "related_to");
        assert_eq!(s, "tech");
        assert_eq!(d, "unknown");
    }
}
