//! Seed substitution categories for conflict detection bootstrap.
//!
//! These categories are populated on first DB init or V14 migration.
//! They provide day-one coverage for common substitution patterns
//! (e.g., "PostgreSQL" vs "MySQL" in the "databases" category).
//! After seeding, categories grow dynamically via user feedback and LLM gossip.

use rusqlite::Connection;

use crate::error::Result;

/// A seed category with its members.
struct SeedCategory {
    name: &'static str,
    conflict_mode: &'static str,
    /// (normalized, display) pairs
    members: &'static [(&'static str, &'static str)],
    /// Optional context hints (JSON array string) for ambiguous members
    context_hints: &'static [(&'static str, &'static str)],
}

const SEED_CATEGORIES: &[SeedCategory] = &[
    SeedCategory {
        name: "databases",
        conflict_mode: "exclusive",
        members: &[
            ("postgresql", "PostgreSQL"),
            ("mysql", "MySQL"),
            ("mariadb", "MariaDB"),
            ("sqlite", "SQLite"),
            ("mongodb", "MongoDB"),
            ("redis", "Redis"),
            ("clickhouse", "ClickHouse"),
            ("cassandra", "Cassandra"),
            ("dynamodb", "DynamoDB"),
            ("cockroachdb", "CockroachDB"),
            ("neo4j", "Neo4j"),
            ("elasticsearch", "Elasticsearch"),
            ("firestore", "Firestore"),
            ("supabase", "Supabase"),
            ("planetscale", "PlanetScale"),
        ],
        context_hints: &[],
    },
    SeedCategory {
        name: "cloud_providers",
        conflict_mode: "exclusive",
        members: &[
            ("aws", "AWS"),
            ("gcp", "GCP"),
            ("azure", "Azure"),
            ("digitalocean", "DigitalOcean"),
            ("vercel", "Vercel"),
            ("netlify", "Netlify"),
            ("cloudflare", "Cloudflare"),
            ("hetzner", "Hetzner"),
        ],
        context_hints: &[],
    },
    SeedCategory {
        name: "programming_languages",
        conflict_mode: "exclusive",
        members: &[
            ("python", "Python"),
            ("javascript", "JavaScript"),
            ("typescript", "TypeScript"),
            ("rust", "Rust"),
            ("go", "Go"),
            ("java", "Java"),
            ("kotlin", "Kotlin"),
            ("swift", "Swift"),
            ("ruby", "Ruby"),
            ("php", "PHP"),
            ("c++", "C++"),
            ("c#", "C#"),
            ("scala", "Scala"),
            ("elixir", "Elixir"),
            ("zig", "Zig"),
        ],
        context_hints: &[
            ("rust", r#"["language","backend","compile","systems"]"#),
            ("go", r#"["language","backend","compile","golang"]"#),
            ("swift", r#"["language","ios","apple","mobile"]"#),
            ("ruby", r#"["language","rails","backend","scripting"]"#),
        ],
    },
    SeedCategory {
        name: "frameworks",
        conflict_mode: "exclusive",
        members: &[
            ("react", "React"),
            ("vue", "Vue"),
            ("angular", "Angular"),
            ("svelte", "Svelte"),
            ("django", "Django"),
            ("flask", "Flask"),
            ("fastapi", "FastAPI"),
            ("spring", "Spring"),
            ("express", "Express"),
            ("nextjs", "Next.js"),
            ("rails", "Rails"),
            ("laravel", "Laravel"),
        ],
        context_hints: &[
            ("spring", r#"["java","boot","framework","dependency"]"#),
            ("express", r#"["nodejs","javascript","backend","api"]"#),
            ("flask", r#"["python","web","backend","api"]"#),
        ],
    },
    SeedCategory {
        name: "roles",
        conflict_mode: "exclusive",
        members: &[
            ("backend", "backend"),
            ("frontend", "frontend"),
            ("fullstack", "fullstack"),
            ("devops", "devops"),
            ("sre", "SRE"),
            ("ml", "ML"),
            ("data", "data"),
            ("platform", "platform"),
            ("mobile", "mobile"),
            ("qa", "QA"),
        ],
        context_hints: &[
            ("backend", r#"["engineer","developer","role","team"]"#),
            ("frontend", r#"["engineer","developer","role","team"]"#),
            ("data", r#"["engineer","scientist","analyst","role"]"#),
            ("platform", r#"["engineer","team","infrastructure","role"]"#),
            ("mobile", r#"["engineer","developer","ios","android","role"]"#),
        ],
    },
    SeedCategory {
        name: "infrastructure",
        conflict_mode: "exclusive",
        members: &[
            ("kubernetes", "Kubernetes"),
            ("docker", "Docker"),
            ("terraform", "Terraform"),
            ("ansible", "Ansible"),
            ("nginx", "nginx"),
            ("apache", "Apache"),
            ("caddy", "Caddy"),
            ("traefik", "Traefik"),
        ],
        context_hints: &[
            ("apache", r#"["webserver","httpd","proxy","server"]"#),
        ],
    },
    SeedCategory {
        name: "editors_tools",
        conflict_mode: "exclusive",
        members: &[
            ("vscode", "VS Code"),
            ("neovim", "Neovim"),
            ("vim", "Vim"),
            ("intellij", "IntelliJ"),
            ("cursor", "Cursor"),
            ("windsurf", "Windsurf"),
        ],
        context_hints: &[
            ("cursor", r#"["editor","ide","ai","coding"]"#),
        ],
    },
    SeedCategory {
        name: "llm_providers",
        conflict_mode: "exclusive",
        members: &[
            ("openai", "OpenAI"),
            ("anthropic", "Anthropic"),
            ("google", "Google"),
            ("mistral", "Mistral"),
            ("meta", "Meta"),
            ("deepseek", "DeepSeek"),
        ],
        context_hints: &[
            ("google", r#"["ai","llm","gemini","model"]"#),
            ("meta", r#"["ai","llm","llama","model"]"#),
        ],
    },
];

/// Populate seed substitution categories into the database.
///
/// Uses INSERT OR IGNORE so this is safe to call repeatedly (idempotent).
pub fn populate_seed_categories(conn: &Connection) -> Result<()> {
    let ts = crate::time::now_secs();

    // We use a zero-byte HLC placeholder for seeds since they don't participate in sync
    let hlc_placeholder: Vec<u8> = vec![0u8; 16];

    for cat in SEED_CATEGORIES {
        let cat_id = format!("seed_{}", cat.name);

        conn.execute(
            "INSERT OR IGNORE INTO substitution_categories
             (id, name, conflict_mode, status, created_at, updated_at, hlc, origin_actor)
             VALUES (?1, ?2, ?3, 'active', ?4, ?4, ?5, 'seed')",
            rusqlite::params![cat_id, cat.name, cat.conflict_mode, ts, hlc_placeholder],
        )?;

        for &(normalized, display) in cat.members {
            let member_id = format!("seed_{}_{}", cat.name, normalized);
            let context = cat
                .context_hints
                .iter()
                .find(|(tok, _)| *tok == normalized)
                .map(|(_, hint)| *hint);

            conn.execute(
                "INSERT OR IGNORE INTO substitution_members
                 (id, category_id, token_normalized, token_display, confidence,
                  source, status, context_hint, created_at, updated_at, hlc, origin_actor)
                 VALUES (?1, ?2, ?3, ?4, 0.95, 'seed', 'active', ?5, ?6, ?6, ?7, 'seed')",
                rusqlite::params![
                    member_id,
                    cat_id,
                    normalized,
                    display,
                    context,
                    ts,
                    hlc_placeholder,
                ],
            )?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_population() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(crate::schema::SCHEMA_SQL).unwrap();

        populate_seed_categories(&conn).unwrap();

        // Check categories were created
        let cat_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM substitution_categories", [], |r| r.get(0))
            .unwrap();
        assert_eq!(cat_count, 8);

        // Check members were created
        let member_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM substitution_members", [], |r| r.get(0))
            .unwrap();
        assert_eq!(member_count, 80);

        // Check specific member
        let pg_cat: String = conn
            .query_row(
                "SELECT c.name FROM substitution_members m
                 JOIN substitution_categories c ON c.id = m.category_id
                 WHERE m.token_normalized = 'postgresql'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(pg_cat, "databases");

        // Idempotent: running again should not error or duplicate
        populate_seed_categories(&conn).unwrap();
        let cat_count2: i64 = conn
            .query_row("SELECT COUNT(*) FROM substitution_categories", [], |r| r.get(0))
            .unwrap();
        assert_eq!(cat_count2, 8);
    }

    #[test]
    fn test_context_hints_populated() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(crate::schema::SCHEMA_SQL).unwrap();

        populate_seed_categories(&conn).unwrap();

        // "go" should have context hints
        let hint: Option<String> = conn
            .query_row(
                "SELECT context_hint FROM substitution_members WHERE token_normalized = 'go'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(hint.is_some());
        assert!(hint.unwrap().contains("golang"));

        // "postgresql" should have no context hints (unambiguous)
        let hint: Option<String> = conn
            .query_row(
                "SELECT context_hint FROM substitution_members WHERE token_normalized = 'postgresql'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(hint.is_none());
    }
}
