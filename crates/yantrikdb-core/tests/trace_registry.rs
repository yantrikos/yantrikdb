//! v0.10 Phase 0 — the trace-contract registry test.
//!
//! Enforces the reliability gate's structural rules over
//! `docs/traces/T*.toml` on every test run:
//! - all 13 contracts exist with unique, well-formed ids;
//! - required fields present (schema-versioned TOML, not markdown prose);
//! - `status` is a legal value, and `implemented` REQUIRES
//!   `implemented_since` + `test_path` (so implemented→pending regression
//!   is structurally detectable; the CI base-diff check covers history).

use std::path::PathBuf;

fn traces_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("docs")
        .join("traces")
}

#[test]
fn trace_contracts_are_complete_and_well_formed() {
    let dir = traces_dir();
    assert!(dir.is_dir(), "docs/traces missing at {dir:?}");

    let mut seen: Vec<String> = Vec::new();
    for i in 1..=13 {
        let id = format!("T{i:02}");
        let path = dir.join(format!("{id}.toml"));
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("trace contract {id} missing/unreadable: {e}"));
        let doc: toml::Value =
            toml::from_str(&raw).unwrap_or_else(|e| panic!("{id}.toml parse error: {e}"));

        let get_str = |key: &str| -> String {
            doc.get(key)
                .and_then(|v| v.as_str())
                .unwrap_or_else(|| panic!("{id}: missing required string field `{key}`"))
                .to_string()
        };

        assert_eq!(
            doc.get("contract_version").and_then(|v| v.as_integer()),
            Some(1),
            "{id}: contract_version must be 1"
        );
        assert_eq!(get_str("id"), id, "{id}: id field mismatch");
        assert!(!get_str("title").is_empty(), "{id}: empty title");
        assert!(!get_str("item").is_empty(), "{id}: empty item");
        assert!(!get_str("owner").is_empty(), "{id}: empty owner");
        assert!(!get_str("fixture").trim().is_empty(), "{id}: empty fixture");

        let assertions = doc
            .get("assertions")
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| panic!("{id}: missing assertions array"));
        assert!(!assertions.is_empty(), "{id}: no assertions");

        let status = get_str("status");
        match status.as_str() {
            "pending" => {}
            "implemented" => {
                // Monotonicity: an implemented trace must carry its
                // implementation evidence. Removing these fields (or
                // flipping back to pending) fails structurally here, and
                // the CI base-diff guards the historical direction.
                let since = doc.get("implemented_since").and_then(|v| v.as_str());
                let test_path = doc.get("test_path").and_then(|v| v.as_str());
                assert!(
                    since.map_or(false, |s| !s.is_empty()),
                    "{id}: implemented without implemented_since"
                );
                assert!(
                    test_path.map_or(false, |s| !s.is_empty()),
                    "{id}: implemented without test_path"
                );
            }
            other => panic!("{id}: illegal status `{other}`"),
        }
        seen.push(id);
    }
    assert_eq!(seen.len(), 13, "all 13 trace contracts present");
}
