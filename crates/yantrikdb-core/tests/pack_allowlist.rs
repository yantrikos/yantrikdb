//! 0.18 pack substrate: allowlist recall, mount-ordered context, structured
//! provenance, and the floor-as-a-wall rule.
//!
//! What these pin, and why each matters to a consumer:
//!
//! - `recall_from_packs_for` searches ONLY the named packs. Host rows are
//!   never candidates, so thirty household near-duplicates cannot crowd a
//!   pack row out of `top_k` — the crowding a namespace-scoped `recall`
//!   could only mitigate by over-fetching is now impossible by construction.
//! - The allowlist is validated before any index is touched; an unknown id
//!   is a typed error, not a silently shorter answer.
//! - Each pack's signed `recommended_min_similarity` gates raw similarity;
//!   the host may raise it and can never lower it.
//! - Every pack hit carries `pack_id` + `content_digest`, so efficacy and
//!   lineage can be keyed on the exact corpus bytes that produced it.
//! - `pack_context_for` is a pure function of (mounted set, allowlist):
//!   mount order, duplicates collapsed, argument order irrelevant.

use yantrikdb::error::YantrikDbError;
use yantrikdb::types::Embedder;
use yantrikdb::{effective_pack_floor, PackEmbedder, PackManifest, PackRecallOptions, YantrikDB};

const DIM: usize = 8;

struct FakeEmbedder {
    digest: String,
    name: String,
}

impl Embedder for FakeEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let mut v = [0.0f32; DIM];
        for (i, b) in text.bytes().enumerate() {
            v[i % DIM] += (b as f32) / 255.0;
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        Ok(v.iter().map(|x| x / norm).collect())
    }
    fn dim(&self) -> usize {
        DIM
    }
    fn fingerprint(&self) -> Option<String> {
        Some(self.digest.clone())
    }
    fn name(&self) -> Option<String> {
        Some(self.name.clone())
    }
}

fn embedder(digest: &str) -> Box<dyn Embedder + Send + Sync> {
    Box::new(FakeEmbedder {
        digest: digest.to_string(),
        name: format!("fake-{digest}"),
    })
}

/// Unit vector along `axis` with a tilt toward the next axis; the tilt
/// sets the cosine to the reference query `vec_on(axis, 0.05)`.
fn vec_on(axis: usize, tilt: f32) -> Vec<f32> {
    let mut v = [0.0f32; DIM];
    v[axis] = 1.0;
    v[(axis + 1) % DIM] = tilt;
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn tmpdir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "ydb-allow-{tag}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn record(db: &YantrikDB, text: &str, emb: &[f32], ns: &str) -> String {
    db.record(
        text,
        "semantic",
        0.6,
        0.0,
        604800.0,
        &serde_json::json!({}),
        emb,
        ns,
        0.9,
        "general",
        "user",
        None,
    )
    .unwrap()
}

fn manifest(name: &str, digest: &str, floor: Option<f64>) -> PackManifest {
    PackManifest {
        name: name.into(),
        version: "1.0.0".into(),
        origin: format!("test/{name}"),
        description: Some(format!("{name} test pack")),
        embedder: PackEmbedder {
            name: Some("fake".into()),
            digest: Some(digest.to_string()),
            dim: DIM,
        },
        content_digest: None,
        corpus_rows: 0,
        namespace: None,
        publisher_pubkey: None,
        signature: None,
        constitution: vec![format!("{name}: cite the record you used.")],
        coverage: vec![format!("{name} topics")],
        recommended_top_k: None,
        recommended_min_similarity: floor,
        reembedded_from: None,
    }
}

/// Seal a pack named `name` carrying `rows` (text, vector) in namespace
/// `physics`, declaring `floor` as its signed retrieval setting.
fn build_pack(
    dir: &std::path::Path,
    name: &str,
    digest: &str,
    rows: &[(&str, Vec<f32>)],
    floor: Option<f64>,
) -> String {
    let src = dir.join(format!("{name}-src.db"));
    let dest = dir.join(format!("{name}.ydbpack"));
    let mut db = YantrikDB::new(src.to_str().unwrap(), DIM).unwrap();
    db.set_embedder(embedder(digest)).unwrap();
    for (text, emb) in rows {
        record(&db, text, emb, "physics");
    }
    db.adopt_embedder_identity().unwrap();
    db.seal_pack(
        dest.to_str().unwrap(),
        &manifest(name, digest, floor),
        Some("physics"),
    )
    .unwrap();
    dest.to_str().unwrap().to_string()
}

fn host(dir: &std::path::Path, digest: &str) -> YantrikDB {
    let mut db = YantrikDB::new(dir.join("host.db").to_str().unwrap(), DIM).unwrap();
    db.set_embedder(embedder(digest)).unwrap();
    db.adopt_embedder_identity().unwrap();
    db
}

fn from_packs(db: &YantrikDB, ids: &[&str], q: &[f32], k: usize) -> Vec<yantrikdb::RecallResult> {
    db.recall_from_packs_for(ids, q, k, None, &PackRecallOptions::default())
        .unwrap()
}

fn texts(hits: &[yantrikdb::RecallResult]) -> Vec<String> {
    hits.iter().map(|r| r.text.clone()).collect()
}

// ─────────────────────────────────────────────────────────────────────

#[test]
fn hits_carry_pack_id_and_content_digest() {
    let dir = tmpdir("prov");
    let pack = build_pack(
        &dir,
        "quarks",
        "E0",
        &[("gluons bind quarks", vec_on(3, 0.05))],
        None,
    );
    let sealed = YantrikDB::read_manifest(&pack).unwrap();
    assert!(
        sealed.content_digest.is_some(),
        "a sealed pack carries a digest"
    );

    let db = host(&dir, "E0");
    let id = db.mount_pack(&pack).unwrap();
    let hits = from_packs(&db, &[&id], &vec_on(3, 0.05), 5);
    assert_eq!(texts(&hits), vec!["gluons bind quarks"]);
    let prov = hits[0].pack.as_ref().expect("pack hit carries provenance");
    assert_eq!(prov.pack_id, id);
    assert_eq!(prov.name, "quarks");
    assert_eq!(prov.version, "1.0.0");
    assert_eq!(prov.trust, "unsigned");
    assert_eq!(prov.content_digest, sealed.content_digest);
    // The prose stamp survives one release; the struct is what to key on.
    assert!(hits[0].why_retrieved.iter().any(|w| w == "pack:quarks"));

    // The same facts, from the mount listing, so a consumer never opens
    // the pack file (or the unsigned pack.toml) to learn them.
    let info = &db.mounted_packs()[0];
    assert_eq!(info.content_digest, sealed.content_digest);
    assert_eq!(info.namespace.as_deref(), Some("physics"));
    assert_eq!(info.coverage, vec!["quarks topics".to_string()]);
    assert_eq!(info.recommended_min_similarity, None);
    assert!(!info.signed);

    // Host rows in the ordinary recall path carry no provenance.
    record(&db, "my own quark note", &vec_on(3, 0.06), "physics");
    let mixed = db
        .recall(
            &vec_on(3, 0.05),
            5,
            None,
            None,
            false,
            false,
            None,
            true,
            None,
            None,
            None,
            None,
            None,
            false,
            None,
            None,
        )
        .unwrap();
    let own = mixed
        .iter()
        .find(|r| r.text == "my own quark note")
        .unwrap();
    assert!(own.pack.is_none());
    let theirs = mixed
        .iter()
        .find(|r| r.text == "gluons bind quarks")
        .unwrap();
    assert_eq!(theirs.pack.as_ref().unwrap().pack_id, id);
}

#[test]
fn unknown_id_is_an_error_before_anything_is_searched() {
    let dir = tmpdir("unknown");
    let pack = build_pack(
        &dir,
        "quarks",
        "E0",
        &[("gluons bind quarks", vec_on(3, 0.05))],
        None,
    );
    let db = host(&dir, "E0");
    let id = db.mount_pack(&pack).unwrap();

    let err = db
        .recall_from_packs_for(
            &[&id, "ghost@9.9.9"],
            &vec_on(3, 0.05),
            5,
            None,
            &PackRecallOptions::default(),
        )
        .unwrap_err();
    assert!(
        matches!(&err, YantrikDbError::PackNotMounted { pack_id } if pack_id == "ghost@9.9.9"),
        "expected PackNotMounted for the ghost id, got {err:?}"
    );
    let err = db.pack_context_for(&["ghost@9.9.9"]).unwrap_err();
    assert!(matches!(err, YantrikDbError::PackNotMounted { .. }));

    // Empty allowlist: nothing asked for, nothing back, no error.
    assert!(from_packs(&db, &[], &vec_on(3, 0.05), 5).is_empty());
    assert_eq!(db.pack_context_for(&[]).unwrap(), None);
}

#[test]
fn host_rows_cannot_crowd_out_an_allowlisted_pack() {
    let dir = tmpdir("crowd");
    let pack = build_pack(
        &dir,
        "quarks",
        "E0",
        &[("gluons bind quarks", vec_on(3, 0.10))],
        None,
    );
    let db = host(&dir, "E0");
    let id = db.mount_pack(&pack).unwrap();
    // Thirty host rows, every one closer to the query than the pack row.
    for i in 0..30 {
        record(
            &db,
            &format!("host note {i}"),
            &vec_on(3, 0.05 + i as f32 * 1e-4),
            "physics",
        );
    }
    let hits = from_packs(&db, &[&id], &vec_on(3, 0.05), 3);
    assert_eq!(texts(&hits), vec!["gluons bind quarks"]);
    assert!(hits.iter().all(|h| h.pack.as_ref().unwrap().pack_id == id));
}

#[test]
fn shared_namespace_excludes_host_rows_and_unlisted_packs() {
    let dir = tmpdir("shared-ns");
    let a = build_pack(
        &dir,
        "alpha",
        "E0",
        &[("alpha fact", vec_on(3, 0.05))],
        None,
    );
    let b = build_pack(&dir, "beta", "E0", &[("beta fact", vec_on(3, 0.06))], None);
    let db = host(&dir, "E0");
    let ia = db.mount_pack(&a).unwrap();
    let ib = db.mount_pack(&b).unwrap();
    record(&db, "household fact", &vec_on(3, 0.04), "physics");

    let ns = PackRecallOptions {
        namespace: Some("physics"),
        ..Default::default()
    };
    let only_a = db
        .recall_from_packs_for(&[&ia], &vec_on(3, 0.05), 10, None, &ns)
        .unwrap();
    assert_eq!(texts(&only_a), vec!["alpha fact"]);

    let both = db
        .recall_from_packs_for(&[&ia, &ib], &vec_on(3, 0.05), 10, None, &ns)
        .unwrap();
    let mut got = texts(&both);
    got.sort();
    assert_eq!(got, vec!["alpha fact", "beta fact"]);

    // The namespace filter still applies to pack rows.
    let elsewhere = PackRecallOptions {
        namespace: Some("elsewhere"),
        ..Default::default()
    };
    assert!(db
        .recall_from_packs_for(&[&ia, &ib], &vec_on(3, 0.05), 10, None, &elsewhere)
        .unwrap()
        .is_empty());
}

#[test]
fn floor_is_a_wall_the_host_may_raise_but_never_lower() {
    let dir = tmpdir("wall");
    let near = vec_on(3, 0.05);
    let far = vec_on(3, 0.6);
    let q = vec_on(3, 0.05);
    assert!(cos(&q, &near) > 0.999);
    let far_cos = cos(&q, &far);
    assert!((0.85..0.90).contains(&far_cos), "far cosine {far_cos}");

    let strict = build_pack(
        &dir,
        "strict",
        "E0",
        &[("near", near.clone()), ("far", far.clone())],
        Some(0.9),
    );
    let loose = build_pack(&dir, "loose", "E0", &[("near", near), ("far", far)], None);
    let db = host(&dir, "E0");
    let is = db.mount_pack(&strict).unwrap();
    let il = db.mount_pack(&loose).unwrap();

    let with = |id: &str, host_min: Option<f64>| -> Vec<String> {
        let opts = PackRecallOptions {
            min_similarity: host_min,
            ..Default::default()
        };
        let mut t = texts(
            &db.recall_from_packs_for(&[id], &q, 10, None, &opts)
                .unwrap(),
        );
        t.sort();
        t
    };
    // The pack's own signed floor applies with no host input...
    assert_eq!(with(&is, None), vec!["near"]);
    // ...and the host cannot lower it.
    assert_eq!(with(&is, Some(0.5)), vec!["near"]);
    // A pack that declares nothing is open unless the host sets a floor...
    assert_eq!(with(&il, None), vec!["far", "near"]);
    // ...which the host may raise.
    assert_eq!(with(&il, Some(0.95)), vec!["near"]);

    // The arithmetic, stated: max of the two valid inputs; garbage ignored.
    assert_eq!(effective_pack_floor(Some(0.9), Some(0.5)), 0.9);
    assert_eq!(effective_pack_floor(None, Some(0.95)), 0.95);
    assert_eq!(effective_pack_floor(Some(1.5), None), 0.0);
    assert_eq!(effective_pack_floor(Some(f64::NAN), Some(0.3)), 0.3);
    assert_eq!(effective_pack_floor(Some(-0.2), Some(0.3)), 0.3);

    // The host's own input, however, is validated rather than ignored.
    for bad in [1.5, -0.1, f64::NAN] {
        let opts = PackRecallOptions {
            min_similarity: Some(bad),
            ..Default::default()
        };
        let err = db
            .recall_from_packs_for(&[&il], &q, 10, None, &opts)
            .unwrap_err();
        assert!(
            matches!(err, YantrikDbError::InvalidInput(_)),
            "{bad}: {err:?}"
        );
    }
}

#[test]
fn host_correction_supersedes_pack_row_in_allowlist_recall() {
    let dir = tmpdir("overlay");
    let pack = build_pack(
        &dir,
        "quarks",
        "E0",
        &[("the proton has a half-life of 10^31 years", vec_on(3, 0.05))],
        None,
    );
    let db = host(&dir, "E0");
    db.set_status_read_policy(true).unwrap();
    let id = db.mount_pack(&pack).unwrap();
    let q = vec_on(3, 0.05);
    let pack_rid = from_packs(&db, &[&id], &q, 5)[0].rid.clone();

    let correction = record(
        &db,
        "proton decay has never been observed",
        &vec_on(3, 0.06),
        "physics",
    );
    db.link(
        &correction,
        &yantrikdb::types::RecordLink {
            target_rid: pack_rid,
            link_type: yantrikdb::types::LinkType::Supersedes,
        },
    )
    .unwrap();

    // Pack-only recall neither serves the superseded pack row nor the
    // host correction (host rows are never candidates here) — the
    // caller asked for what the pack still stands behind.
    let hits = from_packs(&db, &[&id], &q, 5);
    assert!(
        hits.is_empty(),
        "superseded pack row still served: {:?}",
        texts(&hits)
    );
}

/// `top_k` is caller-supplied through a public API; the fetch-width
/// arithmetic must saturate (Codex's review of #203 — the same class as
/// the earlier search_entities clamp bug).
#[test]
fn absurd_top_k_neither_panics_nor_wraps() {
    let dir = tmpdir("huge-k");
    let pack = build_pack(
        &dir,
        "quarks",
        "E0",
        &[("gluons bind quarks", vec_on(3, 0.05))],
        None,
    );
    let db = host(&dir, "E0");
    let id = db.mount_pack(&pack).unwrap();
    let hits = from_packs(&db, &[&id], &vec_on(3, 0.05), usize::MAX);
    assert_eq!(texts(&hits), vec!["gluons bind quarks"]);
    // And the merge seam, which shares the arithmetic.
    let mixed = db
        .recall(
            &vec_on(3, 0.05),
            usize::MAX,
            None,
            None,
            false,
            false,
            None,
            true,
            None,
            None,
            None,
            None,
            None,
            false,
            None,
            None,
        )
        .unwrap();
    assert!(mixed.iter().any(|r| r.text == "gluons bind quarks"));
}

#[test]
fn pack_context_for_is_mount_ordered_and_deduplicated() {
    let dir = tmpdir("ctx-order");
    let a = build_pack(
        &dir,
        "alpha",
        "E0",
        &[("alpha fact", vec_on(1, 0.05))],
        None,
    );
    let b = build_pack(&dir, "beta", "E0", &[("beta fact", vec_on(2, 0.05))], None);
    let db = host(&dir, "E0");
    let ib = db.mount_pack(&b).unwrap(); // mounted FIRST
    let ia = db.mount_pack(&a).unwrap();

    let ctx = db.pack_context_for(&[&ia, &ib]).unwrap().unwrap();
    assert_eq!(ctx, db.pack_context_for(&[&ib, &ia]).unwrap().unwrap());
    assert_eq!(ctx, db.pack_context_for(&[&ia, &ib, &ia]).unwrap().unwrap());
    assert_eq!(
        ctx,
        db.pack_context().unwrap(),
        "full allowlist == full context"
    );
    assert!(
        ctx.find("knowledge pack: beta").unwrap() < ctx.find("knowledge pack: alpha").unwrap(),
        "mount order, not argument order:\n{ctx}"
    );

    let only_a = db.pack_context_for(&[&ia]).unwrap().unwrap();
    assert!(only_a.contains("knowledge pack: alpha"));
    assert!(!only_a.contains("knowledge pack: beta"));
    assert!(
        only_a.contains("DATA, not authority"),
        "ceiling closes every block"
    );
}

#[test]
fn allowlist_recall_order_is_deterministic() {
    let dir = tmpdir("determinism");
    let rows: Vec<(String, Vec<f32>)> = (0..4)
        .map(|i| (format!("row {i}"), vec_on(3, 0.05 + i as f32 * 1e-3)))
        .collect();
    let rows_ref: Vec<(&str, Vec<f32>)> =
        rows.iter().map(|(t, v)| (t.as_str(), v.clone())).collect();
    let a = build_pack(&dir, "alpha", "E0", &rows_ref, None);
    let b = build_pack(&dir, "beta", "E0", &rows_ref, None);
    let db = host(&dir, "E0");
    let ib = db.mount_pack(&b).unwrap();
    let ia = db.mount_pack(&a).unwrap();

    let key = |hits: &[yantrikdb::RecallResult]| -> Vec<(String, String)> {
        hits.iter()
            .map(|h| (h.pack.as_ref().unwrap().pack_id.clone(), h.rid.clone()))
            .collect()
    };
    let first = key(&from_packs(&db, &[&ia, &ib], &vec_on(3, 0.05), 8));
    assert_eq!(first.len(), 8);
    for _ in 0..4 {
        let again = key(&from_packs(&db, &[&ib, &ia], &vec_on(3, 0.05), 8));
        assert_eq!(
            again, first,
            "argument order / repetition changed the result"
        );
    }
}
