use super::*;

// ── embedder window detection: silent truncation must not stay silent ──

/// Embeds a bag-of-chars signature of the text, but only of the first
/// `window` characters — a faithful miniature of what a transformer
/// does when its input window is exceeded.
struct TruncatingEmbedder {
    window: usize,
    dim: usize,
}

impl crate::types::Embedder for TruncatingEmbedder {
    fn embed(
        &self,
        text: &str,
    ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        let seen = &text[..self.window.min(text.len())];
        let mut v = vec![0.0_f32; self.dim];
        for (i, b) in seen.bytes().enumerate() {
            v[(b as usize + i % 3) % self.dim] += 1.0;
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        Ok(v)
    }
    fn dim(&self) -> usize {
        self.dim
    }
}

#[test]
fn detects_a_truncating_embedder_and_counts_the_writes_it_silently_clips() {
    let mut db = YantrikDB::new(":memory:", 32).unwrap();
    db.set_embedder(Box::new(TruncatingEmbedder {
        window: 1000,
        dim: 32,
    }))
    .unwrap();

    let found = db.detect_embedder_window().unwrap().expect("truncation");
    // Binary search resolves to within its 64-char step of the truth.
    assert!(
        (900..=1064).contains(&found),
        "detected window {found} should be near the real 1000"
    );
    assert_eq!(db.embedder_window(), Some(found));

    // A record inside the window is silent; one past it is counted.
    assert_eq!(db.embedder_truncated_write_count(), 0);
    db.embed(&"x".repeat(500)).unwrap();
    assert_eq!(db.embedder_truncated_write_count(), 0, "fits, no warning");
    db.embed(&"y".repeat(5000)).unwrap();
    assert_eq!(
        db.embedder_truncated_write_count(),
        1,
        "a write whose tail is never embedded must be counted, not swallowed"
    );
    let s = db.stats(None).unwrap();
    assert_eq!(s.embedder_truncated_writes, 1);
    assert_eq!(s.embedder_window_chars, Some(found));
}

// ── chunked embeddings: the tail of a long record must be findable ──

/// A [`TruncatingEmbedder`] with an identity, so the probed window can
/// persist to `meta` (keyed to the digest) and be adopted on reopen —
/// the restart half of the chunking contract.
struct FingerprintedTruncatingEmbedder {
    inner: TruncatingEmbedder,
    fp: &'static str,
}

impl crate::types::Embedder for FingerprintedTruncatingEmbedder {
    fn embed(
        &self,
        text: &str,
    ) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error + Send + Sync>> {
        self.inner.embed(text)
    }
    fn dim(&self) -> usize {
        self.inner.dim
    }
    fn fingerprint(&self) -> Option<String> {
        Some(self.fp.to_string())
    }
    fn name(&self) -> Option<String> {
        Some("truncating-test".to_string())
    }
}

/// Text whose HEAD is filler and whose TAIL carries the distinctive
/// content — the exact shape the production defect lost (end-cue
/// retrieval measured at 8% vs 28% for start cues).
fn long_text_with_tail(tail: &str) -> String {
    format!("{}{}", "the quick brown fox sits idle. ".repeat(60), tail)
}

const TAIL: &str = "ZURICH-QUANTUM-9 pairing keys rotate through vault XKCD-221 \
     every equinox; the fallback passphrase lives with the harbormaster.";

fn record_defaults(db: &YantrikDB, text: &str) -> String {
    db.record_text(
        text,
        "episodic",
        0.5,
        0.0,
        604800.0,
        &serde_json::json!({}),
        "default",
        0.8,
        "general",
        "user",
        None,
    )
    .unwrap()
}

#[test]
fn chunked_write_makes_the_tail_findable_and_counts_as_handled() {
    // dim 256, not 32: at dim 32 the hash-trick buckets collide so
    // heavily that a pure-filler window can sit closer to the tail
    // query than the tail window itself (same collision class as the
    // 'Z'/'z' mod-32 case in the distance-margin tests). The span
    // assertion below needs the geometry to be meaningful.
    let mut db = YantrikDB::new(":memory:", 256).unwrap();
    db.set_embedder(Box::new(TruncatingEmbedder {
        window: 400,
        dim: 256,
    }))
    .unwrap();
    db.detect_embedder_window().unwrap().expect("truncation");

    // Decoys that look exactly like the long record's HEAD, so a
    // head-only vector cannot win a tail query on similarity.
    for i in 0..5 {
        record_defaults(
            &db,
            &format!("{}{i}", "the quick brown fox sits idle. ".repeat(11)),
        );
    }
    let rid = record_defaults(&db, &long_text_with_tail(TAIL));

    let s = db.stats(None).unwrap();
    assert_eq!(
        s.embedder_chunked_writes, 1,
        "one write overflowed and was chunked"
    );
    assert!(
        s.chunk_vectors >= 1,
        "durable window vectors must exist (got {})",
        s.chunk_vectors
    );
    assert_eq!(
        s.embedder_truncated_writes, 0,
        "a chunked overflow is handled, not truncation loss"
    );

    // THE defect scenario: query by content that lives only in the tail.
    let hits = db.recall_text(TAIL, 3).unwrap();
    assert_eq!(
        hits.first().map(|r| r.rid.as_str()),
        Some(rid.as_str()),
        "tail cue must find its parent at rank 1 (head-only vectors lose it)"
    );
    // The collapse contract: chunk keys never leak into results.
    for r in &hits {
        assert!(
            !r.rid.contains("#c"),
            "RecallResult.rid must be a real memories rid, got {}",
            r.rid
        );
    }

    // Snippet projection: the span must point INTO the tail — the
    // window that actually matched — not the filler head, so a
    // consumer trimming by best_span ships the matching region.
    let hit = &hits[0];
    let (a, b) = hit
        .best_span
        .expect("a record longer than the window must carry a span");
    assert!(
        hit.text[a..b].contains("ZURICH-QUANTUM-9"),
        "span [{a}, {b}) must cover the matched tail, got: {:?}",
        &hit.text[a..b.min(a + 80)]
    );
    assert!(a > 0, "the head window does not contain the tail cue");
}

#[test]
fn chunk_vectors_survive_reopen_and_the_window_survives_via_meta() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("chunks.db");
    let path = path.to_str().unwrap();

    let found = {
        let mut db = YantrikDB::new(path, 32).unwrap();
        db.set_embedder(Box::new(FingerprintedTruncatingEmbedder {
            inner: TruncatingEmbedder {
                window: 400,
                dim: 32,
            },
            fp: "sha256:trunc-test-1",
        }))
        .unwrap();
        let found = db.detect_embedder_window().unwrap().expect("truncation");
        record_defaults(&db, &long_text_with_tail(TAIL));
        for i in 0..5 {
            record_defaults(
                &db,
                &format!("{}{i}", "the quick brown fox sits idle. ".repeat(11)),
            );
        }
        found
    };

    // Reopen: NO probe this time. The rebuild must re-index the chunk
    // rows (or recall quality silently differs across restarts), and
    // the persisted window must be adopted for the same digest.
    let mut db = YantrikDB::new(path, 32).unwrap();
    db.set_embedder(Box::new(FingerprintedTruncatingEmbedder {
        inner: TruncatingEmbedder {
            window: 400,
            dim: 32,
        },
        fp: "sha256:trunc-test-1",
    }))
    .unwrap();
    assert_eq!(
        db.embedder_window(),
        Some(found),
        "probed window must survive restart via meta (same digest)"
    );

    let hits = db.recall_text(TAIL, 3).unwrap();
    assert!(
        hits.first().is_some_and(|r| r.text.contains("ZURICH")),
        "tail cue must still find its parent after reopen — the rebuild \
         must carry chunk vectors"
    );

    // A DIFFERENT embedder digest must NOT adopt this window.
    let mut db2 = YantrikDB::new(":memory:", 32).unwrap();
    db2.set_embedder(Box::new(FingerprintedTruncatingEmbedder {
        inner: TruncatingEmbedder {
            window: 400,
            dim: 32,
        },
        fp: "sha256:other-model",
    }))
    .unwrap();
    assert_eq!(db2.embedder_window(), None, "fresh DB, never probed");
}

#[test]
fn forget_takes_the_windows_with_it() {
    let mut db = YantrikDB::new(":memory:", 32).unwrap();
    db.set_embedder(Box::new(TruncatingEmbedder {
        window: 400,
        dim: 32,
    }))
    .unwrap();
    db.detect_embedder_window().unwrap().expect("truncation");

    let rid = record_defaults(&db, &long_text_with_tail(TAIL));
    let keeper = record_defaults(&db, "an unrelated small note about harbors");
    assert_eq!(
        db.recall_text(TAIL, 1)
            .unwrap()
            .first()
            .map(|r| r.rid.clone()),
        Some(rid.clone())
    );

    assert!(db.forget(&rid).unwrap());
    let hits = db.recall_text(TAIL, 5).unwrap();
    assert!(
        hits.iter().all(|r| r.rid != rid),
        "a forgotten record must not resurface through its window keys"
    );
    assert!(hits.iter().all(|r| !r.rid.contains("#c")));
    let _ = keeper;
    assert_eq!(
        db.stats(None).unwrap().chunk_vectors,
        0,
        "forget drops the chunk rows too"
    );
}

#[test]
fn correction_that_shrinks_the_text_tombstones_surplus_windows() {
    // dim 256, not 32: at dim 32 the bag-of-chars buckets collide mod 32
    // ('Z' and 'z' share a bucket), which blurs the head/window distance
    // margin this test decides on. At 256 every byte value keeps its own
    // bucket.
    let mut db = YantrikDB::new(":memory:", 256).unwrap();
    db.set_embedder(Box::new(TruncatingEmbedder {
        window: 400,
        dim: 256,
    }))
    .unwrap();
    db.detect_embedder_window().unwrap().expect("truncation");

    // Tail bytes are DISJOINT from the lowercase filler (uppercase +
    // digits + dashes), so under the bag-of-chars mock the final window
    // is near-identical to the tail cue and the head is near-orthogonal
    // — the distances tell window hits and head hits apart decisively.
    let tail = "ZURICH-QUANTUM-9-VAULT-XKCD-221-EQUINOX-77-DELTA-4-HARBORMASTER-".repeat(6);
    let rid = record_defaults(&db, &long_text_with_tail(&tail));
    assert!(db.stats(None).unwrap().chunk_vectors >= 1);

    let cue = db.embed(&tail).unwrap();
    let pre = db.search_state.load().vec_index.search(&cue, 3).unwrap();
    let pre_dist = pre
        .iter()
        .find(|(k, _)| k == &rid)
        .map(|(_, d)| *d)
        .expect("window vector must serve the tail cue before correction");
    assert!(
        pre_dist < 0.3,
        "pre-correction the tail cue hits a near-verbatim window (dist {pre_dist})"
    );

    // Correct to a SHORT text: every old window is surplus.
    db.correct(
        &rid,
        Some("short corrected note, fits the window"),
        None,
        None,
        None,
        "test shrink",
    )
    .unwrap();

    assert_eq!(
        db.stats(None).unwrap().chunk_vectors,
        0,
        "corrected-away windows must not survive in the table"
    );
    let post = db.search_state.load().vec_index.search(&cue, 3).unwrap();
    let post_dist = post.iter().find(|(k, _)| k == &rid).map(|(_, d)| *d);
    assert!(
        post_dist.is_none_or(|d| d > 0.6),
        "after correction the old-tail cue may at most hit the corrected \
         HEAD vector (far), never a stale window (near) — got {post_dist:?}, \
         which means a surplus window survived and is serving vanished text"
    );
}

#[test]
fn rechunk_backfills_records_written_before_the_probe() {
    let mut db = YantrikDB::new(":memory:", 32).unwrap();
    db.set_embedder(Box::new(TruncatingEmbedder {
        window: 400,
        dim: 32,
    }))
    .unwrap();

    // Written BEFORE any window is known: stored head-only, counted as
    // truncated — the state every pre-chunking corpus is in.
    let rid = record_defaults(&db, &long_text_with_tail(TAIL));
    for i in 0..5 {
        record_defaults(
            &db,
            &format!("{}{i}", "the quick brown fox sits idle. ".repeat(11)),
        );
    }
    assert_eq!(db.stats(None).unwrap().chunk_vectors, 0);

    db.detect_embedder_window().unwrap().expect("truncation");
    let (records, vectors) = db.rechunk_long_records().unwrap();
    assert_eq!(records, 1, "exactly the one overflowing record backfills");
    assert!(vectors >= 1);

    let hits = db.recall_text(TAIL, 3).unwrap();
    assert_eq!(
        hits.first().map(|r| r.rid.as_str()),
        Some(rid.as_str()),
        "after backfill the tail must find its parent"
    );

    // Idempotent: a second run has nothing to do.
    assert_eq!(db.rechunk_long_records().unwrap(), (0, 0));
}

#[test]
fn an_embedder_with_no_window_reports_no_truncation() {
    let mut db = YantrikDB::new(":memory:", 32).unwrap();
    db.set_embedder(Box::new(TruncatingEmbedder {
        // Larger than the probe's ceiling: nothing is ever clipped.
        window: 10_000_000,
        dim: 32,
    }))
    .unwrap();
    assert_eq!(db.detect_embedder_window().unwrap(), None);
    assert_eq!(db.embedder_window(), None);
    db.embed(&"z".repeat(200_000)).unwrap();
    assert_eq!(
        db.embedder_truncated_write_count(),
        0,
        "no window, no warning"
    );
    assert_eq!(db.stats(None).unwrap().embedder_window_chars, None);
}
