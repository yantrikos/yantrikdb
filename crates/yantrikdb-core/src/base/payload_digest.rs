//! Canonical payload digest for idempotent writes (v0.10 Item 4a.5).
//!
//! `idempotency_claims.payload_digest` answers exactly one question: a claim
//! already exists for this (origin_actor, namespace, idempotency_key) — is the
//! incoming write the SAME write (a retry, which must be a no-op returning the
//! original rid) or a DIFFERENT write reusing the key (a conflict, which must be
//! refused)? Getting this wrong in either direction is a data-loss bug:
//!   - digest too LOOSE  → a genuinely different write is swallowed as a retry.
//!   - digest too TIGHT  → an honest retry looks like a conflict and is refused.
//!
//! # Why not `serde_json::to_string(&payload)`
//!
//! Hashing the JSON string form looks equivalent and is not. `serde_json::Map`
//! is a `BTreeMap` ONLY while the `preserve_order` feature is off; it is off
//! today purely because nothing in the tree turns it on. Cargo unifies features
//! across the whole dependency graph, so ANY transitive dependency enabling
//! `serde_json/preserve_order` would silently flip `Map` to insertion-ordered
//! `IndexMap` — changing every digest, with no compile error, no test failure at
//! the call site, and idempotency dedup quietly breaking in the field. This
//! module therefore sorts keys EXPLICITLY and never depends on `Map`'s backing
//! store. `digest_is_stable_under_object_key_insertion_order` pins that.
//!
//! # Framing
//!
//! `warrant.rs::content_hash` (the house canonical-hash pattern) separates
//! fields with `b"|"`. That is sound there because claim fields are constrained
//! ids and enum-ish strings. It is NOT sound here: `text`, `source`, `domain`
//! and metadata strings are all free-form and caller-controlled, so separator
//! framing is ambiguous — `text="a|b", domain="c"` and `text="a", domain="b|c"`
//! would feed the hasher identical bytes and collide. Every variable-length
//! field here is LENGTH-PREFIXED instead, which is unambiguous for arbitrary
//! input. This is a deliberate divergence from the house pattern.
//!
//! # Floats are hashed by exact bits, never quantized
//!
//! `warrant.rs::content_hash` rounds weights to millesimals, and copying that
//! here would be a data-loss bug. Its rationale is ARITHMETIC drift: it hashes a
//! derived accumulator input the engine computes. These values are CALLER-
//! SUPPLIED and arrive over JSON/PyO3, which round-trip f64 exactly — a retry
//! re-sends bit-identical floats, so there is no drift to absorb. Quantizing
//! would buy nothing and would digest `importance=0.5` and `importance=0.5004`
//! equal, silently swallowing the second write as a retry. If a caller really
//! does recompute a scalar nondeterministically between attempts, that is a
//! genuinely different write and surfacing it as a conflict is correct.
//!
//! # What is NOT in the digest, and why
//!
//! - the GENERATED embedding (`record_text`, where the engine embeds) — it is
//!   derived from `text` and f32-drifty across embedder builds, so including it
//!   would make an honest retry look like a conflict. Idempotency is decided
//!   BEFORE re-embedding. The CALLER-SUPPLIED embedding (`record`) is a
//!   different matter and IS included: it round-trips exactly, and two writes
//!   with the same text but different vectors recall differently, so they are
//!   different writes.
//! - `rid`, `op_id`, `created_at`, `updated_at`, HLC — per-attempt values. Any
//!   of them in the digest would make EVERY retry a fresh payload and defeat the
//!   whole mechanism.
//! - `route` ('sync' | 'queued') — tracked as its own column on the claim, so
//!   the same logical write does not conflict with itself for being queued.
//! - `kind` — a generated column derived from metadata, already covered.
//!
//! # Known deviation from the design doc: NFC
//!
//! `docs/V0.10_ITEM4_DESIGN.md` specifies "strings NFC-normalized UTF-8". This
//! module hashes exact UTF-8 bytes instead. Rationale for deferring: NFC needs a
//! new `unicode-normalization` dependency (the only normalization crate in the
//! lock today is `unicode-normalization-alignments`, a tokenizers fork reachable
//! only behind the embedder feature — unusable under `--no-default-features`),
//! and the cost of omitting it is bounded: a retry whose text changed
//! normalization mid-flight surfaces as a TYPED CONFLICT carrying `existing_rid`,
//! which the caller can recover from — not silent data loss. T07's fixture is
//! byte-identical repeats, so this does not block the contract. Revisit if a
//! real client hits it (macOS NFD input is the plausible source).

use crate::base::serde_helpers::serialize_f32;
use crate::base::types::RecordInput;

/// Bump to invalidate every stored `payload_digest` when the construction
/// below changes. Stored digests from a previous version will then compare
/// unequal, which surfaces as a conflict rather than a false retry — the
/// fail-safe direction.
pub const PAYLOAD_DIGEST_VERSION: u32 = 1;

/// Distinguishes write shapes so two ops with coincidentally equal fields cannot
/// collide, and decides whether the embedding participates: `record` carries a
/// caller-supplied vector (digested), `record_text` has the engine generate one
/// (excluded, so idempotency is decided before re-embedding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PayloadVariant {
    /// `record()` — caller supplies the embedding, so it is part of the payload.
    Record,
    /// `record_text()` — engine embeds the text; the generated vector is excluded.
    RecordText,
}

impl PayloadVariant {
    fn tag(self) -> u8 {
        match self {
            PayloadVariant::Record => 1,
            PayloadVariant::RecordText => 2,
        }
    }
}

/// A borrowed view of the semantic content of a record-shaped write.
///
/// This is intentionally a view rather than `&RecordInput` so the digest can be
/// computed from the oplog JSON on the replay path too, where no `RecordInput`
/// exists and the field names differ (`type` vs `memory_type`).
#[derive(Debug)]
pub struct PayloadView<'a> {
    pub variant: PayloadVariant,
    pub namespace: &'a str,
    pub text: &'a str,
    pub memory_type: &'a str,
    pub importance: f64,
    pub valence: f64,
    pub half_life: f64,
    pub certainty: f64,
    pub domain: &'a str,
    pub source: &'a str,
    pub emotional_state: Option<&'a str>,
    pub metadata: &'a serde_json::Value,
    /// The CALLER-SUPPLIED embedding. `None` for `record_text`, where the engine
    /// generates the vector and idempotency must be decided before embedding.
    pub embedding: Option<&'a [f32]>,
}

impl<'a> PayloadView<'a> {
    /// Build a view over a `RecordInput`.
    ///
    /// The embedding participates iff `variant` is `Record` — for `RecordText`
    /// the vector in `input` was generated by the engine, not supplied by the
    /// caller, so digesting it would make an embedder re-run look like a
    /// conflict on an honest retry.
    pub fn from_record_input(input: &'a RecordInput, variant: PayloadVariant) -> Self {
        PayloadView {
            variant,
            namespace: &input.namespace,
            text: &input.text,
            memory_type: &input.memory_type,
            importance: input.importance,
            valence: input.valence,
            half_life: input.half_life,
            certainty: input.certainty,
            domain: &input.domain,
            source: &input.source,
            emotional_state: input.emotional_state.as_deref(),
            metadata: &input.metadata,
            embedding: match variant {
                PayloadVariant::Record => Some(&input.embedding),
                PayloadVariant::RecordText => None,
            },
        }
    }
}

// Type tags. Distinct constants so a string can never feed the hasher the same
// bytes as a differently-typed value carrying the same payload.
const T_NONE: u8 = 0;
const T_SOME: u8 = 1;
const T_NUM: u8 = 10;
const T_NAN: u8 = 11;
const J_NULL: u8 = 20;
const J_BOOL: u8 = 21;
const J_NUM_I64: u8 = 22;
const J_NUM_U64: u8 = 23;
const J_NUM_F64: u8 = 24;
const J_NUM_OTHER: u8 = 25;
const J_STR: u8 = 26;
const J_ARR: u8 = 27;
const J_OBJ: u8 = 28;

/// Length-prefixed, so arbitrary caller-controlled bytes are unambiguous.
///
/// Text is hashed as its exact UTF-8 bytes with NO Unicode normalization: a
/// retry re-sends identical bytes, so byte equality is the right test. Two
/// visually identical strings differing in NFC/NFD composition are different
/// payloads here, deliberately — the engine stores what it was given, and
/// declaring them equal would silently drop one.
fn feed_str(h: &mut blake3::Hasher, s: &str) {
    h.update(&(s.len() as u64).to_le_bytes());
    h.update(s.as_bytes());
}

fn feed_opt_str(h: &mut blake3::Hasher, s: Option<&str>) {
    match s {
        None => {
            h.update(&[T_NONE]);
        }
        Some(v) => {
            h.update(&[T_SOME]);
            feed_str(h, v);
        }
    }
}

/// Every f64 in the payload — engine scalars and metadata numbers alike — is
/// hashed by exact bits. See the module docs for why quantization would be a
/// data-loss bug here, despite the `warrant.rs::content_hash` precedent.
///
/// `-0.0` folds onto `0.0`: they compare equal, so they must digest equal, and
/// their bit patterns differ. All NaN payloads collapse to a single tag (NaN has
/// many bit patterns, none of them meaningfully distinct writes); ±inf keep
/// their distinct bits. `validate_scalars` rejects non-finites upstream, but this
/// function is pure and must not depend on its callers having done that.
fn feed_f64(h: &mut blake3::Hasher, x: f64) {
    if x.is_nan() {
        h.update(&[T_NAN]);
        return;
    }
    h.update(&[T_NUM]);
    let normalized = if x == 0.0 { 0.0 } else { x };
    h.update(&normalized.to_bits().to_le_bytes());
}

/// Caller-supplied embedding, via the deterministic byte rule that already
/// exists for vectors (`serde_helpers::serialize_f32`). Length-prefixed like
/// every other variable-length field, and presence-tagged so an absent vector
/// (`record_text`) cannot alias onto an empty one.
fn feed_embedding(h: &mut blake3::Hasher, emb: Option<&[f32]>) {
    match emb {
        None => {
            h.update(&[T_NONE]);
        }
        Some(v) => {
            h.update(&[T_SOME]);
            let bytes = serialize_f32(v);
            h.update(&(bytes.len() as u64).to_le_bytes());
            h.update(&bytes);
        }
    }
}

/// Canonical encoding of a JSON value. Object keys are sorted EXPLICITLY here —
/// never trust `serde_json::Map`'s backing store (see module docs).
fn feed_json(h: &mut blake3::Hasher, v: &serde_json::Value) {
    match v {
        serde_json::Value::Null => {
            h.update(&[J_NULL]);
        }
        serde_json::Value::Bool(b) => {
            h.update(&[J_BOOL]);
            h.update(&[*b as u8]);
        }
        serde_json::Value::Number(n) => {
            // i64/u64/f64 are tagged separately: 1 and 1.0 are distinct JSON
            // numbers that round-trip differently, so they are distinct payloads.
            if let Some(i) = n.as_i64() {
                h.update(&[J_NUM_I64]);
                h.update(&i.to_le_bytes());
            } else if let Some(u) = n.as_u64() {
                h.update(&[J_NUM_U64]);
                h.update(&u.to_le_bytes());
            } else if let Some(f) = n.as_f64() {
                h.update(&[J_NUM_F64]);
                feed_f64(h, f);
            } else {
                // Unrepresentable (only reachable under serde_json's
                // arbitrary_precision feature). Fall back to the literal text so
                // the value still participates rather than silently vanishing.
                h.update(&[J_NUM_OTHER]);
                feed_str(h, &n.to_string());
            }
        }
        serde_json::Value::String(s) => {
            h.update(&[J_STR]);
            feed_str(h, s);
        }
        serde_json::Value::Array(a) => {
            // Array order is semantic in JSON — do NOT sort.
            h.update(&[J_ARR]);
            h.update(&(a.len() as u64).to_le_bytes());
            for e in a {
                feed_json(h, e);
            }
        }
        serde_json::Value::Object(m) => {
            h.update(&[J_OBJ]);
            h.update(&(m.len() as u64).to_le_bytes());
            let mut keys: Vec<&String> = m.keys().collect();
            keys.sort();
            for k in keys {
                feed_str(h, k);
                // Present because it came from `m.keys()`.
                if let Some(val) = m.get(k) {
                    feed_json(h, val);
                }
            }
        }
    }
}

/// Canonical digest of the semantic content of a record-shaped write.
///
/// Pure and total: no I/O, no clock, no failure mode. Stable across processes,
/// platforms and std-hasher changes (blake3, like every other persisted digest
/// in this codebase).
pub fn payload_digest(v: &PayloadView<'_>) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(b"yantrikdb.payload.v");
    h.update(&PAYLOAD_DIGEST_VERSION.to_le_bytes());
    h.update(&[v.variant.tag()]);
    feed_str(&mut h, v.namespace);
    feed_str(&mut h, v.text);
    feed_str(&mut h, v.memory_type);
    feed_f64(&mut h, v.importance);
    feed_f64(&mut h, v.valence);
    feed_f64(&mut h, v.half_life);
    feed_f64(&mut h, v.certainty);
    feed_str(&mut h, v.domain);
    feed_str(&mut h, v.source);
    feed_opt_str(&mut h, v.emotional_state);
    feed_json(&mut h, v.metadata);
    feed_embedding(&mut h, v.embedding);
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn view<'a>(text: &'a str, metadata: &'a serde_json::Value) -> PayloadView<'a> {
        PayloadView {
            variant: PayloadVariant::Record,
            namespace: "default",
            text,
            memory_type: "semantic",
            importance: 0.5,
            valence: 0.0,
            half_life: 604800.0,
            certainty: 0.8,
            domain: "general",
            source: "user",
            emotional_state: None,
            metadata,
            embedding: None,
        }
    }

    #[test]
    fn digest_is_deterministic_and_content_sensitive() {
        let m = json!({"a": 1});
        assert_eq!(
            payload_digest(&view("hello", &m)),
            payload_digest(&view("hello", &m))
        );
        assert_ne!(
            payload_digest(&view("hello", &m)),
            payload_digest(&view("hello!", &m))
        );
    }

    #[test]
    fn digest_is_stable_under_object_key_insertion_order() {
        // THE guard test for the `preserve_order` hazard (see module docs). Under
        // serde_json's default BTreeMap backing this passes trivially; if any
        // transitive dep ever enables `preserve_order`, Map becomes insertion-
        // ordered and this fails LOUDLY instead of silently corrupting dedup.
        let a: serde_json::Value = serde_json::from_str(r#"{"z":1,"a":2,"m":3}"#).unwrap();
        let b: serde_json::Value = serde_json::from_str(r#"{"a":2,"m":3,"z":1}"#).unwrap();
        assert_eq!(
            payload_digest(&view("t", &a)),
            payload_digest(&view("t", &b))
        );
    }

    #[test]
    fn digest_framing_is_unambiguous_for_free_form_fields() {
        // The reason for length-prefixing over the house `|` separator: these two
        // payloads are DIFFERENT writes and must not collide.
        let m = json!({});
        let mut x = view("a|b", &m);
        x.domain = "c";
        let mut y = view("a", &m);
        y.domain = "b|c";
        assert_ne!(payload_digest(&x), payload_digest(&y));
    }

    #[test]
    fn digest_distinguishes_variants() {
        let m = json!({});
        let mut rt = view("same text", &m);
        rt.variant = PayloadVariant::RecordText;
        assert_ne!(payload_digest(&view("same text", &m)), payload_digest(&rt));
    }

    #[test]
    fn floats_are_never_quantized() {
        // The data-loss guard. Quantizing to millesimals (the warrant.rs
        // precedent) would digest these equal and silently swallow the second
        // write as a retry. Caller-supplied floats round-trip exactly, so there
        // is no drift to absorb and no reason to blur them.
        let m = json!({});
        let mut a = view("t", &m);
        a.importance = 0.5;
        let mut b = view("t", &m);
        b.importance = 0.5004;
        assert_ne!(
            payload_digest(&a),
            payload_digest(&b),
            "engine scalar was quantized"
        );

        let m1 = json!({"price": 1.00001});
        let m2 = json!({"price": 1.00002});
        assert_ne!(
            payload_digest(&view("t", &m1)),
            payload_digest(&view("t", &m2)),
            "metadata float was quantized"
        );
    }

    #[test]
    fn negative_zero_digests_as_zero() {
        // -0.0 == 0.0 compares true, so equal values must digest equal; their raw
        // bit patterns differ, so this needs explicit normalization.
        let m = json!({});
        let mut neg = view("t", &m);
        neg.valence = -0.0;
        let mut pos = view("t", &m);
        pos.valence = 0.0;
        assert_eq!(payload_digest(&neg), payload_digest(&pos));
        assert_eq!(
            payload_digest(&view("t", &json!({"k": -0.0}))),
            payload_digest(&view("t", &json!({"k": 0.0})))
        );
    }

    #[test]
    fn non_finite_scalars_do_not_alias() {
        let m = json!({});
        let mut nan = view("t", &m);
        nan.importance = f64::NAN;
        let mut zero = view("t", &m);
        zero.importance = 0.0;
        let mut inf = view("t", &m);
        inf.importance = f64::INFINITY;
        let mut neg_inf = view("t", &m);
        neg_inf.importance = f64::NEG_INFINITY;
        let d_nan = payload_digest(&nan);
        assert_ne!(d_nan, payload_digest(&zero));
        assert_ne!(d_nan, payload_digest(&inf));
        assert_ne!(payload_digest(&inf), payload_digest(&neg_inf));
        // Every NaN bit pattern is the same non-value, so they must not split.
        let mut nan2 = view("t", &m);
        nan2.importance = f64::from_bits(f64::NAN.to_bits() | 1);
        assert!(nan2.importance.is_nan());
        assert_eq!(d_nan, payload_digest(&nan2));
    }

    #[test]
    fn json_types_do_not_collide() {
        // 1 vs 1.0 vs "1" vs true vs null are distinct JSON values.
        let vals = [
            json!({"k": 1}),
            json!({"k": 1.0}),
            json!({"k": "1"}),
            json!({"k": true}),
            json!({"k": null}),
        ];
        let digests: Vec<_> = vals.iter().map(|m| payload_digest(&view("t", m))).collect();
        for i in 0..digests.len() {
            for j in (i + 1)..digests.len() {
                assert_ne!(digests[i], digests[j], "json value {i} and {j} collided");
            }
        }
        // Nesting must not flatten: {"a":{"b":1}} != {"a.b":1}
        assert_ne!(
            payload_digest(&view("t", &json!({"a": {"b": 1}}))),
            payload_digest(&view("t", &json!({"a.b": 1})))
        );
        // Array order is semantic.
        assert_ne!(
            payload_digest(&view("t", &json!({"k": [1, 2]}))),
            payload_digest(&view("t", &json!({"k": [2, 1]})))
        );
    }

    #[test]
    fn every_field_is_covered_by_the_digest() {
        // A field added to PayloadView but not fed to the hasher is invisible to
        // idempotency: a write differing only in that field would be swallowed as
        // a retry. This asserts each field individually changes the digest.
        let m = json!({});
        let base = payload_digest(&view("t", &m));
        let mut v = view("t", &m);
        v.namespace = "other";
        assert_ne!(base, payload_digest(&v), "namespace not covered");
        let mut v = view("t", &m);
        v.memory_type = "episodic";
        assert_ne!(base, payload_digest(&v), "memory_type not covered");
        let mut v = view("t", &m);
        v.importance = 0.9;
        assert_ne!(base, payload_digest(&v), "importance not covered");
        let mut v = view("t", &m);
        v.valence = 0.9;
        assert_ne!(base, payload_digest(&v), "valence not covered");
        let mut v = view("t", &m);
        v.half_life = 1.0;
        assert_ne!(base, payload_digest(&v), "half_life not covered");
        let mut v = view("t", &m);
        v.certainty = 0.1;
        assert_ne!(base, payload_digest(&v), "certainty not covered");
        let mut v = view("t", &m);
        v.domain = "other";
        assert_ne!(base, payload_digest(&v), "domain not covered");
        let mut v = view("t", &m);
        v.source = "inference";
        assert_ne!(base, payload_digest(&v), "source not covered");
        let mut v = view("t", &m);
        v.emotional_state = Some("calm");
        assert_ne!(base, payload_digest(&v), "emotional_state not covered");
        let emb = [0.5f32, 0.25];
        let mut v = view("t", &m);
        v.embedding = Some(&emb);
        assert_ne!(base, payload_digest(&v), "embedding not covered");
        assert_ne!(
            base,
            payload_digest(&view("t", &json!({"x": 1}))),
            "metadata not covered"
        );
        assert_ne!(
            base,
            payload_digest(&view("other text", &m)),
            "text not covered"
        );
    }

    #[test]
    fn optional_field_none_does_not_alias_onto_empty_string() {
        let m = json!({});
        let mut none = view("t", &m);
        none.emotional_state = None;
        let mut empty = view("t", &m);
        empty.emotional_state = Some("");
        assert_ne!(payload_digest(&none), payload_digest(&empty));
    }

    #[test]
    fn caller_supplied_embedding_is_digested_but_generated_one_is_not() {
        let mk = |emb: Vec<f32>| RecordInput {
            idempotency_key: None,
            text: "t".to_string(),
            memory_type: "semantic".to_string(),
            importance: 0.5,
            valence: 0.0,
            half_life: 604800.0,
            metadata: json!({}),
            embedding: emb,
            namespace: "default".to_string(),
            certainty: 0.8,
            domain: "general".to_string(),
            source: "user".to_string(),
            emotional_state: None,
        };
        let a = mk(vec![1.0, 0.0]);
        let b = mk(vec![0.0, 1.0]);

        // record(): the caller CHOSE these vectors. Same text, different vector =
        // different recall behavior = a different write. Treating it as a retry
        // would silently drop b's vector.
        assert_ne!(
            payload_digest(&PayloadView::from_record_input(&a, PayloadVariant::Record)),
            payload_digest(&PayloadView::from_record_input(&b, PayloadVariant::Record)),
            "caller-supplied embedding must be digested"
        );

        // record_text(): the engine GENERATED the vector from the text. An
        // embedder re-run between attempts must not look like a conflict, so it
        // is excluded and idempotency is decided before re-embedding.
        assert_eq!(
            payload_digest(&PayloadView::from_record_input(
                &a,
                PayloadVariant::RecordText
            )),
            payload_digest(&PayloadView::from_record_input(
                &b,
                PayloadVariant::RecordText
            )),
            "generated embedding must not be digested"
        );

        // An absent vector must not alias onto an empty one.
        let empty = mk(vec![]);
        assert_ne!(
            payload_digest(&PayloadView::from_record_input(
                &empty,
                PayloadVariant::Record
            )),
            payload_digest(&PayloadView::from_record_input(
                &empty,
                PayloadVariant::RecordText
            ))
        );
    }

    #[test]
    fn unicode_is_hashed_by_exact_bytes() {
        let m = json!({});
        // NFC "é" (U+00E9) vs NFD "é" (U+0065 U+0301): visually identical,
        // different bytes, therefore different payloads — see feed_str docs.
        assert_ne!(
            payload_digest(&view("caf\u{00e9}", &m)),
            payload_digest(&view("cafe\u{0301}", &m))
        );
        // Multi-byte content round-trips deterministically.
        let s = "日本語 🎉 emoji";
        assert_eq!(payload_digest(&view(s, &m)), payload_digest(&view(s, &m)));
    }
}
