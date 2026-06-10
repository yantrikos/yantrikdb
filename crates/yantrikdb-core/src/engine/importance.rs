//! Write-time importance calibration.
//!
//! ## The problem
//!
//! Importance is a [0, 1] ranking signal, but in practice writers (agents)
//! mark almost everything `1.0`. The 2026-06-10 audit found the recent corpus
//! saturated at the top of the range, which makes importance dead as a
//! discriminator: in the recall score importance appears both in the decay
//! term and in a multiplicative gate, and when every memory is `1.0` that
//! gate is a constant everywhere.
//!
//! ## What calibration does (and what it deliberately does not)
//!
//! Write-time calibration alone *cannot* spread a cluster of identical `1.0`
//! inputs — there is no signal to tell them apart. That differentiation comes
//! from usage feedback (task 32). What this module does is **deflation**:
//! when a namespace shows sustained saturation, incoming high marks are
//! compressed toward a saturation-dependent ceiling, so the scale regains
//! headroom and `1.0` becomes rare and meaningful again ("you've marked
//! everything critical, so critical now means 0.8; a true 1.0 must stand out
//! by not arriving in a saturated stream").
//!
//! ## Properties
//!
//! - **Identity until saturated.** A fresh or low-volume namespace, or any
//!   value outside the high band, passes through unchanged. This is why every
//!   existing exact-importance test (which writes a handful of memories to a
//!   fresh namespace) keeps passing.
//! - **Monotonic.** Higher raw importance never maps to a lower calibrated
//!   value, so the writer's relative ordering is preserved.
//! - **O(1) per write.** Backed by a per-namespace EWMA in a tiny table
//!   (`namespace_importance_stats`), a point-read + upsert keyed on the
//!   namespace primary key. No per-write scan of the corpus.
//! - **Replication-safe.** Calibration runs only at the ingest entry points
//!   (`record`, `record_text`, `record_batch`), never on the replication
//!   apply path (`record_with_rid`), so a follower stores exactly the
//!   calibrated value the leader computed.

use rusqlite::{params, OptionalExtension};

use crate::error::Result;

use super::{now, YantrikDB};

/// EWMA smoothing factor for the per-namespace importance mean.
const EWMA_ALPHA: f64 = 0.15;
/// Minimum number of prior writes in a namespace before calibration engages.
/// Below this there isn't enough signal to call a namespace "saturated".
const MIN_COUNT: u64 = 8;
/// Namespace mean importance above which the namespace is "saturated".
const SATURATION_THRESHOLD: f64 = 0.80;
/// Only raw values above this high-band floor are ever compressed.
const HIGH_FLOOR: f64 = 0.70;
/// The lowest the high-importance ceiling can fall to, at full saturation.
const MIN_CEILING: f64 = 0.75;

/// Pure calibration transform: map a raw importance to a calibrated one given
/// the namespace's running mean (`ewma`) and write `count`.
///
/// Returns `raw` unchanged unless the namespace is saturated *and* the value
/// is in the high band, in which case it is compressed monotonically into
/// `[HIGH_FLOOR, ceiling]` where `ceiling` falls from `1.0` toward
/// `MIN_CEILING` as saturation deepens.
pub(crate) fn calibrate_importance_value(raw: f64, ewma: f64, count: u64) -> f64 {
    if count < MIN_COUNT || ewma <= SATURATION_THRESHOLD || raw <= HIGH_FLOOR {
        return raw;
    }
    // How far the namespace mean sits above the saturation threshold, in [0, 1].
    let sat = ((ewma - SATURATION_THRESHOLD) / (1.0 - SATURATION_THRESHOLD)).clamp(0.0, 1.0);
    // Ceiling for high importance: 1.0 when just-saturated, MIN_CEILING when
    // fully saturated. Always >= HIGH_FLOOR, so the mapping stays monotonic.
    let ceiling = 1.0 - sat * (1.0 - MIN_CEILING);
    // Position of raw within the high band [HIGH_FLOOR, 1.0], in [0, 1].
    let frac = (raw - HIGH_FLOOR) / (1.0 - HIGH_FLOOR);
    (HIGH_FLOOR + frac * (ceiling - HIGH_FLOOR)).clamp(0.0, 1.0)
}

impl YantrikDB {
    /// Calibrate a raw importance against the writing namespace's running
    /// distribution, updating that distribution with the raw value. See the
    /// module docs for semantics. Called once per write at the ingest entry
    /// points.
    pub(crate) fn calibrate_importance(&self, namespace: &str, raw: f64) -> Result<f64> {
        // Key the stats by the normalized namespace so the "" / "default"
        // aliasing can't split a namespace's distribution across two rows.
        let namespace = super::record::normalize_namespace(namespace);

        let conn = self.conn();
        let existing: Option<(f64, i64)> = conn
            .query_row(
                "SELECT ewma, count FROM namespace_importance_stats WHERE namespace = ?1",
                params![namespace],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        let (ewma, count) = existing.unwrap_or((raw, 0));

        let calibrated = calibrate_importance_value(raw, ewma, count as u64);

        // Advance the EWMA toward the RAW value — we track what writers ask
        // for (their intent), not our deflated output, so saturation is
        // measured honestly.
        let new_ewma = if count == 0 {
            raw
        } else {
            (1.0 - EWMA_ALPHA) * ewma + EWMA_ALPHA * raw
        };
        conn.execute(
            "INSERT INTO namespace_importance_stats (namespace, ewma, count, updated_at) \
             VALUES (?1, ?2, 1, ?3) \
             ON CONFLICT(namespace) DO UPDATE SET ewma = ?2, count = count + 1, updated_at = ?3",
            params![namespace, new_ewma, now()],
        )?;

        if (calibrated - raw).abs() > f64::EPSILON {
            tracing::debug!(
                target: "yantrikdb::audit::importance",
                namespace,
                raw,
                calibrated,
                ewma = new_ewma,
                count = count + 1,
                "deflated saturated importance",
            );
        }
        Ok(calibrated)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Task 32 — usage-corrected importance (use-it-or-lose-it)
//
// Calibration (above) sets the importance *prior* at write time. Over time
// that prior should be corrected by usage: a memory written at high
// importance that is never retrieved is probably not actually that important,
// so its stored importance should revert toward a baseline. This runs as a
// background maintenance pass (driven by the sleep cycle, task 24, or invoked
// directly), NOT on the recall hot path — so it changes the durable prior
// that recall already reads, with no per-query cost and no ranking-formula
// surgery.
//
// It deliberately does only the half that is observable without an external
// signal: deflating *unused* high marks. "Retrieved and acted on" requires
// downstream feedback (the explicit `recall` feedback loop) which the engine
// cannot synthesize on its own without circularity.
// ─────────────────────────────────────────────────────────────────────────

/// Importance that unused high marks revert toward.
const REVERSION_BASELINE: f64 = 0.5;
/// Seconds since last access at which reversion reaches full strength (90d).
const REVERSION_TENURE_SECS: f64 = 90.0 * 86_400.0;
/// Strongest reversion: an untouched, never-accessed high mark at/after full
/// tenure is pulled this fraction of the way to baseline.
const MAX_REVERSION: f64 = 0.6;
/// Cap on rids echoed back in a recalibration report.
const SAMPLE_CAP: usize = 50;

/// Pure usage correction: given a memory's prior importance, lifetime access
/// count, and seconds since it was last accessed, return the importance it
/// should revert to.
///
/// Properties:
/// - **Only deflates.** Marks at or below baseline are returned unchanged —
///   we never inflate a low-importance memory just because it sits unused.
/// - **Identity when fresh.** Zero time since access ⇒ returns the prior, so
///   a just-written or just-recalled memory is untouched.
/// - **Access-resistant.** More lifetime accesses slow reversion (diminishing
///   returns via `ln`), so a frequently-used memory keeps its importance.
/// - **Bounded & monotonic** in staleness.
/// - **Idempotent.** The result reverts toward a staleness-anchored *target*
///   via `min`, so re-running the pass at the same staleness is a no-op — the
///   correction does not compound across repeated maintenance cycles. It also
///   never inflates: the value can only move down toward the target, never up.
pub(crate) fn usage_corrected_importance(
    prior: f64,
    access_count: i64,
    secs_since_access: f64,
) -> f64 {
    if prior <= REVERSION_BASELINE {
        return prior;
    }
    let staleness = (secs_since_access / REVERSION_TENURE_SECS).clamp(0.0, 1.0);
    // Each access adds resistance (diminishing); never-accessed ⇒ resistance 1.
    let resistance = 1.0 + (access_count.max(0) as f64).ln_1p();
    let reversion = (MAX_REVERSION * staleness / resistance).clamp(0.0, MAX_REVERSION);
    // Target importance for this staleness, anchored to the top of the range
    // (1.0) so the target is a fixed function of (staleness, access) — not of
    // the current value. Taking the min makes the pass idempotent and
    // non-inflating: a memory only ever settles toward its staleness target.
    let target = REVERSION_BASELINE + (1.0 - REVERSION_BASELINE) * (1.0 - reversion);
    prior.min(target)
}

/// Outcome of a [`YantrikDB::recalibrate_unused_importance`] pass.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct ImportanceRecalibrationReport {
    pub dry_run: bool,
    /// Active memories above baseline that were examined.
    pub scanned: usize,
    /// Memories whose importance was (or would be) reverted downward.
    pub adjusted: usize,
    /// Sum of the downward importance drift across adjusted memories.
    pub total_drift: f64,
    /// Sample of adjusted rids for operator spot-checking.
    pub sample_rids: Vec<String>,
}

impl YantrikDB {
    /// Revert stale, never-/rarely-accessed, high-importance memories toward
    /// the baseline (use-it-or-lose-it). Background maintenance, not a recall
    /// hot-path operation. Run with `dry_run = true` to preview.
    ///
    /// Updates both the durable `memories.importance` column and the scoring
    /// cache so recall sees the corrected prior immediately.
    pub fn recalibrate_unused_importance(
        &self,
        dry_run: bool,
    ) -> Result<ImportanceRecalibrationReport> {
        let mut report = ImportanceRecalibrationReport {
            dry_run,
            ..Default::default()
        };
        let now_ts = now();

        // Scan only candidates that can possibly change: active + above baseline.
        let rows: Vec<(String, f64, i64, f64)> = {
            let conn = self.conn();
            let mut stmt = conn.prepare(
                "SELECT rid, importance, access_count, last_access FROM memories \
                 WHERE consolidation_status = 'active' AND importance > ?1",
            )?;
            let rows = stmt
                .query_map(params![REVERSION_BASELINE], |r| {
                    Ok((
                        r.get::<_, String>(0)?,
                        r.get::<_, f64>(1)?,
                        r.get::<_, i64>(2)?,
                        r.get::<_, f64>(3)?,
                    ))
                })?
                .collect::<std::result::Result<Vec<_>, _>>()?;
            rows
        };
        report.scanned = rows.len();

        let mut pending: Vec<(String, f64)> = Vec::new();
        for (rid, importance, access_count, last_access) in rows {
            let secs = (now_ts - last_access).max(0.0);
            let corrected = usage_corrected_importance(importance, access_count, secs);
            if importance - corrected > 1e-6 {
                report.adjusted += 1;
                report.total_drift += importance - corrected;
                if report.sample_rids.len() < SAMPLE_CAP {
                    report.sample_rids.push(rid.clone());
                }
                pending.push((rid, corrected));
            }
        }

        if dry_run || pending.is_empty() {
            return Ok(report);
        }

        // Apply durably in one transaction.
        {
            let conn = self.conn();
            conn.execute_batch("SAVEPOINT importance_recal")?;
            let apply: Result<()> = (|| {
                for (rid, corrected) in &pending {
                    conn.execute(
                        "UPDATE memories SET importance = ?1 WHERE rid = ?2",
                        params![corrected, rid],
                    )?;
                }
                Ok(())
            })();
            match apply {
                Ok(()) => conn.execute_batch("RELEASE importance_recal")?,
                Err(e) => {
                    let _ = conn
                        .execute_batch("ROLLBACK TO importance_recal; RELEASE importance_recal");
                    return Err(e);
                }
            }
        }

        // Keep the scoring cache in step so recall reflects the new prior at
        // once rather than waiting for eviction.
        {
            let mut cache = self.scoring_cache.write();
            for (rid, corrected) in &pending {
                if let Some(row) = cache.get_mut(rid) {
                    row.importance = *corrected;
                }
            }
        }

        tracing::info!(
            target: "yantrikdb::audit::importance",
            scanned = report.scanned,
            adjusted = report.adjusted,
            total_drift = report.total_drift,
            "unused-importance recalibration complete",
        );

        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_below_min_count() {
        // Not enough writes yet — pass through even at max importance.
        assert_eq!(calibrate_importance_value(1.0, 1.0, MIN_COUNT - 1), 1.0);
    }

    #[test]
    fn identity_when_not_saturated() {
        // Plenty of writes, but the mean is moderate — no deflation.
        assert_eq!(calibrate_importance_value(1.0, 0.5, 100), 1.0);
        // Exactly at the threshold is treated as not-yet-saturated.
        assert_eq!(
            calibrate_importance_value(1.0, SATURATION_THRESHOLD, 100),
            1.0
        );
    }

    #[test]
    fn identity_for_low_band_values() {
        // Even in a saturated namespace, modest importances are untouched.
        assert_eq!(calibrate_importance_value(HIGH_FLOOR, 1.0, 100), HIGH_FLOOR);
        assert_eq!(calibrate_importance_value(0.5, 1.0, 100), 0.5);
    }

    #[test]
    fn deflates_high_values_when_saturated() {
        let c = calibrate_importance_value(1.0, 1.0, 100);
        assert!(c < 1.0, "max importance deflates under full saturation: {c}");
        assert!(c >= MIN_CEILING, "but never below the floor ceiling: {c}");
    }

    #[test]
    fn deflation_is_monotonic() {
        // Under saturation, higher raw still maps to higher calibrated.
        let lo = calibrate_importance_value(0.75, 0.95, 100);
        let mid = calibrate_importance_value(0.90, 0.95, 100);
        let hi = calibrate_importance_value(1.00, 0.95, 100);
        assert!(lo < mid, "lo {lo} < mid {mid}");
        assert!(mid < hi, "mid {mid} < hi {hi}");
        assert!(hi < 1.0, "even the top is deflated: {hi}");
    }

    #[test]
    fn deeper_saturation_deflates_harder() {
        // A more saturated namespace pushes the same raw value lower.
        let mild = calibrate_importance_value(1.0, 0.85, 100);
        let severe = calibrate_importance_value(1.0, 1.0, 100);
        assert!(severe < mild, "severe {severe} < mild {mild}");
    }

    #[test]
    fn result_stays_in_unit_interval() {
        for &ewma in &[0.0, 0.5, 0.81, 0.9, 1.0] {
            for &raw in &[0.0, 0.5, 0.7, 0.85, 1.0] {
                let c = calibrate_importance_value(raw, ewma, 100);
                assert!((0.0..=1.0).contains(&c), "raw={raw} ewma={ewma} -> {c}");
            }
        }
    }

    // ── Task 32: usage correction ──

    #[test]
    fn usage_identity_when_fresh() {
        // Just accessed ⇒ no reversion, regardless of how high the prior is.
        assert_eq!(usage_corrected_importance(1.0, 0, 0.0), 1.0);
        assert_eq!(usage_corrected_importance(0.9, 5, 0.0), 0.9);
    }

    #[test]
    fn usage_never_inflates_low_marks() {
        // At or below baseline, stale or not, the value is untouched.
        let ancient = REVERSION_TENURE_SECS * 4.0;
        assert_eq!(usage_corrected_importance(0.5, 0, ancient), 0.5);
        assert_eq!(usage_corrected_importance(0.3, 0, ancient), 0.3);
    }

    #[test]
    fn usage_reverts_unused_high_importance() {
        let c = usage_corrected_importance(1.0, 0, REVERSION_TENURE_SECS);
        assert!(c < 1.0, "an unused high mark deflates: {c}");
        assert!(c >= REVERSION_BASELINE, "but never below baseline: {c}");
    }

    #[test]
    fn usage_access_resists_reversion() {
        let unused = usage_corrected_importance(1.0, 0, REVERSION_TENURE_SECS);
        let used = usage_corrected_importance(1.0, 50, REVERSION_TENURE_SECS);
        assert!(used > unused, "frequent access slows reversion: {used} > {unused}");
    }

    #[test]
    fn usage_monotonic_in_staleness() {
        let prior = 1.0;
        let young = usage_corrected_importance(prior, 0, REVERSION_TENURE_SECS * 0.25);
        let old = usage_corrected_importance(prior, 0, REVERSION_TENURE_SECS * 0.75);
        assert!(old < young, "more staleness ⇒ more reversion: {old} < {young}");
        assert!(young <= prior && old >= REVERSION_BASELINE);
    }
}
