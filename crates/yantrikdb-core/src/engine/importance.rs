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
}
