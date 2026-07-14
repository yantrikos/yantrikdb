//! Test seams (v0.10 Phase 0): failpoints and the injectable clock.
//!
//! Everything here is gated behind the NON-DEFAULT `testing` cargo feature.
//! Production builds compile the hooks to inlined no-ops — there is no
//! runtime registry, no mutable clock, and no environment sniffing unless
//! the feature is explicitly enabled (sol Q2/Q3 converged design: never
//! ship a hidden mutable production clock; failpoints are hand-rolled
//! because we only need the park-and-handshake pattern, not the `fail`
//! crate's dependency surface).
//!
//! ## Failpoints
//! A named call site — e.g. `fail_point("link.between_row_and_oplog")` —
//! checks a process-global registry. If armed with `Park`, it prints a
//! `FAILPOINT:<name>` handshake line to stdout (flushed), then parks the
//! thread forever; the parent test process waits for the handshake with a
//! bounded timeout and kills the child. This is what makes kill-mid-
//! transaction tests deterministic instead of scheduler roulette.
//! Registry state is process-global: kill-harness tests run the parked
//! side in a CHILD process (see the `YANTRIKDB_FAILPOINTS` env format),
//! and in-process tests must clear state via [`clear_fail_points`] /
//! the RAII [`FailPointGuard`].
//!
//! Env format (parsed once, first use): `YANTRIKDB_FAILPOINTS=name1,name2`
//! — every named point parks.
//!
//! ## Injectable clock
//! [`set_test_clock`] / [`advance_test_clock`] override the SEMANTIC
//! wall-clock (`crate::time::now_secs` and everything derived from it:
//! decay, aged-status, rate-limiter windows). Monotonic `Instant`-based
//! deadlines and performance timers are intentionally NOT overridable.
//! Use [`TestClockGuard`] so a panicking test cannot leak a frozen clock
//! into the next test; clock-mutating tests must be serialized or run in
//! child processes.

#[cfg(feature = "testing")]
mod enabled {
    use std::collections::HashSet;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};

    static FAIL_POINTS: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    /// f64 bits of the override; 0 = no override (0.0 secs is not a valid
    /// semantic wall-clock time for this engine, so the sentinel is safe).
    static TEST_CLOCK_BITS: AtomicU64 = AtomicU64::new(0);

    fn registry() -> &'static Mutex<HashSet<String>> {
        FAIL_POINTS.get_or_init(|| {
            let mut set = HashSet::new();
            if let Ok(env) = std::env::var("YANTRIKDB_FAILPOINTS") {
                for name in env.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    set.insert(name.to_string());
                }
            }
            Mutex::new(set)
        })
    }

    pub fn fail_point(name: &str) {
        let armed = registry().lock().map(|s| s.contains(name)).unwrap_or(false);
        if armed {
            // Handshake: the parent waits for this exact line, then kills us.
            println!("FAILPOINT:{name}");
            let _ = std::io::stdout().flush();
            loop {
                std::thread::park();
            }
        }
    }

    pub fn set_fail_point(name: &str) {
        registry().lock().unwrap().insert(name.to_string());
    }

    pub fn clear_fail_points() {
        registry().lock().unwrap().clear();
    }

    /// RAII: arms a failpoint, clears ALL failpoints on drop (panic-safe).
    pub struct FailPointGuard;
    impl FailPointGuard {
        pub fn arm(name: &str) -> Self {
            set_fail_point(name);
            FailPointGuard
        }
    }
    impl Drop for FailPointGuard {
        fn drop(&mut self) {
            clear_fail_points();
        }
    }

    pub fn set_test_clock(secs: f64) {
        TEST_CLOCK_BITS.store(secs.to_bits(), Ordering::SeqCst);
    }

    pub fn advance_test_clock(delta_secs: f64) {
        let cur = test_clock();
        let base = cur.unwrap_or_else(crate::time::real_now_secs);
        set_test_clock(base + delta_secs);
    }

    pub fn reset_test_clock() {
        TEST_CLOCK_BITS.store(0, Ordering::SeqCst);
    }

    pub fn test_clock() -> Option<f64> {
        let bits = TEST_CLOCK_BITS.load(Ordering::SeqCst);
        if bits == 0 {
            None
        } else {
            Some(f64::from_bits(bits))
        }
    }

    /// RAII: sets the clock, resets it on drop (panic-safe).
    pub struct TestClockGuard;
    impl TestClockGuard {
        pub fn set(secs: f64) -> Self {
            set_test_clock(secs);
            TestClockGuard
        }
    }
    impl Drop for TestClockGuard {
        fn drop(&mut self) {
            reset_test_clock();
        }
    }
}

#[cfg(feature = "testing")]
pub use enabled::*;

/// No-op in production builds: inlined away entirely.
#[cfg(not(feature = "testing"))]
#[inline(always)]
pub fn fail_point(_name: &str) {}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;

    #[test]
    fn clock_override_roundtrip_and_guard_reset() {
        {
            let _g = TestClockGuard::set(1_000_000.5);
            assert_eq!(test_clock(), Some(1_000_000.5));
            advance_test_clock(10.0);
            assert_eq!(test_clock(), Some(1_000_010.5));
        }
        assert_eq!(test_clock(), None, "guard resets on drop");
    }

    #[test]
    fn unarmed_fail_point_is_a_no_op() {
        clear_fail_points();
        fail_point("not.armed"); // must return, not park
    }
}
