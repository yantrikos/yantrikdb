/// Platform-aware time utilities.
/// On native targets, uses `std::time::SystemTime` / `Instant`.
/// On wasm32, uses `js_sys::Date::now()`.
///
/// **v0.10 Phase 0:** `now_secs`/`now_ms` are the engine's SEMANTIC
/// wall-clock seam. Under the non-default `testing` cargo feature they
/// consult the injectable test clock (`crate::testing::set_test_clock`)
/// so decay / aged-status / rate-limiter traces can script time
/// deterministically. Production builds compile straight to the system
/// clock. The monotonic [`Instant`] below is for deadlines/timing and is
/// intentionally NOT overridable.

/// The real system clock, never overridden (used by the test-clock's own
/// `advance` bootstrap; not for engine semantics).
#[cfg(not(target_arch = "wasm32"))]
pub fn real_now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

#[cfg(target_arch = "wasm32")]
pub fn real_now_secs() -> f64 {
    js_sys::Date::now() / 1000.0
}

pub fn now_secs() -> f64 {
    #[cfg(feature = "testing")]
    if let Some(t) = crate::testing::test_clock() {
        return t;
    }
    real_now_secs()
}

/// Milliseconds since epoch — derives from the SAME override as
/// [`now_secs`] (sol Q2 containment: the two must never disagree).
pub fn now_ms() -> u64 {
    (now_secs() * 1000.0) as u64
}

/// Monotonic-ish instant for measuring elapsed time.
#[cfg(not(target_arch = "wasm32"))]
pub struct Instant(std::time::Instant);

#[cfg(not(target_arch = "wasm32"))]
impl Instant {
    pub fn now() -> Self {
        Instant(std::time::Instant::now())
    }
    pub fn elapsed_ms(&self) -> u64 {
        self.0.elapsed().as_millis() as u64
    }
}

#[cfg(target_arch = "wasm32")]
pub struct Instant(f64);

#[cfg(target_arch = "wasm32")]
impl Instant {
    pub fn now() -> Self {
        Instant(js_sys::Date::now())
    }
    pub fn elapsed_ms(&self) -> u64 {
        (js_sys::Date::now() - self.0).max(0.0) as u64
    }
}
