//! SIMD-accelerated vector kernels with runtime CPU dispatch (v0.9.3).
//!
//! The recall hot path is dominated by f32→f64 dot products (dim ≈ 64–512,
//! hundreds of comparisons per query at ef_search=200). The scalar version
//! is a strict-FP serial dependency chain — LLVM cannot auto-vectorize an
//! f64 reduction because FP addition is not associative — so an explicit
//! kernel with independent lane accumulators is worth 4–8× on this loop.
//!
//! Dispatch policy ("detect and auto-use", CPU edition):
//! - x86_64: runtime `is_x86_feature_detected!` for AVX2+FMA (std caches the
//!   CPUID probe in an atomic, so the check is a relaxed load after the
//!   first call). Wheels are built for baseline x86-64 (SSE2), so compile-
//!   time `target_feature` can't be assumed — runtime dispatch is what lets
//!   one published wheel use the machine it lands on.
//! - aarch64 and everything else: the unrolled scalar path (4 independent
//!   f64 accumulators — same instruction-level parallelism idea, letting
//!   the CPU overlap adds without violating FP semantics).
//!
//! Numerical note: lane-split accumulation changes the SUMMATION ORDER
//! relative to the historical strictly-sequential sum, so results can
//! differ from it in the last ulps (~1e-15 relative). Cosine scores are
//! consumed as rankings with far coarser tolerances; the unit test below
//! pins the kernels to the sequential reference within 1e-9.

/// Dot product of two f32 slices with f64 accumulation, best available
/// kernel for this CPU. Slices must be equal length (callers validate
/// dimensions at the API boundary; mismatch here would read short).
#[inline]
pub fn dot_f64(a: &[f32], b: &[f32]) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
        {
            // SAFETY: feature presence just verified at runtime.
            return unsafe { dot_f64_avx2_fma(a, b) };
        }
    }
    dot_f64_unrolled(a, b)
}

/// Portable unrolled kernel: 4 independent f64 accumulators break the
/// serial FP dependency chain (ILP), then one deterministic combine.
#[inline]
fn dot_f64_unrolled(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let chunks = n / 4;
    let (mut s0, mut s1, mut s2, mut s3) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..chunks {
        let base = i * 4;
        s0 += a[base] as f64 * b[base] as f64;
        s1 += a[base + 1] as f64 * b[base + 1] as f64;
        s2 += a[base + 2] as f64 * b[base + 2] as f64;
        s3 += a[base + 3] as f64 * b[base + 3] as f64;
    }
    let mut tail = 0.0f64;
    for i in (chunks * 4)..n {
        tail += a[i] as f64 * b[i] as f64;
    }
    (s0 + s1) + (s2 + s3) + tail
}

/// AVX2+FMA kernel: 2×4-lane f64 accumulators (8 f32 elements per
/// iteration via `vcvtps2pd` widening), FMA contraction, horizontal sum
/// at the end. f64 accumulation preserves the precision contract of the
/// historical scalar implementation.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available (runtime-detected in
/// [`dot_f64`]).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f64_avx2_fma(a: &[f32], b: &[f32]) -> f64 {
    use std::arch::x86_64::*;

    let n = a.len().min(b.len());
    let mut acc0 = _mm256_setzero_pd();
    let mut acc1 = _mm256_setzero_pd();

    let chunks = n / 8;
    let ap = a.as_ptr();
    let bp = b.as_ptr();
    for i in 0..chunks {
        let base = i * 8;
        // Widen 4 f32 -> 4 f64 per 128-bit load; two loads cover 8 elements.
        let a_lo = _mm256_cvtps_pd(_mm_loadu_ps(ap.add(base)));
        let b_lo = _mm256_cvtps_pd(_mm_loadu_ps(bp.add(base)));
        acc0 = _mm256_fmadd_pd(a_lo, b_lo, acc0);
        let a_hi = _mm256_cvtps_pd(_mm_loadu_ps(ap.add(base + 4)));
        let b_hi = _mm256_cvtps_pd(_mm_loadu_ps(bp.add(base + 4)));
        acc1 = _mm256_fmadd_pd(a_hi, b_hi, acc1);
    }

    // Horizontal sum of both accumulators.
    let acc = _mm256_add_pd(acc0, acc1);
    let lo = _mm256_castpd256_pd128(acc);
    let hi = _mm256_extractf128_pd(acc, 1);
    let sum2 = _mm_add_pd(lo, hi);
    let sum1 = _mm_add_sd(sum2, _mm_unpackhi_pd(sum2, sum2));
    let mut total = _mm_cvtsd_f64(sum1);

    // Scalar tail (< 8 elements).
    for i in (chunks * 8)..n {
        total += *a.get_unchecked(i) as f64 * *b.get_unchecked(i) as f64;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Strictly-sequential reference (the historical implementation).
    fn dot_reference(a: &[f32], b: &[f32]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum()
    }

    fn test_vec(seed: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((seed + i as f32) * 0.7311 + (i as f32) * 0.311).sin())
            .collect()
    }

    #[test]
    fn kernels_match_sequential_reference_across_dims_and_tails() {
        // Cover the dims the engine actually ships (64/256/384/512) plus
        // every tail-length class (n % 8 in 0..8) and tiny inputs.
        for n in [0, 1, 3, 5, 7, 8, 9, 15, 16, 63, 64, 100, 256, 384, 385, 512] {
            let a = test_vec(1.0, n);
            let b = test_vec(9.0, n);
            let reference = dot_reference(&a, &b);
            let dispatched = dot_f64(&a, &b);
            let unrolled = dot_f64_unrolled(&a, &b);
            assert!(
                (dispatched - reference).abs() <= 1e-9 * (1.0 + reference.abs()),
                "dispatched kernel diverged at n={n}: {dispatched} vs {reference}"
            );
            assert!(
                (unrolled - reference).abs() <= 1e-9 * (1.0 + reference.abs()),
                "unrolled kernel diverged at n={n}: {unrolled} vs {reference}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_kernel_matches_reference_when_available() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma"))
        {
            eprintln!("skipping: avx2+fma not available on this CPU");
            return;
        }
        for n in [7, 8, 64, 384, 385] {
            let a = test_vec(2.0, n);
            let b = test_vec(5.0, n);
            let reference = dot_reference(&a, &b);
            let simd = unsafe { dot_f64_avx2_fma(&a, &b) };
            assert!(
                (simd - reference).abs() <= 1e-9 * (1.0 + reference.abs()),
                "avx2 kernel diverged at n={n}: {simd} vs {reference}"
            );
        }
    }
}
