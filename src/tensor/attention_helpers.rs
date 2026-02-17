// =============================================================================
// tensor/attention_helpers.rs — Helpers for attention mechanisms
// =============================================================================

/// AVX2-accelerated f32 dot product for attention score computation.
/// Processes 16 floats per iteration with dual 256-bit accumulators
/// to hide pipeline latency and maximise throughput.
pub fn dot_f32(a: &[f32], b: &[f32], len: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { dot_f32_avx2(a, b, len) };
        }
    }
    dot_f32_scalar(a, b, len)
}

#[inline(always)]
fn dot_f32_scalar(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;
    let chunks = len / 4;
    for c in 0..chunks {
        let i = c * 4;
        s0 += a[i]     * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for i in (chunks * 4)..len {
        s0 += a[i] * b[i];
    }
    (s0 + s1) + (s2 + s3)
}

/// AVX2 dot product: 16 floats/iteration with dual 256-bit accumulators.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_f32_avx2(a: &[f32], b: &[f32], len: usize) -> f32 {
    use std::arch::x86_64::*;
    let chunks16 = len / 16;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    for c in 0..chunks16 {
        let base = c * 16;
        let va0 = _mm256_loadu_ps(a.as_ptr().add(base));
        let vb0 = _mm256_loadu_ps(b.as_ptr().add(base));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va0, vb0));
        let va1 = _mm256_loadu_ps(a.as_ptr().add(base + 8));
        let vb1 = _mm256_loadu_ps(b.as_ptr().add(base + 8));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(va1, vb1));
    }

    // Handle remaining 8-float chunk
    let rem_start = chunks16 * 16;
    if rem_start + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(rem_start));
        let vb = _mm256_loadu_ps(b.as_ptr().add(rem_start));
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(va, vb));
    }

    // Combine dual accumulators + horizontal sum
    let combined = _mm256_add_ps(acc0, acc1);
    let hi = _mm256_extractf128_ps(combined, 1);
    let lo = _mm256_castps256_ps128(combined);
    let sum4 = _mm_add_ps(lo, hi);
    let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
    let mut result = _mm_cvtss_f32(sum1);

    // Scalar tail
    let done = if rem_start + 8 <= len { rem_start + 8 } else { rem_start };
    for i in done..len {
        result += a[i] * b[i];
    }
    result
}
