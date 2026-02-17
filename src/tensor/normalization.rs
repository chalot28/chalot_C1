// =============================================================================
// tensor/normalization.rs — RMSNorm and Softmax implementations
// =============================================================================

/// RMS normalization with AVX2 acceleration.
pub fn rmsnorm(out: &mut [f32], x: &[f32], weight: &[f32]) {
    let dim = x.len();
    assert_eq!(out.len(), dim);
    assert_eq!(weight.len(), dim);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { rmsnorm_avx2(out, x, weight, dim); }
            return;
        }
    }
    rmsnorm_scalar(out, x, weight, dim);
}

#[inline(always)]
fn rmsnorm_scalar(out: &mut [f32], x: &[f32], weight: &[f32], dim: usize) {
    // Use f64 for accumulation to improve stability and prevent underflow/overflow
    // in deep layers (critical for reasoning tasks).
    let ss: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / (dim as f64);
    let inv_rms = 1.0 / (ss + 1e-5).sqrt() as f32;
    for i in 0..dim {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

/// AVX2 RMS norm: vectorized sum-of-squares + vectorized normalise.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rmsnorm_avx2(out: &mut [f32], x: &[f32], weight: &[f32], dim: usize) {
    use std::arch::x86_64::*;
    let chunks = dim / 8;

    // Vectorized sum-of-squares
    let mut ss_vec = _mm256_setzero_ps();
    for c in 0..chunks {
        let v = _mm256_loadu_ps(x.as_ptr().add(c * 8));
        ss_vec = _mm256_add_ps(ss_vec, _mm256_mul_ps(v, v));
    }
    let hi = _mm256_extractf128_ps(ss_vec, 1);
    let lo = _mm256_castps256_ps128(ss_vec);
    let sum4 = _mm_add_ps(lo, hi);
    let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
    let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
    let mut ss = _mm_cvtss_f32(sum1);
    for i in (chunks * 8)..dim {
        ss += x[i] * x[i];
    }

    let inv_rms = 1.0 / (ss / dim as f32 + 1e-5f32).sqrt();
    let inv_v = _mm256_set1_ps(inv_rms);

    // Vectorized normalise: out[i] = x[i] * inv_rms * weight[i]
    for c in 0..chunks {
        let xv = _mm256_loadu_ps(x.as_ptr().add(c * 8));
        let wv = _mm256_loadu_ps(weight.as_ptr().add(c * 8));
        let r = _mm256_mul_ps(_mm256_mul_ps(xv, inv_v), wv);
        _mm256_storeu_ps(out.as_mut_ptr().add(c * 8), r);
    }
    for i in (chunks * 8)..dim {
        out[i] = x[i] * inv_rms * weight[i];
    }
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------
/// Softmax with AVX2-accelerated max-finding, fast exp, and normalisation.
pub fn softmax(x: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { softmax_avx2(x); }
            return;
        }
    }
    softmax_scalar(x);
}

#[inline(always)]
fn softmax_scalar(x: &mut [f32]) {
    // Safety: handle NaNs by defaulting to a safe minimum
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b));
    if max == f32::NEG_INFINITY {
        return; // All values are likely NaN or -Inf
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// AVX2 softmax: vectorized max + fast exp + vectorized normalise.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn softmax_avx2(x: &mut [f32]) {
    use std::arch::x86_64::*;
    let n = x.len();
    if n == 0 { return; }
    let chunks = n / 8;

    // Pass 1: find max (SIMD)
    let mut max_v = _mm256_set1_ps(f32::NEG_INFINITY);
    for c in 0..chunks {
        let v = _mm256_loadu_ps(x.as_ptr().add(c * 8));
        max_v = _mm256_max_ps(max_v, v);
    }
    let hi = _mm256_extractf128_ps(max_v, 1);
    let lo = _mm256_castps256_ps128(max_v);
    let m4 = _mm_max_ps(lo, hi);
    let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    let m1 = _mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1));
    let mut max_val = _mm_cvtss_f32(m1);
    for i in (chunks * 8)..n {
        max_val = max_val.max(x[i]);
    }
    let max_broadcast = _mm256_set1_ps(max_val);

    // Pass 2: exp(x - max) and accumulate sum (SIMD fast exp)
    let mut sum_v = _mm256_setzero_ps();
    for c in 0..chunks {
        let v = _mm256_loadu_ps(x.as_ptr().add(c * 8));
        let shifted = _mm256_sub_ps(v, max_broadcast);
        let exp_v = fast_exp_avx2(shifted);
        _mm256_storeu_ps(x.as_mut_ptr().add(c * 8), exp_v);
        sum_v = _mm256_add_ps(sum_v, exp_v);
    }
    let hs = _mm256_extractf128_ps(sum_v, 1);
    let ls = _mm256_castps256_ps128(sum_v);
    let s4 = _mm_add_ps(ls, hs);
    let s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    let s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    let mut sum = _mm_cvtss_f32(s1);
    for i in (chunks * 8)..n {
        let v = (x[i] - max_val).exp();
        x[i] = v;
        sum += v;
    }

    // Pass 3: normalise (SIMD)
    let inv_v = _mm256_set1_ps(1.0 / sum);
    for c in 0..chunks {
        let v = _mm256_loadu_ps(x.as_ptr().add(c * 8));
        _mm256_storeu_ps(x.as_mut_ptr().add(c * 8), _mm256_mul_ps(v, inv_v));
    }
    let inv_s = 1.0 / sum;
    for i in (chunks * 8)..n {
        x[i] *= inv_s;
    }
}

/// Fast exp approximation for AVX2 (used by softmax).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fast_exp_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    
    // Constants for polynomial approximation of exp(x)
    // exp(x) ≈ 2^n · exp(r) where x = n·ln(2) + r, |r| < ln(2)/2
    let ln2 = _mm256_set1_ps(0.693147180559945309);
    let inv_ln2 = _mm256_set1_ps(1.442695040888963407); // 1/ln(2)
    
    // Polynomial coefficients (5th order Remez approximation)
    let c0 = _mm256_set1_ps(1.0);
    let c1 = _mm256_set1_ps(1.0);
    let c2 = _mm256_set1_ps(0.5);
    let c3 = _mm256_set1_ps(0.166666666666);
    let c4 = _mm256_set1_ps(0.041666666666);
    let c5 = _mm256_set1_ps(0.008333333333);
    
    // Decompose: x = n·ln(2) + r
    let n = _mm256_floor_ps(_mm256_mul_ps(x, inv_ln2));
    let r = _mm256_sub_ps(x, _mm256_mul_ps(n, ln2));
    
    // exp(r) ≈ Horner: ((((c5·r + c4)·r + c3)·r + c2)·r + c1)·r + c0
    let mut poly = c5;
    poly = _mm256_add_ps(_mm256_mul_ps(poly, r), c4);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, r), c3);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, r), c2);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, r), c1);
    poly = _mm256_add_ps(_mm256_mul_ps(poly, r), c0);
    
    // 2^n via IEEE-754 exponent bit manipulation
    let ni = _mm256_cvtps_epi32(n);
    let pow2n = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23),
    );
    
    _mm256_mul_ps(poly, pow2n)
}

/// Compute softmax then return indices + renormalised weights of top-k entries.
pub fn softmax_top_k(scores: &mut [f32], k: usize) -> Vec<(usize, f32)> {
    softmax(scores);
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    indexed.truncate(k);
    // Re-normalise selected weights
    let sum: f32 = indexed.iter().map(|(_, w)| w).sum();
    if sum > 1e-12 {
        for item in &mut indexed {
            item.1 /= sum;
        }
    }
    indexed
}
