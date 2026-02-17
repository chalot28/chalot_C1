// =============================================================================
// tensor/activations.rs — Activation functions
// =============================================================================

/// Sigmoid: 1 / (1 + exp(-x))
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// ReLU: max(0, x)
#[allow(dead_code)]
#[inline(always)]
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

/// SiLU (Swish): x * sigmoid(x)
/// Used extensively in modern architectures.
#[allow(dead_code)]
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// GELU (Gaussian Error Linear Unit): approximation variant.
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
}

/// SwiGLU fused: gate[i] ← silu(gate[i]) * up[i]
/// Combines gating and multiplication in-place for expert/FFN layers.
#[inline(always)]
pub fn swiglu_fused(gate: &mut [f32], up: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { swiglu_fused_avx2(gate, up); }
            return;
        }
    }
    swiglu_fused_scalar(gate, up);
}

#[inline(always)]
fn swiglu_fused_scalar(gate: &mut [f32], up: &[f32]) {
    assert_eq!(gate.len(), up.len());
    for i in 0..gate.len() {
        let g = gate[i];
        let silu_g = g / (1.0 + (-g).exp());
        gate[i] = silu_g * up[i];
    }
}

/// AVX2-accelerated SwiGLU: vectorized sigmoid + multiply.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn swiglu_fused_avx2(gate: &mut [f32], up: &[f32]) {
    use std::arch::x86_64::*;
    let n = gate.len();
    let chunks = n / 8;
    let one = _mm256_set1_ps(1.0);
    
    for c in 0..chunks {
        let base = c * 8;
        let g = _mm256_loadu_ps(gate.as_ptr().add(base));
        let u = _mm256_loadu_ps(up.as_ptr().add(base));
        
        // SiLU(g) = g / (1 + exp(-g)) via fast_exp if available, else scalar fallback
        let neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
        let exp_neg_g = fast_exp_avx2(neg_g);
        let denom = _mm256_add_ps(one, exp_neg_g);
        let silu_g = _mm256_div_ps(g, denom);
        
        // result = silu_g * up[i]
        let result = _mm256_mul_ps(silu_g, u);
        _mm256_storeu_ps(gate.as_mut_ptr().add(base), result);
    }
    
    // Scalar tail
    for i in (chunks * 8)..n {
        let g = gate[i];
        let silu_g = g / (1.0 + (-g).exp());
        gate[i] = silu_g * up[i];
    }
}

/// Fast exp approximation for AVX2 (imported from parent module for SwiGLU).
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

/// Logit soft-capping: cap × tanh(x / cap).
/// Bounds attention scores to (-cap, +cap), preventing entropy collapse
/// in deep layers. Used by Gemma 2 and modern reasoning models.
#[inline(always)]
pub fn logit_soft_cap(x: f32, cap: f32) -> f32 {
    cap * (x / cap).tanh()
}
