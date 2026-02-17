// =============================================================================
// tensor/rope.rs — Rotary Position Embedding
// =============================================================================

/// Apply RoPE in-place to q and k vectors for a given position `pos`.
/// `head_dim` is dim / n_heads.
#[inline(always)]
#[allow(dead_code)]
pub fn apply_rope(q: &mut [f32], k: &mut [f32], head_dim: usize, pos: usize) {
    let half = head_dim / 2;
    for i in 0..half {
        let freq = 1.0 / 10000.0f32.powf((2 * i) as f32 / head_dim as f32);
        let theta = pos as f32 * freq;
        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Apply to q
        let q0 = q[2 * i];
        let q1 = q[2 * i + 1];
        q[2 * i] = q0 * cos_t - q1 * sin_t;
        q[2 * i + 1] = q0 * sin_t + q1 * cos_t;

        // Apply to k
        let k0 = k[2 * i];
        let k1 = k[2 * i + 1];
        k[2 * i] = k0 * cos_t - k1 * sin_t;
        k[2 * i + 1] = k0 * sin_t + k1 * cos_t;
    }
}
