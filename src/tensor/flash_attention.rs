// =============================================================================
// tensor/flash_attention.rs — Flash Attention with CPU Tiling
// =============================================================================
//
// Standard attention has O(N²) memory complexity and causes cache thrashing
// for long sequences (N > 512).
//
// Flash Attention uses tiling to fit attention computation into L1/L2 cache:
//   1. Divide Q, K, V into tiles (e.g., 64×64)
//   2. Compute attention block-by-block
//   3. Fuse operations (softmax + accumulation) to minimize memory traffic
//
// Benefits:
//   - 2-4× faster for long sequences (512-2048 tokens)
//   - Reduced memory usage (no full N×N attention matrix)
//   - Better cache utilization (tiles fit in L1/L2)
//
// Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention"
//            (Dao et al., 2022)
// =============================================================================

/// Tile size for Q, K, V matrices (tuned for L1 cache ~32KB)
/// A 64×64 f32 tile = 16KB, leaves room for other data
#[allow(dead_code)]
const TILE_SIZE: usize = 64;

/// Flash Attention: tiled multi-head attention computation
///
/// Computes: O = softmax(Q @ K^T / sqrt(d)) @ V
///
/// * `q` - Query matrix [n_heads, seq_len, head_dim]
/// * `k` - Key matrix [n_heads, seq_len, head_dim]
/// * `v` - Value matrix [n_heads, seq_len, head_dim]
/// * `out` - Output matrix [n_heads, seq_len, head_dim]
/// * `n_heads` - Number of attention heads
/// * `seq_len` - Sequence length (current position + 1)
/// * `head_dim` - Dimension per head
#[allow(dead_code)]
pub fn flash_attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_heads: usize,
    seq_len: usize,
    head_dim: usize,
) {
    let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();
    
    // For each head independently
    for h in 0..n_heads {
        let q_head = &q[h * seq_len * head_dim..(h + 1) * seq_len * head_dim];
        let k_head = &k[h * seq_len * head_dim..(h + 1) * seq_len * head_dim];
        let v_head = &v[h * seq_len * head_dim..(h + 1) * seq_len * head_dim];
        let out_head = &mut out[h * seq_len * head_dim..(h + 1) * seq_len * head_dim];
        
        flash_attention_single_head(
            q_head,
            k_head,
            v_head,
            out_head,
            seq_len,
            head_dim,
            inv_sqrt_d,
        );
    }
}

/// Flash attention for a single head (tiled implementation)
#[allow(dead_code)]
fn flash_attention_single_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    inv_sqrt_d: f32,
) {
    // Initialize output to zero
    for i in 0..seq_len * head_dim {
        out[i] = 0.0;
    }
    
    // Compute tile size (smaller of TILE_SIZE or sequence length)
    let tile_size = TILE_SIZE.min(seq_len);
    let num_tiles = (seq_len + tile_size - 1) / tile_size;
    
    // Allocate temporary buffers for this head
    let mut scores = vec![0.0f32; tile_size * tile_size];
    let mut row_max = vec![f32::NEG_INFINITY; tile_size];
    let mut row_sum = vec![0.0f32; tile_size];
    let mut local_out = vec![0.0f32; tile_size * head_dim];
    
    // Process Q tiles (rows)
    for q_tile_idx in 0..num_tiles {
        let q_start = q_tile_idx * tile_size;
        let q_end = (q_start + tile_size).min(seq_len);
        let q_tile_len = q_end - q_start;
        
        // Reset accumulators for this Q tile
        row_max.fill(f32::NEG_INFINITY);
        row_sum.fill(0.0);
        local_out.fill(0.0);
        
        // Process K/V tiles (columns)
        for kv_tile_idx in 0..num_tiles {
            let kv_start = kv_tile_idx * tile_size;
            let kv_end = (kv_start + tile_size).min(seq_len);
            let kv_tile_len = kv_end - kv_start;
            
            // Causal mask: only process if kv_end <= q_end (upper triangular)
            if kv_start > q_end {
                break;
            }
            
            // Step 1: Compute attention scores for this tile
            // scores[i, j] = q[q_start + i] @ k[kv_start + j] / sqrt(d)
            for i in 0..q_tile_len {
                for j in 0..kv_tile_len {
                    // Apply causal mask
                    if q_start + i < kv_start + j {
                        scores[i * tile_size + j] = f32::NEG_INFINITY;
                        continue;
                    }
                    
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[(q_start + i) * head_dim + d] * k[(kv_start + j) * head_dim + d];
                    }
                    scores[i * tile_size + j] = dot * inv_sqrt_d;
                }
            }
            
            // Step 2: Online softmax (fused with max finding)
            // For numerical stability, we track per-row max and sum
            for i in 0..q_tile_len {
                // Find max in this tile for row i
                let mut tile_max = f32::NEG_INFINITY;
                for j in 0..kv_tile_len {
                    tile_max = tile_max.max(scores[i * tile_size + j]);
                }
                
                // Update global row max (for online softmax)
                let prev_max = row_max[i];
                let new_max = prev_max.max(tile_max);
                row_max[i] = new_max;
                
                // Compute exp and sum for this tile
                let mut tile_sum = 0.0f32;
                for j in 0..kv_tile_len {
                    let exp_val = (scores[i * tile_size + j] - new_max).exp();
                    scores[i * tile_size + j] = exp_val;
                    tile_sum += exp_val;
                }
                
                // Update global sum with correction factor
                let correction = (prev_max - new_max).exp();
                row_sum[i] = row_sum[i] * correction + tile_sum;
            }
            
            // Step 3: Accumulate weighted values
            // out[i] += sum_j scores[i,j] * v[kv_start + j]
            for i in 0..q_tile_len {
                for j in 0..kv_tile_len {
                    let weight = scores[i * tile_size + j];
                    for d in 0..head_dim {
                        local_out[i * head_dim + d] += weight * v[(kv_start + j) * head_dim + d];
                    }
                }
            }
        }
        
        // Step 4: Normalize and write output for this Q tile
        for i in 0..q_tile_len {
            let norm = 1.0 / row_sum[i].max(1e-12);
            for d in 0..head_dim {
                out[(q_start + i) * head_dim + d] = local_out[i * head_dim + d] * norm;
            }
        }
    }
}

/// Flash attention with logit capping (for stability in deep networks)
#[allow(dead_code)]
pub fn flash_attention_forward_capped(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_heads: usize,
    seq_len: usize,
    head_dim: usize,
    logit_cap: f32,
) {
    let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();
    
    for h in 0..n_heads {
        let q_head = &q[h * seq_len * head_dim..(h + 1) * seq_len * head_dim];
        let k_head = &k[h * seq_len * head_dim..(h + 1) * seq_len * head_dim];
        let v_head = &v[h * seq_len * head_dim..(h + 1) * seq_len * head_dim];
        let out_head = &mut out[h * seq_len * head_dim..(h + 1) * seq_len * head_dim];
        
        flash_attention_single_head_capped(
            q_head,
            k_head,
            v_head,
            out_head,
            seq_len,
            head_dim,
            inv_sqrt_d,
            logit_cap,
        );
    }
}

/// Flash attention with logit soft-capping
fn flash_attention_single_head_capped(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_len: usize,
    head_dim: usize,
    inv_sqrt_d: f32,
    logit_cap: f32,
) {
    let tanh_cap = logit_cap;
    
    for i in 0..seq_len * head_dim {
        out[i] = 0.0;
    }
    
    let tile_size = TILE_SIZE.min(seq_len);
    let num_tiles = (seq_len + tile_size - 1) / tile_size;
    
    let mut scores = vec![0.0f32; tile_size * tile_size];
    let mut row_max = vec![f32::NEG_INFINITY; tile_size];
    let mut row_sum = vec![0.0f32; tile_size];
    let mut local_out = vec![0.0f32; tile_size * head_dim];
    
    for q_tile_idx in 0..num_tiles {
        let q_start = q_tile_idx * tile_size;
        let q_end = (q_start + tile_size).min(seq_len);
        let q_tile_len = q_end - q_start;
        
        row_max.fill(f32::NEG_INFINITY);
        row_sum.fill(0.0);
        local_out.fill(0.0);
        
        for kv_tile_idx in 0..num_tiles {
            let kv_start = kv_tile_idx * tile_size;
            let kv_end = (kv_start + tile_size).min(seq_len);
            let kv_tile_len = kv_end - kv_start;
            
            if kv_start > q_end {
                break;
            }
            
            // Compute scores with logit capping
            for i in 0..q_tile_len {
                for j in 0..kv_tile_len {
                    if q_start + i < kv_start + j {
                        scores[i * tile_size + j] = f32::NEG_INFINITY;
                        continue;
                    }
                    
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q[(q_start + i) * head_dim + d] * k[(kv_start + j) * head_dim + d];
                    }
                    
                    // Apply soft-cap: tanh(x / cap) * cap
                    let scaled = dot * inv_sqrt_d;
                    let capped = (scaled / tanh_cap).tanh() * tanh_cap;
                    scores[i * tile_size + j] = capped;
                }
            }
            
            // Online softmax
            for i in 0..q_tile_len {
                let mut tile_max = f32::NEG_INFINITY;
                for j in 0..kv_tile_len {
                    tile_max = tile_max.max(scores[i * tile_size + j]);
                }
                
                let prev_max = row_max[i];
                let new_max = prev_max.max(tile_max);
                row_max[i] = new_max;
                
                let mut tile_sum = 0.0f32;
                for j in 0..kv_tile_len {
                    let exp_val = (scores[i * tile_size + j] - new_max).exp();
                    scores[i * tile_size + j] = exp_val;
                    tile_sum += exp_val;
                }
                
                let correction = (prev_max - new_max).exp();
                row_sum[i] = row_sum[i] * correction + tile_sum;
            }
            
            // Accumulate values
            for i in 0..q_tile_len {
                for j in 0..kv_tile_len {
                    let weight = scores[i * tile_size + j];
                    for d in 0..head_dim {
                        local_out[i * head_dim + d] += weight * v[(kv_start + j) * head_dim + d];
                    }
                }
            }
        }
        
        // Normalize and write output
        for i in 0..q_tile_len {
            let norm = 1.0 / row_sum[i].max(1e-12);
            for d in 0..head_dim {
                out[(q_start + i) * head_dim + d] = local_out[i * head_dim + d] * norm;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_flash_attention_small() {
        let n_heads = 1;
        let seq_len = 4;
        let head_dim = 8;
        
        let q = vec![1.0; seq_len * head_dim];
        let k = vec![1.0; seq_len * head_dim];
        let v = vec![1.0; seq_len * head_dim];
        let mut out = vec![0.0; seq_len * head_dim];
        
        flash_attention_forward(&q, &k, &v, &mut out, n_heads, seq_len, head_dim);
        
        // Output should be non-zero
        assert!(out.iter().any(|&x| x != 0.0));
    }
    
    #[test]
    fn test_flash_attention_with_cap() {
        let n_heads = 1;
        let seq_len = 8;
        let head_dim = 16;
        let logit_cap = 30.0;
        
        let q = vec![0.5; seq_len * head_dim];
        let k = vec![0.5; seq_len * head_dim];
        let v = vec![1.0; seq_len * head_dim];
        let mut out = vec![0.0; seq_len * head_dim];
        
        flash_attention_forward_capped(&q, &k, &v, &mut out, n_heads, seq_len, head_dim, logit_cap);
        
        // Output should be reasonable
        for &val in &out {
            assert!(val.is_finite());
            assert!(val >= 0.0);
        }
    }
}
