// =============================================================================
// tensor/bitnet.rs — BitNet b1.58 Ternary Quantization (The 1-bit Era)
// =============================================================================
//
// BitNet b1.58 uses ternary weights: {-1, 0, 1}
// This eliminates expensive multiplication operations, using only addition/subtraction.
//
// Key advantages:
// - 2-4× faster inference (no multiplications)
// - 2-3× less RAM (from Int4 ~4-bit to ~1.58-bit average)
// - Simpler hardware requirements (CPU/GPU compute units)
//
// Encoding: We pack 5 ternary values into 1 byte:
//   3^5 = 243 < 256, so we can use base-3 encoding
//   This gives us ~1.58 bits per weight on average
// =============================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ---------------------------------------------------------------------------
// Ternary Quantization
// ---------------------------------------------------------------------------

/// Quantize f32 vector to ternary {-1, 0, 1} with AbsMean scaling.
/// Returns the scale factor.
#[allow(dead_code)]
pub fn quantize_f32_to_ternary(input: &[f32], out: &mut [i8]) -> f32 {
    assert_eq!(input.len(), out.len());
    
    // Calculate absolute mean for threshold
    let abs_mean = input.iter().map(|&v| v.abs()).sum::<f32>() / input.len() as f32;
    let threshold = abs_mean * 0.5; // Threshold for zero
    
    // Quantize to {-1, 0, 1}
    for (i, &val) in input.iter().enumerate() {
        out[i] = if val.abs() < threshold {
            0
        } else if val > 0.0 {
            1
        } else {
            -1
        };
    }
    
    abs_mean
}

/// Pack ternary values (stored as i8) into compressed format.
/// We pack 5 ternary values per byte using base-3 encoding.
/// Input: i8 array with values {-1, 0, 1}
/// Output: packed byte array (length = ceil(input.len() / 5))
#[allow(dead_code)]
pub fn pack_ternary(input: &[i8]) -> Vec<u8> {
    let n = input.len();
    let packed_len = (n + 4) / 5; // Ceiling division
    let mut packed = vec![0u8; packed_len];
    
    for i in 0..packed_len {
        let base = i * 5;
        let mut value = 0u8;
        
        for j in 0..5 {
            if base + j >= n {
                break;
            }
            
            // Convert {-1, 0, 1} to {0, 1, 2} for base-3 encoding
            let ternary = match input[base + j] {
                -1 => 0u8,
                0 => 1u8,
                1 => 2u8,
                _ => 1u8, // Default to 0 for invalid values
            };
            
            value += ternary * 3u8.pow(j as u32);
        }
        
        packed[i] = value;
    }
    
    packed
}

/// Unpack ternary values from compressed format.
/// Inverse of pack_ternary.
#[allow(dead_code)]
pub fn unpack_ternary(packed: &[u8], out: &mut [i8], n: usize) {
    for i in 0..packed.len() {
        let base = i * 5;
        let mut value = packed[i];
        
        for j in 0..5 {
            if base + j >= n {
                break;
            }
            
            let digit = value % 3;
            value /= 3;
            
            // Convert {0, 1, 2} back to {-1, 0, 1}
            out[base + j] = match digit {
                0 => -1,
                1 => 0,
                2 => 1,
                _ => 0,
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Ternary Matrix-Vector Multiplication (No Multiplication!)
// ---------------------------------------------------------------------------

/// BitNet b1.58 matrix-vector product: w (ternary) × x (f32).
/// This uses ONLY addition and subtraction - NO multiplication!
///
/// * `out`       — output buffer [out_dim]
/// * `w_packed`  — packed ternary weights
/// * `w_scales`  — per-row scale factors [out_dim]
/// * `x`         — input activation vector [in_dim]
/// * `out_dim`   — number of output features
/// * `in_dim`    — number of input features
#[allow(dead_code)]
pub fn matmul_ternary(
    out: &mut [f32],
    w_packed: &[u8],
    w_scales: &[f32],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    assert!(out.len() >= out_dim);
    assert!(x.len() >= in_dim);
    assert!(w_scales.len() >= out_dim);
    
    let packed_per_row = (in_dim + 4) / 5;
    assert!(w_packed.len() >= out_dim * packed_per_row);
    
    // Temporary buffer for unpacked row
    let mut w_row = vec![0i8; in_dim];
    
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                return matmul_ternary_avx2(out, w_packed, w_scales, x, &mut w_row, out_dim, in_dim, packed_per_row);
            }
        }
    }
    
    matmul_ternary_scalar(out, w_packed, w_scales, x, &mut w_row, out_dim, in_dim, packed_per_row);
}

/// Scalar implementation of ternary matmul
#[allow(dead_code)]
fn matmul_ternary_scalar(
    out: &mut [f32],
    w_packed: &[u8],
    w_scales: &[f32],
    x: &[f32],
    w_row: &mut [i8],
    out_dim: usize,
    in_dim: usize,
    packed_per_row: usize,
) {
    for i in 0..out_dim {
        // Unpack this row
        let row_start = i * packed_per_row;
        unpack_ternary(&w_packed[row_start..row_start + packed_per_row], w_row, in_dim);
        
        // Compute dot product using only addition/subtraction
        #[allow(unused_assignments, unused_variables)]
        let mut _acc = 0.0f32;
        for j in 0..in_dim {
            match w_row[j] {
                1 => _acc += x[j],      // Add
                -1 => _acc -= x[j],     // Subtract
                _ => {}                // Zero: do nothing
            }
        }
        
        // Scale the result
        out[i] = _acc * w_scales[i];
    }
}

/// AVX2 SIMD implementation of ternary matmul
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(dead_code)]
unsafe fn matmul_ternary_avx2(
    out: &mut [f32],
    w_packed: &[u8],
    w_scales: &[f32],
    x: &[f32],
    w_row: &mut [i8],
    out_dim: usize,
    in_dim: usize,
    packed_per_row: usize,
) {
    let chunks8 = in_dim / 8;
    
    for i in 0..out_dim {
        // Unpack this row
        let row_start = i * packed_per_row;
        unpack_ternary(&w_packed[row_start..row_start + packed_per_row], w_row, in_dim);
        
        // SIMD accumulation: process 8 elements at a time
        let mut acc_pos = _mm256_setzero_ps(); // Accumulator for +1 weights
        let mut acc_neg = _mm256_setzero_ps(); // Accumulator for -1 weights
        
        for c in 0..chunks8 {
            let base = c * 8;
            #[allow(unused_variables)]
            let x_vec = _mm256_loadu_ps(x.as_ptr().add(base));
            
            // Process each element
            for j in 0..8 {
                let weight = w_row[base + j];
                if weight == 1 {
                    // Multiply by 1: just add x[j]
                    let x_val = _mm256_set1_ps(x[base + j]);
                    acc_pos = _mm256_add_ps(acc_pos, x_val);
                } else if weight == -1 {
                    // Multiply by -1: add to negative accumulator
                    let x_val = _mm256_set1_ps(x[base + j]);
                    acc_neg = _mm256_add_ps(acc_neg, x_val);
                }
            }
        }
        
        // Horizontal sum of both accumulators
        let diff = _mm256_sub_ps(acc_pos, acc_neg);
        let hi = _mm256_extractf128_ps(diff, 1);
        let lo = _mm256_castps256_ps128(diff);
        let sum4 = _mm_add_ps(lo, hi);
        let sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
        let sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
        let mut acc = _mm_cvtss_f32(sum1);
        
        // Handle remaining elements
        for j in (chunks8 * 8)..in_dim {
            match w_row[j] {
                1 => acc += x[j],
                -1 => acc -= x[j],
                _ => {}
            }
        }
        
        out[i] = acc * w_scales[i];
    }
}

// ---------------------------------------------------------------------------
// Activation Quantization for BitNet
// ---------------------------------------------------------------------------

/// Quantize activations for BitNet (typically to Int8, but can be ternary too).
/// For BitNet b1.58, we quantize activations to Int8 for better precision.
#[allow(dead_code)]
pub fn quantize_activation_bitnet(input: &[f32], out: &mut [i8]) -> f32 {
    assert_eq!(input.len(), out.len());
    
    // Use absolute maximum scaling (symmetric quantization)
    let abs_max = input.iter().map(|&v| v.abs()).fold(0.0f32, f32::max).max(1e-12);
    let scale = abs_max / 127.0;
    let inv_scale = 127.0 / abs_max;
    
    for (i, &val) in input.iter().enumerate() {
        out[i] = (val * inv_scale).round().clamp(-127.0, 127.0) as i8;
    }
    
    scale
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ternary_quantization() {
        let input = vec![0.5, -0.3, 0.1, -0.8, 0.0, 0.9];
        let mut out = vec![0i8; 6];
        
        let scale = quantize_f32_to_ternary(&input, &mut out);
        
        // Check that output is ternary
        for &val in &out {
            assert!(val == -1 || val == 0 || val == 1);
        }
        
        assert!(scale > 0.0);
    }
    
    #[test]
    fn test_ternary_packing() {
        let input = vec![1, 0, -1, 1, 0, 1, -1, 0, 1, 0];
        let packed = pack_ternary(&input);
        
        let mut unpacked = vec![0i8; 10];
        unpack_ternary(&packed, &mut unpacked, 10);
        
        assert_eq!(input, unpacked);
    }
    
    #[test]
    fn test_ternary_matmul() {
        let in_dim = 4;
        let out_dim = 2;
        
        // Ternary weights: [[1, 0, -1, 1], [0, 1, 1, -1]]
        let w_ternary = vec![1i8, 0, -1, 1, 0, 1, 1, -1];
        let w_packed = pack_ternary(&w_ternary);
        let w_scales = vec![1.0, 1.0];
        
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0; 2];
        
        matmul_ternary(&mut out, &w_packed, &w_scales, &x, out_dim, in_dim);
        
        // Expected: [1*1 + 0*2 + (-1)*3 + 1*4, 0*1 + 1*2 + 1*3 + (-1)*4]
        //         = [1 + 0 - 3 + 4, 0 + 2 + 3 - 4]
        //         = [2, 1]
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!((out[1] - 1.0).abs() < 1e-5);
    }
}
