// =============================================================================
// tests.rs — Unit tests for tensor operations
// =============================================================================
//
// This module contains unit tests for quantization and matrix multiplication
// operations, including:
//
// - Int8 quantization roundtrip tests
// - Int8 matrix-vector product correctness tests
// - Int4 pack/unpack correctness tests
// - Int4 quantization and matmul tests
// - Softmax, RMS norm, and sampling utilities
//
// =============================================================================

#[cfg(test)]
mod tests {
    use super::super::quantization::*;
    use super::super::matmul::*;
    use super::super::{rmsnorm, softmax, sample_argmax, softmax_top_k};

    #[test]
    fn test_quantize_i8_roundtrip() {
        let x = vec![1.0f32, 0.5, -0.5, 0.0, -1.0];
        let mut q = vec![0i8; 5];
        let scale = quantize_f32_to_i8(&x, &mut q);
        // 1.0 should map to 127
        assert_eq!(q[0], 127);
        // -1.0 → -127
        assert_eq!(q[4], -127);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_matmul_int8_identity_like() {
        // W = 2×3 "identity-ish" in int8, x = [1, 2, 3]
        let w: Vec<i8> = vec![
            127, 0, 0, // row 0
            0, 127, 0, // row 1
        ];
        let w_scales = vec![1.0 / 127.0; 2]; // so dequant ≈ 1.0 per element
        let x = vec![1.0f32, 2.0, 3.0];
        let mut out = vec![0.0f32; 2];

        matmul_int8(&mut out, &w, &w_scales, &x, 2, 3);

        // out[0] ≈ 1.0, out[1] ≈ 2.0  (with quantisation noise)
        assert!((out[0] - 1.0).abs() < 0.1, "out[0]={}", out[0]);
        assert!((out[1] - 2.0).abs() < 0.1, "out[1]={}", out[1]);
    }

    #[test]
    fn test_rmsnorm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let mut out = vec![0.0f32; 4];
        rmsnorm(&mut out, &x, &w);
        // After norm, values should be scaled
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / 4.0 + 1e-5).sqrt();
        assert!((out[0] - 1.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_softmax() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn test_sample_argmax() {
        let logits = vec![0.1, 0.9, 0.5, 0.3];
        assert_eq!(sample_argmax(&logits), 1);
    }

    #[test]
    fn test_int4_pack_unpack() {
        for lo in -8i8..=7 {
            for hi in -8i8..=7 {
                let byte = pack_int4(lo, hi);
                let (lo2, hi2) = unpack_int4(byte);
                assert_eq!(lo, lo2, "lo mismatch for ({}, {})", lo, hi);
                assert_eq!(hi, hi2, "hi mismatch for ({}, {})", lo, hi);
            }
        }
    }

    #[test]
    fn test_quantize_i4_roundtrip() {
        let x = vec![1.0f32, -0.5, 0.0, 0.25, -1.0, 0.7, -0.3, 0.9];
        let n = x.len();
        let n_groups = (n + INT4_GROUP_SIZE - 1) / INT4_GROUP_SIZE;
        let packed_len = (n + 1) / 2;
        let mut out = vec![0u8; packed_len];
        let mut scales = vec![0.0f32; n_groups];
        quantize_f32_to_i4(&x, &mut out, &mut scales);
        assert!(scales[0] > 0.0);
        // 1.0 and -1.0 should map to extremes (±7)
        let (lo, _) = unpack_int4(out[0]);
        assert!(lo > 0, "1.0 should quantise to positive int4");
    }

    #[test]
    fn test_matmul_int4_basic() {
        let in_dim = 64;
        let out_dim = 2;
        let group = INT4_GROUP_SIZE;
        let n_groups = (in_dim + group - 1) / group;
        let packed_row = in_dim / 2;
        // Create simple weights: row 0 = all +7 (max), row 1 = all 0
        let mut w_packed = vec![0u8; out_dim * packed_row];
        for j in 0..packed_row {
            w_packed[j] = pack_int4(7, 7); // row 0: all 7s
        }
        // row 1 stays zero (pack_int4(0,0) = 0x88)
        for j in 0..packed_row {
            w_packed[packed_row + j] = pack_int4(0, 0);
        }
        let w_scales = vec![1.0f32 / 7.0; out_dim * n_groups];
        let x: Vec<f32> = vec![1.0; in_dim];
        let mut out = vec![0.0f32; out_dim];
        matmul_int4(&mut out, &w_packed, &w_scales, &x, out_dim, in_dim);
        // Row 0: sum of 64 * (7/7 * 1.0) with quantisation ≈ 64ish
        assert!(out[0].abs() > 10.0, "row 0 should have large output: {}", out[0]);
        assert!(out[1].abs() < 1.0, "row 1 should be near zero: {}", out[1]);
    }

    #[test]
    fn test_softmax_top_k() {
        let mut scores = vec![1.0, 3.0, 2.0, 0.5];
        let top2 = softmax_top_k(&mut scores, 2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, 1); // index of highest score
        assert_eq!(top2[1].0, 2); // index of second highest
        let sum: f32 = top2.iter().map(|(_, w)| w).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
}
