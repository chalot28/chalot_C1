// =============================================================================
// matmul.rs — Quantized matrix-vector multiplication kernels
// =============================================================================
//
// This module implements the critical matrix-vector product operations using
// low-precision integer arithmetic:
//
// - **matmul_int8**: Int8 weight × Int8 quantized-input kernel with
//   per-row weight scales. Includes scalar fallback, SSE2, AVX2, and NEON
//   implementations with runtime CPU feature detection.
//
// - **matmul_int4**: Int4 weight × Int8 quantized-input kernel with
//   per-group weight scales. AVX2 implementation unpacks nibbles to Int8
//   in SIMD registers for efficient processing.
//
// Both kernels use 2-row parallelism in SIMD paths to maximize cache reuse
// of the quantized input vector.
//
// =============================================================================

use super::quantization::{quantize_f32_to_i8, unpack_int4, INT4_GROUP_SIZE};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ---------------------------------------------------------------------------
// matmul_int8  — THE critical kernel
// ---------------------------------------------------------------------------
//
// Computes:  out[i] = dequant( dot(w_row[i], x_q) )  for i in 0..out_dim
//
// Where w is [out_dim x in_dim] stored as Int8 with per-row scales,
//       x is an f32 vector that we quantise on-the-fly.
//
// Complexity: O(out_dim * in_dim)  — dominant cost of every layer.
// ---------------------------------------------------------------------------

/// Quantized matrix–vector product.
///
/// * `out`       — output buffer, length = `out_dim`
/// * `w`         — weight matrix (Int8), contiguous [out_dim * in_dim]
/// * `w_scales`  — per-row f32 scales, length = `out_dim`
/// * `x`         — input activation vector (f32), length = `in_dim`
pub fn matmul_int8(
    out: &mut [f32],
    w: &[i8],
    w_scales: &[f32],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    assert!(out.len() >= out_dim);
    assert!(w.len() >= out_dim * in_dim);
    assert!(w_scales.len() >= out_dim);
    assert!(x.len() >= in_dim);

    // --- Step 1: quantise input f32 → i8 on stack (reusable scratch) ---
    // For dim=256 this is only 256 bytes — fine on the stack.
    let mut x_q = vec![0i8; in_dim]; // TODO: use stack array when const generic
    let x_scale = quantize_f32_to_i8(x, &mut x_q);

    // --- Step 2: dot product per output row ---
    // Portable scalar path (SIMD specialisation below when available).
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        matmul_int8_scalar(out, w, w_scales, &x_q, x_scale, out_dim, in_dim);
    }

    // x86-64: prefer AVX2 (2× throughput) when available, fallback to SSE2
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                matmul_int8_avx2(out, w, w_scales, &x_q, x_scale, out_dim, in_dim);
            }
        } else {
            unsafe {
                matmul_int8_sse2(out, w, w_scales, &x_q, x_scale, out_dim, in_dim);
            }
        }
    }

    // AArch64 NEON path
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            matmul_int8_neon(out, w, w_scales, &x_q, x_scale, out_dim, in_dim);
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar fallback
// ---------------------------------------------------------------------------
#[allow(dead_code)]
fn matmul_int8_scalar(
    out: &mut [f32],
    w: &[i8],
    w_scales: &[f32],
    x_q: &[i8],
    x_scale: f32,
    out_dim: usize,
    in_dim: usize,
) {
    for i in 0..out_dim {
        let row = &w[i * in_dim..(i + 1) * in_dim];
        let mut acc: i32 = 0;
        for j in 0..in_dim {
            acc += (row[j] as i32) * (x_q[j] as i32);
        }
        // Dequantize: int_result * w_scale * x_scale
        out[i] = (acc as f32) * w_scales[i] * x_scale;
    }
}

// ---------------------------------------------------------------------------
// x86-64 SSE2 intrinsics path
// ---------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn matmul_int8_sse2(
    out: &mut [f32],
    w: &[i8],
    w_scales: &[f32],
    x_q: &[i8],
    x_scale: f32,
    out_dim: usize,
    in_dim: usize,
) {
    let chunks = in_dim / 16;
    let _remainder = in_dim % 16;

    for i in 0..out_dim {
        let row_off = i * in_dim;
        let mut acc = _mm_setzero_si128(); // 4 × i32 accumulator

        for c in 0..chunks {
            let base = c * 16;
            // Load 16 × i8 from weight and input
            let vw = _mm_loadu_si128(w.as_ptr().add(row_off + base) as *const __m128i);
            let vx = _mm_loadu_si128(x_q.as_ptr().add(base) as *const __m128i);

            // SSE2 does not have i8 dot product — we widen to i16 and multiply.
            // Unpack low  8×i8 → 8×i16
            let w_lo = _mm_srai_epi16(_mm_unpacklo_epi8(vw, vw), 8);
            let x_lo = _mm_srai_epi16(_mm_unpacklo_epi8(vx, vx), 8);
            // Unpack high 8×i8 → 8×i16
            let w_hi = _mm_srai_epi16(_mm_unpackhi_epi8(vw, vw), 8);
            let x_hi = _mm_srai_epi16(_mm_unpackhi_epi8(vx, vx), 8);

            // _mm_madd_epi16: pairs of i16→i32 multiply-add (gives 4×i32)
            let prod_lo = _mm_madd_epi16(w_lo, x_lo); // 4 × i32
            let prod_hi = _mm_madd_epi16(w_hi, x_hi); // 4 × i32

            acc = _mm_add_epi32(acc, _mm_add_epi32(prod_lo, prod_hi));
        }

        // Horizontal sum of 4 × i32 → scalar
        let hi64 = _mm_shuffle_epi32(acc, 0b_00_01_10_11);
        let sum2 = _mm_add_epi32(acc, hi64);
        let hi32 = _mm_shufflelo_epi16(sum2, 0b_01_00_11_10);
        let sum1 = _mm_add_epi32(sum2, hi32);
        let mut dot: i32 = _mm_cvtsi128_si32(sum1);

        // Scalar tail for remainder
        for j in (chunks * 16)..in_dim {
            dot += (w[row_off + j] as i32) * (x_q[j] as i32);
        }

        out[i] = (dot as f32) * w_scales[i] * x_scale;
    }
}

// ---------------------------------------------------------------------------
// x86-64 AVX2 intrinsics path — 2× throughput over SSE2
// Processes 32 elements/iteration + 2-row parallelism for cache reuse
// ---------------------------------------------------------------------------
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn matmul_int8_avx2(
    out: &mut [f32],
    w: &[i8],
    w_scales: &[f32],
    x_q: &[i8],
    x_scale: f32,
    out_dim: usize,
    in_dim: usize,
) {
    let chunks = in_dim / 32;
    let rem_start = chunks * 32;

    // Process pairs of output rows — x_q stays in L1 cache
    let pairs = out_dim / 2;
    let has_tail = out_dim % 2 != 0;

    for p in 0..pairs {
        let i0 = p * 2;
        let i1 = i0 + 1;
        let row0 = i0 * in_dim;
        let row1 = i1 * in_dim;
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();

        for c in 0..chunks {
            let base = c * 32;
            // Load 32 × i8 from input (shared between both rows)
            let vx_raw = _mm256_loadu_si256(x_q.as_ptr().add(base) as *const __m256i);
            let vx_lo = _mm256_castsi256_si128(vx_raw);
            let vx_hi = _mm256_extracti128_si256(vx_raw, 1);
            let vx_lo16 = _mm256_cvtepi8_epi16(vx_lo);
            let vx_hi16 = _mm256_cvtepi8_epi16(vx_hi);

            // Row 0: load weights, sign-extend, madd
            let vw0 = _mm256_loadu_si256(w.as_ptr().add(row0 + base) as *const __m256i);
            let vw0_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vw0));
            let vw0_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vw0, 1));
            let p0_lo = _mm256_madd_epi16(vw0_lo16, vx_lo16);
            let p0_hi = _mm256_madd_epi16(vw0_hi16, vx_hi16);
            acc0 = _mm256_add_epi32(acc0, _mm256_add_epi32(p0_lo, p0_hi));

            // Row 1: same x, different weights
            let vw1 = _mm256_loadu_si256(w.as_ptr().add(row1 + base) as *const __m256i);
            let vw1_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vw1));
            let vw1_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vw1, 1));
            let p1_lo = _mm256_madd_epi16(vw1_lo16, vx_lo16);
            let p1_hi = _mm256_madd_epi16(vw1_hi16, vx_hi16);
            acc1 = _mm256_add_epi32(acc1, _mm256_add_epi32(p1_lo, p1_hi));
        }

        // Horizontal sum: 8 × i32 → scalar (row 0)
        let hi0 = _mm256_extracti128_si256(acc0, 1);
        let lo0 = _mm256_castsi256_si128(acc0);
        let s4_0 = _mm_add_epi32(lo0, hi0);
        let sh0 = _mm_shuffle_epi32(s4_0, 0b_00_01_10_11);
        let s2_0 = _mm_add_epi32(s4_0, sh0);
        let sh0b = _mm_shufflelo_epi16(s2_0, 0b_01_00_11_10);
        let s1_0 = _mm_add_epi32(s2_0, sh0b);
        let mut dot0: i32 = _mm_cvtsi128_si32(s1_0);

        // Horizontal sum: 8 × i32 → scalar (row 1)
        let hi1 = _mm256_extracti128_si256(acc1, 1);
        let lo1 = _mm256_castsi256_si128(acc1);
        let s4_1 = _mm_add_epi32(lo1, hi1);
        let sh1 = _mm_shuffle_epi32(s4_1, 0b_00_01_10_11);
        let s2_1 = _mm_add_epi32(s4_1, sh1);
        let sh1b = _mm_shufflelo_epi16(s2_1, 0b_01_00_11_10);
        let s1_1 = _mm_add_epi32(s2_1, sh1b);
        let mut dot1: i32 = _mm_cvtsi128_si32(s1_1);

        // Scalar tail (shared load)
        for j in rem_start..in_dim {
            let xv = x_q[j] as i32;
            dot0 += (w[row0 + j] as i32) * xv;
            dot1 += (w[row1 + j] as i32) * xv;
        }

        out[i0] = (dot0 as f32) * w_scales[i0] * x_scale;
        out[i1] = (dot1 as f32) * w_scales[i1] * x_scale;
    }

    // Handle odd last row
    if has_tail {
        let i = out_dim - 1;
        let row_off = i * in_dim;
        let mut acc = _mm256_setzero_si256();

        for c in 0..chunks {
            let base = c * 32;
            let vw = _mm256_loadu_si256(w.as_ptr().add(row_off + base) as *const __m256i);
            let vx = _mm256_loadu_si256(x_q.as_ptr().add(base) as *const __m256i);
            let vw_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vw));
            let vx_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vx));
            let vw_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vw, 1));
            let vx_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vx, 1));
            let prod_lo = _mm256_madd_epi16(vw_lo16, vx_lo16);
            let prod_hi = _mm256_madd_epi16(vw_hi16, vx_hi16);
            acc = _mm256_add_epi32(acc, _mm256_add_epi32(prod_lo, prod_hi));
        }

        let hi128 = _mm256_extracti128_si256(acc, 1);
        let lo128 = _mm256_castsi256_si128(acc);
        let sum4 = _mm_add_epi32(lo128, hi128);
        let shuf = _mm_shuffle_epi32(sum4, 0b_00_01_10_11);
        let sum2 = _mm_add_epi32(sum4, shuf);
        let shuf2 = _mm_shufflelo_epi16(sum2, 0b_01_00_11_10);
        let sum1 = _mm_add_epi32(sum2, shuf2);
        let mut dot: i32 = _mm_cvtsi128_si32(sum1);

        for j in rem_start..in_dim {
            dot += (w[row_off + j] as i32) * (x_q[j] as i32);
        }

        out[i] = (dot as f32) * w_scales[i] * x_scale;
    }
}

// ---------------------------------------------------------------------------
// AArch64 NEON intrinsics path
// ---------------------------------------------------------------------------
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn matmul_int8_neon(
    out: &mut [f32],
    w: &[i8],
    w_scales: &[f32],
    x_q: &[i8],
    x_scale: f32,
    out_dim: usize,
    in_dim: usize,
) {
    let chunks = in_dim / 16;

    for i in 0..out_dim {
        let row_off = i * in_dim;
        let mut acc0 = vdupq_n_s32(0);
        let mut acc1 = vdupq_n_s32(0);

        for c in 0..chunks {
            let base = row_off + c * 16;
            let bx = c * 16;

            let vw = vld1q_s8(w.as_ptr().add(base));
            let vx = vld1q_s8(x_q.as_ptr().add(bx));

            // Widen to i16 and multiply-accumulate
            let w_lo = vmovl_s8(vget_low_s8(vw));
            let x_lo = vmovl_s8(vget_low_s8(vx));
            let w_hi = vmovl_s8(vget_high_s8(vw));
            let x_hi = vmovl_s8(vget_high_s8(vx));

            // i16 × i16 → i32 widening multiply-accumulate
            acc0 = vmlal_s16(acc0, vget_low_s16(w_lo), vget_low_s16(x_lo));
            acc0 = vmlal_s16(acc0, vget_high_s16(w_lo), vget_high_s16(x_lo));
            acc1 = vmlal_s16(acc1, vget_low_s16(w_hi), vget_low_s16(x_hi));
            acc1 = vmlal_s16(acc1, vget_high_s16(w_hi), vget_high_s16(x_hi));
        }

        let total = vaddq_s32(acc0, acc1);
        let mut dot: i32 = vaddvq_s32(total);

        // Scalar remainder
        for j in (chunks * 16)..in_dim {
            dot += (w[row_off + j] as i32) * (x_q[j] as i32);
        }

        out[i] = (dot as f32) * w_scales[i] * x_scale;
    }
}

// ---------------------------------------------------------------------------
// matmul_int4 — Int4 weight × Int8 quantised-input kernel (AVX2 optimized)
// ---------------------------------------------------------------------------
// AVX2 path: unpacks Int4→Int8 in SIMD registers using nibble extraction,
// then uses i8→i16 widening + _mm256_madd_epi16 for accumulation.
// Processes 32 elements (16 packed bytes) per SIMD iteration with
// 2-row parallelism for cache reuse of the quantised input vector.
// ---------------------------------------------------------------------------

/// Quantized Int4 matrix–vector product with per-group weight scales.
pub fn matmul_int4(
    out: &mut [f32],
    w_packed: &[u8],
    w_scales: &[f32],
    x: &[f32],
    out_dim: usize,
    in_dim: usize,
) {
    let group = INT4_GROUP_SIZE;
    let n_groups = (in_dim + group - 1) / group;
    let packed_row = (in_dim + 1) / 2;

    assert!(out.len() >= out_dim);
    assert!(w_packed.len() >= out_dim * packed_row);
    assert!(w_scales.len() >= out_dim * n_groups);
    assert!(x.len() >= in_dim);

    // Quantise input to Int8 (shared between SIMD and scalar paths)
    let mut x_q = vec![0i8; in_dim];
    let x_scale = quantize_f32_to_i8(x, &mut x_q);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                matmul_int4_avx2(
                    out, w_packed, w_scales, &x_q, x_scale,
                    out_dim, in_dim, group, n_groups, packed_row,
                );
            }
            return;
        }
    }
    matmul_int4_scalar(
        out, w_packed, w_scales, &x_q, x_scale,
        out_dim, in_dim, group, n_groups, packed_row,
    );
}

fn matmul_int4_scalar(
    out: &mut [f32],
    w_packed: &[u8],
    w_scales: &[f32],
    x_q: &[i8],
    x_scale: f32,
    out_dim: usize,
    in_dim: usize,
    group: usize,
    n_groups: usize,
    packed_row: usize,
) {
    for i in 0..out_dim {
        let w_off = i * packed_row;
        let s_off = i * n_groups;
        let mut total = 0.0f32;

        for g in 0..n_groups {
            let g_start = g * group;
            let g_end = (g_start + group).min(in_dim);
            let group_scale = w_scales[s_off + g];
            let mut acc: i32 = 0;

            let mut j = g_start;
            while j + 1 < g_end {
                let byte = w_packed[w_off + j / 2];
                let (w0, w1) = unpack_int4(byte);
                acc += (w0 as i32) * (x_q[j] as i32);
                acc += (w1 as i32) * (x_q[j + 1] as i32);
                j += 2;
            }
            if j < g_end {
                let byte = w_packed[w_off + j / 2];
                let (w0, _) = unpack_int4(byte);
                acc += (w0 as i32) * (x_q[j] as i32);
            }

            total += (acc as f32) * group_scale * x_scale;
        }

        out[i] = total;
    }
}

/// AVX2 Int4 matmul: unpacks 16 packed bytes → 32 i8 weights via nibble
/// extraction, then uses i8→i16 widening + madd_epi16 for accumulation.
/// 2-row parallelism for cache reuse of x_q.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn matmul_int4_avx2(
    out: &mut [f32],
    w_packed: &[u8],
    w_scales: &[f32],
    x_q: &[i8],
    x_scale: f32,
    out_dim: usize,
    in_dim: usize,
    group: usize,
    n_groups: usize,
    packed_row: usize,
) {
    let mask_0f = _mm_set1_epi8(0x0F);
    let offset8 = _mm_set1_epi8(8);

    // Process pairs of output rows for cache reuse of x_q
    let pairs = out_dim / 2;
    let has_tail = out_dim % 2 != 0;

    for p in 0..pairs {
        let i0 = p * 2;
        let i1 = i0 + 1;
        let w_off0 = i0 * packed_row;
        let w_off1 = i1 * packed_row;
        let s_off0 = i0 * n_groups;
        let s_off1 = i1 * n_groups;
        let mut total0 = 0.0f32;
        let mut total1 = 0.0f32;

        for g in 0..n_groups {
            let g_start = g * group;
            let g_end = (g_start + group).min(in_dim);
            let g_len = g_end - g_start;
            let gs0 = w_scales[s_off0 + g];
            let gs1 = w_scales[s_off1 + g];
            let packed_g_start = g_start / 2;

            let mut acc0 = _mm256_setzero_si256();
            let mut acc1 = _mm256_setzero_si256();
            let simd_iters = g_len / 32;

            for c in 0..simd_iters {
                let byte_off = packed_g_start + c * 16;
                let elem_off = g_start + c * 32;

                // Load 32 x_q values (shared between both rows)
                let vx_raw = _mm256_loadu_si256(x_q.as_ptr().add(elem_off) as *const __m256i);
                let vx_lo = _mm256_castsi256_si128(vx_raw);
                let vx_hi = _mm256_extracti128_si256(vx_raw, 1);
                let vx_lo16 = _mm256_cvtepi8_epi16(vx_lo);
                let vx_hi16 = _mm256_cvtepi8_epi16(vx_hi);

                // Row 0: unpack Int4 → Int8 via nibble extraction
                let packed0 = _mm_loadu_si128(w_packed.as_ptr().add(w_off0 + byte_off) as *const __m128i);
                let lo0 = _mm_sub_epi8(_mm_and_si128(packed0, mask_0f), offset8);
                let hi0 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(packed0, 4), mask_0f), offset8);
                let w0_all = _mm256_set_m128i(
                    _mm_unpackhi_epi8(lo0, hi0),
                    _mm_unpacklo_epi8(lo0, hi0),
                );
                let w0_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w0_all));
                let w0_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w0_all, 1));
                let p0_lo = _mm256_madd_epi16(w0_lo16, vx_lo16);
                let p0_hi = _mm256_madd_epi16(w0_hi16, vx_hi16);
                acc0 = _mm256_add_epi32(acc0, _mm256_add_epi32(p0_lo, p0_hi));

                // Row 1: same x_q, different weights
                let packed1 = _mm_loadu_si128(w_packed.as_ptr().add(w_off1 + byte_off) as *const __m128i);
                let lo1 = _mm_sub_epi8(_mm_and_si128(packed1, mask_0f), offset8);
                let hi1 = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(packed1, 4), mask_0f), offset8);
                let w1_all = _mm256_set_m128i(
                    _mm_unpackhi_epi8(lo1, hi1),
                    _mm_unpacklo_epi8(lo1, hi1),
                );
                let w1_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w1_all));
                let w1_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w1_all, 1));
                let p1_lo = _mm256_madd_epi16(w1_lo16, vx_lo16);
                let p1_hi = _mm256_madd_epi16(w1_hi16, vx_hi16);
                acc1 = _mm256_add_epi32(acc1, _mm256_add_epi32(p1_lo, p1_hi));
            }

            // Horizontal sum for row 0: 8×i32 → scalar
            let hi128_0 = _mm256_extracti128_si256(acc0, 1);
            let lo128_0 = _mm256_castsi256_si128(acc0);
            let s4_0 = _mm_add_epi32(lo128_0, hi128_0);
            let sh0 = _mm_shuffle_epi32(s4_0, 0b_00_01_10_11);
            let s2_0 = _mm_add_epi32(s4_0, sh0);
            let shb0 = _mm_shufflelo_epi16(s2_0, 0b_01_00_11_10);
            let s1_0 = _mm_add_epi32(s2_0, shb0);
            let mut gdot0: i32 = _mm_cvtsi128_si32(s1_0);

            // Horizontal sum for row 1
            let hi128_1 = _mm256_extracti128_si256(acc1, 1);
            let lo128_1 = _mm256_castsi256_si128(acc1);
            let s4_1 = _mm_add_epi32(lo128_1, hi128_1);
            let sh1 = _mm_shuffle_epi32(s4_1, 0b_00_01_10_11);
            let s2_1 = _mm_add_epi32(s4_1, sh1);
            let shb1 = _mm_shufflelo_epi16(s2_1, 0b_01_00_11_10);
            let s1_1 = _mm_add_epi32(s2_1, shb1);
            let mut gdot1: i32 = _mm_cvtsi128_si32(s1_1);

            // Scalar tail for remainder elements
            let simd_done = simd_iters * 32;
            let mut j = g_start + simd_done;
            while j + 1 < g_end {
                let byte0 = w_packed[w_off0 + j / 2];
                let (wa0, wa1) = unpack_int4(byte0);
                gdot0 += (wa0 as i32) * (x_q[j] as i32);
                gdot0 += (wa1 as i32) * (x_q[j + 1] as i32);
                let byte1 = w_packed[w_off1 + j / 2];
                let (wb0, wb1) = unpack_int4(byte1);
                gdot1 += (wb0 as i32) * (x_q[j] as i32);
                gdot1 += (wb1 as i32) * (x_q[j + 1] as i32);
                j += 2;
            }
            if j < g_end {
                let byte0 = w_packed[w_off0 + j / 2];
                let (wa, _) = unpack_int4(byte0);
                gdot0 += (wa as i32) * (x_q[j] as i32);
                let byte1 = w_packed[w_off1 + j / 2];
                let (wb, _) = unpack_int4(byte1);
                gdot1 += (wb as i32) * (x_q[j] as i32);
            }

            total0 += (gdot0 as f32) * gs0 * x_scale;
            total1 += (gdot1 as f32) * gs1 * x_scale;
        }

        out[i0] = total0;
        out[i1] = total1;
    }

    // Handle odd last row
    if has_tail {
        let i = out_dim - 1;
        let w_off = i * packed_row;
        let s_off = i * n_groups;
        let mut total = 0.0f32;

        for g in 0..n_groups {
            let g_start = g * group;
            let g_end = (g_start + group).min(in_dim);
            let g_len = g_end - g_start;
            let gs = w_scales[s_off + g];
            let packed_g_start = g_start / 2;

            let mut acc = _mm256_setzero_si256();
            let simd_iters = g_len / 32;

            for c in 0..simd_iters {
                let byte_off = packed_g_start + c * 16;
                let elem_off = g_start + c * 32;

                let vx_raw = _mm256_loadu_si256(x_q.as_ptr().add(elem_off) as *const __m256i);
                let vx_lo = _mm256_castsi256_si128(vx_raw);
                let vx_hi = _mm256_extracti128_si256(vx_raw, 1);
                let vx_lo16 = _mm256_cvtepi8_epi16(vx_lo);
                let vx_hi16 = _mm256_cvtepi8_epi16(vx_hi);

                let packed_w = _mm_loadu_si128(w_packed.as_ptr().add(w_off + byte_off) as *const __m128i);
                let lo_n = _mm_sub_epi8(_mm_and_si128(packed_w, mask_0f), offset8);
                let hi_n = _mm_sub_epi8(_mm_and_si128(_mm_srli_epi16(packed_w, 4), mask_0f), offset8);
                let w_il = _mm256_set_m128i(
                    _mm_unpackhi_epi8(lo_n, hi_n),
                    _mm_unpacklo_epi8(lo_n, hi_n),
                );
                let w_lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(w_il));
                let w_hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(w_il, 1));
                let p_lo = _mm256_madd_epi16(w_lo16, vx_lo16);
                let p_hi = _mm256_madd_epi16(w_hi16, vx_hi16);
                acc = _mm256_add_epi32(acc, _mm256_add_epi32(p_lo, p_hi));
            }

            let hi128 = _mm256_extracti128_si256(acc, 1);
            let lo128 = _mm256_castsi256_si128(acc);
            let s4 = _mm_add_epi32(lo128, hi128);
            let sh = _mm_shuffle_epi32(s4, 0b_00_01_10_11);
            let s2 = _mm_add_epi32(s4, sh);
            let shb = _mm_shufflelo_epi16(s2, 0b_01_00_11_10);
            let s1 = _mm_add_epi32(s2, shb);
            let mut gdot: i32 = _mm_cvtsi128_si32(s1);

            let simd_done = simd_iters * 32;
            let mut j = g_start + simd_done;
            while j + 1 < g_end {
                let byte = w_packed[w_off + j / 2];
                let (w0, w1) = unpack_int4(byte);
                gdot += (w0 as i32) * (x_q[j] as i32);
                gdot += (w1 as i32) * (x_q[j + 1] as i32);
                j += 2;
            }
            if j < g_end {
                let byte = w_packed[w_off + j / 2];
                let (w0, _) = unpack_int4(byte);
                gdot += (w0 as i32) * (x_q[j] as i32);
            }

            total += (gdot as f32) * gs * x_scale;
        }

        out[i] = total;
    }
}
