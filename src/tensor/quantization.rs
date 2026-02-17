// =============================================================================
// quantization.rs — Quantization functions for f32 → Int8 and Int4
// =============================================================================
//
// This module provides quantization utilities for converting floating-point
// tensors to low-precision integer representations:
//
// - **Int8 quantization**: Symmetric per-tensor quantization with AVX2 SIMD
//   acceleration for abs_max finding and vectorized quantization.
//
// - **Int4 quantization**: Per-group quantization with nibble packing, used
//   for memory-efficient weight storage (2 weights per byte).
//
// =============================================================================

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ---------------------------------------------------------------------------
// Int8 Quantization
// ---------------------------------------------------------------------------

/// Quantise an f32 vector to int8 in-place (symmetric per-tensor).
/// Returns the scale factor used.  `out` must be same length as `input`.
/// AVX2-accelerated abs_max finding + vectorized quantization when available.
pub fn quantize_f32_to_i8(input: &[f32], out: &mut [i8]) -> f32 {
    assert_eq!(input.len(), out.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { quantize_f32_to_i8_avx2(input, out) };
        }
    }
    quantize_f32_to_i8_scalar(input, out)
}

#[inline(always)]
fn quantize_f32_to_i8_scalar(input: &[f32], out: &mut [i8]) -> f32 {
    let abs_max = input
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, f32::max)
        .max(1e-12);
    let scale = abs_max / 127.0;
    let inv = 1.0 / scale;
    for (o, &v) in out.iter_mut().zip(input.iter()) {
        *o = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// AVX2 accelerated quantization: SIMD abs_max + vectorized i32→i8 packing.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_f32_to_i8_avx2(input: &[f32], out: &mut [i8]) -> f32 {
    let n = input.len();
    let chunks = n / 8;

    // SIMD abs_max: 8 floats per iteration
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFF_FFFFu32 as i32));
    let mut max_vec = _mm256_setzero_ps();
    for c in 0..chunks {
        let v = _mm256_loadu_ps(input.as_ptr().add(c * 8));
        let abs_v = _mm256_and_ps(v, sign_mask);
        max_vec = _mm256_max_ps(max_vec, abs_v);
    }
    let hi = _mm256_extractf128_ps(max_vec, 1);
    let lo = _mm256_castps256_ps128(max_vec);
    let m4 = _mm_max_ps(lo, hi);
    let m2 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    let m1 = _mm_max_ss(m2, _mm_shuffle_ps(m2, m2, 1));
    let mut abs_max = _mm_cvtss_f32(m1);
    for i in (chunks * 8)..n {
        abs_max = abs_max.max(input[i].abs());
    }
    abs_max = abs_max.max(1e-12);

    let scale = abs_max / 127.0;
    let inv = 1.0 / scale;
    let inv_v = _mm256_set1_ps(inv);
    let min_v = _mm256_set1_ps(-127.0);
    let max_v = _mm256_set1_ps(127.0);

    // SIMD quantize: 8 floats → 8 i8 per iteration
    for c in 0..chunks {
        let v = _mm256_loadu_ps(input.as_ptr().add(c * 8));
        let scaled = _mm256_mul_ps(v, inv_v);
        let clamped = _mm256_max_ps(_mm256_min_ps(scaled, max_v), min_v);
        let i32s = _mm256_cvtps_epi32(clamped);
        // Pack i32 → i16 → i8 preserving element order
        let lo128 = _mm256_castsi256_si128(i32s);
        let hi128 = _mm256_extracti128_si256(i32s, 1);
        let packed16 = _mm_packs_epi32(lo128, hi128);
        let packed8 = _mm_packs_epi16(packed16, packed16);
        // Store low 8 bytes (the 8 i8 values)
        let val = _mm_cvtsi128_si64(packed8);
        (out.as_mut_ptr().add(c * 8) as *mut i64).write_unaligned(val);
    }
    for i in (chunks * 8)..n {
        out[i] = (input[i] * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

// ---------------------------------------------------------------------------
// Int4 Quantization
// ---------------------------------------------------------------------------

pub const INT4_GROUP_SIZE: usize = 64;

/// Unpack one byte into two signed 4-bit integers.
#[inline(always)]
pub fn unpack_int4(byte: u8) -> (i8, i8) {
    let lo = (byte & 0x0F) as i8 - 8;
    let hi = ((byte >> 4) & 0x0F) as i8 - 8;
    (lo, hi)
}

/// Pack two signed 4-bit values into one byte.
#[inline(always)]
pub fn pack_int4(lo: i8, hi: i8) -> u8 {
    let lo_u = ((lo + 8) as u8) & 0x0F;
    let hi_u = ((hi + 8) as u8) & 0x0F;
    lo_u | (hi_u << 4)
}

/// Quantise f32 → Int4 with per-group scales.
/// `out` length = ceil(input.len()/2), `scales` length = ceil(input.len()/GROUP).
pub fn quantize_f32_to_i4(input: &[f32], out: &mut [u8], scales: &mut [f32]) {
    let n = input.len();
    let n_groups = (n + INT4_GROUP_SIZE - 1) / INT4_GROUP_SIZE;
    assert!(scales.len() >= n_groups);
    assert!(out.len() >= (n + 1) / 2);

    for g in 0..n_groups {
        let start = g * INT4_GROUP_SIZE;
        let end = (start + INT4_GROUP_SIZE).min(n);

        let abs_max = input[start..end]
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max)
            .max(1e-12);
        let scale = abs_max / 7.0;
        scales[g] = scale;
        let inv = 1.0 / scale;

        let mut j = start;
        while j < end {
            let v0 = (input[j] * inv).round().clamp(-8.0, 7.0) as i8;
            let v1 = if j + 1 < end {
                (input[j + 1] * inv).round().clamp(-8.0, 7.0) as i8
            } else {
                0i8
            };
            out[j / 2] = pack_int4(v0, v1);
            j += 2;
        }
    }
}
