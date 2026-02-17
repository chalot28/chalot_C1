// =============================================================================
// tensor.rs — Zero-copy TensorView + Int8 quantized MatMul kernel
// =============================================================================
use std::fmt;

// ---------------------------------------------------------------------------
// TensorView: chỉ chứa pointer + shape, không sở hữu data
// ---------------------------------------------------------------------------
/// A non-owning view into a contiguous buffer interpreted as an N-D tensor.
/// Lifetime `'a` ties the view to the backing store (mmap / Vec / slice).
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct TensorView<'a> {
    /// Raw byte slice that backs this tensor.
    data: &'a [u8],
    /// Logical shape — e.g. [rows, cols].
    shape: [usize; 4],
    /// Number of dimensions actually used (1..=4).
    ndim: usize,
}

#[allow(dead_code)]
impl<'a> TensorView<'a> {
    // -- constructors --------------------------------------------------------

    /// 1-D tensor view (vector).
    pub fn new_1d(data: &'a [u8], d0: usize) -> Self {
        assert!(
            data.len() >= d0,
            "TensorView::new_1d — buffer too small: {} < {}",
            data.len(),
            d0
        );
        Self {
            data,
            shape: [d0, 0, 0, 0],
            ndim: 1,
        }
    }

    /// 2-D tensor view (matrix).
    pub fn new_2d(data: &'a [u8], rows: usize, cols: usize) -> Self {
        let need = rows * cols;
        assert!(
            data.len() >= need,
            "TensorView::new_2d — buffer too small: {} < {}",
            data.len(),
            need
        );
        Self {
            data,
            shape: [rows, cols, 0, 0],
            ndim: 2,
        }
    }

    // -- accessors -----------------------------------------------------------

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        let mut n = 1usize;
        for i in 0..self.ndim {
            n *= self.shape[i];
        }
        n
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape[..self.ndim]
    }

    /// Interpret backing bytes as `&[i8]`.
    pub fn as_i8(&self) -> &'a [i8] {
        let len = self.numel();
        // SAFETY: i8 and u8 have identical layout; pointer alignment is 1.
        unsafe { std::slice::from_raw_parts(self.data.as_ptr() as *const i8, len) }
    }

    /// Interpret backing bytes as `&[u8]` (raw).
    pub fn as_bytes(&self) -> &'a [u8] {
        &self.data[..self.numel()]
    }

    /// Return a row slice for a 2-D tensor (as i8).
    pub fn row_i8(&self, r: usize) -> &'a [i8] {
        assert!(self.ndim == 2, "row_i8 requires 2-D tensor");
        let cols = self.shape[1];
        let start = r * cols;
        &self.as_i8()[start..start + cols]
    }
}

impl fmt::Debug for TensorView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TensorView(shape={:?}, bytes={})",
            &self.shape[..self.ndim],
            self.data.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Scale view — casts raw bytes → &[f32] (little-endian assumed)
// ---------------------------------------------------------------------------
/// Reinterpret a byte slice as `&[f32]`. Caller must guarantee alignment + size.
pub fn bytes_as_f32(data: &[u8]) -> &[f32] {
    assert!(data.len() % 4 == 0, "bytes_as_f32: length not multiple of 4");
    let ptr = data.as_ptr();
    // On x86/ARM little-endian this is safe when ptr is 4-byte aligned.
    // mmap returns page-aligned memory so the first float is fine.
    // For interior slices we check alignment at runtime.
    assert!(
        ptr as usize % std::mem::align_of::<f32>() == 0,
        "bytes_as_f32: pointer not aligned to 4 bytes"
    );
    unsafe { std::slice::from_raw_parts(ptr as *const f32, data.len() / 4) }
}

/// Mutable version.
#[allow(dead_code)]
pub fn bytes_as_f32_mut(data: &mut [u8]) -> &mut [f32] {
    assert!(data.len() % 4 == 0);
    let ptr = data.as_mut_ptr();
    assert!(ptr as usize % std::mem::align_of::<f32>() == 0);
    unsafe { std::slice::from_raw_parts_mut(ptr as *mut f32, data.len() / 4) }
}

// ---------------------------------------------------------------------------
// Quantise helpers
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
    use std::arch::x86_64::*;
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
/// * `out_dim`   — number of rows in W (output dimension)
/// * `in_dim`    — number of cols in W (input dimension)
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
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

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
    use std::arch::x86_64::*;

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
    use std::arch::aarch64::*;

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
// Utility: RMS Norm (used in every layer)
// ---------------------------------------------------------------------------
/// In-place RMS normalisation: x[i] = (x[i] / rms) * weight[i]
/// AVX2-accelerated with vectorized sum-of-squares and normalisation.
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
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / (dim as f32);
    let inv_rms = 1.0 / (ss + 1e-5f32).sqrt();
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
// Utility: Softmax
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
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
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

// ---------------------------------------------------------------------------
// Utility: RoPE (Rotary Position Embedding)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Top-K Sampler
// ---------------------------------------------------------------------------
/// Greedy argmax over logits. Returns the token id.
#[inline(always)]
pub fn sample_argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

/// Top-K sampling with temperature.
#[allow(dead_code)]
pub fn sample_top_k(logits: &mut [f32], k: usize, temperature: f32) -> usize {
    // Apply temperature
    if temperature > 0.0 {
        let inv_t = 1.0 / temperature;
        for v in logits.iter_mut() {
            *v *= inv_t;
        }
    }

    // Find top-k indices
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    indices.truncate(k);

    // Softmax over top-k
    let max_val = logits[indices[0]];
    let mut probs: Vec<f32> = indices.iter().map(|&i| (logits[i] - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // Simple random selection using a basic LCG (no external deps)
    // In production, use a proper RNG.
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let r = (seed as f32) / (u32::MAX as f32);

    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return indices[i];
        }
    }
    indices[indices.len() - 1]
}

// ===========================================================================
// Int4 Quantization Support (4-bit with group-wise scales)
// ===========================================================================
//
// Pack 2 × signed 4-bit weights per byte.  Group-wise f32 scales give
// accuracy close to Int8 while halving storage.
//
// Storage: value_stored = value + 8  (maps [-8,7] → [0,15])
//   byte = (lo_stored & 0xF) | ((hi_stored & 0xF) << 4)
// ===========================================================================

/// Number of elements sharing one f32 scale factor.
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

// ---------------------------------------------------------------------------
// matmul_int4 — Int4 weight × Int8 quantised-input kernel (AVX2 optimized)
// ---------------------------------------------------------------------------
// AVX2 path: unpacks Int4→Int8 in SIMD registers using nibble extraction,
// then uses i8→i16 widening + _mm256_madd_epi16 for accumulation.
// Processes 32 elements (16 packed bytes) per SIMD iteration with
// 2-row parallelism for cache reuse of the quantised input vector.
// ---------------------------------------------------------------------------
/// Quantized Int4 matrix–vector product with per-group weight scales.
///
/// * `w_packed`  — [out_dim × packed_cols] packed Int4 bytes
/// * `w_scales`  — [out_dim × n_groups] f32 per-group scales
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
    use std::arch::x86_64::*;

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

// ---------------------------------------------------------------------------
// Expert router helpers
// ---------------------------------------------------------------------------

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

/// Sigmoid activation.
#[inline(always)]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// ReLU activation.
#[inline(always)]
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

// ===========================================================================
// ADVANCED ACTIVATION FUNCTIONS
// ===========================================================================

/// SiLU (Sigmoid Linear Unit): x × σ(x). Core activation for SwiGLU.
#[inline(always)]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GELU (Gaussian Error Linear Unit) — exact approximation.
/// Superior gradient flow for deep reasoning layers vs ReLU.
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh())
}

/// Fused SwiGLU: gate[i] = SiLU(gate[i]) × up[i].
/// AVX2-accelerated with fast vectorized exp for SiLU computation.
pub fn swiglu_fused(gate: &mut [f32], up: &[f32]) {
    assert_eq!(gate.len(), up.len());

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
    for i in 0..gate.len() {
        let g = gate[i];
        gate[i] = (g / (1.0 + (-g).exp())) * up[i];
    }
}

/// AVX2 SwiGLU with fast vectorized exp for sigmoid computation.
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

        // SiLU(g) = g * σ(g) = g / (1 + exp(-g))
        let neg_g = _mm256_sub_ps(_mm256_setzero_ps(), g);
        let exp_neg_g = fast_exp_avx2(neg_g);
        let sigmoid_g = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg_g));
        let silu_g = _mm256_mul_ps(g, sigmoid_g);

        // SwiGLU: silu(gate) * up
        let result = _mm256_mul_ps(silu_g, u);
        _mm256_storeu_ps(gate.as_mut_ptr().add(base), result);
    }
    // Scalar tail
    for i in (chunks * 8)..n {
        let g = gate[i];
        gate[i] = (g / (1.0 + (-g).exp())) * up[i];
    }
}

/// Fast vectorized exp approximation using range reduction + degree-5 Taylor.
/// Max relative error ~2×10⁻⁶ for |x| < 88. Used by softmax and SwiGLU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn fast_exp_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let ln2   = _mm256_set1_ps(0.693147180559945_f32);
    let log2e = _mm256_set1_ps(1.44269504088896_f32);
    let c0 = _mm256_set1_ps(1.0);
    let c1 = _mm256_set1_ps(1.0);
    let c2 = _mm256_set1_ps(0.5);
    let c3 = _mm256_set1_ps(0.166666667);
    let c4 = _mm256_set1_ps(0.041666666);
    let c5 = _mm256_set1_ps(0.008333333);

    // Clamp to avoid overflow/underflow
    let x = _mm256_max_ps(
        _mm256_set1_ps(-88.0),
        _mm256_min_ps(_mm256_set1_ps(88.0), x),
    );

    // n = round(x * log2e) — integer part of exponent
    let n = _mm256_round_ps(
        _mm256_mul_ps(x, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );

    // r = x - n * ln2 — reduced range: |r| ≤ ln2/2 ≈ 0.347
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

// ===========================================================================
// ATTENTION HELPERS
// ===========================================================================

/// Logit soft-capping: cap × tanh(x / cap).
/// Bounds attention scores to (-cap, +cap), preventing entropy collapse
/// in deep layers. Used by Gemma 2 and modern reasoning models.
#[inline(always)]
pub fn logit_soft_cap(x: f32, cap: f32) -> f32 {
    cap * (x / cap).tanh()
}

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

// ===========================================================================
// ADVANCED SAMPLING STRATEGIES
// ===========================================================================

/// Top-p (nucleus) sampling: sample from the smallest token set whose
/// cumulative probability exceeds `p`. Superior to fixed top-k for reasoning
/// because the candidate pool adapts to model confidence.
pub fn sample_top_p(logits: &mut [f32], p: f32, temperature: f32) -> usize {
    let n = logits.len();
    if n == 0 { return 0; }

    if temperature > 0.0 && temperature != 1.0 {
        let inv_t = 1.0 / temperature;
        for v in logits.iter_mut() { *v *= inv_t; }
    }

    softmax(logits);

    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

    let mut cumulative = 0.0f32;
    let mut cutoff = n;
    for (rank, &idx) in indices.iter().enumerate() {
        cumulative += logits[idx];
        if cumulative > p {
            cutoff = rank + 1;
            break;
        }
    }
    indices.truncate(cutoff);

    let mut probs: Vec<f32> = indices.iter().map(|&i| logits[i]).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 1e-12 {
        for val in probs.iter_mut() { *val /= sum; }
    }

    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let r = (seed as f32) / (u32::MAX as f32);

    let mut cum = 0.0f32;
    for (i, &prob) in probs.iter().enumerate() {
        cum += prob;
        if r < cum { return indices[i]; }
    }
    indices[indices.len() - 1]
}

/// Apply repetition penalty: reduce probability of recently generated tokens.
/// penalty > 1.0 discourages repetition; typical range: 1.05–1.3
pub fn apply_repetition_penalty(logits: &mut [f32], history: &[usize], penalty: f32) {
    for &tok in history {
        if tok < logits.len() {
            if logits[tok] > 0.0 {
                logits[tok] /= penalty;
            } else {
                logits[tok] *= penalty;
            }
        }
    }
}

/// Min-P sampling: keeps tokens with prob ≥ min_p × max_prob.
/// Adaptive cutoff that automatically scales with model confidence.
/// More principled than fixed top-k or top-p for variable-difficulty tasks.
pub fn sample_min_p(logits: &mut [f32], min_p: f32, temperature: f32) -> usize {
    let n = logits.len();
    if n == 0 { return 0; }

    if temperature > 0.0 && temperature != 1.0 {
        let inv_t = 1.0 / temperature;
        for v in logits.iter_mut() { *v *= inv_t; }
    }

    softmax(logits);

    let max_prob = logits.iter().cloned().fold(0.0f32, f32::max);
    let threshold = max_prob * min_p;

    let mut indices: Vec<usize> = (0..n).filter(|&i| logits[i] >= threshold).collect();
    if indices.is_empty() {
        return logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
    }

    let mut probs: Vec<f32> = indices.iter().map(|&i| logits[i]).collect();
    let sum: f32 = probs.iter().sum();
    if sum > 1e-12 {
        for val in probs.iter_mut() { *val /= sum; }
    }

    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let r = (seed as f32) / (u32::MAX as f32);

    let mut cum = 0.0f32;
    for (i, &prob) in probs.iter().enumerate() {
        cum += prob;
        if r < cum { return indices[i]; }
    }
    indices[indices.len() - 1]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_roundtrip() {
        let x = vec![1.0f32, -0.5, 0.0, 0.25, -1.0];
        let mut q = vec![0i8; x.len()];
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
