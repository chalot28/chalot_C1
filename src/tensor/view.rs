// =============================================================================
// tensor/view.rs — Zero-copy TensorView + byte conversion utilities
// =============================================================================

use std::fmt;

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
