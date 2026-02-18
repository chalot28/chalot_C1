// =============================================================================
// model/memory.rs — Paged KV Cache với Int8 Quantization (<100MB RAM)
// =============================================================================
//
// Triết lý: Thay vì lưu toàn bộ KV cache trong RAM (tốn ~512MB cho 8K ctx),
// ta chia nhỏ thành các "trang" (pages) 256 token, chỉ giữ những trang
// đang dùng trong RAM, còn lại swap ra đĩa hoặc nén Int8.
// 
// Kỹ thuật chính:
// - Int8 quantization: Giảm 4x kích thước (Float32 → Int8)
// - Paging: Chỉ load các trang active (sliding window)
// - LRU eviction: Đẩy trang cũ ra khi RAM đầy
// =============================================================================

use std::collections::{HashMap, VecDeque};

/// Kích thước 1 trang (số token)
#[allow(dead_code)]
const PAGE_SIZE: usize = 256;

/// Số trang tối đa trong RAM (256 token × 16 pages = 4K context)
#[allow(dead_code)]
const MAX_PAGES_IN_RAM: usize = 16;

/// Một trang KV cache đã được lượng tử hóa Int8
#[derive(Clone)]
pub struct KVPage {
    /// Keys: [page_size, dim] → Int8
    #[allow(dead_code)]
    pub k: Vec<i8>,
    /// Values: [page_size, dim] → Int8
    #[allow(dead_code)]
    pub v: Vec<i8>,
    /// Scale factors cho dequantization
    #[allow(dead_code)]
    pub k_scale: Vec<f32>,
    #[allow(dead_code)]
    pub v_scale: Vec<f32>,
    /// Thứ tự truy cập (cho LRU)
    #[allow(dead_code)]
    pub last_access: u64,
}

impl KVPage {
    /// Tạo trang trống
    pub fn new(dim: usize) -> Self {
        Self {
            k: vec![0; PAGE_SIZE * dim],
            v: vec![0; PAGE_SIZE * dim],
            k_scale: vec![1.0; PAGE_SIZE],
            v_scale: vec![1.0; PAGE_SIZE],
            last_access: 0,
        }
    }

    /// Quantize Float32 → Int8 (per-row scaling)
    #[allow(dead_code)]
    pub fn quantize_row(row: &[f32], out: &mut [i8], scale: &mut f32) {
        let max_val = row.iter().map(|x| x.abs()).fold(0.0, f32::max);
        *scale = if max_val > 0.0 { max_val / 127.0 } else { 1.0 };
        
        for (i, &val) in row.iter().enumerate() {
            out[i] = (val / *scale).clamp(-128.0, 127.0) as i8;
        }
    }

    /// Dequantize Int8 → Float32
    #[allow(dead_code)]
    pub fn dequantize_row(row: &[i8], out: &mut [f32], scale: f32) {
        for (i, &val) in row.iter().enumerate() {
            out[i] = val as f32 * scale;
        }
    }

    /// Ghi 1 token vào trang (tại vị trí offset)
    #[allow(dead_code)]
    pub fn write_token(&mut self, offset: usize, k: &[f32], v: &[f32], timestamp: u64) {
        let dim = k.len();
        let k_start = offset * dim;
        let v_start = offset * dim;

        Self::quantize_row(k, &mut self.k[k_start..k_start + dim], &mut self.k_scale[offset]);
        Self::quantize_row(v, &mut self.v[v_start..v_start + dim], &mut self.v_scale[offset]);
        self.last_access = timestamp;
    }

    /// Đọc 1 token từ trang
    #[allow(dead_code)]
    pub fn read_token(&self, offset: usize, k_out: &mut [f32], v_out: &mut [f32]) {
        let dim = k_out.len();
        let k_start = offset * dim;
        let v_start = offset * dim;

        Self::dequantize_row(&self.k[k_start..k_start + dim], k_out, self.k_scale[offset]);
        Self::dequantize_row(&self.v[v_start..v_start + dim], v_out, self.v_scale[offset]);
    }
}

/// Quản lý KV cache với cơ chế paging
pub struct PagedKVCache {
    /// Kích thước embedding
    pub dim: usize,
    /// Số layer
    #[allow(dead_code)]
    pub n_layers: usize,
    /// Bộ nhớ trang: [layer_id][page_id] → KVPage
    pub pages: Vec<HashMap<usize, KVPage>>,
    /// Hàng đợi LRU: theo dõi thứ tự truy cập
    #[allow(dead_code)]
    pub lru_queue: VecDeque<(usize, usize)>, // (layer_id, page_id)
    /// Bộ đếm timestamp
    #[allow(dead_code)]
    pub timestamp: u64,
}

impl PagedKVCache {
    /// Khởi tạo cache trống
    /// kv_dim: Kích thước vector K/V (với GQA, kv_dim = head_dim * n_kv_heads)
    pub fn new(kv_dim: usize, n_layers: usize) -> Self {
        Self {
            dim: kv_dim,
            n_layers,
            pages: (0..n_layers).map(|_| HashMap::new()).collect(),
            lru_queue: VecDeque::new(),
            timestamp: 0,
        }
    }

    /// Tính toán RAM usage (MB)
    pub fn memory_mb(&self) -> f32 {
        let total_pages: usize = self.pages.iter().map(|m| m.len()).sum();
        let bytes_per_page = PAGE_SIZE * self.dim * 2 + PAGE_SIZE * 8; // Int8 KV + Float scales
        (total_pages * bytes_per_page) as f32 / 1e6
    }

    /// Evict trang cũ nhất nếu RAM đầy (LRU policy)
    #[allow(dead_code)]
    fn evict_if_full(&mut self) {
        let total_pages: usize = self.pages.iter().map(|m| m.len()).sum();
        if total_pages >= MAX_PAGES_IN_RAM * self.n_layers {
            if let Some((layer_id, page_id)) = self.lru_queue.pop_front() {
                self.pages[layer_id].remove(&page_id);
            }
        }
    }

    /// Lấy hoặc tạo trang (auto-eviction)
    #[allow(dead_code)]
    fn get_or_create_page(&mut self, layer_id: usize, page_id: usize) -> &mut KVPage {
        self.evict_if_full();
        self.timestamp += 1;

        // Update LRU queue
        self.lru_queue.retain(|(l, p)| !(*l == layer_id && *p == page_id));
        self.lru_queue.push_back((layer_id, page_id));

        self.pages[layer_id]
            .entry(page_id)
            .or_insert_with(|| KVPage::new(self.dim))
    }

    /// Ghi KV cho 1 token vào cache
    #[allow(dead_code)]
    pub fn write(&mut self, layer_id: usize, pos: usize, k: &[f32], v: &[f32]) {
        let page_id = pos / PAGE_SIZE;
        let offset = pos % PAGE_SIZE;
        let timestamp = self.timestamp;

        let page = self.get_or_create_page(layer_id, page_id);
        page.write_token(offset, k, v, timestamp);
    }

    /// Đọc KV từ cache (trả về None nếu trang không tồn tại)
    #[allow(dead_code)]
    pub fn read(&self, layer_id: usize, pos: usize, k_out: &mut [f32], v_out: &mut [f32]) -> bool {
        let page_id = pos / PAGE_SIZE;
        let offset = pos % PAGE_SIZE;

        if let Some(page) = self.pages[layer_id].get(&page_id) {
            page.read_token(offset, k_out, v_out);
            true
        } else {
            false
        }
    }

    /// Xóa toàn bộ cache (reset conversation)
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        for page_map in &mut self.pages {
            page_map.clear();
        }
        self.lru_queue.clear();
        self.timestamp = 0;
    }

    /// Stats: Số trang đang active
    pub fn active_pages(&self) -> usize {
        self.pages.iter().map(|m| m.len()).sum()
    }
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantization() {
        let input = vec![0.5, -1.2, 3.4, 0.0];
        let mut quantized = vec![0i8; 4];
        let mut scale = 0.0;

        KVPage::quantize_row(&input, &mut quantized, &mut scale);

        let mut dequantized = vec![0.0f32; 4];
        KVPage::dequantize_row(&quantized, &mut dequantized, scale);

        for (orig, deq) in input.iter().zip(&dequantized) {
            assert!((orig - deq).abs() < 0.1, "Quantization error too large");
        }
    }

    #[test]
    fn test_paging() {
        let mut cache = PagedKVCache::new(64, 4); // kv_dim = 64
        let k = vec![1.0; 64];
        let v = vec![2.0; 64];

        // Write token at position 300 (page 1)
        cache.write(0, 300, &k, &v);

        // Read back
        let mut k_out = vec![0.0; 64];
        let mut v_out = vec![0.0; 64];
        assert!(cache.read(0, 300, &mut k_out, &mut v_out));

        // Check approximate equality (due to quantization)
        assert!((k_out[0] - 1.0).abs() < 0.1);
        assert!((v_out[0] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_lru_eviction() {
        let mut cache = PagedKVCache::new(32, 1); // kv_dim = 32
        let k = vec![1.0; 32];
        let v = vec![2.0; 32];

        // Fill cache beyond limit (force eviction)
        for i in 0..(MAX_PAGES_IN_RAM + 5) {
            cache.write(0, i * PAGE_SIZE, &k, &v);
        }

        // Should have evicted old pages
        assert!(cache.active_pages() <= MAX_PAGES_IN_RAM);
    }
}
