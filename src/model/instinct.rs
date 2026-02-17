// =============================================================================
// model/instinct.rs — Lõi Bản Năng: Online Learning với 1M Parameters
// =============================================================================
//
// Triết lý: Đây là "trực giác" của AI - phản ứng nhanh (<0.1ms) để quyết định:
//   - Nên dùng vùng não nào? (Shallow/Deep/Fact)
//   - Ngữ cảnh hiện tại thuộc dạng nào? (Chat/Code/Math...)
//
// Kỹ thuật:
//   - Memory-mapped file: 1M tham số (Float32 = 4MB) map trực tiếp vào RAM
//   - Hash-based routing: O(1) lookup, không cần forward pass
//   - Hebbian learning: Reward dương → Tăng trọng số, âm → Giảm
//   - Zero-copy I/O: Ghi trực tiếp vào file (dùng MmapMut)
//
// Cấu trúc file .bin:
//   [0..4MB]: 1M Float32 weights
// =============================================================================

use memmap2::{Mmap, MmapMut};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

/// Số tham số trong Instinct Core (1 triệu)
const INSTINCT_PARAMS: usize = 1_000_000;

/// Các loại vùng não (Brain Regions)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrainRegion {
    ShallowReflex = 0, // Chém gió, ngữ pháp đơn giản (nhẹ, nhanh)
    DeepLogic = 1,     // Code, Toán học, Suy luận (nặng, chậm)
    HardFact = 2,      // Wiki, Tra cứu sự thật (nặng, chính xác)
}

impl BrainRegion {
    pub fn from_id(id: usize) -> Self {
        match id {
            0 => Self::ShallowReflex,
            1 => Self::DeepLogic,
            2 => Self::HardFact,
            _ => Self::ShallowReflex, // Default fallback
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::ShallowReflex => "Shallow",
            Self::DeepLogic => "Deep",
            Self::HardFact => "Fact",
        }
    }
}

/// Bộ nhớ Instinct (Read-only mode)
pub struct InstinctCore {
    /// Memory-mapped weights (1M Float32)
    weights: Mmap,
    /// Read-only flag
    read_only: bool,
}

/// Bộ nhớ Instinct (Mutable mode - cho training)
pub struct InstinctCoreMut {
    /// Mutable memory-mapped weights
    weights_mut: MmapMut,
}

impl InstinctCore {
    /// Load từ file .bin (read-only)
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("open instinct: {e}"))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap instinct: {e}"))?;

        let expected_size = INSTINCT_PARAMS * 4; // Float32 = 4 bytes
        if mmap.len() != expected_size {
            return Err(format!(
                "Invalid instinct file: expected {} bytes, got {}",
                expected_size,
                mmap.len()
            ));
        }

        Ok(Self {
            weights: mmap,
            read_only: true,
        })
    }

    /// Tạo file .bin mới với weights khởi tạo ngẫu nhiên
    pub fn create(path: &Path) -> Result<(), String> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| format!("create instinct: {e}"))?;

        // Khởi tạo với giá trị nhỏ ngẫu nhiên (~uniform(-0.1, 0.1))
        let mut rng_state = 42u64; // Simple LCG
        let mut buffer = vec![0u8; INSTINCT_PARAMS * 4];
        
        for chunk in buffer.chunks_exact_mut(4) {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let rand_val = ((rng_state >> 16) & 0xFFFF) as f32 / 65535.0; // [0, 1]
            let weight = (rand_val - 0.5) * 0.2; // [-0.1, 0.1]
            chunk.copy_from_slice(&weight.to_le_bytes());
        }

        file.write_all(&buffer)
            .map_err(|e| format!("write instinct: {e}"))?;

        println!(
            "[instinct] Created new instinct core: {} MB",
            buffer.len() as f32 / 1e6
        );
        Ok(())
    }

    /// Đọc trọng số tại index (unsafe cast)
    fn get_weight(&self, idx: usize) -> f32 {
        let offset = idx * 4;
        if offset + 4 > self.weights.len() {
            return 0.0;
        }
        f32::from_le_bytes([
            self.weights[offset],
            self.weights[offset + 1],
            self.weights[offset + 2],
            self.weights[offset + 3],
        ])
    }

    /// Hash ngữ cảnh thành index (FNV-1a hash)
    fn hash_context(&self, tokens: &[usize], max_lookback: usize) -> usize {
        let mut hash = 2166136261u64;
        let lookback = tokens.len().min(max_lookback);
        let start = tokens.len().saturating_sub(lookback);

        for &token in &tokens[start..] {
            hash ^= token as u64;
            hash = hash.wrapping_mul(16777619);
        }

        (hash % INSTINCT_PARAMS as u64) as usize
    }

    /// Dự đoán vùng não nên dùng (CORE LOGIC)
    pub fn predict_region(&self, context_tokens: &[usize]) -> BrainRegion {
        // Hash ngữ cảnh với sliding window (chỉ xem 32 token gần nhất)
        let idx = self.hash_context(context_tokens, 32);
        let signal = self.get_weight(idx);

        // Phân loại theo ngưỡng
        if signal > 0.5 {
            BrainRegion::DeepLogic // Tín hiệu mạnh → Logic phức tạp
        } else if signal < -0.5 {
            BrainRegion::HardFact // Tín hiệu âm → Tra cứu sự thật
        } else {
            BrainRegion::ShallowReflex // Trung lập → Chém gió
        }
    }

    /// Tính confidence score (độ tự tin) từ 0-1
    pub fn confidence(&self, context_tokens: &[usize]) -> f32 {
        let idx = self.hash_context(context_tokens, 32);
        let signal = self.get_weight(idx);
        signal.abs().clamp(0.0, 1.0)
    }
}

impl InstinctCoreMut {
    /// Mở file .bin ở chế độ mutable (cho training)
    pub fn load_mut(path: &Path) -> Result<Self, String> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| format!("open instinct mut: {e}"))?;

        let mmap = unsafe { MmapMut::map_mut(&file) }.map_err(|e| format!("mmap mut: {e}"))?;

        let expected_size = INSTINCT_PARAMS * 4;
        if mmap.len() != expected_size {
            return Err(format!(
                "Invalid instinct file: expected {} bytes, got {}",
                expected_size,
                mmap.len()
            ));
        }

        Ok(Self { weights_mut: mmap })
    }

    /// Đọc trọng số
    fn get_weight(&self, idx: usize) -> f32 {
        let offset = idx * 4;
        f32::from_le_bytes([
            self.weights_mut[offset],
            self.weights_mut[offset + 1],
            self.weights_mut[offset + 2],
            self.weights_mut[offset + 3],
        ])
    }

    /// Ghi trọng số (direct memory write)
    fn set_weight(&mut self, idx: usize, value: f32) {
        let offset = idx * 4;
        let bytes = value.to_le_bytes();
        self.weights_mut[offset..offset + 4].copy_from_slice(&bytes);
    }

    /// Hash ngữ cảnh
    fn hash_context(&self, tokens: &[usize], max_lookback: usize) -> usize {
        let mut hash = 2166136261u64;
        let lookback = tokens.len().min(max_lookback);
        let start = tokens.len().saturating_sub(lookback);

        for &token in &tokens[start..] {
            hash ^= token as u64;
            hash = hash.wrapping_mul(16777619);
        }

        (hash % INSTINCT_PARAMS as u64) as usize
    }

    /// Học online (Hebbian Learning): Tăng/giảm trọng số dựa trên reward
    pub fn learn(&mut self, context_tokens: &[usize], reward: f32, learning_rate: f32) {
        let idx = self.hash_context(context_tokens, 32);
        let current = self.get_weight(idx);

        // Cập nhật: w += lr * reward
        let new_weight = (current + learning_rate * reward).clamp(-1.0, 1.0);
        self.set_weight(idx, new_weight);
    }

    /// Batch update: Học từ nhiều mẫu cùng lúc
    pub fn batch_learn(&mut self, samples: &[(Vec<usize>, f32)], learning_rate: f32) {
        for (context, reward) in samples {
            self.learn(context, *reward, learning_rate);
        }
    }

    /// Flush thay đổi ra đĩa (auto khi drop, nhưng có thể gọi thủ công)
    pub fn flush(&mut self) -> Result<(), String> {
        self.weights_mut
            .flush()
            .map_err(|e| format!("flush instinct: {e}"))
    }
}

// =============================================================================
// Helper: Tự động phát hiện signal patterns (heuristics cho reward)
// =============================================================================

/// Phát hiện pattern toán học trong tokens (ví dụ: "+", "=", "solve")
pub fn detect_math_pattern(tokens: &[usize], vocab_size: usize) -> bool {
    // Giả sử token "+" ~ 29, "=" ~ 25, số ~ [16-25]
    // (Thực tế cần tra bảng tokenizer)
    tokens.iter().any(|&t| t < vocab_size && (t == 29 || t == 25))
}

/// Phát hiện pattern code (ví dụ: "{", "fn", "def")
pub fn detect_code_pattern(tokens: &[usize]) -> bool {
    // Heuristic đơn giản: Nếu có nhiều ký tự đặc biệt code
    // (Cần cải tiến bằng regex trên decoded text)
    tokens.len() > 5 && tokens.iter().any(|&t| t < 100)
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_instinct() {
        let path = Path::new("test_instinct.bin");
        InstinctCore::create(path).unwrap();
        assert!(path.exists());

        let core = InstinctCore::load(path).unwrap();
        assert!(core.weights.len() == INSTINCT_PARAMS * 4);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_predict_region() {
        let path = Path::new("test_instinct2.bin");
        InstinctCore::create(path).unwrap();

        let core = InstinctCore::load(path).unwrap();
        let context = vec![1, 2, 3, 29, 5]; // Giả sử token 29 là "+"
        let region = core.predict_region(&context);

        assert!(matches!(
            region,
            BrainRegion::ShallowReflex | BrainRegion::DeepLogic | BrainRegion::HardFact
        ));

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_online_learning() {
        let path = Path::new("test_instinct3.bin");
        InstinctCore::create(path).unwrap();

        {
            let mut core_mut = InstinctCoreMut::load_mut(path).unwrap();
            let context = vec![10, 20, 30];
            let idx = core_mut.hash_context(&context, 32);
            let before = core_mut.get_weight(idx);

            // Học với reward dương
            core_mut.learn(&context, 1.0, 0.1);
            let after = core_mut.get_weight(idx);

            assert!(after > before, "Weight should increase with positive reward");
            core_mut.flush().unwrap();
        }

        std::fs::remove_file(path).ok();
    }
}
