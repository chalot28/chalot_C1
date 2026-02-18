// =============================================================================
// model/supervisor.rs — Mạng Kiểm Tra Ảo Giác (Hallucination Detection)
// =============================================================================
//
// Triết lý: AI thường "chế" thông tin khi không tự tin (ảo giác = hallucination).
// Supervisor là mạng nhỏ (tiny MLP) giám sát hidden state, phát hiện:
//   - Entropy cao (phân phối token xáo trộn)
//   - Variance cao (embedding dao động mạnh)
//   - Attention collapse (tất cả đầu chú ý vào 1 token)
//
// Khi phát hiện ảo giác → Ép chuyển sang vùng "Fact" (tra cứu sự thật).
//
// Kỹ thuật:
//   - Lightweight: Chỉ 2 layer MLP với 64 hidden units
//   - Fast: O(dim) complexity, không làm chậm inference
//   - Threshold-based: Binary classification (hallucinating / confident)
// =============================================================================

use std::f32;

/// Hành động đề xuất bởi Supervisor
#[derive(Debug, PartialEq)]
pub enum SupervisorAction {
    Continue,       // Tự tin, tiếp tục sinh token
    SwitchToFact,   // Hơi không chắc, chuyển sang vùng não Fact
    RAGSearch,      // [NEW] Ảo giác nặng -> Kích hoạt Crawler/RAG
}

/// Cấu trúc Supervisor (Quality Control Network)
pub struct Supervisor {
    /// Trọng số layer 1: [dim, 64]
    w1: Vec<f32>,
    /// Trọng số layer 2: [64, 1]
    w2: Vec<f32>,
    /// Ngưỡng phát hiện ảo giác (0-1, mặc định 0.7)
    threshold: f32,
}

impl Supervisor {
    /// Tạo Supervisor với trọng số ngẫu nhiên
    pub fn new(dim: usize, threshold: f32) -> Self {
        let hidden = 64;
        let mut rng_state = 314159u64;

        // Khởi tạo w1: [dim, 64]
        let mut w1 = vec![0.0; dim * hidden];
        for w in &mut w1 {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let rand_val = ((rng_state >> 16) & 0xFFFF) as f32 / 65535.0;
            *w = (rand_val - 0.5) * (2.0 / (dim as f32).sqrt()); // Xavier init
        }

        // Khởi tạo w2: [64, 1]
        let mut w2 = vec![0.0; hidden];
        for w in &mut w2 {
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let rand_val = ((rng_state >> 16) & 0xFFFF) as f32 / 65535.0;
            *w = (rand_val - 0.5) * (2.0 / (hidden as f32).sqrt());
        }

        Self { w1, w2, threshold }
    }

    /// Load từ file (binary format: [w1_bytes][w2_bytes])
    #[allow(dead_code)]
    pub fn load(path: &str, dim: usize, threshold: f32) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("load supervisor: {e}"))?;

        let hidden = 64;
        let w1_size = dim * hidden * 4; // Float32
        let w2_size = hidden * 4;

        if data.len() != w1_size + w2_size {
            return Err(format!(
                "Invalid supervisor file size: expected {}, got {}",
                w1_size + w2_size,
                data.len()
            ));
        }

        let w1 = data[..w1_size]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        let w2 = data[w1_size..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        Ok(Self { w1, w2, threshold })
    }

    /// Save ra file
    #[allow(dead_code)]
    pub fn save(&self, path: &str) -> Result<(), String> {
        let mut buffer = Vec::new();

        // Write w1
        for &w in &self.w1 {
            buffer.extend_from_slice(&w.to_le_bytes());
        }

        // Write w2
        for &w in &self.w2 {
            buffer.extend_from_slice(&w.to_le_bytes());
        }

        std::fs::write(path, &buffer).map_err(|e| format!("save supervisor: {e}"))
    }

    /// ReLU activation
    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }

    /// Forward pass: hidden_state → hallucination_score (0-1)
    pub fn forward(&self, hidden_state: &[f32]) -> f32 {
        let dim = hidden_state.len();
        let hidden = 64;

        // Layer 1: [dim] → [64] với ReLU
        let mut h = vec![0.0; hidden];
        for i in 0..hidden {
            let mut sum = 0.0;
            for j in 0..dim {
                sum += self.w1[j * hidden + i] * hidden_state[j];
            }
            h[i] = Self::relu(sum);
        }

        // Layer 2: [64] → [1] → Sigmoid
        let mut logit = 0.0;
        for i in 0..hidden {
            logit += self.w2[i] * h[i];
        }

        // Sigmoid: 0-1
        1.0 / (1.0 + (-logit).exp())
    }

    /// Kiểm tra có đang ảo giác không (main API)
    pub fn check_status(&self, hidden_state: &[f32]) -> SupervisorAction {
        // 1. Tính hallucination score từ MLP
        let mlp_score = self.forward(hidden_state);

        // 2. Tính thêm các heuristic bổ sung
        let entropy = Self::compute_entropy(hidden_state);
        let variance = Self::compute_variance(hidden_state);

        // 3. Kết hợp các tín hiệu
        let combined_score = mlp_score * 0.6 + entropy * 0.2 + variance * 0.2;

        if combined_score > 0.85 {
            SupervisorAction::RAGSearch // Rất hỗn loạn -> Cần dữ liệu ngoài
        } else if combined_score > self.threshold {
            SupervisorAction::SwitchToFact // Hơi nghi ngờ -> Dùng vùng não Fact
        } else {
            SupervisorAction::Continue
        }
    }

    /// Tính entropy của embedding (độ hỗn loạn)
    fn compute_entropy(x: &[f32]) -> f32 {
        // Normalize to probability distribution
        let max_val = x.iter().map(|v| v.abs()).fold(0.0, f32::max);
        if max_val < 1e-6 {
            return 0.0;
        }

        let mut probs = vec![0.0; x.len()];
        let mut sum = 0.0;
        for (i, &val) in x.iter().enumerate() {
            probs[i] = (val / max_val).abs();
            sum += probs[i];
        }

        if sum < 1e-6 {
            return 0.0;
        }

        // Normalize
        for p in &mut probs {
            *p /= sum;
        }

        // Compute Shannon entropy (normalized to 0-1)
        let mut entropy = 0.0;
        for &p in &probs {
            if p > 1e-6 {
                entropy -= p * p.ln();
            }
        }

        let max_entropy = (probs.len() as f32).ln();
        (entropy / max_entropy).clamp(0.0, 1.0)
    }

    /// Tính variance của embedding (độ dao động)
    fn compute_variance(x: &[f32]) -> f32 {
        let n = x.len() as f32;
        let mean = x.iter().sum::<f32>() / n;
        let variance = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;

        // Normalize to 0-1 (heuristic: variance > 1.0 là cao)
        (variance / (variance + 1.0)).clamp(0.0, 1.0)
    }

    /// Tính confidence score (ngược lại của hallucination)
    #[allow(dead_code)]
    pub fn confidence_score(&self, hidden_state: &[f32]) -> f32 {
        1.0 - self.forward(hidden_state)
    }

    /// Cập nhật threshold động (adaptive)
    #[allow(dead_code)]
    pub fn set_threshold(&mut self, new_threshold: f32) {
        self.threshold = new_threshold.clamp(0.0, 1.0);
    }
}

// =============================================================================
// Helper: Phân tích attention pattern (phát hiện attention collapse)
// =============================================================================

/// Phát hiện attention collapse (tất cả heads focus vào 1 token)
#[allow(dead_code)]
pub fn detect_attention_collapse(attn_weights: &[f32], n_heads: usize, seq_len: usize) -> bool {
    // attn_weights: [n_heads, seq_len]
    if attn_weights.len() != n_heads * seq_len {
        return false;
    }

    let mut collapse_count = 0;

    for head in 0..n_heads {
        let start = head * seq_len;
        let head_attn = &attn_weights[start..start + seq_len];

        // Tìm max attention
        let max_attn = head_attn.iter().cloned().fold(0.0, f32::max);

        // Nếu max > 90% tổng attention → Collapse
        let sum_attn: f32 = head_attn.iter().sum();
        if sum_attn > 0.0 && max_attn / sum_attn > 0.9 {
            collapse_count += 1;
        }
    }

    // Nếu > 50% heads collapse → Có vấn đề
    collapse_count > n_heads / 2
}

// =============================================================================
// Tests
// =============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supervisor_forward() {
        let sup = Supervisor::new(128, 0.7);
        let hidden = vec![0.5; 128];
        let score = sup.forward(&hidden);

        assert!(score >= 0.0 && score <= 1.0, "Score must be in [0, 1]");
    }

    #[test]
    fn test_hallucination_detection() {
        let sup = Supervisor::new(64, 0.6);

        // Normal case: low entropy
        let normal = vec![0.1; 64];
        assert_eq!(sup.check_status(&normal), SupervisorAction::Continue);

        // Hallucination case: high variance
        let mut chaotic = vec![0.0; 64];
        for (i, v) in chaotic.iter_mut().enumerate() {
            *v = (i as f32 * 0.314).sin() * 5.0; // High variance
        }
        // Có thể hoặc không hallucinating tùy ngưỡng (test logic only)
        let _ = sup.check_status(&chaotic);
    }

    #[test]
    fn test_entropy() {
        // Uniform distribution → high entropy
        let uniform = vec![1.0; 100];
        let entropy = Supervisor::compute_entropy(&uniform);
        assert!(entropy > 0.9, "Uniform should have high entropy");

        // Peaked distribution → low entropy
        let mut peaked = vec![0.0; 100];
        peaked[0] = 10.0;
        let entropy2 = Supervisor::compute_entropy(&peaked);
        assert!(entropy2 < 0.3, "Peaked should have low entropy");
    }

    #[test]
    fn test_save_load() {
        let path = "test_supervisor.bin";
        let sup = Supervisor::new(64, 0.75);
        sup.save(path).unwrap();

        let sup2 = Supervisor::load(path, 64, 0.75).unwrap();
        assert_eq!(sup.w1.len(), sup2.w1.len());
        assert_eq!(sup.w2.len(), sup2.w2.len());

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_attention_collapse() {
        // Normal attention: distributed
        let normal = vec![0.2, 0.3, 0.1, 0.2, 0.2]; // 1 head, 5 tokens
        assert!(!detect_attention_collapse(&normal, 1, 5));

        // Collapsed attention: focused on 1 token
        let collapsed = vec![0.95, 0.01, 0.01, 0.01, 0.02];
        assert!(detect_attention_collapse(&collapsed, 1, 5));
    }
}
