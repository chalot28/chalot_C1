// =============================================================================
// model/config.rs — Runtime Model Configuration
// =============================================================================

use super::constants::{MAX_SEQ_LEN, DEPTH_ROUTER_HIDDEN};
use super::header::FileHeader;
use crate::tensor::INT4_GROUP_SIZE;

/// ModelConfig — runtime-friendly copy of header values
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub dim: usize,
    pub hidden_dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub is_quantized: bool,
    #[allow(dead_code)]
    pub is_bitnet: bool,      // [NEW] BitNet b1.58 mode (-1, 0, 1)
    #[allow(dead_code)]
    pub n_kv_heads: usize,    // [NEW] Group Query Attention (GQA)
    // MoE
    pub n_experts: usize,
    pub top_k: usize,
    // Int4
    pub int4_group_size: usize,
    // Adaptive depth
    pub depth_router_layer: usize,
    // Architecture
    pub tri_layer_mode: bool,
    #[allow(dead_code)]
    pub speculative_steps: usize, // [NEW] Số token dự đoán trước (Drafting)
    // RoPE
    pub rope_theta: f32,      // RoPE frequency base (10000 for default, 1000000 for Qwen)
}

impl ModelConfig {
    pub fn from_header(h: &FileHeader) -> Self {
        let dim = h.dim as usize;
        let n_heads = h.n_heads as usize;
        let n_experts = if h.n_experts >= 2 { h.n_experts as usize } else { 1 };
        let top_k = if h.top_k >= 1 { h.top_k as usize } else { n_experts.min(2) };
        let group = if h.int4_group_size > 0 { h.int4_group_size as usize } else { INT4_GROUP_SIZE };
        let max_seq = if h.max_seq_len > 0 { h.max_seq_len as usize } else { MAX_SEQ_LEN };
        // Mặc định n_kv_heads = n_heads nếu không được quy định (Multi-head Attention truyền thống)
        let n_kv_heads = n_heads; 

        Self {
            dim,
            hidden_dim: h.hidden_dim as usize,
            n_layers: h.n_layers as usize,
            n_heads,
            head_dim: dim / n_heads,
            n_kv_heads,
            vocab_size: h.vocab_size as usize,
            max_seq_len: max_seq,
            is_quantized: (h.flags & 1) != 0,
            is_bitnet: (h.flags & 0b1000) != 0, // Giả sử bit 3 là cờ BitNet
            n_experts,
            top_k: top_k.min(n_experts),
            int4_group_size: group,
            depth_router_layer: h.depth_router_layer as usize,
            tri_layer_mode: false, // Mặc định tắt cho các model cũ
            speculative_steps: 0,
            rope_theta: h.rope_theta_f32(),
        }
    }

    pub fn is_moe(&self) -> bool {
        self.n_experts >= 2
    }

    /// Kiểm tra xem có dùng GQA không
    #[allow(dead_code)]
    pub fn is_gqa(&self) -> bool {
        self.n_kv_heads < self.n_heads
    }

    /// Total parameter count (approximate).
    pub fn param_count(&self) -> usize {
        let dim = self.dim;
        let hidden = self.hidden_dim;
        let vocab = self.vocab_size;
        let n_layers = self.n_layers;
        let n_exp = self.n_experts;

        let embed = vocab * dim;
        let per_layer_attn = 3 * dim * dim + dim * dim; // QKV + out
        let per_layer_norms = 2 * dim;
        let per_expert = 2 * hidden * dim + dim * hidden; // gate+up + down
        let router_per_layer = if n_exp >= 2 { n_exp * dim + n_exp } else { 0 };
        let per_layer = per_layer_attn + per_layer_norms + router_per_layer + n_exp * per_expert;
        let output = vocab * dim;
        let depth_router = if self.depth_router_layer > 0 {
            dim * DEPTH_ROUTER_HIDDEN + DEPTH_ROUTER_HIDDEN + DEPTH_ROUTER_HIDDEN + 1
        } else {
            0
        };

        embed + n_layers * per_layer + output + depth_router + dim
    }

    /// Cấu hình tối ưu cho 100MB RAM: BitNet b1.58 + Tri-Layer Dense
    #[allow(dead_code)]
    pub fn mobile_bitnet_config() -> Self {
        Self {
            dim: 192,           // Mỏng lại (Gốc 512) -> Tiết kiệm RAM KV Cache cực lớn
            hidden_dim: 512,    // Hidden cũng nhỏ lại
            n_layers: 24,       // Tăng số lớp (Gốc 12) -> Suy luận sâu hơn
            n_heads: 6,
            n_kv_heads: 2,      // GQA: 3 query heads share 1 KV head -> Giảm 3x KV Cache
            head_dim: 32,
            vocab_size: 32000,  // Ví dụ
            max_seq_len: 2048,
            is_bitnet: true,    // [OPTIMIZATION] Ternary weights (-1, 0, 1)
            tri_layer_mode: true, // Cờ bật chế độ 3 tầng
            speculative_steps: 3, // Dự đoán trước 3 token bằng ShallowReflex
            rope_theta: 10000.0,
            ..unsafe { std::mem::zeroed() } // Hack để điền các field còn lại (hoặc điền đầy đủ)
        }
    }

    /// Cấu hình cho Qwen2.5-0.5B với MoE up-cycling (Pixel 5 tối ưu)
    /// Qwen gốc: 0.5B params → Up-cycle thành ~1.5B params MoE
    /// Memory footprint: ~750MB on disk, ~250MB runtime (Top-2 activation)
    #[allow(dead_code)]
    pub fn qwen_0_5b_moe_config() -> Self {
        Self {
            dim: 896,              // Qwen2.5-0.5B hidden dimension
            hidden_dim: 4864,      // FFN intermediate size
            n_layers: 24,          // Qwen2.5-0.5B có 24 transformer blocks
            n_heads: 14,           // 14 attention heads
            n_kv_heads: 2,         // [OPTIMIZATION] Qwen2.5 dùng GQA (14/2 = 7x reduction in KV Cache)
            head_dim: 64,          // 896 / 14 = 64
            vocab_size: 151936,    // Qwen tokenizer vocab size
            max_seq_len: 2048,     // Giới hạn context cho mobile
            is_quantized: true,    // Int8 attention + Int4 experts
            is_bitnet: false,      // Qwen gốc không phải BitNet (trừ khi fine-tune lại)
            n_experts: 8,          // Up-cycle: Nhân bản FFN thành 8 experts
            top_k: 2,              // Chỉ kích hoạt 2 experts/token
            int4_group_size: 32,   // Group size cho Int4 quantization
            depth_router_layer: 8, // Adaptive depth sau layer 8
            tri_layer_mode: true,  // Bật chế độ Brain Map 3 tầng
            speculative_steps: 2,  // Enable speculative decoding
            rope_theta: 1000000.0, // Qwen2.5 uses rope_theta=1000000
        }
    }

    /// Tính toán layer ranges cho Brain Map (phân vùng Qwen 24-layer)
    /// - Shallow Reflex: Layer 0-5 (6 layers) - Ngữ pháp, từ vựng
    /// - Deep Logic: Layer 6-17 (12 layers) - Suy luận, code
    /// - Hard Fact: Layer 18-23 (6 layers) - Kiến thức tra cứu
    #[allow(dead_code)]
    pub fn qwen_brain_map_ranges(&self) -> [(usize, usize); 3] {
        if self.n_layers == 24 {
            [(0, 6), (6, 18), (18, 24)] // Shallow, Deep, Fact
        } else {
            // Fallback cho các model khác
            let third = self.n_layers / 3;
            [(0, third), (third, 2 * third), (2 * third, self.n_layers)]
        }
    }
}
