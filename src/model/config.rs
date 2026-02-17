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
    // MoE
    pub n_experts: usize,
    pub top_k: usize,
    // Int4
    pub int4_group_size: usize,
    // Adaptive depth
    pub depth_router_layer: usize,
    // Architecture
    pub tri_layer_mode: bool,
}

impl ModelConfig {
    pub fn from_header(h: &FileHeader) -> Self {
        let dim = h.dim as usize;
        let n_heads = h.n_heads as usize;
        let n_experts = if h.n_experts >= 2 { h.n_experts as usize } else { 1 };
        let top_k = if h.top_k >= 1 { h.top_k as usize } else { n_experts.min(2) };
        let group = if h.int4_group_size > 0 { h.int4_group_size as usize } else { INT4_GROUP_SIZE };
        let max_seq = if h.max_seq_len > 0 { h.max_seq_len as usize } else { MAX_SEQ_LEN };

        Self {
            dim,
            hidden_dim: h.hidden_dim as usize,
            n_layers: h.n_layers as usize,
            n_heads,
            head_dim: dim / n_heads,
            vocab_size: h.vocab_size as usize,
            max_seq_len: max_seq,
            is_quantized: (h.flags & 1) != 0,
            n_experts,
            top_k: top_k.min(n_experts),
            int4_group_size: group,
            depth_router_layer: h.depth_router_layer as usize,
            tri_layer_mode: false, // Mặc định tắt cho các model cũ
        }
    }

    pub fn is_moe(&self) -> bool {
        self.n_experts >= 2
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

    /// Cấu hình tối ưu cho 100MB RAM theo kiến trúc "3 Tầng - Ống dẫn"
    pub fn mobile_tri_layer_config() -> Self {
        Self {
            dim: 192,           // Mỏng lại (Gốc 512) -> Tiết kiệm RAM KV Cache cực lớn
            hidden_dim: 512,    // Hidden cũng nhỏ lại
            n_layers: 24,       // Tăng số lớp (Gốc 12) -> Suy luận sâu hơn
            n_heads: 6,
            head_dim: 32,
            vocab_size: 32000,  // Ví dụ
            max_seq_len: 2048,
            tri_layer_mode: true, // Cờ bật chế độ 3 tầng
            ..unsafe { std::mem::zeroed() } // Hack để điền các field còn lại (hoặc điền đầy đủ)
        }
    }
}
