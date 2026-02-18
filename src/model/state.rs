// =============================================================================
// model/state.rs — Inference State and Statistics
// =============================================================================

use super::config::ModelConfig;
use super::constants::DEPTH_ROUTER_HIDDEN;

/// InferenceState — pre-allocated scratchpad
pub struct InferenceState {
    // Transformer state
    pub x: Vec<f32>,
    pub xb: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub att_out: Vec<f32>,
    pub att_scores: Vec<f32>,
    pub hb: Vec<f32>,
    pub hb2: Vec<f32>,
    pub logits: Vec<f32>,
    pub key_cache: Vec<f32>,
    pub value_cache: Vec<f32>,
    pub tmp: Vec<f32>,
    pub rope_cos: Vec<f32>,
    pub rope_sin: Vec<f32>,

    // MoE buffers
    pub expert_scores: Vec<f32>,   // [n_experts]
    pub expert_out: Vec<f32>,      // [dim] weighted sum accumulator

    // Depth router buffers
    pub depth_hidden: Vec<f32>,    // [DEPTH_ROUTER_HIDDEN]
    pub active_layers: usize,     // determined at runtime

    // Stats (per-generation)
    pub experts_evaluated: usize,
    pub layers_evaluated: usize,
}

impl InferenceState {
    pub fn new(cfg: &ModelConfig) -> Self {
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let vocab = cfg.vocab_size;
        let kv_size = cfg.n_layers * cfg.max_seq_len * dim;
        let half_head = cfg.head_dim / 2;
        let rope_len = cfg.max_seq_len * half_head;

        // Pre-compute RoPE frequency table
        let mut rope_cos = vec![0.0f32; rope_len];
        let mut rope_sin = vec![0.0f32; rope_len];
        for pos in 0..cfg.max_seq_len {
            for i in 0..half_head {
                let freq = 1.0 / 10000.0f32.powf((2 * i) as f32 / cfg.head_dim as f32);
                let theta = pos as f32 * freq;
                rope_cos[pos * half_head + i] = theta.cos();
                rope_sin[pos * half_head + i] = theta.sin();
            }
        }

        Self {
            x: vec![0.0; dim],
            xb: vec![0.0; dim],
            q: vec![0.0; dim],
            k: vec![0.0; dim],
            v: vec![0.0; dim],
            att_out: vec![0.0; dim],
            att_scores: vec![0.0; cfg.max_seq_len],
            hb: vec![0.0; hidden],
            hb2: vec![0.0; hidden],
            logits: vec![0.0; vocab],
            key_cache: vec![0.0; kv_size],
            value_cache: vec![0.0; kv_size],
            tmp: vec![0.0; dim.max(hidden)],
            rope_cos,
            rope_sin,
            expert_scores: vec![0.0; cfg.n_experts],
            expert_out: vec![0.0; dim],
            depth_hidden: vec![0.0; DEPTH_ROUTER_HIDDEN],
            active_layers: cfg.n_layers,
            experts_evaluated: 0,
            layers_evaluated: 0,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        (self.x.len()
            + self.xb.len()
            + self.q.len()
            + self.k.len()
            + self.v.len()
            + self.att_out.len()
            + self.att_scores.len()
            + self.hb.len()
            + self.hb2.len()
            + self.logits.len()
            + self.key_cache.len()
            + self.value_cache.len()
            + self.tmp.len()
            + self.rope_cos.len()
            + self.rope_sin.len()
            + self.expert_scores.len()
            + self.expert_out.len()
            + self.depth_hidden.len())
            * 4
    }
}

/// SparseLoadStats — tracks how much of the model is actually paged in
#[derive(Debug, Default, Clone)]
pub struct SparseLoadStats {
    pub total_expert_blocks: usize,
    pub experts_accessed: usize,
    pub layers_skipped: usize,
    pub total_forwards: usize,
}

impl SparseLoadStats {
    pub fn expert_sparsity(&self) -> f64 {
        if self.total_expert_blocks == 0 {
            return 0.0;
        }
        1.0 - (self.experts_accessed as f64 / self.total_expert_blocks as f64)
    }

    #[allow(dead_code)]
    pub fn avg_layers_used(&self) -> f64 {
        if self.total_forwards == 0 {
            return 0.0;
        }
        (self.total_expert_blocks - self.layers_skipped) as f64 / self.total_forwards as f64
    }
}
