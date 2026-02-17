// =============================================================================
// model/weight_index.rs — Weight offsets into mmap'd file
// =============================================================================

use super::config::ModelConfig;
use super::constants::{HEADER_SIZE, DEPTH_ROUTER_HIDDEN};

/// Per-expert weight offsets within a layer (for Int4 packed weights).
#[derive(Debug, Clone)]
pub struct ExpertWeightIndex {
    // Gate+Up packed: [2*hidden_dim × dim/2] Int4 packed bytes
    pub gate_up_offset: usize,
    pub gate_up_packed_bytes: usize,
    // Gate+Up group scales: [2*hidden_dim × n_groups_dim] f32
    pub gate_up_scales_offset: usize,
    pub gate_up_scales_count: usize, // number of f32 scale values
    // Down packed: [dim × hidden_dim/2] Int4 packed bytes
    pub down_offset: usize,
    pub down_packed_bytes: usize,
    // Down group scales: [dim × n_groups_hidden] f32
    pub down_scales_offset: usize,
    pub down_scales_count: usize,
}

#[derive(Debug, Clone)]
pub struct LayerWeightIndex {
    // RMS norm (attention): [dim] f32
    pub rms_attn_offset: usize,
    pub rms_bytes: usize,

    // Attention QKV: [3*dim*dim] Int8 + [3*dim] f32 scales
    pub attn_qkv_offset: usize,
    pub attn_qkv_bytes: usize,
    pub attn_qkv_scales_offset: usize,

    // Attention output: [dim*dim] Int8 + [dim] f32 scales
    pub attn_out_offset: usize,
    pub attn_out_bytes: usize,
    pub attn_out_scales_offset: usize,

    // RMS norm (FFN): [dim] f32
    pub rms_ffn_offset: usize,

    // Expert router weights (only if MoE): [n_experts * dim] f32 + [n_experts] f32 bias
    pub router_w_offset: usize,
    pub router_b_offset: usize,
    pub router_bytes: usize, // total bytes for router w + b

    // Per-expert weights
    pub experts: Vec<ExpertWeightIndex>,
}

#[derive(Debug)]
pub struct WeightIndex {
    // Token embeddings: [vocab × dim] Int8
    pub embed_offset: usize,
    pub embed_bytes: usize,
    // Embed scales: [vocab] f32
    pub embed_scales_offset: usize,

    // Per-layer offsets
    pub layers: Vec<LayerWeightIndex>,

    // Depth router (optional)
    pub depth_router_w1_offset: usize, // [dim × DEPTH_ROUTER_HIDDEN] f32
    pub depth_router_b1_offset: usize, // [DEPTH_ROUTER_HIDDEN] f32
    pub depth_router_w2_offset: usize, // [DEPTH_ROUTER_HIDDEN] f32
    pub depth_router_b2_offset: usize, // [1] f32
    pub depth_router_bytes: usize,

    // Final RMS norm: [dim] f32
    pub final_norm_offset: usize,

    // Output projection: [vocab × dim] Int8
    pub output_proj_offset: usize,
    pub output_proj_bytes: usize,
    pub output_proj_scales_offset: usize,

    // Total bytes in file
    pub total_bytes: usize,
}

impl WeightIndex {
    pub fn build(cfg: &ModelConfig) -> Self {
        let dim = cfg.dim;
        let _hidden = cfg.hidden_dim;
        let vocab = cfg.vocab_size;
        let n_layers = cfg.n_layers;
        let _n_experts = cfg.n_experts;
        let group = cfg.int4_group_size;
        let hdr = if cfg.is_moe() || group > 0 { HEADER_SIZE } else { 128 };

        let mut cursor = hdr;

        // 1. Token embeddings (Int8 + per-row scales)
        let embed_offset = cursor;
        let embed_bytes = vocab * dim;
        cursor += embed_bytes;
        let embed_scales_offset = cursor;
        cursor += vocab * 4;

        // 2. Layers
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            let l = Self::build_layer(cfg, &mut cursor);
            layers.push(l);
        }

        // 3. Depth router (if enabled)
        let depth_router_w1_offset;
        let depth_router_b1_offset;
        let depth_router_w2_offset;
        let depth_router_b2_offset;
        let depth_router_bytes;

        if cfg.depth_router_layer > 0 {
            depth_router_w1_offset = cursor;
            cursor += dim * DEPTH_ROUTER_HIDDEN * 4;
            depth_router_b1_offset = cursor;
            cursor += DEPTH_ROUTER_HIDDEN * 4;
            depth_router_w2_offset = cursor;
            cursor += DEPTH_ROUTER_HIDDEN * 4;
            depth_router_b2_offset = cursor;
            cursor += 4;
            depth_router_bytes = cursor - depth_router_w1_offset;
        } else {
            depth_router_w1_offset = 0;
            depth_router_b1_offset = 0;
            depth_router_w2_offset = 0;
            depth_router_b2_offset = 0;
            depth_router_bytes = 0;
        }

        // 4. Final RMS norm
        let final_norm_offset = cursor;
        cursor += dim * 4;

        // 5. Output projection (Int8 + per-row scales)
        let output_proj_offset = cursor;
        let output_proj_bytes = vocab * dim;
        cursor += output_proj_bytes;
        let output_proj_scales_offset = cursor;
        cursor += vocab * 4;

        WeightIndex {
            embed_offset,
            embed_bytes,
            embed_scales_offset,
            layers,
            depth_router_w1_offset,
            depth_router_b1_offset,
            depth_router_w2_offset,
            depth_router_b2_offset,
            depth_router_bytes,
            final_norm_offset,
            output_proj_offset,
            output_proj_bytes,
            output_proj_scales_offset,
            total_bytes: cursor,
        }
    }

    fn build_layer(cfg: &ModelConfig, cursor: &mut usize) -> LayerWeightIndex {
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim;
        let n_experts = cfg.n_experts;
        let group = cfg.int4_group_size;
        let _ = (hidden, n_experts); // used below via computed values

        // --- RMS norm (attention) ---
        let rms_attn_offset = *cursor;
        let rms_bytes = dim * 4;
        *cursor += rms_bytes;

        // --- Attention QKV: [3*dim*dim] Int8 ---
        let attn_qkv_offset = *cursor;
        let attn_qkv_bytes = 3 * dim * dim;
        *cursor += attn_qkv_bytes;
        let attn_qkv_scales_offset = *cursor;
        *cursor += 3 * dim * 4;

        // --- Attention output: [dim*dim] Int8 ---
        let attn_out_offset = *cursor;
        let attn_out_bytes = dim * dim;
        *cursor += attn_out_bytes;
        let attn_out_scales_offset = *cursor;
        *cursor += dim * 4;

        // --- RMS norm (FFN) ---
        let rms_ffn_offset = *cursor;
        *cursor += rms_bytes;

        // --- Expert router (if MoE) ---
        let router_w_offset;
        let router_b_offset;
        let router_bytes;

        if n_experts >= 2 {
            router_w_offset = *cursor;
            *cursor += n_experts * dim * 4; // f32
            router_b_offset = *cursor;
            *cursor += n_experts * 4;
            router_bytes = n_experts * dim * 4 + n_experts * 4;
        } else {
            router_w_offset = 0;
            router_b_offset = 0;
            router_bytes = 0;
        }

        // --- Per-expert Int4 weights ---
        let n_groups_dim = (dim + group - 1) / group;
        let n_groups_hidden = (hidden + group - 1) / group;
        let packed_dim = (dim + 1) / 2;       // bytes per row when in_dim = dim
        let packed_hidden = (hidden + 1) / 2;  // bytes per row when in_dim = hidden

        let mut experts = Vec::with_capacity(n_experts);
        for _ in 0..n_experts {
            // Gate+Up: [2*hidden × dim/2] Int4 packed
            let gate_up_offset = *cursor;
            let gate_up_rows = 2 * hidden;
            let gate_up_packed_bytes = gate_up_rows * packed_dim;
            *cursor += gate_up_packed_bytes;
            let gate_up_scales_offset = *cursor;
            let gate_up_scales_count = gate_up_rows * n_groups_dim;
            *cursor += gate_up_scales_count * 4;

            // Down: [dim × hidden/2] Int4 packed
            let down_offset = *cursor;
            let down_packed_bytes = dim * packed_hidden;
            *cursor += down_packed_bytes;
            let down_scales_offset = *cursor;
            let down_scales_count = dim * n_groups_hidden;
            *cursor += down_scales_count * 4;

            experts.push(ExpertWeightIndex {
                gate_up_offset,
                gate_up_packed_bytes,
                gate_up_scales_offset,
                gate_up_scales_count,
                down_offset,
                down_packed_bytes,
                down_scales_offset,
                down_scales_count,
            });
        }

        LayerWeightIndex {
            rms_attn_offset,
            rms_bytes,
            attn_qkv_offset,
            attn_qkv_bytes,
            attn_qkv_scales_offset,
            attn_out_offset,
            attn_out_bytes,
            attn_out_scales_offset,
            rms_ffn_offset,
            router_w_offset,
            router_b_offset,
            router_bytes,
            experts,
        }
    }

    /// Total bytes used by expert weights across all layers (for stats).
    pub fn expert_weight_bytes(&self) -> usize {
        self.layers
            .iter()
            .flat_map(|l| l.experts.iter())
            .map(|e| {
                e.gate_up_packed_bytes
                    + e.gate_up_scales_count * 4
                    + e.down_packed_bytes
                    + e.down_scales_count * 4
            })
            .sum()
    }

    /// Bytes that are ALWAYS accessed (embedding, attention, norms, output).
    pub fn dense_weight_bytes(&self) -> usize {
        self.total_bytes - self.expert_weight_bytes() - self.depth_router_bytes
    }
}
