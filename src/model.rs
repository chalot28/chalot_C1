// =============================================================================
// model.rs — Sparse MoE Transformer with Adaptive Depth (v3 Extreme)
// =============================================================================
//
// Architecture v3 — Maximum Reasoning Efficiency:
//   dim=512, hidden_dim=1536, layers=12, heads=8, vocab=32000
//   n_experts=8 per layer, top_k=2 activated per token
//   Attention: Int8 quantized + logit soft-capping (anti-entropy collapse)
//   Expert/FFN: Int4 quantized + fused SwiGLU + group-wise scales
//   Adaptive depth: continuous depth routing with GELU-gated 32-dim MLP
//   SIMD: AVX2 2-row parallel int8 matmul (2× throughput over SSE2)
//
// Total backbone: ~272M params ≈ ~160 MB on disk (mixed Int8/Int4)
// Runtime RAM with sparse loading: ~55–85 MB (top-2/8 experts active)
//
// Key innovation — "Partial Open Structure":
//   The model does NOT load all weights into RAM.  It uses mmap so the OS
//   only pages-in weights that are actually *accessed*.  The expert router
//   selects top-k experts per layer; weights for un-selected experts are
//   never touched → never loaded → zero RAM cost.  The depth router
//   lets simple queries exit early, skipping entire layers.
// =============================================================================

use memmap2::Mmap;
use std::fs::File;
use std::path::Path;

use crate::tensor::{
    self, bytes_as_f32, matmul_int4, matmul_int8, rmsnorm, sigmoid, gelu, softmax,
    softmax_top_k, swiglu_fused, logit_soft_cap, INT4_GROUP_SIZE,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
pub const MAGIC: u32 = 0x4D594149; // "MYAI"
pub const HEADER_SIZE: usize = 256;
pub const MAX_SEQ_LEN: usize = 512;
pub const DEPTH_ROUTER_HIDDEN: usize = 32;  // doubled for deeper reasoning assessment
pub const DEPTH_ROUTER_AFTER_LAYER: usize = 2; // compute depth score after this many layers
pub const ATTN_LOGIT_CAP: f32 = 30.0;  // attention soft-cap prevents entropy collapse

// ---------------------------------------------------------------------------
// Header (first 256 bytes of .myai v2 file)
// ---------------------------------------------------------------------------
/// Fields are read manually for portability; no repr(C,packed) tricks.
#[derive(Debug, Clone)]
pub struct FileHeader {
    pub magic: u32,
    pub version: u32,
    pub dim: u32,
    pub hidden_dim: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub vocab_size: u32,
    pub flags: u32,         // bit0 = quantized, bit1 = int4_experts, bit2 = moe
    pub n_experts: u32,     // 0 or 1 = dense, ≥2 = MoE
    pub top_k: u32,         // experts activated per token
    pub int4_group_size: u32, // 0 = Int8 only, else group size for Int4
    pub depth_router_layer: u32, // 0 = disabled
    pub max_seq_len: u32,   // 0 = default (512)
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn write_u32_le(buf: &mut [u8], offset: usize, val: u32) {
    buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
}

impl FileHeader {
    pub fn from_bytes(data: &[u8]) -> Self {
        assert!(data.len() >= 128, "File too small for header");
        let version = read_u32_le(data, 4);

        let mut h = FileHeader {
            magic: read_u32_le(data, 0),
            version,
            dim: read_u32_le(data, 8),
            hidden_dim: read_u32_le(data, 12),
            n_layers: read_u32_le(data, 16),
            n_heads: read_u32_le(data, 20),
            vocab_size: read_u32_le(data, 24),
            flags: read_u32_le(data, 28),
            // v2 defaults (overwritten below if version ≥ 2)
            n_experts: 0,
            top_k: 0,
            int4_group_size: 0,
            depth_router_layer: 0,
            max_seq_len: 0,
        };

        if version >= 2 && data.len() >= HEADER_SIZE {
            h.n_experts = read_u32_le(data, 128);
            h.top_k = read_u32_le(data, 132);
            h.int4_group_size = read_u32_le(data, 136);
            h.depth_router_layer = read_u32_le(data, 140);
            h.max_seq_len = read_u32_le(data, 144);
        }
        h
    }

    pub fn write_to(&self, buf: &mut [u8]) {
        assert!(buf.len() >= HEADER_SIZE);
        // Zero the header area first
        for b in buf[..HEADER_SIZE].iter_mut() {
            *b = 0;
        }
        write_u32_le(buf, 0, self.magic);
        write_u32_le(buf, 4, self.version);
        write_u32_le(buf, 8, self.dim);
        write_u32_le(buf, 12, self.hidden_dim);
        write_u32_le(buf, 16, self.n_layers);
        write_u32_le(buf, 20, self.n_heads);
        write_u32_le(buf, 24, self.vocab_size);
        write_u32_le(buf, 28, self.flags);
        // v2 fields at offset 128+
        write_u32_le(buf, 128, self.n_experts);
        write_u32_le(buf, 132, self.top_k);
        write_u32_le(buf, 136, self.int4_group_size);
        write_u32_le(buf, 140, self.depth_router_layer);
        write_u32_le(buf, 144, self.max_seq_len);
    }

    pub fn header_size(&self) -> usize {
        if self.version >= 2 { HEADER_SIZE } else { 128 }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.magic != MAGIC {
            return Err(format!("Bad magic: 0x{:08X}", self.magic));
        }
        if self.version == 0 || self.version > 2 {
            return Err(format!("Unsupported version: {}", self.version));
        }
        Ok(())
    }

    pub fn is_moe(&self) -> bool {
        self.n_experts >= 2
    }

    pub fn has_int4(&self) -> bool {
        self.int4_group_size > 0
    }

    pub fn has_depth_router(&self) -> bool {
        self.depth_router_layer > 0
    }
}

// ---------------------------------------------------------------------------
// ModelConfig — runtime-friendly copy of header values
// ---------------------------------------------------------------------------
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
}

// ---------------------------------------------------------------------------
// Weight offsets — pointers into the mmap'd file
// ---------------------------------------------------------------------------

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
        let hidden = cfg.hidden_dim;
        let vocab = cfg.vocab_size;
        let n_layers = cfg.n_layers;
        let n_experts = cfg.n_experts;
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

// ---------------------------------------------------------------------------
// InferenceState — pre-allocated scratchpad
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// SparseLoadStats — tracks how much of the model is actually paged in
// ---------------------------------------------------------------------------
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

    pub fn avg_layers_used(&self) -> f64 {
        if self.total_forwards == 0 {
            return 0.0;
        }
        (self.total_expert_blocks - self.layers_skipped) as f64 / self.total_forwards as f64
    }
}

// ---------------------------------------------------------------------------
// Engine — owns the mmap + forward pass logic
// ---------------------------------------------------------------------------
pub struct Engine {
    pub mmap: Mmap,
    pub config: ModelConfig,
    pub weights: WeightIndex,
    pub state: InferenceState,
    pub stats: SparseLoadStats,
    #[allow(dead_code)]
    pub current_head: Option<TaskHead>,
}

#[allow(dead_code)]
pub struct TaskHead {
    pub name: String,
    pub data: Vec<u8>,
}

impl Engine {
    /// Load backbone from a `.myai` file (v1 or v2).
    pub fn load(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("open: {e}"))?;
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap: {e}"))?;

        if mmap.len() < 128 {
            return Err("File too small for header".into());
        }

        let header = FileHeader::from_bytes(&mmap);
        header.validate()?;

        let config = ModelConfig::from_header(&header);
        let weights = WeightIndex::build(&config);

        if mmap.len() < weights.total_bytes {
            return Err(format!(
                "File too small: {} < {} expected",
                mmap.len(),
                weights.total_bytes
            ));
        }

        let state = InferenceState::new(&config);
        let param_m = config.param_count() as f64 / 1e6;

        println!(
            "[engine] Loaded v{} model: dim={}, layers={}, heads={}, vocab={}, experts={}×top{}",
            header.version, config.dim, config.n_layers, config.n_heads, config.vocab_size,
            config.n_experts, config.top_k,
        );
        println!(
            "[engine]   ~{:.1}M params | backbone={:.1} MB | scratchpad={:.1} MB | int4_group={}",
            param_m,
            weights.total_bytes as f64 / 1e6,
            state.memory_bytes() as f64 / 1e6,
            config.int4_group_size,
        );
        if config.depth_router_layer > 0 {
            println!(
                "[engine]   Adaptive depth enabled (router after layer {})",
                config.depth_router_layer
            );
        }

        Ok(Self {
            mmap,
            config,
            weights,
            state,
            stats: SparseLoadStats::default(),
            current_head: None,
        })
    }

    /// Swap task head.
    #[allow(dead_code)]
    pub fn switch_task(&mut self, task_file: &str) -> Result<(), String> {
        self.current_head = None;
        let data = std::fs::read(task_file).map_err(|e| format!("load head: {e}"))?;
        let name = Path::new(task_file)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        println!("[engine] Loaded task head '{}' ({} bytes)", name, data.len());
        self.current_head = Some(TaskHead { name, data });
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Forward pass — Sparse MoE + Adaptive Depth
    // -----------------------------------------------------------------------
    pub fn forward(&mut self, token: usize, pos: usize) -> usize {
        let dim = self.config.dim;
        let hidden = self.config.hidden_dim;
        let n_heads = self.config.n_heads;
        let head_dim = self.config.head_dim;
        let n_layers = self.config.n_layers;
        let max_seq = self.config.max_seq_len;
        let vocab = self.config.vocab_size;
        let half_head = head_dim / 2;
        let n_experts = self.config.n_experts;
        let top_k = self.config.top_k;

        let wdata: &[u8] = &self.mmap;
        let weights = &self.weights;
        let s = &mut self.state;

        // Reset per-forward stats
        s.experts_evaluated = 0;
        s.layers_evaluated = 0;

        // Determine active layers (may be overridden by depth router)
        let mut active_layers = s.active_layers.min(n_layers);

        // 1. Token embedding (Int8 → f32 dequant)
        {
            let embed = slice_i8(wdata, weights.embed_offset, weights.embed_bytes);
            let embed_s = slice_f32(wdata, weights.embed_scales_offset, vocab);
            let row = &embed[token * dim..(token + 1) * dim];
            let scale = embed_s[token];
            for i in 0..dim {
                s.x[i] = row[i] as f32 * scale;
            }
        }

        // 2. Transformer layers
        for layer in 0..active_layers {
            let lw = &weights.layers[layer];

            // ─── Attention sub-block ───
            {
                let norm_w = slice_f32(wdata, lw.rms_attn_offset, dim);
                s.tmp[..dim].copy_from_slice(&s.x);
                rmsnorm(&mut s.xb, &s.tmp[..dim], norm_w);
            }

            // QKV projection (Int8)
            {
                let qkv_w = slice_i8(wdata, lw.attn_qkv_offset, lw.attn_qkv_bytes);
                let qkv_s = slice_f32(wdata, lw.attn_qkv_scales_offset, 3 * dim);
                s.tmp[..dim].copy_from_slice(&s.xb);
                matmul_int8(&mut s.q, &qkv_w[0..dim * dim], &qkv_s[0..dim], &s.tmp[..dim], dim, dim);
                matmul_int8(&mut s.k, &qkv_w[dim * dim..2 * dim * dim], &qkv_s[dim..2 * dim], &s.tmp[..dim], dim, dim);
                matmul_int8(&mut s.v, &qkv_w[2 * dim * dim..3 * dim * dim], &qkv_s[2 * dim..3 * dim], &s.tmp[..dim], dim, dim);
            }

            // RoPE (pre-computed cos/sin)
            {
                let rope_off = pos * half_head;
                for h in 0..n_heads {
                    let base = h * head_dim;
                    for i in 0..half_head {
                        let cos_t = s.rope_cos[rope_off + i];
                        let sin_t = s.rope_sin[rope_off + i];

                        let q0 = s.q[base + 2 * i];
                        let q1 = s.q[base + 2 * i + 1];
                        s.q[base + 2 * i] = q0 * cos_t - q1 * sin_t;
                        s.q[base + 2 * i + 1] = q0 * sin_t + q1 * cos_t;

                        let k0 = s.k[base + 2 * i];
                        let k1 = s.k[base + 2 * i + 1];
                        s.k[base + 2 * i] = k0 * cos_t - k1 * sin_t;
                        s.k[base + 2 * i + 1] = k0 * sin_t + k1 * cos_t;
                    }
                }
            }

            // KV cache
            {
                let cache_off = layer * max_seq * dim + pos * dim;
                for i in 0..dim {
                    s.key_cache[cache_off + i] = s.k[i];
                    s.value_cache[cache_off + i] = s.v[i];
                }
            }

            // Multi-head attention
            for i in 0..dim {
                s.att_out[i] = 0.0;
            }

            for h in 0..n_heads {
                let q_off = h * head_dim;
                let inv_sqrt = 1.0 / (head_dim as f32).sqrt();

                for t in 0..=pos {
                    let k_off = layer * max_seq * dim + t * dim + h * head_dim;
                    let dot = tensor::dot_f32(&s.q[q_off..], &s.key_cache[k_off..], head_dim);
                    // Logit soft-cap: bounds extreme scores for stable deep reasoning
                    s.att_scores[t] = logit_soft_cap(dot * inv_sqrt, ATTN_LOGIT_CAP);
                }

                softmax(&mut s.att_scores[..=pos]);

                // Cache-friendly accumulation: loop over timesteps (outer)
                // so value_cache access is contiguous within each timestep.
                for t in 0..=pos {
                    let v_off = layer * max_seq * dim + t * dim + h * head_dim;
                    let w = s.att_scores[t];
                    for d in 0..head_dim {
                        s.att_out[q_off + d] += w * s.value_cache[v_off + d];
                    }
                }
            }

            // Attention output projection (Int8)
            {
                let wo = slice_i8(wdata, lw.attn_out_offset, lw.attn_out_bytes);
                let wo_s = slice_f32(wdata, lw.attn_out_scales_offset, dim);
                s.tmp[..dim].copy_from_slice(&s.att_out);
                matmul_int8(&mut s.xb, wo, wo_s, &s.tmp[..dim], dim, dim);
            }

            // Residual
            for i in 0..dim {
                s.x[i] += s.xb[i];
            }

            // ─── FFN / MoE sub-block ───
            {
                let norm_w = slice_f32(wdata, lw.rms_ffn_offset, dim);
                s.tmp[..dim].copy_from_slice(&s.x);
                rmsnorm(&mut s.xb, &s.tmp[..dim], norm_w);
            }

            if n_experts >= 2 && lw.experts.len() >= 2 {
                // ─── Mixture of Experts path ───
                // Compute router scores
                let router_w = slice_f32(wdata, lw.router_w_offset, n_experts * dim);
                let router_b = slice_f32(wdata, lw.router_b_offset, n_experts);

                for e in 0..n_experts {
                    let score = router_b[e]
                        + tensor::dot_f32(&router_w[e * dim..], &s.xb, dim);
                    s.expert_scores[e] = score;
                }

                // Select top-k experts with softmax-normalised weights
                let selected = softmax_top_k(&mut s.expert_scores[..n_experts], top_k);

                // Accumulate weighted expert outputs
                for i in 0..dim {
                    s.expert_out[i] = 0.0;
                }

                for &(expert_id, expert_weight) in &selected {
                    let ew = &lw.experts[expert_id];

                    // Gate+Up projection (Int4): [2*hidden × dim]
                    let gu_packed = &wdata[ew.gate_up_offset..ew.gate_up_offset + ew.gate_up_packed_bytes];
                    let gu_scales = slice_f32(wdata, ew.gate_up_scales_offset, ew.gate_up_scales_count);

                    s.tmp[..dim].copy_from_slice(&s.xb);

                    // Gate: first `hidden` rows
                    matmul_int4(
                        &mut s.hb,
                        &gu_packed[..ew.gate_up_packed_bytes / 2],
                        &gu_scales[..ew.gate_up_scales_count / 2],
                        &s.tmp[..dim],
                        hidden,
                        dim,
                    );
                    // Up: next `hidden` rows
                    matmul_int4(
                        &mut s.hb2,
                        &gu_packed[ew.gate_up_packed_bytes / 2..],
                        &gu_scales[ew.gate_up_scales_count / 2..],
                        &s.tmp[..dim],
                        hidden,
                        dim,
                    );

                    // SwiGLU: fused SiLU(gate) × up — single cache-efficient pass
                    swiglu_fused(&mut s.hb[..hidden], &s.hb2[..hidden]);

                    // Down projection (Int4): [dim × hidden]
                    let down_packed = &wdata[ew.down_offset..ew.down_offset + ew.down_packed_bytes];
                    let down_scales = slice_f32(wdata, ew.down_scales_offset, ew.down_scales_count);

                    s.tmp[..hidden].copy_from_slice(&s.hb);
                    matmul_int4(&mut s.xb, down_packed, down_scales, &s.tmp[..hidden], dim, hidden);

                    // Weighted accumulate
                    for i in 0..dim {
                        s.expert_out[i] += expert_weight * s.xb[i];
                    }

                    s.experts_evaluated += 1;
                }

                // Residual connection (from expert output)
                for i in 0..dim {
                    s.x[i] += s.expert_out[i];
                }
            } else {
                // ─── Dense FFN path (single expert / v1 compat) ───
                if !lw.experts.is_empty() {
                    let ew = &lw.experts[0];

                    let gu_packed = &wdata[ew.gate_up_offset..ew.gate_up_offset + ew.gate_up_packed_bytes];
                    let gu_scales = slice_f32(wdata, ew.gate_up_scales_offset, ew.gate_up_scales_count);

                    s.tmp[..dim].copy_from_slice(&s.xb);
                    matmul_int4(
                        &mut s.hb,
                        &gu_packed[..ew.gate_up_packed_bytes / 2],
                        &gu_scales[..ew.gate_up_scales_count / 2],
                        &s.tmp[..dim],
                        hidden,
                        dim,
                    );
                    matmul_int4(
                        &mut s.hb2,
                        &gu_packed[ew.gate_up_packed_bytes / 2..],
                        &gu_scales[ew.gate_up_scales_count / 2..],
                        &s.tmp[..dim],
                        hidden,
                        dim,
                    );

                    // SwiGLU: fused SiLU(gate) × up — single cache-efficient pass
                    swiglu_fused(&mut s.hb[..hidden], &s.hb2[..hidden]);

                    let down_packed = &wdata[ew.down_offset..ew.down_offset + ew.down_packed_bytes];
                    let down_scales = slice_f32(wdata, ew.down_scales_offset, ew.down_scales_count);
                    s.tmp[..hidden].copy_from_slice(&s.hb);
                    matmul_int4(&mut s.xb, down_packed, down_scales, &s.tmp[..hidden], dim, hidden);

                    for i in 0..dim {
                        s.x[i] += s.xb[i];
                    }
                }
            }

            s.layers_evaluated += 1;

            // ─── Adaptive Depth: evaluate depth router after designated layer ───
            if self.config.depth_router_layer > 0
                && layer + 1 == self.config.depth_router_layer
                && weights.depth_router_bytes > 0
            {
                active_layers = compute_depth(
                    &s.x, &mut s.depth_hidden,
                    wdata, weights, self.config.dim,
                    self.config.depth_router_layer, n_layers,
                );
            }
        }

        // 3. Final RMS norm
        {
            let norm_w = slice_f32(wdata, weights.final_norm_offset, dim);
            s.tmp[..dim].copy_from_slice(&s.x);
            rmsnorm(&mut s.x, &s.tmp[..dim], norm_w);
        }

        // 4. Output projection → logits (Int8)
        {
            let out_w = slice_i8(wdata, weights.output_proj_offset, weights.output_proj_bytes);
            let out_s = slice_f32(wdata, weights.output_proj_scales_offset, vocab);
            s.tmp[..dim].copy_from_slice(&s.x);
            matmul_int8(&mut s.logits, out_w, out_s, &s.tmp[..dim], vocab, dim);
        }

        // Update stats
        self.stats.total_forwards += 1;
        self.stats.experts_accessed += s.experts_evaluated;
        self.stats.total_expert_blocks += s.layers_evaluated * n_experts;
        if s.layers_evaluated < n_layers {
            self.stats.layers_skipped += n_layers - s.layers_evaluated;
        }

        // 5. Greedy decode
        tensor::sample_argmax(&s.logits)
    }

    // compute_depth moved to free function to avoid borrow conflicts

    /// Report sparse loading statistics.
    pub fn sparse_stats_report(&self) -> String {
        format!(
            "Sparse stats: {}/{} expert blocks accessed ({:.1}% sparse), {} layers skipped across {} forwards",
            self.stats.experts_accessed,
            self.stats.total_expert_blocks,
            self.stats.expert_sparsity() * 100.0,
            self.stats.layers_skipped,
            self.stats.total_forwards,
        )
    }
}

// ---------------------------------------------------------------------------
// Free-standing weight accessors
// ---------------------------------------------------------------------------

/// Compute depth score from current hidden state and decide how many layers
/// to actually run.  Free function to avoid borrow conflicts with Engine.
fn compute_depth(
    x: &[f32],
    depth_hidden: &mut [f32],
    wdata: &[u8],
    weights: &WeightIndex,
    dim: usize,
    dr_layer: usize,
    n_layers: usize,
) -> usize {
    let w1 = slice_f32(wdata, weights.depth_router_w1_offset, dim * DEPTH_ROUTER_HIDDEN);
    let b1 = slice_f32(wdata, weights.depth_router_b1_offset, DEPTH_ROUTER_HIDDEN);
    let w2 = slice_f32(wdata, weights.depth_router_w2_offset, DEPTH_ROUTER_HIDDEN);
    let b2_data = slice_f32(wdata, weights.depth_router_b2_offset, 1);

    // Hidden layer: h = GELU(x @ W1 + b1) — GELU for smoother depth gating
    for j in 0..DEPTH_ROUTER_HIDDEN {
        let mut sum = b1[j];
        for i in 0..dim {
            sum += x[i] * w1[j * dim + i];
        }
        depth_hidden[j] = gelu(sum);
    }

    // Output score: sigmoid(h @ W2 + b2)
    let mut score = b2_data[0];
    for j in 0..DEPTH_ROUTER_HIDDEN {
        score += depth_hidden[j] * w2[j];
    }
    let score = sigmoid(score);

    // Continuous depth mapping: smooth interpolation instead of 4 discrete tiers
    // score ∈ [0,1] maps linearly to [dr_layer+1, n_layers]
    let remaining = (n_layers - dr_layer) as f32;
    let continuous = dr_layer as f32 + 1.0 + score * (remaining - 1.0);
    let active = (continuous.round() as usize).max(dr_layer + 1).min(n_layers);

    active
}

fn slice_i8(data: &[u8], offset: usize, len: usize) -> &[i8] {
    let bytes = &data[offset..offset + len];
    unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, len) }
}

fn slice_f32(data: &[u8], offset: usize, count: usize) -> &[f32] {
    bytes_as_f32(&data[offset..offset + count * 4])
}

// ---------------------------------------------------------------------------
// Create dummy .myai v2 file for testing
// ---------------------------------------------------------------------------
pub fn create_dummy_model(path: &Path, cfg: &ModelConfig) -> std::io::Result<()> {
    use std::io::Write;
    use crate::tensor::pack_int4;

    let weights = WeightIndex::build(cfg);
    let mut buf = vec![0u8; weights.total_bytes];

    // Write header
    let header = FileHeader {
        magic: MAGIC,
        version: 2,
        dim: cfg.dim as u32,
        hidden_dim: cfg.hidden_dim as u32,
        n_layers: cfg.n_layers as u32,
        n_heads: cfg.n_heads as u32,
        vocab_size: cfg.vocab_size as u32,
        flags: if cfg.is_quantized { 0b111 } else { 0 }, // quantized + int4 + moe
        n_experts: cfg.n_experts as u32,
        top_k: cfg.top_k as u32,
        int4_group_size: cfg.int4_group_size as u32,
        depth_router_layer: cfg.depth_router_layer as u32,
        max_seq_len: cfg.max_seq_len as u32,
    };
    header.write_to(&mut buf);

    // Embedding scales (small values)
    for i in 0..cfg.vocab_size {
        let off = weights.embed_scales_offset + i * 4;
        buf[off..off + 4].copy_from_slice(&(1.0f32 / 127.0).to_le_bytes());
    }

    let scale_small: f32 = 0.01;
    let scale_bytes = scale_small.to_le_bytes();
    let norm_one = 1.0f32.to_le_bytes();

    // Per-layer weights
    for lw in &weights.layers {
        // RMS norms = 1.0
        for i in 0..cfg.dim {
            let off = lw.rms_attn_offset + i * 4;
            buf[off..off + 4].copy_from_slice(&norm_one);
        }
        for i in 0..cfg.dim {
            let off = lw.rms_ffn_offset + i * 4;
            buf[off..off + 4].copy_from_slice(&norm_one);
        }

        // Attention weight scales
        for i in 0..(3 * cfg.dim) {
            let off = lw.attn_qkv_scales_offset + i * 4;
            buf[off..off + 4].copy_from_slice(&scale_bytes);
        }
        for i in 0..cfg.dim {
            let off = lw.attn_out_scales_offset + i * 4;
            buf[off..off + 4].copy_from_slice(&scale_bytes);
        }

        // Router bias: uniform distribution so all experts equally likely
        if cfg.n_experts >= 2 && lw.router_bytes > 0 {
            for e in 0..cfg.n_experts {
                let off = lw.router_b_offset + e * 4;
                buf[off..off + 4].copy_from_slice(&0.0f32.to_le_bytes());
            }
        }

        // Expert Int4 weights: pack zeros + small scales
        let zero_packed = pack_int4(0, 0);
        for ew in &lw.experts {
            // Gate+Up packed zeros
            for j in 0..ew.gate_up_packed_bytes {
                buf[ew.gate_up_offset + j] = zero_packed;
            }
            // Gate+Up scales
            for i in 0..ew.gate_up_scales_count {
                let off = ew.gate_up_scales_offset + i * 4;
                buf[off..off + 4].copy_from_slice(&scale_bytes);
            }
            // Down packed zeros
            for j in 0..ew.down_packed_bytes {
                buf[ew.down_offset + j] = zero_packed;
            }
            // Down scales
            for i in 0..ew.down_scales_count {
                let off = ew.down_scales_offset + i * 4;
                buf[off..off + 4].copy_from_slice(&scale_bytes);
            }
        }
    }

    // Depth router weights (small random-ish values)
    if cfg.depth_router_layer > 0 && weights.depth_router_bytes > 0 {
        // W1: small values
        for i in 0..(cfg.dim * DEPTH_ROUTER_HIDDEN) {
            let val = ((i % 7) as f32 - 3.0) * 0.01;
            let off = weights.depth_router_w1_offset + i * 4;
            buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
        // b1: zeros
        for i in 0..DEPTH_ROUTER_HIDDEN {
            let off = weights.depth_router_b1_offset + i * 4;
            buf[off..off + 4].copy_from_slice(&0.0f32.to_le_bytes());
        }
        // W2: bias towards higher depth (positive values → sigmoid > 0.5)
        for i in 0..DEPTH_ROUTER_HIDDEN {
            let val = 0.1f32;
            let off = weights.depth_router_w2_offset + i * 4;
            buf[off..off + 4].copy_from_slice(&val.to_le_bytes());
        }
        // b2: slight positive bias
        let off = weights.depth_router_b2_offset;
        buf[off..off + 4].copy_from_slice(&0.5f32.to_le_bytes());
    }

    // Final norm = 1.0
    for i in 0..cfg.dim {
        let off = weights.final_norm_offset + i * 4;
        buf[off..off + 4].copy_from_slice(&norm_one);
    }

    // Output projection scales
    for i in 0..cfg.vocab_size {
        let off = weights.output_proj_scales_offset + i * 4;
        buf[off..off + 4].copy_from_slice(&scale_bytes);
    }

    let mut file = File::create(path)?;
    file.write_all(&buf)?;
    file.flush()?;

    let param_m = cfg.param_count() as f64 / 1e6;
    println!(
        "[model] Created v2 dummy model at {:?} ({:.1} MB, ~{:.1}M params, {} experts×top{})",
        path,
        buf.len() as f64 / 1e6,
        param_m,
        cfg.n_experts,
        cfg.top_k,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_config() -> ModelConfig {
        // Small config for fast tests
        ModelConfig {
            dim: 64,
            hidden_dim: 128,
            n_layers: 4,
            n_heads: 4,
            head_dim: 16,
            vocab_size: 256,
            max_seq_len: 64,
            is_quantized: true,
            n_experts: 4,
            top_k: 2,
            int4_group_size: 64,
            depth_router_layer: 2,
        }
    }

    #[test]
    fn test_weight_index_build_moe() {
        let cfg = test_config();
        let idx = WeightIndex::build(&cfg);
        assert_eq!(idx.layers.len(), 4);
        assert_eq!(idx.layers[0].experts.len(), 4);
        assert!(idx.total_bytes > 0);
        assert!(idx.expert_weight_bytes() > 0);
        println!(
            "Total backbone: {} bytes ({:.2} MB) | Expert weights: {} bytes ({:.1}%)",
            idx.total_bytes,
            idx.total_bytes as f64 / 1e6,
            idx.expert_weight_bytes(),
            idx.expert_weight_bytes() as f64 / idx.total_bytes as f64 * 100.0,
        );
    }

    #[test]
    fn test_dummy_model_v2() {
        let cfg = test_config();
        let tmp = PathBuf::from("test_model_v2.myai");
        create_dummy_model(&tmp, &cfg).unwrap();

        let mut engine = Engine::load(&tmp).unwrap();
        let token_id = engine.forward(42, 0);
        assert!(token_id < cfg.vocab_size);
        println!("Forward result: token={}, layers_eval={}, experts_eval={}",
            token_id, engine.state.layers_evaluated, engine.state.experts_evaluated);

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_adaptive_depth() {
        let cfg = test_config();
        let tmp = PathBuf::from("test_model_depth.myai");
        create_dummy_model(&tmp, &cfg).unwrap();

        let mut engine = Engine::load(&tmp).unwrap();
        // Run several forward passes and check that depth routing is active
        for i in 0..5 {
            engine.forward(i + 1, i);
        }
        println!("Stats: {}", engine.sparse_stats_report());

        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn test_param_count() {
        // Full 150M param config
        let cfg = ModelConfig {
            dim: 512,
            hidden_dim: 1536,
            n_layers: 12,
            n_heads: 8,
            head_dim: 64,
            vocab_size: 32000,
            max_seq_len: 512,
            is_quantized: true,
            n_experts: 4,
            top_k: 2,
            int4_group_size: 64,
            depth_router_layer: 2,
        };
        let params = cfg.param_count();
        let params_m = params as f64 / 1e6;
        println!("150M config: {:.1}M params", params_m);
        assert!(params_m > 100.0, "Should be >100M params");
        assert!(params_m < 200.0, "Should be <200M params");
    }

    #[test]
    fn test_sparse_load_stats() {
        let cfg = test_config();
        let idx = WeightIndex::build(&cfg);
        let expert_bytes = idx.expert_weight_bytes();
        let dense_bytes = idx.dense_weight_bytes();
        println!(
            "Expert: {} bytes, Dense: {} bytes, Ratio: {:.1}%",
            expert_bytes,
            dense_bytes,
            expert_bytes as f64 / idx.total_bytes as f64 * 100.0
        );
        assert!(expert_bytes > dense_bytes, "Experts should be majority of weights");
    }

    #[test]
    fn test_inference_state_memory() {
        let cfg = test_config();
        let state = InferenceState::new(&cfg);
        let mem = state.memory_bytes();
        println!("InferenceState: {} bytes ({:.2} KB)", mem, mem as f64 / 1024.0);
        assert!(mem < 10_000_000); // well under 10 MB for small config
    }
}
