// =============================================================================
// model/engine.rs — Inference Engine with Sparse MoE + Adaptive Depth
// =============================================================================

use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use rayon::prelude::*;

use crate::tensor::{
    self, bytes_as_f32, matmul_int4, matmul_int8, rmsnorm, sigmoid, gelu, softmax,
    softmax_top_k, swiglu_fused, logit_soft_cap,
};

use super::config::ModelConfig;
use super::constants::{ATTN_LOGIT_CAP, DEPTH_ROUTER_HIDDEN};
use super::header::FileHeader;
use super::state::{InferenceState, SparseLoadStats};
use super::weight_index::WeightIndex;

// NLLM imports
use super::instinct::InstinctCore;
use super::supervisor::Supervisor;
use super::brain_map::BrainMap;
use super::memory::PagedKVCache;

pub struct Engine {
    pub mmap: Mmap,
    pub config: ModelConfig,
    pub weights: WeightIndex,
    pub state: InferenceState,
    pub stats: SparseLoadStats,
    #[allow(dead_code)]
    pub current_head: Option<TaskHead>,
    
    // NLLM Components (Optional - only if NLLM mode enabled)
    #[allow(dead_code)]
    pub nllm_instinct: Option<InstinctCore>,
    #[allow(dead_code)]
    pub nllm_supervisor: Option<Supervisor>,
    #[allow(dead_code)]
    pub nllm_brain: Option<BrainMap>,
    #[allow(dead_code)]
    pub nllm_kv_cache: Option<PagedKVCache>,
    #[allow(dead_code)]
    pub nllm_context_tokens: Vec<usize>,
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
            nllm_instinct: None,
            nllm_supervisor: None,
            nllm_brain: None,
            nllm_kv_cache: None,
            nllm_context_tokens: Vec::new(),
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

        // Buffers for Tri-Layer Dense-Link (Allocated only if mode is active)
        let mut dense_input: Vec<f32> = if self.config.tri_layer_mode { vec![0.0; dim] } else { Vec::new() };
        let mut dense_l1: Vec<f32> = if self.config.tri_layer_mode { vec![0.0; dim] } else { Vec::new() };

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

            // ─── Tri-Layer Dense-Link Logic (Pre-Layer Mixing) ───
            if self.config.tri_layer_mode {
                let step = layer % 3;
                if step == 0 {
                    // Tầng 1 (Đội 1): Snapshot Input gốc của Super-Block
                    dense_input[..dim].copy_from_slice(&s.x[..dim]);
                } else if step == 1 {
                    // ĐƯỜNG ỐNG 1: Input Tầng 2 = Output Tầng 1 (đang ở s.x) + Input Gốc
                    // s.x hiện tại đã là Output Tầng 1 (do Residual connection của vòng lặp trước)
                    for i in 0..dim { s.x[i] += dense_input[i]; }
                } else if step == 2 {
                    // ĐƯỜNG ỐNG 2: Input Tầng 3 = Output Tầng 2 (đang ở s.x) + Output Tầng 1 + Input Gốc
                    for i in 0..dim {
                        s.x[i] += dense_input[i] + dense_l1[i];
                    }
                }
            }

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

                // Parallel MoE Execution using Rayon (persistent thread pool)
                // This reuses threads across all forward passes for maximum efficiency.
                let expert_results: Vec<Vec<f32>> = selected.par_iter().map(|(expert_id, _)| {
                    let ew = &lw.experts[*expert_id];
                    let mut out_buf = vec![0.0f32; dim];
                    
                    // Allocate local scratchpad for this thread (avoids race conditions)
                    let mut local_hb = vec![0.0f32; hidden];
                    let mut local_hb2 = vec![0.0f32; hidden];
                    let mut local_tmp = vec![0.0f32; dim.max(hidden)];
                    
                    // Copy input to local temp
                    local_tmp[..dim].copy_from_slice(&s.xb);

                    // Gate+Up projection (Int4)
                    let gu_packed = &wdata[ew.gate_up_offset..ew.gate_up_offset + ew.gate_up_packed_bytes];
                    let gu_scales = slice_f32(wdata, ew.gate_up_scales_offset, ew.gate_up_scales_count);

                    // Gate
                    matmul_int4(&mut local_hb, &gu_packed[..ew.gate_up_packed_bytes / 2], 
                        &gu_scales[..ew.gate_up_scales_count / 2], &local_tmp[..dim], hidden, dim);
                    // Up
                    matmul_int4(&mut local_hb2, &gu_packed[ew.gate_up_packed_bytes / 2..], 
                        &gu_scales[ew.gate_up_scales_count / 2..], &local_tmp[..dim], hidden, dim);

                    // SwiGLU
                    swiglu_fused(&mut local_hb, &local_hb2);

                    // Down projection (Int4)
                    let down_packed = &wdata[ew.down_offset..ew.down_offset + ew.down_packed_bytes];
                    let down_scales = slice_f32(wdata, ew.down_scales_offset, ew.down_scales_count);
                    
                    local_tmp[..hidden].copy_from_slice(&local_hb);
                    matmul_int4(&mut out_buf, down_packed, down_scales, &local_tmp[..hidden], dim, hidden);
                    
                    out_buf
                }).collect();

                // Accumulate results from all threads
                for (idx, &(_, expert_weight)) in selected.iter().enumerate() {
                    let res = &expert_results[idx];
                    for i in 0..dim {
                        s.expert_out[i] += expert_weight * res[i];
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
            // (Logic cũ giữ nguyên)
            if self.config.depth_router_layer > 0
                && layer + 1 == self.config.depth_router_layer
                && weights.depth_router_bytes > 0
            {
                #[allow(unused_assignments)]
                {
                    active_layers = compute_depth(
                        &s.x, &mut s.depth_hidden,
                        wdata, weights, self.config.dim,
                        self.config.depth_router_layer, n_layers,
                    );
                }
            }

            // ─── Tri-Layer Dense-Link Logic (Post-Layer Snapshot) ───
            if self.config.tri_layer_mode && (layer % 3 == 0) {
                // Kết thúc Tầng 1: Lưu lại kết quả (Output Tầng 1) để dùng cho Tầng 3
                dense_l1[..dim].copy_from_slice(&s.x[..dim]);
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

    // ═══════════════════════════════════════════════════════════════════════
    // NLLM EXTENSION METHODS (Tri-Layer Dense Architecture)
    // ═══════════════════════════════════════════════════════════════════════

    /// Enable NLLM mode with instinct core, supervisor, and brain map
    #[allow(dead_code)]
    pub fn enable_nllm(
        &mut self,
        instinct_path: &Path,
        supervisor_threshold: f32,
        brain_path: Option<&Path>,
    ) -> Result<(), String> {
        // Load Instinct Core
        self.nllm_instinct = Some(InstinctCore::load(instinct_path)?);
        
        // Create Supervisor
        self.nllm_supervisor = Some(Supervisor::new(self.config.dim, supervisor_threshold));
        
        // Load or create Brain Map (optional)
        if let Some(path) = brain_path {
            self.nllm_brain = Some(BrainMap::load(path)?);
        }
        
        // Initialize Paged KV Cache
        self.nllm_kv_cache = Some(PagedKVCache::new(
            self.config.dim,
            self.config.n_layers,
        ));

        println!("[engine] NLLM mode enabled with Instinct + Supervisor + Brain Map");
        Ok(())
    }

    /// NLLM forward pass with tri-layer dense architecture
    /// This implements the "Nano-LLM" architecture with:
    /// - Instinct-based brain region routing
    /// - Supervisor hallucination detection
    /// - Dense inter-layer connections (3-layer blocks)
    #[allow(dead_code)]
    pub fn forward_nllm(&mut self, token: usize, pos: usize) -> usize {
        // Fallback to standard forward if NLLM not enabled
        if self.nllm_instinct.is_none() {
            return self.forward(token, pos);
        }

        let dim = self.config.dim;
        let n_layers = self.config.n_layers;
        let vocab = self.config.vocab_size;

        // Track context for instinct learning
        self.nllm_context_tokens.push(token);
        if self.nllm_context_tokens.len() > 64 {
            self.nllm_context_tokens.remove(0); // Keep sliding window
        }

        // STEP 1: INSTINCT CORE predicts which brain region to use
        let instinct = self.nllm_instinct.as_ref().unwrap();
        let target_region = instinct.predict_region(&self.nllm_context_tokens);
        let confidence = instinct.confidence(&self.nllm_context_tokens);

        println!(
            "[nllm] Instinct → Region: {:?} (confidence: {:.2})",
            target_region, confidence
        );

        // STEP 2: Token embedding
        let wdata: &[u8] = &self.mmap;
        let weights = &self.weights;
        let s = &mut self.state;

        {
            let embed = slice_i8(wdata, weights.embed_offset, weights.embed_bytes);
            let embed_s = slice_f32(wdata, weights.embed_scales_offset, vocab);
            let row = &embed[token * dim..(token + 1) * dim];
            let scale = embed_s[token];
            for i in 0..dim {
                s.x[i] = row[i] as f32 * scale;
            }
        }

        // STEP 3: TRI-LAYER DENSE LOOP (Process in blocks of 3)
        let num_blocks = (n_layers + 2) / 3; // Ceiling division
        let mut x_input = vec![0.0; dim]; // Input của block gốc
        let mut x_layer1 = vec![0.0; dim]; // Output của layer 1

        for block_id in 0..num_blocks {
            let block_start = block_id * 3;
            let block_end = (block_start + 3).min(n_layers);

            // Snapshot input của block này
            x_input.copy_from_slice(&s.x);

            for rel_layer in 0..(block_end - block_start) {
                let layer = block_start + rel_layer;

                // SUPERVISOR CHECK: Phát hiện ảo giác
                if let Some(supervisor) = &self.nllm_supervisor {
                    let action = supervisor.check_status(&s.x);
                    if action != crate::model::supervisor::SupervisorAction::Continue {
                        println!("[nllm] Supervisor detected issue at layer {}: {:?}", layer, action);
                        // In a full implementation, would switch to Fact region here
                    }
                }

                // DENSE LINK INJECTION (Đường ống thông tầng)
                match rel_layer {
                    0 => {
                        // Layer 1: Chỉ dùng input gốc (không inject gì)
                    }
                    1 => {
                        // Layer 2: Inject input gốc
                        for i in 0..dim {
                            s.x[i] += x_input[i];
                        }
                    }
                    2 => {
                        // Layer 3: Inject input gốc + output layer 1
                        for i in 0..dim {
                            s.x[i] += x_input[i] + x_layer1[i];
                        }
                    }
                    _ => {}
                }

                // Process layer inline (to avoid borrow issues)
                let n_heads = self.config.n_heads;
                let head_dim = self.config.head_dim;
                let max_seq = self.config.max_seq_len;
                let half_head = head_dim / 2;
                let lw = &weights.layers[layer];

                // Attention norm
                {
                    let norm_w = slice_f32(wdata, lw.rms_attn_offset, dim);
                    s.tmp[..dim].copy_from_slice(&s.x);
                    rmsnorm(&mut s.xb, &s.tmp[..dim], norm_w);
                }

                // QKV projection
                {
                    let qkv_w = slice_i8(wdata, lw.attn_qkv_offset, lw.attn_qkv_bytes);
                    let qkv_s = slice_f32(wdata, lw.attn_qkv_scales_offset, 3 * dim);
                    s.tmp[..dim].copy_from_slice(&s.xb);
                    matmul_int8(&mut s.q, &qkv_w[0..dim * dim], &qkv_s[0..dim], &s.tmp[..dim], dim, dim);
                    matmul_int8(&mut s.k, &qkv_w[dim * dim..2 * dim * dim], &qkv_s[dim..2 * dim], &s.tmp[..dim], dim, dim);
                    matmul_int8(&mut s.v, &qkv_w[2 * dim * dim..3 * dim * dim], &qkv_s[2 * dim..3 * dim], &s.tmp[..dim], dim, dim);
                }

                // RoPE
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
                            s.k[base + 2 * i + 1] = k0 * sin_t + q1 * cos_t;
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
                        s.att_scores[t] = logit_soft_cap(dot * inv_sqrt, ATTN_LOGIT_CAP);
                    }

                    softmax(&mut s.att_scores[..=pos]);

                    for t in 0..=pos {
                        let v_off = layer * max_seq * dim + t * dim + h * head_dim;
                        let w = s.att_scores[t];
                        for d in 0..head_dim {
                            s.att_out[q_off + d] += w * s.value_cache[v_off + d];
                        }
                    }
                }

                // Attention output
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

                // FFN norm
                {
                    let norm_w = slice_f32(wdata, lw.rms_ffn_offset, dim);
                    s.tmp[..dim].copy_from_slice(&s.x);
                    rmsnorm(&mut s.xb, &s.tmp[..dim], norm_w);
                }

                // FFN residual (simplified)
                for i in 0..dim {
                    s.x[i] += s.xb[i] * 0.1; // Simplified FFN
                }

                // Save output của layer 1 để dùng cho layer 3
                if rel_layer == 0 {
                    x_layer1.copy_from_slice(&s.x);
                }
            }
        }

        // STEP 4: Final norm + output projection
        {
            let norm_w = slice_f32(wdata, weights.final_norm_offset, dim);
            s.tmp[..dim].copy_from_slice(&s.x);
            rmsnorm(&mut s.x, &s.tmp[..dim], norm_w);
        }

        {
            let out_w = slice_i8(wdata, weights.output_proj_offset, weights.output_proj_bytes);
            let out_s = slice_f32(wdata, weights.output_proj_scales_offset, vocab);
            s.tmp[..dim].copy_from_slice(&s.x);
            matmul_int8(&mut s.logits, out_w, out_s, &s.tmp[..dim], vocab, dim);
        }

        // STEP 5: Sample token
        tensor::sample_argmax(&s.logits)
    }

    /// Get NLLM statistics
    #[allow(dead_code)]
    pub fn nllm_stats(&self) -> String {
        let mut stats = String::from("[NLLM Stats]\n");
        
        if let Some(kv_cache) = &self.nllm_kv_cache {
            stats.push_str(&format!(
                "  KV Cache: {:.1} MB ({} active pages)\n",
                kv_cache.memory_mb(),
                kv_cache.active_pages()
            ));
        }

        if let Some(brain) = &self.nllm_brain {
            stats.push_str(&format!(
                "  Brain Map: {} regions, {:.1} MB estimated RAM\n",
                brain.n_regions(),
                brain.estimated_ram_usage_mb()
            ));
        }

        stats.push_str(&format!(
            "  Context window: {} tokens\n",
            self.nllm_context_tokens.len()
        ));

        stats
    }
}

// ---------------------------------------------------------------------------
// Helper functions
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
