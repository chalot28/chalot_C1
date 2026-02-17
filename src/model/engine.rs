// =============================================================================
// model/engine.rs — Inference Engine with Sparse MoE + Adaptive Depth
// =============================================================================

use memmap2::Mmap;
use std::fs::File;
use std::path::Path;
use std::thread;

use crate::tensor::{
    self, bytes_as_f32, matmul_int4, matmul_int8, rmsnorm, sigmoid, gelu, softmax,
    softmax_top_k, swiglu_fused, logit_soft_cap,
};

use super::config::ModelConfig;
use super::constants::{ATTN_LOGIT_CAP, DEPTH_ROUTER_HIDDEN};
use super::header::FileHeader;
use super::state::{InferenceState, SparseLoadStats};
use super::weight_index::WeightIndex;

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

                // Parallel MoE Execution: Run selected experts on separate threads
                // This significantly improves latency on multi-core CPUs.
                let expert_results: Vec<Vec<f32>> = thread::scope(|scope| {
                    let handles: Vec<_> = selected.iter().enumerate().map(|(idx, &(expert_id, _))| {
                        let ew = &lw.experts[expert_id];
                        let input_x = &s.xb; // Read-only shared ref
                        
                        // Spawn thread for this expert
                        scope.spawn(move || {
                            let mut out_buf = vec![0.0f32; dim];
                            // Allocate local scratchpad for this thread (avoids race conditions)
                            let mut local_hb = vec![0.0f32; hidden];
                            let mut local_hb2 = vec![0.0f32; hidden];
                            let mut local_tmp = vec![0.0f32; dim.max(hidden)];
                            
                            // Copy input to local temp
                            local_tmp[..dim].copy_from_slice(input_x);

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
                        })
                    }).collect();
                    
                    handles.into_iter().map(|h| h.join().unwrap()).collect()
                });

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
