// =============================================================================
// model/dummy.rs — Create Dummy Test Models
// =============================================================================

use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::tensor::pack_int4;

use super::config::ModelConfig;
use super::constants::{MAGIC, DEPTH_ROUTER_HIDDEN};
use super::header::FileHeader;
use super::weight_index::WeightIndex;

/// Create dummy .myai v2 file for testing
pub fn create_dummy_model(path: &Path, cfg: &ModelConfig) -> std::io::Result<()> {
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
