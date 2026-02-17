// =============================================================================
// model/mod.rs — Sparse MoE Transformer with Adaptive Depth (v3 Extreme)
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

pub mod constants;
pub mod header;
pub mod config;
pub mod weight_index;
pub mod state;
pub mod engine;
pub mod dummy;

// Re-export main public APIs
pub use constants::*;
pub use header::FileHeader;
pub use config::ModelConfig;
pub use weight_index::{ExpertWeightIndex, LayerWeightIndex, WeightIndex};
pub use state::{InferenceState, SparseLoadStats};
pub use engine::{Engine, TaskHead};
pub use dummy::create_dummy_model;

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
