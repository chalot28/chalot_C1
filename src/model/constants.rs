// =============================================================================
// model/constants.rs — Model Constants
// =============================================================================

pub const MAGIC: u32 = 0x4D594149; // "MYAI"
pub const HEADER_SIZE: usize = 256;
pub const MAX_SEQ_LEN: usize = 512;
pub const DEPTH_ROUTER_HIDDEN: usize = 32;  // doubled for deeper reasoning assessment
pub const DEPTH_ROUTER_AFTER_LAYER: usize = 2; // compute depth score after this many layers
pub const ATTN_LOGIT_CAP: f32 = 30.0;  // attention soft-cap prevents entropy collapse

// Qwen2.5-0.5B specific constants
#[allow(dead_code)]
pub const QWEN_DIM: usize = 896;
#[allow(dead_code)]
pub const QWEN_HIDDEN_DIM: usize = 4864;
#[allow(dead_code)]
pub const QWEN_N_LAYERS: usize = 24;
#[allow(dead_code)]
pub const QWEN_N_HEADS: usize = 14;
#[allow(dead_code)]
pub const QWEN_VOCAB_SIZE: usize = 151936;
#[allow(dead_code)]
pub const QWEN_MAX_SEQ_LEN: usize = 2048;

// MoE up-cycling config for Qwen
#[allow(dead_code)]
pub const QWEN_N_EXPERTS: usize = 8;  // Up-cycle from dense to 8 experts
#[allow(dead_code)]
pub const QWEN_TOP_K: usize = 2;      // Activate top-2 experts per token
#[allow(dead_code)]
pub const QWEN_EXPERT_NOISE_STD: f32 = 0.01; // Gaussian noise for expert diversification

// Brain Map partitioning for Qwen 24-layer
#[allow(dead_code)]
pub const QWEN_SHALLOW_START: usize = 0;
#[allow(dead_code)]
pub const QWEN_SHALLOW_END: usize = 6;
#[allow(dead_code)]
pub const QWEN_DEEP_START: usize = 6;
#[allow(dead_code)]
pub const QWEN_DEEP_END: usize = 18;
#[allow(dead_code)]
pub const QWEN_FACT_START: usize = 18;
#[allow(dead_code)]
pub const QWEN_FACT_END: usize = 24;
