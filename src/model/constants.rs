// =============================================================================
// model/constants.rs — Model Constants
// =============================================================================

pub const MAGIC: u32 = 0x4D594149; // "MYAI"
pub const HEADER_SIZE: usize = 256;
pub const MAX_SEQ_LEN: usize = 512;
pub const DEPTH_ROUTER_HIDDEN: usize = 32;  // doubled for deeper reasoning assessment
pub const DEPTH_ROUTER_AFTER_LAYER: usize = 2; // compute depth score after this many layers
pub const ATTN_LOGIT_CAP: f32 = 30.0;  // attention soft-cap prevents entropy collapse
