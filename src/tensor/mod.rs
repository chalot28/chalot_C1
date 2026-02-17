// =============================================================================
// tensor/mod.rs — Module exports
// =============================================================================

mod view;
mod quantization;
mod matmul;
mod activations;
mod normalization;
mod rope;
mod sampling;
mod attention_helpers;
mod bitnet;
mod flash_attention;

#[cfg(test)]
mod tests;

// Re-export public API
#[allow(unused_imports)]
pub use view::{TensorView, bytes_as_f32, bytes_as_f32_mut};
pub use quantization::{
    INT4_GROUP_SIZE,
    quantize_f32_to_i8,
    quantize_f32_to_i4,
    pack_int4,
    unpack_int4,
};
pub use matmul::{matmul_int8, matmul_int4};
pub use bitnet::{
    quantize_f32_to_ternary,
    pack_ternary,
    unpack_ternary,
    matmul_ternary,
    quantize_activation_bitnet,
};
pub use activations::{sigmoid, relu, silu, gelu, swiglu_fused, logit_soft_cap};
pub use flash_attention::{flash_attention_forward, flash_attention_forward_capped};
pub use normalization::{rmsnorm, softmax, softmax_top_k};
pub use rope::apply_rope;
pub use sampling::{
    sample_argmax,
    sample_top_k,
    sample_top_p,
    sample_min_p,
    apply_repetition_penalty,
};
pub use attention_helpers::dot_f32;
