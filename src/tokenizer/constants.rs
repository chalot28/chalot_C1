// =============================================================================
// tokenizer/constants.rs — Token constants
// =============================================================================
//
// Token layout (vocab_size = 8192):
//   0       = <PAD>
//   1       = <BOS>  (Begin of Sequence)
//   2       = <EOS>  (End of Sequence)
//   3..258  = individual bytes (0x00..0xFF)
//   259+    = learned BPE merge pairs
// =============================================================================

pub const PAD_TOKEN: u32 = 0;
pub const BOS_TOKEN: u32 = 1;
pub const EOS_TOKEN: u32 = 2;
pub const BYTE_OFFSET: u32 = 3; // byte 0x00 → token 3, ..., 0xFF → token 258
pub const MERGE_OFFSET: u32 = 259;
