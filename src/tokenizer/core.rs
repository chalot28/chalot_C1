// =============================================================================
// tokenizer/core.rs — Core tokenizer struct and encoding/decoding
// =============================================================================

use std::collections::HashMap;
use super::constants::*;

/// Byte-level BPE tokenizer (zero external deps)
/// Memory: vocab table ≈ 50–200 KB depending on merge count.
pub struct Tokenizer {
    /// BPE merge rules: (token_a, token_b) → merged_token_id, ordered by priority.
    pub(crate) merges: Vec<(u32, u32)>,
    /// Merge lookup: (left_id, right_id) → merge priority (lower = higher priority).
    pub(crate) merge_ranks: HashMap<(u32, u32), usize>,
    /// Merge result: (left_id, right_id) → merged_id (precomputed at load time).
    pub(crate) merge_results: HashMap<(u32, u32), u32>,
    /// Reverse vocabulary: token_id → byte sequence (for decoding).
    pub(crate) vocab: Vec<Vec<u8>>,
    /// Forward lookup: byte sequence → token_id (for fast encode of known tokens).
    pub(crate) token_map: HashMap<Vec<u8>, u32>,
    /// GPT-2 style byte→token_id table: maps raw byte value to its initial token ID.
    /// Only populated in HF mode.
    pub(crate) byte_to_token: [u32; 256],
    /// Maximum vocabulary size.
    pub vocab_size: usize,
    /// Whether this tokenizer was loaded from HuggingFace binary vocab.
    /// In HF mode, token IDs map directly to vocab entries (no byte offset).
    pub(crate) hf_mode: bool,
}

impl Tokenizer {
    /// Create a new tokenizer with byte-level fallback (no merges yet).
    pub fn new(vocab_size: usize) -> Self {
        let mut vocab = Vec::with_capacity(vocab_size.min(512));
        let mut token_map = HashMap::new();

        // Special tokens
        let specials = [b"<PAD>".as_slice(), b"<BOS>", b"<EOS>"];
        for (id, &bytes) in specials.iter().enumerate() {
            vocab.push(bytes.to_vec());
            token_map.insert(bytes.to_vec(), id as u32);
        }

        // Byte tokens (3..258)
        for b in 0u8..=255 {
            let v = vec![b];
            token_map.insert(v.clone(), BYTE_OFFSET + b as u32);
            vocab.push(v);
        }

        Tokenizer {
            merges: Vec::new(),
            merge_ranks: HashMap::new(),
            merge_results: HashMap::new(),
            vocab,
            token_map,
            byte_to_token: [0u32; 256],
            vocab_size,
            hf_mode: false,
        }
    }

    // -- Encoding -----------------------------------------------------------

    /// Encode text → token IDs using byte-level BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // HF mode with BPE merges: proper byte-pair encoding
        if self.hf_mode && !self.merges.is_empty() {
            return self.encode_bpe_hf(text);
        }

        // HF mode without merges: fallback to greedy longest-match
        if self.hf_mode {
            return self.encode_greedy(text);
        }

        // Legacy mode: custom byte offsets with merge rules
        let mut final_tokens = Vec::new();
        
        // Simple regex-like split: separate alphanumeric from punctuation/whitespace
        let mut current_chunk = String::new();
        for c in text.chars() {
            if !current_chunk.is_empty() {
                let last_char = current_chunk.chars().last().unwrap();
                if c.is_alphanumeric() != last_char.is_alphanumeric() || c.is_whitespace() {
                    final_tokens.extend(self.encode_chunk(&current_chunk));
                    current_chunk.clear();
                }
            }
            current_chunk.push(c);
        }
        if !current_chunk.is_empty() {
            final_tokens.extend(self.encode_chunk(&current_chunk));
        }
        
        final_tokens
    }

    /// BPE encoding for HuggingFace vocab with merge rules.
    /// This properly implements byte-pair encoding:
    /// 1. Start with each byte mapped to its unicode token (GPT-2 style)
    /// 2. Iteratively merge the highest-priority adjacent pair
    fn encode_bpe_hf(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        if bytes.is_empty() {
            return Vec::new();
        }

        // Step 1: Map each byte to its token ID via GPT-2 byte-to-unicode table
        let mut tokens: Vec<u32> = bytes.iter().map(|&b| {
            self.byte_to_token[b as usize]
        }).collect();

        // Step 2: Iteratively apply BPE merges (highest priority = lowest rank)
        loop {
            if tokens.len() < 2 {
                break;
            }

            // Find the pair with the lowest rank (highest priority)
            let mut best_rank = usize::MAX;
            let mut best_idx = 0;

            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i], tokens[i + 1]);
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_idx = i;
                    }
                }
            }

            if best_rank == usize::MAX {
                break; // No more merges applicable
            }

            // Apply the merge
            let pair = (tokens[best_idx], tokens[best_idx + 1]);
            if let Some(&merged_id) = self.merge_results.get(&pair) {
                tokens[best_idx] = merged_id;
                tokens.remove(best_idx + 1);
            } else {
                break; // Shouldn't happen, but safety check
            }
        }

        tokens
    }

    /// Greedy longest-match encoding for HuggingFace vocab.
    /// Finds the longest matching token at each position in the text.
    fn encode_greedy(&self, text: &str) -> Vec<u32> {
        let bytes = text.as_bytes();
        let mut tokens = Vec::new();
        let mut pos = 0;

        while pos < bytes.len() {
            let mut best_len = 0;
            let mut best_id = None;

            // Try increasingly longer byte sequences starting at pos
            let max_check = (bytes.len() - pos).min(64); // Max token length to check
            for len in 1..=max_check {
                let candidate = &bytes[pos..pos + len];
                if let Some(&id) = self.token_map.get(candidate) {
                    best_len = len;
                    best_id = Some(id);
                }
            }

            if let Some(id) = best_id {
                tokens.push(id);
                pos += best_len;
            } else {
                // Fallback: emit byte as single-byte token if exists, else skip
                let single = &bytes[pos..pos + 1];
                if let Some(&id) = self.token_map.get(single) {
                    tokens.push(id);
                } else {
                    // Unknown byte — try to find any single-byte match
                    // This shouldn't happen if the vocab has all byte tokens
                }
                pos += 1;
            }
        }

        tokens
    }

    /// Internal helper: Encode a single pre-tokenized chunk
    fn encode_chunk(&self, text: &str) -> Vec<u32> {
        // Start: each byte → its own token
        let mut tokens: Vec<u32> = text.bytes().map(|b| BYTE_OFFSET + b as u32).collect();

        // Apply merges in priority order (lower index = higher priority)
        for (idx, &(a, b)) in self.merges.iter().enumerate() {
            let merged = MERGE_OFFSET + idx as u32;
            if merged as usize >= self.vocab_size {
                break;
            }
            let mut i = 0;
            while i + 1 < tokens.len() {
                if tokens[i] == a && tokens[i + 1] == b {
                    tokens[i] = merged;
                    tokens.remove(i + 1);
                    // Don't advance i — check if new token merges with next
                    if i > 0 { i -= 1; }
                } else {
                    i += 1;
                }
            }
        }
        tokens
    }

    /// Encode with BOS/EOS wrapping.
    #[allow(dead_code)]
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let mut tokens = vec![BOS_TOKEN];
        tokens.extend(self.encode(text));
        tokens.push(EOS_TOKEN);
        tokens
    }

    // -- Decoding -----------------------------------------------------------

    /// Decode token IDs → String.
    #[allow(dead_code)]
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(tokens.len());
        for &t in tokens {
            let id = t as usize;
            if id < self.vocab.len() {
                // In HF mode, all tokens are valid (no special token skipping)
                if !self.hf_mode {
                    // Skip special tokens in legacy mode
                    if t == PAD_TOKEN || t == BOS_TOKEN || t == EOS_TOKEN {
                        continue;
                    }
                }
                bytes.extend_from_slice(&self.vocab[id]);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode a single token to its string representation.
    #[allow(dead_code)]
    pub fn decode_token(&self, t: u32) -> String {
        let id = t as usize;
        if id < self.vocab.len() {
            String::from_utf8_lossy(&self.vocab[id]).into_owned()
        } else {
            format!("<UNK:{}>", t)
        }
    }

    /// Look up a token ID by its text representation.
    /// Useful for finding special tokens like `<|im_end|>`.
    #[allow(dead_code)]
    pub fn find_token_id(&self, text: &str) -> Option<u32> {
        self.token_map.get(text.as_bytes()).copied()
    }

    // -- Stats --------------------------------------------------------------

    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }

    pub fn memory_bytes(&self) -> usize {
        let vocab_mem: usize = self.vocab.iter().map(|v| v.len() + 24).sum();
        let merge_mem = self.merges.len() * 8;
        let rank_mem = self.merge_ranks.len() * 24; // HashMap overhead
        let result_mem = self.merge_results.len() * 20;
        let map_mem = self.token_map.len() * 48; // rough estimate
        vocab_mem + merge_mem + rank_mem + result_mem + map_mem
    }
}
