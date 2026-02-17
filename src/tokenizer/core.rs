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
    /// Reverse vocabulary: token_id → byte sequence (for decoding).
    pub(crate) vocab: Vec<Vec<u8>>,
    /// Forward lookup: byte sequence → token_id (for fast encode of known tokens).
    pub(crate) token_map: HashMap<Vec<u8>, u32>,
    /// Maximum vocabulary size.
    pub vocab_size: usize,
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
            vocab,
            token_map,
            vocab_size,
        }
    }

    // -- Encoding -----------------------------------------------------------

    /// Encode text → token IDs using byte-level BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Pre-tokenization: Split by whitespace and punctuation to preserve grammar structure.
        // This mimics GPT-4/Llama style pre-tokenization for better associative reasoning.
        let mut final_tokens = Vec::new();
        
        // Simple regex-like split: separate alphanumeric from punctuation/whitespace
        let mut current_chunk = String::new();
        for c in text.chars() {
            // If category changes (e.g. letter -> symbol), flush chunk
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
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::with_capacity(tokens.len());
        for &t in tokens {
            let id = t as usize;
            if id < self.vocab.len() {
                // Skip special tokens in output
                if t == PAD_TOKEN || t == BOS_TOKEN || t == EOS_TOKEN {
                    continue;
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

    // -- Stats --------------------------------------------------------------

    pub fn num_merges(&self) -> usize {
        self.merges.len()
    }

    pub fn memory_bytes(&self) -> usize {
        let vocab_mem: usize = self.vocab.iter().map(|v| v.len() + 24).sum();
        let merge_mem = self.merges.len() * 8;
        let map_mem = self.token_map.len() * 48; // rough estimate
        vocab_mem + merge_mem + map_mem
    }
}
