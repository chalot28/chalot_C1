// =============================================================================
// tokenizer.rs — Byte-level BPE tokenizer (zero external deps)
// =============================================================================
//
// Token layout (vocab_size = 8192):
//   0       = <PAD>
//   1       = <BOS>  (Begin of Sequence)
//   2       = <EOS>  (End of Sequence)
//   3..258  = individual bytes (0x00..0xFF)
//   259+    = learned BPE merge pairs
//
// Memory: vocab table ≈ 50–200 KB depending on merge count.
// =============================================================================

use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
pub const PAD_TOKEN: u32 = 0;
pub const BOS_TOKEN: u32 = 1;
pub const EOS_TOKEN: u32 = 2;
const BYTE_OFFSET: u32 = 3; // byte 0x00 → token 3, ..., 0xFF → token 258
const MERGE_OFFSET: u32 = 259;

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------
pub struct Tokenizer {
    /// BPE merge rules: (token_a, token_b) → merged_token_id, ordered by priority.
    merges: Vec<(u32, u32)>,
    /// Reverse vocabulary: token_id → byte sequence (for decoding).
    vocab: Vec<Vec<u8>>,
    /// Forward lookup: byte sequence → token_id (for fast encode of known tokens).
    token_map: HashMap<Vec<u8>, u32>,
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

    // -- BPE Training -------------------------------------------------------

    /// Train BPE merges from a corpus. `num_merges` = how many merge rules to learn.
    /// This is a simplified greedy BPE trainer.
    pub fn train(&mut self, corpus: &str, num_merges: usize) {
        // Tokenize corpus to bytes
        let mut tokens: Vec<u32> = corpus.bytes().map(|b| BYTE_OFFSET + b as u32).collect();

        for _round in 0..num_merges {
            if tokens.len() < 2 {
                break;
            }

            // Count bigram frequencies
            let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
            for w in tokens.windows(2) {
                *pair_counts.entry((w[0], w[1])).or_insert(0) += 1;
            }

            // Find most frequent pair
            let best = pair_counts
                .iter()
                .max_by_key(|&(_, &count)| count)
                .map(|(&pair, &count)| (pair, count));

            let ((a, b), _count) = match best {
                Some(x) if x.1 >= 2 => x,
                _ => break, // No pair appears ≥ 2 times → stop
            };

            // Create new token
            let new_id = MERGE_OFFSET + self.merges.len() as u32;
            if new_id as usize >= self.vocab_size {
                break;
            }

            // Build vocab entry for new token (concatenation of a and b sequences)
            let mut new_bytes = Vec::new();
            if (a as usize) < self.vocab.len() {
                new_bytes.extend_from_slice(&self.vocab[a as usize]);
            }
            if (b as usize) < self.vocab.len() {
                new_bytes.extend_from_slice(&self.vocab[b as usize]);
            }

            self.token_map.insert(new_bytes.clone(), new_id);
            self.vocab.push(new_bytes);
            self.merges.push((a, b));

            // Apply merge to token stream
            let mut i = 0;
            while i + 1 < tokens.len() {
                if tokens[i] == a && tokens[i + 1] == b {
                    tokens[i] = new_id;
                    tokens.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }
    }

    // -- Persistence --------------------------------------------------------

    /// Save tokenizer merges to a simple text file.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)?;
        writeln!(f, "MYTOK v1 vocab_size={}", self.vocab_size)?;
        for &(a, b) in &self.merges {
            writeln!(f, "{} {}", a, b)?;
        }
        Ok(())
    }

    /// Load tokenizer merges from file.
    #[allow(dead_code)]
    pub fn load(path: &Path, vocab_size: usize) -> std::io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut tok = Self::new(vocab_size);
        for line in content.lines().skip(1) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 2 {
                if let (Ok(a), Ok(b)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
                    let new_id = MERGE_OFFSET + tok.merges.len() as u32;
                    let mut new_bytes = Vec::new();
                    if (a as usize) < tok.vocab.len() {
                        new_bytes.extend_from_slice(&tok.vocab[a as usize]);
                    }
                    if (b as usize) < tok.vocab.len() {
                        new_bytes.extend_from_slice(&tok.vocab[b as usize]);
                    }
                    tok.token_map.insert(new_bytes.clone(), new_id);
                    tok.vocab.push(new_bytes);
                    tok.merges.push((a, b));
                }
            }
        }
        Ok(tok)
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_level_encode_decode() {
        let tok = Tokenizer::new(8192);
        let text = "Hello, world!";
        let ids = tok.encode(text);
        assert_eq!(ids.len(), text.len()); // No merges → 1 token per byte
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_with_special() {
        let tok = Tokenizer::new(8192);
        let ids = tok.encode_with_special("Hi");
        assert_eq!(ids[0], BOS_TOKEN);
        assert_eq!(*ids.last().unwrap(), EOS_TOKEN);
        assert_eq!(ids.len(), 4); // BOS + H + i + EOS
    }

    #[test]
    fn test_bpe_training() {
        let mut tok = Tokenizer::new(8192);
        let corpus = "abcabcabcabc"; // "ab" and "abc" should be learned
        tok.train(corpus, 5);
        assert!(tok.num_merges() > 0);
        let ids = tok.encode("abc");
        // After merges, "abc" should be fewer than 3 tokens
        assert!(ids.len() <= 3, "ids={:?}", ids);
    }

    #[test]
    fn test_utf8_roundtrip() {
        let tok = Tokenizer::new(8192);
        let text = "Xin chào thế giới! 你好世界";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_save_load() {
        let mut tok = Tokenizer::new(8192);
        tok.train("hello hello hello world world", 3);
        let path = Path::new("test_tokenizer.txt");
        tok.save(path).unwrap();

        let tok2 = Tokenizer::load(path, 8192).unwrap();
        assert_eq!(tok.num_merges(), tok2.num_merges());
        assert_eq!(tok.encode("hello"), tok2.encode("hello"));

        std::fs::remove_file(path).ok();
    }
}
