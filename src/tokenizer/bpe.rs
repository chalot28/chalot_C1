// =============================================================================
// tokenizer/bpe.rs — BPE Training logic
// =============================================================================

use std::collections::HashMap;
use super::constants::*;
use super::core::Tokenizer;

impl Tokenizer {
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
}
