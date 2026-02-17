// =============================================================================
// tokenizer/persistence.rs — Save/Load tokenizer
// =============================================================================

use std::path::Path;
use super::constants::*;
use super::core::Tokenizer;

impl Tokenizer {
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
}
