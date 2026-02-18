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

    /// Load tokenizer merges from text file.
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
                    tok.vocab_decoded.push(new_bytes.clone());
                    tok.vocab.push(new_bytes);
                    tok.merges.push((a, b));
                }
            }
        }
        Ok(tok)
    }

    /// Load binary HuggingFace vocab (.mytok v2 format).
    /// Format:
    ///   Header (16 bytes): magic(u32) + version(u32) + vocab_size(u32) + num_merges(u32)
    ///   Vocab: for each token: len(u16) + bytes[len]
    ///   Merges: for each merge: token_a(u32) + token_b(u32)
    pub fn load_hf_vocab(path: &Path) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        if data.len() < 16 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "File too small"));
        }

        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let _version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let vocab_size = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let num_merges = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;

        if magic != 0x544F4B45 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Bad magic: 0x{:08X}, expected 0x544F4B45", magic),
            ));
        }

        let mut vocab = Vec::with_capacity(vocab_size);
        let mut token_map = std::collections::HashMap::new();
        let mut cursor = 16usize;

        // Read vocab entries
        for token_id in 0..vocab_size {
            if cursor + 2 > data.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("EOF reading vocab entry {}", token_id),
                ));
            }
            let len = u16::from_le_bytes([data[cursor], data[cursor + 1]]) as usize;
            cursor += 2;

            if cursor + len > data.len() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("EOF reading vocab bytes for token {}", token_id),
                ));
            }
            let bytes = data[cursor..cursor + len].to_vec();
            cursor += len;

            token_map.insert(bytes.clone(), token_id as u32);
            vocab.push(bytes);
        }

        // Read merges
        let mut merges = Vec::with_capacity(num_merges);
        for _ in 0..num_merges {
            if cursor + 8 > data.len() {
                break;
            }
            let a = u32::from_le_bytes([data[cursor], data[cursor+1], data[cursor+2], data[cursor+3]]);
            let b = u32::from_le_bytes([data[cursor+4], data[cursor+5], data[cursor+6], data[cursor+7]]);
            cursor += 8;
            merges.push((a, b));
        }

        println!("[tokenizer] Loaded HF vocab: {} tokens, {} merges", vocab_size, merges.len());

        // Build merge lookup tables for efficient BPE encoding
        let mut merge_ranks = std::collections::HashMap::with_capacity(merges.len());
        let mut merge_results = std::collections::HashMap::with_capacity(merges.len());

        for (rank, &(a, b)) in merges.iter().enumerate() {
            merge_ranks.insert((a, b), rank);

            // Compute the merged token: concat bytes of a and b, look up in token_map
            let a_idx = a as usize;
            let b_idx = b as usize;
            if a_idx < vocab.len() && b_idx < vocab.len() {
                let mut merged_bytes = vocab[a_idx].clone();
                merged_bytes.extend_from_slice(&vocab[b_idx]);
                if let Some(&merged_id) = token_map.get(&merged_bytes) {
                    merge_results.insert((a, b), merged_id);
                }
            }
        }

        let resolved = merge_results.len();
        println!("[tokenizer] Built merge tables: {}/{} merges resolved", resolved, merges.len());

        // Build GPT-2 byte-to-unicode mapping table:
        // Maps raw byte values (0-255) to their token IDs in the Qwen vocab.
        // GPT-2/Qwen uses a special unicode encoding where:
        // - Printable bytes (33-126, 161-172, 174-255) map to themselves as unicode chars
        // - Non-printable bytes map to unicode chars starting at U+0100
        let mut byte_to_token = [0u32; 256];
        // Also build the REVERSE mapping: unicode_char → raw_byte_value
        // This is needed to decode tokens back to proper text
        let mut unicode_to_byte = std::collections::HashMap::new();
        {
            // Build the GPT-2 bytes_to_unicode table
            let mut printable = Vec::new();
            for b in 33u16..=126 { printable.push(b); }
            for b in 161u16..=172 { printable.push(b); }
            for b in 174u16..=255 { printable.push(b); }

            let mut n: u16 = 0;
            for byte_val in 0u16..=255 {
                let unicode_char = if printable.contains(&byte_val) {
                    // Printable: maps to itself as unicode codepoint
                    char::from_u32(byte_val as u32).unwrap_or('?')
                } else {
                    // Non-printable: maps to U+0100 + offset
                    let ch = char::from_u32(256 + n as u32).unwrap_or('?');
                    n += 1;
                    ch
                };

                // Store reverse mapping for decode
                unicode_to_byte.insert(unicode_char, byte_val as u8);

                // Encode the unicode char to UTF-8 and look up in token_map
                let mut buf = [0u8; 4];
                let utf8 = unicode_char.encode_utf8(&mut buf);
                if let Some(&id) = token_map.get(utf8.as_bytes()) {
                    byte_to_token[byte_val as usize] = id;
                }
            }

            // Verify: count how many bytes have valid mappings
            let mapped = byte_to_token.iter().filter(|&&id| {
                // A valid mapping exists if the byte's token has non-empty text in vocab
                let idx = id as usize;
                idx < vocab.len() && !vocab[idx].is_empty()
            }).count();
            println!("[tokenizer] Byte-to-token mapping: {}/256 bytes mapped", mapped);
        }

        // Build vocab_decoded: convert GPT-2 unicode representations back to raw bytes.
        // The vocab entries store GPT-2 byte-to-unicode chars (e.g. Ġ for space, Ċ for newline).
        // vocab_decoded[id] contains the actual raw bytes that the token represents.
        let mut vocab_decoded = Vec::with_capacity(vocab.len());
        for token_bytes in &vocab {
            let text = String::from_utf8_lossy(token_bytes);
            let raw_bytes: Vec<u8> = text.chars().flat_map(|c| {
                if let Some(&byte) = unicode_to_byte.get(&c) {
                    // GPT-2 unicode char → raw byte value
                    vec![byte]
                } else {
                    // Not a GPT-2 byte char (e.g. special tokens like <|im_end|>)
                    // Keep as UTF-8 encoded bytes
                    let mut buf = [0u8; 4];
                    let s = c.encode_utf8(&mut buf);
                    s.as_bytes().to_vec()
                }
            }).collect();
            vocab_decoded.push(raw_bytes);
        }
        println!("[tokenizer] Built vocab_decoded: {} entries with GPT-2 byte reversal", vocab_decoded.len());

        Ok(Tokenizer {
            merges,
            merge_ranks,
            merge_results,
            vocab,
            token_map,
            byte_to_token,
            vocab_decoded,
            vocab_size,
            hf_mode: true,
        })
    }
}
