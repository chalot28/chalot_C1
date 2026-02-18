// =============================================================================
// model/header.rs — File Header (v1/v2 .myai format)
// =============================================================================

use super::constants::{MAGIC, HEADER_SIZE};

/// Fields are read manually for portability; no repr(C,packed) tricks.
#[derive(Debug, Clone)]
pub struct FileHeader {
    pub magic: u32,
    pub version: u32,
    pub dim: u32,
    pub hidden_dim: u32,
    pub n_layers: u32,
    pub n_heads: u32,
    pub vocab_size: u32,
    pub flags: u32,         // bit0 = quantized, bit1 = int4_experts, bit2 = moe
    pub n_experts: u32,     // 0 or 1 = dense, ≥2 = MoE
    pub top_k: u32,         // experts activated per token
    pub int4_group_size: u32, // 0 = Int8 only, else group size for Int4
    pub depth_router_layer: u32, // 0 = disabled
    pub max_seq_len: u32,   // 0 = default (512)
}

fn read_u32_le(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ])
}

fn write_u32_le(buf: &mut [u8], offset: usize, val: u32) {
    buf[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
}

impl FileHeader {
    pub fn from_bytes(data: &[u8]) -> Self {
        assert!(data.len() >= 128, "File too small for header");
        let version = read_u32_le(data, 4);

        let mut h = FileHeader {
            magic: read_u32_le(data, 0),
            version,
            dim: read_u32_le(data, 8),
            hidden_dim: read_u32_le(data, 12),
            n_layers: read_u32_le(data, 16),
            n_heads: read_u32_le(data, 20),
            vocab_size: read_u32_le(data, 24),
            flags: read_u32_le(data, 28),
            // v2 defaults (overwritten below if version ≥ 2)
            n_experts: 0,
            top_k: 0,
            int4_group_size: 0,
            depth_router_layer: 0,
            max_seq_len: 0,
        };

        if version >= 2 && data.len() >= HEADER_SIZE {
            h.n_experts = read_u32_le(data, 128);
            h.top_k = read_u32_le(data, 132);
            h.int4_group_size = read_u32_le(data, 136);
            h.depth_router_layer = read_u32_le(data, 140);
            h.max_seq_len = read_u32_le(data, 144);
        }
        h
    }

    pub fn write_to(&self, buf: &mut [u8]) {
        assert!(buf.len() >= HEADER_SIZE);
        // Zero the header area first
        for b in buf[..HEADER_SIZE].iter_mut() {
            *b = 0;
        }
        write_u32_le(buf, 0, self.magic);
        write_u32_le(buf, 4, self.version);
        write_u32_le(buf, 8, self.dim);
        write_u32_le(buf, 12, self.hidden_dim);
        write_u32_le(buf, 16, self.n_layers);
        write_u32_le(buf, 20, self.n_heads);
        write_u32_le(buf, 24, self.vocab_size);
        write_u32_le(buf, 28, self.flags);
        // v2 fields at offset 128+
        write_u32_le(buf, 128, self.n_experts);
        write_u32_le(buf, 132, self.top_k);
        write_u32_le(buf, 136, self.int4_group_size);
        write_u32_le(buf, 140, self.depth_router_layer);
        write_u32_le(buf, 144, self.max_seq_len);
    }

    #[allow(dead_code)]
    pub fn header_size(&self) -> usize {
        if self.version >= 2 { HEADER_SIZE } else { 128 }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.magic != MAGIC {
            return Err(format!("Bad magic: 0x{:08X}", self.magic));
        }
        if self.version == 0 || self.version > 2 {
            return Err(format!("Unsupported version: {}", self.version));
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub fn is_moe(&self) -> bool {
        self.n_experts >= 2
    }

    #[allow(dead_code)]
    pub fn has_int4(&self) -> bool {
        self.int4_group_size > 0
    }

    #[allow(dead_code)]
    pub fn has_depth_router(&self) -> bool {
        self.depth_router_layer > 0
    }
}
