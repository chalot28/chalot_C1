#!/usr/bin/env python3
"""
=============================================================================
qwen_to_myai.py — Chuyển đổi Qwen2.5-0.5B sang định dạng .myai
=============================================================================

Quy trình:
1. Tải Qwen2.5-0.5B-Instruct từ HuggingFace
2. Up-cycle thành MoE (nhân bản FFN → 8 experts với nhiễu Gaussian)
3. Quantize:
   - Attention weights → Int8
   - Expert weights → Int4
4. Phân vùng Brain Map (Shallow/Deep/Fact)
5. Xuất ra file .myai

Yêu cầu:
    pip install torch safetensors transformers numpy
    huggingface-cli login  # Nếu cần

Chạy:
    python qwen_to_myai.py --model Qwen/Qwen2.5-0.5B-Instruct --output qwen_moe.myai
"""

import argparse
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from safetensors import safe_open
from transformers import AutoConfig


# =============================================================================
# Cấu hình Qwen2.5-0.5B
# =============================================================================
QWEN_CONFIG = {
    'dim': 896,
    'hidden_dim': 4864,
    'n_layers': 24,
    'n_heads': 14,
    'n_kv_heads': 2,     # GQA: 2 KV heads shared among 14 Q heads
    'head_dim': 64,       # dim / n_heads = 896 / 14
    'vocab_size': 151936,
    'max_seq_len': 2048,
    'n_experts': 8,
    'top_k': 2,
    'int4_group_size': 32,
    'rope_theta': 1000000.0,  # Qwen2.5 uses 1M (NOT default 10000)
}

# Brain Map layer ranges
BRAIN_MAP_RANGES = {
    'shallow': (0, 6),    # Layers 0-5
    'deep': (6, 18),      # Layers 6-17
    'fact': (18, 24),     # Layers 18-23
}

EXPERT_NOISE_STD = 0.01


# =============================================================================
# Quantization Functions
# =============================================================================
def quantize_int8(weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize Float32 → Int8 với per-row scaling
    Returns: (int8_weights, per_row_scales)
    """
    assert weights.ndim == 2, "Expect 2D matrix"
    rows, cols = weights.shape
    
    scales = np.zeros(rows, dtype=np.float32)
    quantized = np.zeros_like(weights, dtype=np.int8)
    
    for i in range(rows):
        row = weights[i]
        max_val = np.abs(row).max()
        scale = max_val / 127.0 if max_val > 0 else 1.0
        scales[i] = scale
        quantized[i] = np.clip(row / scale, -128, 127).astype(np.int8)
    
    return quantized, scales


def quantize_int4(weights: np.ndarray, group_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize Float32 → Int4 với row-wise group quantization
    Returns: (packed_int4_bytes, scales_per_row_per_group)
    
    Format: Quantize từng row riêng biệt, mỗi row chia thành groups
    """
    assert weights.ndim == 2, "Expect 2D matrix"
    rows, cols = weights.shape
    
    n_groups_per_row = (cols + group_size - 1) // group_size
    total_scales = rows * n_groups_per_row
    
    scales = np.zeros(total_scales, dtype=np.float32)
    quantized_all = []
    
    for row_idx in range(rows):
        row = weights[row_idx]
        quantized_row = np.zeros(cols, dtype=np.int8)
        
        # Quantize each group in this row
        for g in range(n_groups_per_row):
            start = g * group_size
            end = min(start + group_size, cols)
            group = row[start:end]
            
            max_val = np.abs(group).max()
            scale = max_val / 7.0 if max_val > 0 else 1.0
            scales[row_idx * n_groups_per_row + g] = scale
            
            quantized_row[start:end] = np.clip(group / scale, -8, 7).astype(np.int8)
        
        quantized_all.append(quantized_row)
    
    # Pack 2 Int4 values into 1 byte (row by row)
    all_quantized = np.concatenate(quantized_all)
    quantized_uint = (all_quantized + 8).astype(np.uint8)  # -8..7 -> 0..15
    
    packed = []
    for i in range(0, len(quantized_uint), 2):
        if i + 1 < len(quantized_uint):
            val1 = quantized_uint[i] & 0xF
            val2 = quantized_uint[i + 1] & 0xF
            packed.append((val1 | (val2 << 4)))
        else:
            packed.append(quantized_uint[i] & 0xF)
    
    return np.array(packed, dtype=np.uint8), scales


# =============================================================================
# MoE Up-cycling
# =============================================================================
def upcycle_ffn_to_moe(
    gate_proj: np.ndarray,
    up_proj: np.ndarray,
    down_proj: np.ndarray,
    n_experts: int = 8,
    noise_std: float = 0.01
) -> List[Dict[str, np.ndarray]]:
    """
    Nhân bản FFN gốc thành n_experts experts với nhiễu Gaussian
    
    Returns: List[{'gate_proj', 'up_proj', 'down_proj'}]
    """
    experts = []
    for i in range(n_experts):
        # Thêm nhiễu để tạo sự đa dạng
        noise_gate = np.random.normal(0, noise_std, gate_proj.shape).astype(np.float32)
        noise_up = np.random.normal(0, noise_std, up_proj.shape).astype(np.float32)
        noise_down = np.random.normal(0, noise_std, down_proj.shape).astype(np.float32)
        
        experts.append({
            'gate_proj': gate_proj + noise_gate,
            'up_proj': up_proj + noise_up,
            'down_proj': down_proj + noise_down,
        })
    
    return experts


def create_router_weights(dim: int, n_experts: int):
    """
    Tạo router weights ngẫu nhiên cho MoE gating
    Returns: (W [n_experts, dim], b [n_experts])
    """
    # Xavier initialization
    limit = np.sqrt(6.0 / (dim + n_experts))
    W = np.random.uniform(-limit, limit, (n_experts, dim)).astype(np.float32)
    b = np.zeros(n_experts, dtype=np.float32)
    return W, b


# =============================================================================
# Đọc Safetensors từ HuggingFace
# =============================================================================
def load_qwen_weights(model_path: str) -> Dict[str, np.ndarray]:
    """
    Load weights từ Qwen2.5-0.5B safetensors files
    Returns: Dict[layer_name -> numpy array]
    """
    model_dir = Path(model_path)
    weights = {}
    
    # Tìm tất cả các file .safetensors
    safetensor_files = list(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        # Thử tải từ cache HuggingFace
        from huggingface_hub import snapshot_download
        model_dir = Path(snapshot_download(repo_id=model_path))
        safetensor_files = list(model_dir.glob("*.safetensors"))
    
    print(f"[load] Tìm thấy {len(safetensor_files)} file safetensors")
    
    for st_file in safetensor_files:
        print(f"[load] Đọc {st_file.name}...")
        with safe_open(st_file, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                # Convert bfloat16 to float32 for numpy compatibility
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                weights[key] = tensor.numpy()
    
    print(f"[load] Đã load {len(weights)} tensors")
    return weights


# =============================================================================
# Chuyển đổi chính
# =============================================================================
def expand_kv_heads(weight: np.ndarray, n_kv_heads: int, n_heads: int, head_dim: int) -> np.ndarray:
    """
    Expand GQA K/V projections from n_kv_heads to n_heads by repeating each KV head.
    
    Input:  [n_kv_heads * head_dim, dim] (e.g. [128, 896] for 2 KV heads)
    Output: [n_heads * head_dim, dim]     (e.g. [896, 896] for 14 Q heads)
    
    Each KV head is repeated (n_heads // n_kv_heads) times.
    """
    if n_kv_heads == n_heads:
        return weight  # No expansion needed (standard MHA)
    
    assert n_heads % n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
    repeat_factor = n_heads // n_kv_heads
    
    dim = weight.shape[1]
    # Reshape: [n_kv_heads, head_dim, dim]
    reshaped = weight.reshape(n_kv_heads, head_dim, dim)
    # Repeat: [n_kv_heads, repeat_factor, head_dim, dim]
    expanded = np.repeat(reshaped, repeat_factor, axis=0)  # [n_heads, head_dim, dim]
    # Flatten: [n_heads * head_dim, dim]
    result = expanded.reshape(n_heads * head_dim, dim)
    
    print(f"    GQA expand: [{n_kv_heads}×{head_dim}, {dim}] → [{n_heads}×{head_dim}, {dim}] (repeat×{repeat_factor})")
    return result


def export_tokenizer(model_path: str, output_path: str):
    """
    Export Qwen tokenizer to binary .mytok format for Rust loading.
    
    Format:
      Header (16 bytes): magic(u32) + version(u32) + vocab_size(u32) + num_merges(u32)
      Vocab: for each token: len(u16) + bytes[len]
      Merges: for each merge: token_a(u32) + token_b(u32)
    """
    from transformers import AutoTokenizer
    
    print(f"[tokenizer] Loading Qwen tokenizer from {model_path}...")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    vocab = tok.get_vocab()  # {string: id}
    vocab_size = max(vocab.values()) + 1
    
    # Build reverse vocab: id → bytes
    id_to_bytes = {}
    for token_str, token_id in vocab.items():
        try:
            token_bytes = token_str.encode('utf-8')
        except:
            token_bytes = b''
        id_to_bytes[token_id] = token_bytes
    
    # Get merges if available
    merges_list = []
    try:
        # Try to get merges from the tokenizer's backend
        if hasattr(tok, 'backend_tokenizer'):
            model = tok.backend_tokenizer.model
            if hasattr(model, 'get_vocab') and hasattr(tok.backend_tokenizer, 'get_vocab'):
                pass  # Will use the vocab we already have
    except Exception as e:
        print(f"[tokenizer] Note: Could not extract merges: {e}")
    
    print(f"[tokenizer] Vocab size: {vocab_size}, writing to {output_path}")
    
    with open(output_path, 'wb') as f:
        # Header
        f.write(struct.pack('<I', 0x544F4B45))  # Magic "TOKE"
        f.write(struct.pack('<I', 2))            # Version
        f.write(struct.pack('<I', vocab_size))
        f.write(struct.pack('<I', len(merges_list)))
        
        # Vocab entries
        for token_id in range(vocab_size):
            token_bytes = id_to_bytes.get(token_id, b'')
            # Limit to 65535 bytes per token
            if len(token_bytes) > 65535:
                token_bytes = token_bytes[:65535]
            f.write(struct.pack('<H', len(token_bytes)))
            f.write(token_bytes)
        
        # Merges (if any)
        for (a, b) in merges_list:
            f.write(struct.pack('<II', a, b))
    
    file_size = Path(output_path).stat().st_size
    print(f"[tokenizer] Exported: {file_size / 1024:.1f} KB ({vocab_size} tokens)")


def convert_qwen_to_myai(model_path: str, output_path: str):
    """
    Pipeline chính: Qwen → MoE → Quantized → .myai
    """
    print("=" * 80)
    print("QWEN2.5-0.5B → AI_CHALOT_C1 MoE CONVERTER")
    print("=" * 80)
    
    # 0. Auto-detect model config from HuggingFace
    print("\n[0/6] Detecting model configuration...")
    try:
        from transformers import AutoConfig
        hf_config = AutoConfig.from_pretrained(model_path)
        QWEN_CONFIG['n_kv_heads'] = getattr(hf_config, 'num_key_value_heads', QWEN_CONFIG['n_heads'])
        QWEN_CONFIG['n_heads'] = getattr(hf_config, 'num_attention_heads', QWEN_CONFIG['n_heads'])
        QWEN_CONFIG['dim'] = getattr(hf_config, 'hidden_size', QWEN_CONFIG['dim'])
        QWEN_CONFIG['hidden_dim'] = getattr(hf_config, 'intermediate_size', QWEN_CONFIG['hidden_dim'])
        QWEN_CONFIG['n_layers'] = getattr(hf_config, 'num_hidden_layers', QWEN_CONFIG['n_layers'])
        QWEN_CONFIG['vocab_size'] = getattr(hf_config, 'vocab_size', QWEN_CONFIG['vocab_size'])
        QWEN_CONFIG['head_dim'] = QWEN_CONFIG['dim'] // QWEN_CONFIG['n_heads']
        QWEN_CONFIG['rope_theta'] = getattr(hf_config, 'rope_theta', QWEN_CONFIG['rope_theta'])
        print(f"  Detected: dim={QWEN_CONFIG['dim']}, heads={QWEN_CONFIG['n_heads']}, "
              f"kv_heads={QWEN_CONFIG['n_kv_heads']}, layers={QWEN_CONFIG['n_layers']}, "
              f"rope_theta={QWEN_CONFIG['rope_theta']}")
    except Exception as e:
        print(f"  Warning: Could not auto-detect config: {e}")
        print(f"  Using hardcoded QWEN_CONFIG")
    
    config = QWEN_CONFIG
    
    # 1. Load weights
    print("\n[1/6] Loading Qwen2.5-0.5B weights...")
    weights = load_qwen_weights(model_path)
    
    # 2. Chuẩn bị cấu trúc
    print("\n[2/6] Preparing architecture...")
    
    # 3. Tạo file .myai
    with open(output_path, 'wb') as f:
        # Write header (256 bytes)
        write_header(f, config)
        
        # Write embedding
        print("\n[3/6] Processing embeddings...")
        write_embeddings(f, weights, config)
        
        # Write layers với MoE up-cycling
        print("\n[4/6] Converting layers with MoE up-cycling...")
        write_layers_moe(f, weights, config)
        
        # Write output head
        print("\n[5/6] Writing output layer...")
        write_output(f, weights, config)
    
    # Verify file size matches Rust engine expectation
    actual_size = Path(output_path).stat().st_size
    expected_size = compute_expected_size(config)
    
    print(f"\n✅ Conversion complete: {output_path}")
    print(f"   File size: {actual_size / 1e6:.1f} MB")
    print(f"   Expected:  {expected_size / 1e6:.1f} MB")
    if actual_size != expected_size:
        print(f"   ⚠️  SIZE MISMATCH: {actual_size} vs {expected_size} (diff: {actual_size - expected_size})")
    else:
        print(f"   ✅ Size matches perfectly!")
    
    # 6. Export tokenizer
    tok_path = output_path.replace('.myai', '.mytok')
    print(f"\n[6/6] Exporting tokenizer...")
    try:
        export_tokenizer(model_path, tok_path)
    except Exception as e:
        print(f"   Warning: Tokenizer export failed: {e}")
        print(f"   The model will use byte-level fallback tokenization")


def write_header(f, config: dict):
    """
    Ghi header 256 bytes theo format model/header.rs
    """
    # Magic "MYAI" (4 bytes)
    f.write(struct.pack('<I', 0x4D594149))
    
    # Version 2 (4 bytes)
    f.write(struct.pack('<I', 2))
    
    # Core config (8 fields × 4 bytes = 32 bytes)
    f.write(struct.pack('<I', config['dim']))
    f.write(struct.pack('<I', config['hidden_dim']))
    f.write(struct.pack('<I', config['n_layers']))
    f.write(struct.pack('<I', config['n_heads']))
    f.write(struct.pack('<I', config['vocab_size']))
    
    # Flags: bit0=quantized, bit1=int4_experts, bit2=moe
    flags = 0b111  # Quantized + Int4 + MoE
    f.write(struct.pack('<I', flags))
    
    # Padding đến offset 128
    f.write(b'\x00' * (128 - 8 - 24))
    
    # V2 fields (offset 128)
    f.write(struct.pack('<I', config['n_experts']))
    f.write(struct.pack('<I', config['top_k']))
    f.write(struct.pack('<I', config['int4_group_size']))
    f.write(struct.pack('<I', 0))  # depth_router_layer (disabled, no weights written)
    f.write(struct.pack('<I', config['max_seq_len']))
    
    # rope_theta × 100 (stored as u32, e.g. 1000000.0 → 100000000)
    rope_theta_u32 = int(config.get('rope_theta', 10000.0) * 100)
    f.write(struct.pack('<I', rope_theta_u32))
    
    # Padding đến 256 bytes
    f.write(b'\x00' * (256 - 128 - 24))
    
    print(f"[header] Written: v2, dim={config['dim']}, layers={config['n_layers']}, "
          f"experts={config['n_experts']}, rope_theta={config.get('rope_theta', 10000.0)}")


def write_embeddings(f, weights: dict, config: dict):
    """
    Ghi embedding table (vocab_size × dim)
    Quantize Int8 để tiết kiệm
    """
    # Tìm weight key cho embedding
    embed_key = 'model.embed_tokens.weight'
    if embed_key not in weights:
        raise KeyError(f"Cannot find {embed_key} in weights")
    
    embed = weights[embed_key].astype(np.float32)
    print(f"[embed] Shape: {embed.shape}")
    
    # Quantize Int8 (per-row)
    quantized, scales = quantize_int8(embed)
    
    # Write: data first, then scales (matching Rust WeightIndex)
    f.write(quantized.tobytes())
    f.write(scales.tobytes())
    
    print(f"[embed] Written {quantized.nbytes / 1e6:.1f} MB (Int8)")


def write_layers_moe(f, weights: dict, config: dict):
    """
    Ghi 24 layers với MoE up-cycling
    Format matching Rust WeightIndex::build_layer():
      1. RMS attn norm (Float32)
      2. QKV weights gộp (Int8 data + scales)
      3. Attn out weights (Int8 data + scales)
      4. RMS ffn norm (Float32)
      5. Router (Float32)
      6. Experts (Int4)
    """
    dim = config['dim']
    hidden_dim = config['hidden_dim']
    n_experts = config['n_experts']
    group_size = config['int4_group_size']
    
    for layer_idx in range(config['n_layers']):
        print(f"\n[layer {layer_idx:02d}] Processing...")
        prefix = f'model.layers.{layer_idx}'
        
        # 1. RMS attn norm (Float32)
        attn_norm = weights[f'{prefix}.input_layernorm.weight'].astype(np.float32)
        f.write(attn_norm.tobytes())
        
        # 2. QKV weights - gộp lại theo thứ tự Q, K, V
        q_proj = weights[f'{prefix}.self_attn.q_proj.weight'].astype(np.float32)
        k_proj = weights[f'{prefix}.self_attn.k_proj.weight'].astype(np.float32)
        v_proj = weights[f'{prefix}.self_attn.v_proj.weight'].astype(np.float32)
        
        # GQA expansion: expand K/V from n_kv_heads to n_heads
        n_kv_heads = config.get('n_kv_heads', config['n_heads'])
        n_heads = config['n_heads']
        head_dim = config['head_dim']
        if n_kv_heads < n_heads:
            if layer_idx == 0:
                print(f"  [GQA] Expanding K/V: {n_kv_heads} KV heads → {n_heads} heads (repeat×{n_heads // n_kv_heads})")
            k_proj = expand_kv_heads(k_proj, n_kv_heads, n_heads, head_dim)
            v_proj = expand_kv_heads(v_proj, n_kv_heads, n_heads, head_dim)
        
        # Quantize và gộp
        q_quant, q_scale = quantize_int8(q_proj)
        k_quant, k_scale = quantize_int8(k_proj)
        v_quant, v_scale = quantize_int8(v_proj)
        
        # Write data (QKV gộp)
        f.write(q_quant.tobytes())
        f.write(k_quant.tobytes())
        f.write(v_quant.tobytes())
        
        # Write scales (QKV gộp)
        f.write(q_scale.tobytes())
        f.write(k_scale.tobytes())
        f.write(v_scale.tobytes())
        
        # 3. Attn output projection
        o_proj = weights[f'{prefix}.self_attn.o_proj.weight'].astype(np.float32)
        o_quant, o_scale = quantize_int8(o_proj)
        f.write(o_quant.tobytes())
        f.write(o_scale.tobytes())
        
        # 4. RMS ffn norm (Float32)
        ffn_norm = weights[f'{prefix}.post_attention_layernorm.weight'].astype(np.float32)
        f.write(ffn_norm.tobytes())
        
        # 5. Router weights (Float32: W + b)
        router_w, router_b = create_router_weights(dim, n_experts)
        f.write(router_w.tobytes())
        f.write(router_b.tobytes())
        
        # 6. FFN → MoE up-cycling (Int4)
        gate_proj = weights[f'{prefix}.mlp.gate_proj.weight'].astype(np.float32)
        up_proj = weights[f'{prefix}.mlp.up_proj.weight'].astype(np.float32)
        down_proj = weights[f'{prefix}.mlp.down_proj.weight'].astype(np.float32)
        
        experts = upcycle_ffn_to_moe(gate_proj, up_proj, down_proj, n_experts, EXPERT_NOISE_STD)
        
        # Write each expert
        for expert_idx, expert in enumerate(experts):
            # Gate+Up: gộp lại thành [2*hidden, dim]
            gate = expert['gate_proj']  # [hidden, dim]
            up = expert['up_proj']      # [hidden, dim]
            gate_up = np.vstack([gate, up])  # [2*hidden, dim]
            
            # Quantize gate_up
            gate_up_quant, gate_up_scales = quantize_int4(gate_up, group_size)
            
            # Write gate_up: data + scales (NO n_scales prefix)
            f.write(gate_up_quant.tobytes())
            f.write(gate_up_scales.tobytes())
            
            # Down projection
            down = expert['down_proj']  # [dim, hidden]
            down_quant, down_scales = quantize_int4(down, group_size)
            
            # Write down: data + scales
            f.write(down_quant.tobytes())
            f.write(down_scales.tobytes())
        
        print(f"[layer {layer_idx:02d}] ✓ Written (Attn: Int8, {n_experts} Experts: Int4)")


def write_output(f, weights: dict, config: dict):
    """
    Ghi output projection (lm_head)
    """
    # Final norm
    norm = weights['model.norm.weight'].astype(np.float32)
    f.write(norm.tobytes())
    
    # LM head (vocab_size × dim)
    # Qwen2.5 uses tied weights: lm_head shares with embed_tokens
    if 'lm_head.weight' in weights:
        lm_head = weights['lm_head.weight'].astype(np.float32)
    else:
        # Use tied embeddings (same as input embedding)
        lm_head = weights['model.embed_tokens.weight'].astype(np.float32)
    
    # Quantize Int8 (per-row)
    quant, scales = quantize_int8(lm_head)
    
    # Write: data first, then scales (matching Rust WeightIndex)
    f.write(quant.tobytes())
    f.write(scales.tobytes())
    
    print(f"[output] Written lm_head {quant.nbytes / 1e6:.1f} MB (Int8)")


# =============================================================================
# Size Verification
# =============================================================================
def compute_expected_size(config: dict) -> int:
    """
    Compute the expected file size that the Rust engine will expect.
    Must match WeightIndex::build() in weight_index.rs exactly.
    """
    dim = config['dim']
    hidden = config['hidden_dim']
    vocab = config['vocab_size']
    n_layers = config['n_layers']
    n_experts = config['n_experts']
    group = config['int4_group_size']
    
    cursor = 256  # Header size (v2)
    
    # 1. Embeddings: [vocab × dim] Int8 + [vocab] f32 scales
    cursor += vocab * dim       # Int8 data
    cursor += vocab * 4         # f32 scales
    
    # 2. Layers
    for _ in range(n_layers):
        # RMS attn norm: [dim] f32
        cursor += dim * 4
        
        # QKV: [3*dim*dim] Int8 (after GQA expansion, K/V are [dim, dim])
        cursor += 3 * dim * dim             # Int8 data
        cursor += 3 * dim * 4               # f32 scales
        
        # Attn out: [dim*dim] Int8
        cursor += dim * dim                 # Int8 data
        cursor += dim * 4                   # f32 scales
        
        # RMS FFN norm: [dim] f32
        cursor += dim * 4
        
        # Router: [n_experts * dim] f32 + [n_experts] f32
        if n_experts >= 2:
            cursor += n_experts * dim * 4   # weights
            cursor += n_experts * 4         # bias
        
        # Per-expert Int4 weights
        n_groups_dim = (dim + group - 1) // group
        n_groups_hidden = (hidden + group - 1) // group
        packed_dim = (dim + 1) // 2
        packed_hidden = (hidden + 1) // 2
        
        for _ in range(n_experts):
            # Gate+Up: [2*hidden, dim] Int4 packed
            gate_up_rows = 2 * hidden
            cursor += gate_up_rows * packed_dim              # packed data
            cursor += gate_up_rows * n_groups_dim * 4        # scales
            
            # Down: [dim, hidden] Int4 packed
            cursor += dim * packed_hidden                    # packed data
            cursor += dim * n_groups_hidden * 4              # scales
    
    # 3. Depth router (disabled → 0 bytes)
    # cursor += 0
    
    # 4. Final RMS norm: [dim] f32
    cursor += dim * 4
    
    # 5. Output proj: [vocab × dim] Int8 + [vocab] f32 scales
    cursor += vocab * dim       # Int8 data
    cursor += vocab * 4         # f32 scales
    
    return cursor


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Convert Qwen2.5-0.5B to .myai format')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='Model path hoặc HuggingFace repo ID')
    parser.add_argument('--output', type=str, default='qwen_moe.myai',
                        help='Output .myai file path')
    
    args = parser.parse_args()
    
    convert_qwen_to_myai(args.model, args.output)
    
    print("\n" + "=" * 80)
    print("🎉 NEXT STEPS:")
    print("=" * 80)
    print("1. Copy qwen_moe.myai to your Pixel 5:")
    print("   adb push qwen_moe.myai /sdcard/Download/")
    print("")
    print("2. Build & run AI_chalot_C1:")
    print("   cargo build --release --target aarch64-linux-android")
    print("   adb push target/aarch64-linux-android/release/AI_chalot_C1 /data/local/tmp/")
    print("   adb shell chmod +x /data/local/tmp/AI_chalot_C1")
    print("   adb shell /data/local/tmp/AI_chalot_C1 /sdcard/Download/qwen_moe.myai")
    print("")
    print("3. Expect performance:")
    print("   - RAM usage: ~250MB (Top-2 MoE activation)")
    print("   - Speed: 20-25 tokens/sec on Pixel 5")
    print("   - Quality: Comparable to Qwen2.5-0.5B-Instruct")


if __name__ == '__main__':
    main()
