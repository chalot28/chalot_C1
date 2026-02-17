#!/usr/bin/env python3
"""
validate_myai.py - Kiểm tra tính hợp lệ của file .myai sau conversion

Usage:
    python validate_myai.py qwen_moe.myai
"""

import argparse
import struct
from pathlib import Path


def read_u32(f):
    """Đọc 4 bytes little-endian"""
    return struct.unpack('<I', f.read(4))[0]


def validate_myai_header(filepath: str):
    """
    Validate header của file .myai
    Returns: (is_valid, config_dict, error_message)
    """
    path = Path(filepath)
    
    if not path.exists():
        return False, {}, f"File không tồn tại: {filepath}"
    
    if path.stat().st_size < 256:
        return False, {}, f"File quá nhỏ ({path.stat().st_size} bytes < 256 bytes header)"
    
    with open(filepath, 'rb') as f:
        # Read header (256 bytes)
        magic = read_u32(f)
        version = read_u32(f)
        dim = read_u32(f)
        hidden_dim = read_u32(f)
        n_layers = read_u32(f)
        n_heads = read_u32(f)
        vocab_size = read_u32(f)
        flags = read_u32(f)
        
        # Skip to offset 128 for v2 fields
        f.seek(128)
        n_experts = read_u32(f)
        top_k = read_u32(f)
        int4_group_size = read_u32(f)
        depth_router_layer = read_u32(f)
        max_seq_len = read_u32(f)
        
        config = {
            'magic': magic,
            'version': version,
            'dim': dim,
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'n_heads': n_heads,
            'vocab_size': vocab_size,
            'flags': flags,
            'n_experts': n_experts,
            'top_k': top_k,
            'int4_group_size': int4_group_size,
            'depth_router_layer': depth_router_layer,
            'max_seq_len': max_seq_len,
        }
        
        # Validate magic
        if magic != 0x4D594149:  # "MYAI"
            return False, config, f"Magic number không hợp lệ: 0x{magic:08X} (expected 0x4D594149)"
        
        # Validate version
        if version not in [1, 2]:
            return False, config, f"Version không được hỗ trợ: {version}"
        
        # Validate Qwen config
        if dim != 896:
            return False, config, f"dim không đúng: {dim} (expected 896 cho Qwen2.5-0.5B)"
        
        if n_layers != 24:
            return False, config, f"n_layers không đúng: {n_layers} (expected 24 cho Qwen2.5-0.5B)"
        
        if n_heads != 14:
            return False, config, f"n_heads không đúng: {n_heads} (expected 14 cho Qwen2.5-0.5B)"
        
        if vocab_size != 151936:
            return False, config, f"vocab_size không đúng: {vocab_size} (expected 151936 cho Qwen)"
        
        if n_experts < 2:
            return False, config, f"n_experts phải >= 2 cho MoE (got {n_experts})"
        
        # Validate flags (should have quantized + int4 + moe bits)
        is_quantized = (flags & 0b001) != 0
        is_int4 = (flags & 0b010) != 0
        is_moe = (flags & 0b100) != 0
        
        if not (is_quantized and is_int4 and is_moe):
            return False, config, f"Flags không đúng: 0b{flags:03b} (expected 0b111 = quantized+int4+moe)"
        
        # Estimate expected file size
        # Embeddings: vocab × dim × 1 byte (Int8) + 4 bytes scale
        embed_size = vocab_size * dim + 4
        
        # Per layer:
        #   Attention: 4 × (dim × dim × 1 + 4)  [QKVO Int8]
        #   Norms: 2 × dim × 4                  [Float32]
        #   Router: dim × n_experts × 4         [Float32]
        #   Experts: n_experts × 3 × (hidden×dim×0.5 + scales)  [Int4 packed]
        attn_per_layer = 4 * (dim * dim + 4)
        norms_per_layer = 2 * dim * 4
        router_per_layer = dim * n_experts * 4
        
        # Int4: 2 values per byte, plus scales
        expert_weights_per_expert = 2 * hidden_dim * dim + hidden_dim * dim  # gate+up+down
        packed_size = expert_weights_per_expert // 2  # Int4 packing
        n_groups = (expert_weights_per_expert + int4_group_size - 1) // int4_group_size
        scales_size = n_groups * 4 * 3 + 12  # 3 projections + overhead
        expert_per_layer = n_experts * (packed_size + scales_size)
        
        layer_size = attn_per_layer + norms_per_layer + router_per_layer + expert_per_layer
        
        # Output: norm + lm_head
        output_size = dim * 4 + vocab_size * dim + 4
        
        expected_size = 256 + embed_size + n_layers * layer_size + output_size
        actual_size = path.stat().st_size
        size_ratio = actual_size / expected_size
        
        if size_ratio < 0.5 or size_ratio > 2.0:
            return False, config, f"File size không hợp lý: {actual_size/1e6:.1f}MB (expected ~{expected_size/1e6:.1f}MB, ratio {size_ratio:.2f})"
        
        return True, config, None


def main():
    parser = argparse.ArgumentParser(description='Validate .myai file format')
    parser.add_argument('file', type=str, help='Path to .myai file')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MYAI FILE VALIDATOR")
    print("=" * 60)
    print(f"File: {args.file}\n")
    
    is_valid, config, error = validate_myai_header(args.file)
    
    if is_valid:
        print("✅ VALIDATION PASSED")
        print("\nConfiguration:")
        print(f"  Magic:       0x{config['magic']:08X} (MYAI)")
        print(f"  Version:     {config['version']}")
        print(f"  Dimension:   {config['dim']}")
        print(f"  Hidden dim:  {config['hidden_dim']}")
        print(f"  Layers:      {config['n_layers']}")
        print(f"  Heads:       {config['n_heads']}")
        print(f"  Vocab size:  {config['vocab_size']:,}")
        print(f"  Max seq len: {config['max_seq_len']}")
        print(f"\nMoE Config:")
        print(f"  Experts:     {config['n_experts']}")
        print(f"  Top-K:       {config['top_k']}")
        print(f"  Int4 group:  {config['int4_group_size']}")
        print(f"\nFlags:")
        print(f"  Quantized:   {'✓' if (config['flags'] & 1) else '✗'}")
        print(f"  Int4:        {'✓' if (config['flags'] & 2) else '✗'}")
        print(f"  MoE:         {'✓' if (config['flags'] & 4) else '✗'}")
        
        file_size_mb = Path(args.file).stat().st_size / 1e6
        print(f"\nFile size: {file_size_mb:.1f} MB")
        
        # Estimate RAM usage
        # Embeddings + 1 layer active + output + KV cache
        embed_ram = config['vocab_size'] * config['dim'] // 4  # Int8 → bytes
        layer_ram = config['dim'] * config['hidden_dim'] * 2 * config['top_k'] // 2  # Top-K experts, Int4
        output_ram = config['vocab_size'] * config['dim'] // 4
        kv_cache_ram = config['n_layers'] * config['dim'] * 256 * 2 // 4  # 256 tokens cached per layer, Int8
        
        total_ram_mb = (embed_ram + layer_ram + output_ram + kv_cache_ram) / 1e6
        
        print(f"\nEstimated runtime RAM: ~{total_ram_mb:.1f} MB")
        print(f"  (Embeddings: ~{embed_ram/1e6:.1f}MB, Active layer: ~{layer_ram/1e6:.1f}MB,")
        print(f"   Output: ~{output_ram/1e6:.1f}MB, KV cache: ~{kv_cache_ram/1e6:.1f}MB)")
        
        return 0
    else:
        print("❌ VALIDATION FAILED")
        print(f"\nError: {error}")
        print("\nPartial config:")
        for key, val in config.items():
            if key == 'magic':
                print(f"  {key}: 0x{val:08X}")
            elif key == 'flags':
                print(f"  {key}: 0b{val:03b}")
            else:
                print(f"  {key}: {val}")
        return 1


if __name__ == '__main__':
    exit(main())
