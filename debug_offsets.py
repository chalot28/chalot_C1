#!/usr/bin/env python3
"""Debug: Calculate exact byte offsets"""

dim = 896
hidden = 4864
vocab = 151936
n_experts = 8
group = 32

cursor = 256  # After header

print("BYTE OFFSETS:")
print("=" * 70)

# 1. Embeddings
print(f"\n[{cursor:,}] Embeddings data (Int8): vocab={vocab}, dim={dim}")
embed_data = vocab * dim
cursor += embed_data
print(f"  → {cursor:,}")

print(f"[{cursor:,}] Embeddings scales (Float32): vocab={vocab}")
embed_scales = vocab * 4
cursor += embed_scales
print(f"  → {cursor:,}")

# 2. Layers
for layer in range(24):
    print(f"\nLayer {layer}:")
    
    # RMS attn
    print(f"  [{cursor:,}] RMS attn: {dim * 4} bytes")
    cursor += dim * 4
    
    # QKV data
    print(f"  [{cursor:,}] QKV data (Int8): {3 * dim * dim} bytes")
    cursor += 3 * dim * dim
    
    # QKV scales
    print(f"  [{cursor:,}] QKV scales (Float32): {3 * dim * 4} bytes")
    cursor += 3 * dim * 4
    
    # Attn out data
    print(f"  [{cursor:,}] Attn out data (Int8): {dim * dim} bytes")
    cursor += dim * dim
    
    # Attn out scales
    print(f"  [{cursor:,}] Attn out scales (Float32): {dim * 4} bytes")
    cursor += dim * 4
    
    # RMS ffn
    print(f"  [{cursor:,}] RMS ffn: {dim * 4} bytes")
    cursor += dim * 4
    
    # Router
    router_w = n_experts * dim * 4
    router_b = n_experts * 4
    print(f"  [{cursor:,}] Router W: {router_w} bytes")
    cursor += router_w
    print(f"  [{cursor:,}] Router b: {router_b} bytes")
    cursor += router_b
    
    # Experts
    for exp_idx in range(n_experts):
        # Gate+Up
        n_groups_dim = (dim + group - 1) // group
        gate_up_rows = 2 * hidden
        gate_up_data = gate_up_rows * ((dim + 1) // 2)
        gate_up_scales = gate_up_rows * n_groups_dim * 4
        
        print(f"  [{cursor:,}] Expert {exp_idx} gate+up data: {gate_up_data} bytes")
        cursor += gate_up_data
        print(f"  [{cursor:,}] Expert {exp_idx} gate+up scales: {gate_up_scales} bytes")
        cursor += gate_up_scales
        
        # Down
        n_groups_hidden = (hidden + group - 1) // group
        down_data = dim * ((hidden + 1) // 2)
        down_scales = dim * n_groups_hidden * 4
        
        print(f"  [{cursor:,}] Expert {exp_idx} down data: {down_data} bytes")
        cursor += down_data
        print(f"  [{cursor:,}] Expert {exp_idx} down scales: {down_scales} bytes")
        cursor += down_scales

# Final norm
print(f"\n[{cursor:,}] Final norm: {dim * 4} bytes")
cursor += dim * 4

# Output
print(f"[{cursor:,}] Output data (Int8): {vocab * dim} bytes")
cursor += vocab * dim

print(f"[{cursor:,}] Output scales (Float32): {vocab * 4} bytes")
cursor += vocab * 4

print("=" * 70)
print(f"TOTAL: {cursor:,} bytes ({cursor/1e6:.1f} MB)")
print("=" * 70)
