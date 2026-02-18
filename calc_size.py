#!/usr/bin/env python3
"""Calculate expected .myai file size for Qwen2.5-0.5B"""

dim = 896
hidden = 4864
n_layers = 24
vocab = 151936
n_experts = 8
group_size = 32

def calc_int4_size(rows, cols, group_size):
    """Calculate Int4 matrix size with group-wise quantization"""
    packed_bytes = rows * ((cols + 1) // 2)  # packed Int4 data
    n_groups = (cols + group_size - 1) // group_size
    scales_bytes = rows * n_groups * 4  # Float32 scales
    return packed_bytes + scales_bytes

print("=" * 70)
print("Qwen2.5-0.5B → .myai SIZE CALCULATION")
print("=" * 70)

total = 0

# Header
header_size = 256
total += header_size
print(f"Header: {header_size} bytes")

# Embeddings (Int8 + scales)
embed_data = vocab * dim  # Int8
embed_scales = vocab * 4  # Float32
embed_total = embed_data + embed_scales
total += embed_total
print(f"Embeddings: {embed_total:,} bytes ({embed_total/1e6:.1f} MB)")

# Layers
print(f"\nPer-layer breakdown:")
layer_size = 0

# RMS attn norm
rms_attn = dim * 4
layer_size += rms_attn
print(f"  RMS attn norm: {rms_attn:,} bytes")

# QKV (Int8)
qkv_data = 3 * dim * dim
qkv_scales = 3 * dim * 4
qkv_total = qkv_data + qkv_scales
layer_size += qkv_total
print(f"  QKV (Int8): {qkv_total:,} bytes")

# Attn out (Int8)
out_data = dim * dim
out_scales = dim * 4
out_total = out_data + out_scales
layer_size += out_total
print(f"  Attn out (Int8): {out_total:,} bytes")

# RMS ffn norm
rms_ffn = dim * 4
layer_size += rms_ffn
print(f"  RMS ffn norm: {rms_ffn:,} bytes")

# Router (W + b)
router_w = n_experts * dim * 4
router_b = n_experts * 4
router_total = router_w + router_b
layer_size += router_total
print(f"  Router: {router_total:,} bytes")

# Experts (Int4)
expert_size = 0

# Gate+Up [2*hidden, dim]
gate_up_size = calc_int4_size(2 * hidden, dim, group_size)
expert_size += gate_up_size
print(f"  Expert gate+up (Int4): {gate_up_size:,} bytes")

# Down [dim, hidden]
down_size = calc_int4_size(dim, hidden, group_size)
expert_size += down_size
print(f"  Expert down (Int4): {down_size:,} bytes")

print(f"  Total per expert: {expert_size:,} bytes")

experts_total = expert_size * n_experts
layer_size += experts_total
print(f"  All {n_experts} experts: {experts_total:,} bytes")

print(f"\nTotal per layer: {layer_size:,} bytes ({layer_size/1e6:.1f} MB)")

layers_total = layer_size * n_layers
total += layers_total
print(f"All {n_layers} layers: {layers_total:,} bytes ({layers_total/1e6:.1f} MB)")

# Final norm
final_norm = dim * 4
total += final_norm
print(f"\nFinal norm: {final_norm:,} bytes")

# Output (Int8)
output_data = vocab * dim
output_scales = vocab * 4
output_total = output_data + output_scales
total += output_total
print(f"Output projection (Int8): {output_total:,} bytes ({output_total/1e6:.1f} MB)")

print("=" * 70)
print(f"TOTAL EXPECTED: {total:,} bytes ({total/1e6:.1f} MB)")
print("=" * 70)
