#!/usr/bin/env python3
"""
=============================================================================
train_distill.py — Knowledge Distillation: Qwen2.5 Teacher → MoE Student
=============================================================================

Pipeline:
  1. Load Qwen2.5-0.5B-Instruct làm TEACHER (frozen, inference only)
  2. Load PyTorch MoE student model (mirror architecture của .myai)
  3. Sinh mini-batches từ dataset text
  4. Forward qua teacher → lấy soft logits
  5. Forward qua student → tính KD loss (KL divergence) + CE loss
  6. Backward + optimizer step
  7. Save checkpoint .pt sau mỗi epoch
  8. Export sang .myai format

Chạy:
    python train_distill.py \
        --teacher  Qwen/Qwen2.5-0.5B-Instruct \
        --student  qwen_moe.myai \
        --corpus   corpus.txt \
        --epochs   3 \
        --batch    2 \
        --seq_len  256 \
        --lr       3e-4 \
        --output   trained_student.myai
"""

import argparse
import struct
import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List, Dict, Tuple


# =============================================================================
# Cấu hình cố định cho Qwen2.5-0.5B MoE student
# =============================================================================
STUDENT_CONFIG = {
    'dim':            896,
    'hidden_dim':     4864,
    'n_layers':       24,
    'n_heads':        14,
    'n_kv_heads':     2,        # GQA
    'head_dim':       64,
    'vocab_size':     151936,
    'n_experts':      8,
    'top_k':          2,
    'int4_group_size': 32,
    'rope_theta':     1_000_000.0,
    'max_seq_len':    2048,
}

# =============================================================================
# Cấu hình nhỏ ~20M params — dùng để test nhanh trên CPU yếu
# dim=256, hidden=512, 6 layers, 4 experts → 18.8M params
# vocab=32000 (dùng tokenizer BPE nhỏ hoặc Qwen subset)
# Ưu điểm: train 200 samples x3 epochs ≈ 5–10 phút
# =============================================================================
TINY_CONFIG = {
    'dim':            256,
    'hidden_dim':     512,
    'n_layers':       6,
    'n_heads':        4,
    'n_kv_heads':     2,        # GQA
    'head_dim':       64,
    'vocab_size':     32000,    # Vocab nhỏ hơn → embedding nhẹ hơn
    'n_experts':      4,
    'top_k':          2,
    'int4_group_size': 32,
    'rope_theta':     10_000.0,
    'max_seq_len':    512,
}

HEADER_SIZE   = 256          # bytes — phải khớp với Rust constants.rs
INT4_GROUP    = 32


# =============================================================================
# RoPE  — precompute cos/sin cache
# =============================================================================
def build_rope_cache(seq_len: int, head_dim: int, theta: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
    t     = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, freqs)            # [seq, half]
    cos   = freqs.cos()
    sin   = freqs.sin()
    return cos, sin   # [seq, half]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, n_heads, T, head_dim]"""
    B, H, T, D = x.shape
    half = D // 2
    x1   = x[..., :half]
    x2   = x[..., half:]
    cos_ = cos[:T, :].unsqueeze(0).unsqueeze(0)   # [1,1,T,half]
    sin_ = sin[:T, :].unsqueeze(0).unsqueeze(0)
    # Standard RoPE: [x1*cos - x2*sin, x1*sin + x2*cos]
    # Interleaved version (Qwen style):
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    r_even = x_even * cos_ - x_odd * sin_
    r_odd  = x_even * sin_ + x_odd * cos_
    # Interleave back
    out = torch.zeros_like(x)
    out[..., 0::2] = r_even
    out[..., 1::2] = r_odd
    return out


# =============================================================================
# RMSNorm
# =============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps    = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


# =============================================================================
# GQA Attention
# =============================================================================
class GQAttention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.dim      = cfg['dim']
        self.n_heads  = cfg['n_heads']
        self.n_kv     = cfg['n_kv_heads']
        self.head_dim = cfg['head_dim']
        self.repeat   = self.n_heads // self.n_kv

        # Full-dim projections (mở rộng = n_heads đầu Q, n_kv đầu K/V)
        self.q_proj  = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj  = nn.Linear(self.dim, self.n_kv   * self.head_dim, bias=False)
        self.v_proj  = nn.Linear(self.dim, self.n_kv   * self.head_dim, bias=False)
        self.o_proj  = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)   # [B, H, T, D]
        k = self.k_proj(x).view(B, T, self.n_kv,    self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv,    self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand KV for GQA
        k = k.repeat_interleave(self.repeat, dim=1)   # [B, n_heads, T, D]
        v = v.repeat_interleave(self.repeat, dim=1)

        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T, T]

        # Causal mask
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        if attn_mask is not None:
            scores = scores + attn_mask

        attn = F.softmax(scores, dim=-1)
        out  = torch.matmul(attn, v)                           # [B, H, T, D]
        out  = out.transpose(1, 2).contiguous().view(B, T, -1) # [B, T, dim]
        return self.o_proj(out)


# =============================================================================
# MoE Expert + Router
# =============================================================================
class Expert(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up   = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SparseMoE(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        dim    = cfg['dim']
        hidden = cfg['hidden_dim']
        n_exp  = cfg['n_experts']
        top_k  = cfg['top_k']

        self.top_k   = top_k
        self.n_exp   = n_exp
        self.router  = nn.Linear(dim, n_exp, bias=True)
        self.experts = nn.ModuleList([Expert(dim, hidden) for _ in range(n_exp)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, dim]"""
        B, T, D = x.shape
        flat = x.view(-1, D)                       # [B*T, D]

        logits = self.router(flat)                 # [B*T, n_exp]
        probs  = F.softmax(logits, dim=-1)

        # Top-k selection
        top_vals, top_idx = probs.topk(self.top_k, dim=-1)   # [B*T, top_k]
        # Renormalize selected weights
        top_vals = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-9)

        out = torch.zeros_like(flat)
        for k in range(self.top_k):
            expert_id     = top_idx[:, k]           # [B*T]
            expert_weight = top_vals[:, k]           # [B*T]

            # Route tokens to their assigned expert (grouped by expert)
            for e_id in range(self.n_exp):
                mask = (expert_id == e_id).nonzero(as_tuple=True)[0]
                if mask.numel() == 0:
                    continue
                tokens    = flat[mask]             # [n_tokens, D]
                e_out     = self.experts[e_id](tokens)
                w         = expert_weight[mask].unsqueeze(1)
                out[mask] = out[mask] + e_out * w

        return out.view(B, T, D)


# =============================================================================
# Transformer Block
# =============================================================================
class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.norm_attn = RMSNorm(cfg['dim'])
        self.attn      = GQAttention(cfg)
        self.norm_ffn  = RMSNorm(cfg['dim'])
        self.moe       = SparseMoE(cfg)

    def forward(self, x: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), cos, sin)
        x = x + self.moe(self.norm_ffn(x))
        return x


# =============================================================================
# Student MoE Model
# =============================================================================
class StudentMoE(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        dim     = cfg['dim']
        vocab   = cfg['vocab_size']

        self.embed   = nn.Embedding(vocab, dim)
        self.layers  = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg['n_layers'])])
        self.norm    = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

        # Tie weights (embedding ↔ lm_head) — như Qwen gốc
        self.lm_head.weight = self.embed.weight

        self._build_rope()
        self._init_weights()

    def _build_rope(self):
        cos, sin = build_rope_cache(
            self.cfg['max_seq_len'],
            self.cfg['head_dim'],
            self.cfg['rope_theta'],
            device=torch.device('cpu')
        )
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

    def _init_weights(self):
        """Xavier init — router cần init đặc biệt để tránh expert collapse"""
        for name, p in self.named_parameters():
            if 'router' in name and 'weight' in name:
                nn.init.zeros_(p)   # Router bắt đầu từ 0 → uniform routing
            elif p.dim() >= 2:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, T, vocab]"""
        B, T = input_ids.shape
        x   = self.embed(input_ids)
        cos = self.rope_cos[:T]
        sin = self.rope_sin[:T]
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)

    def load_from_myai(self, myai_path: str):
        """
        Load pre-trained weights từ .myai file vào PyTorch model.
        Đây là bước quan trọng — load weights Qwen gốc đã convert.
        """
        print(f"[student] Loading weights from {myai_path}...")
        with open(myai_path, 'rb') as f:
            data = f.read()

        buf   = np.frombuffer(data, dtype=np.uint8)
        dim   = self.cfg['dim']
        vocab = self.cfg['vocab_size']
        n_lay = self.cfg['n_layers']
        group = self.cfg['int4_group_size']

        cursor = HEADER_SIZE

        # --- 1. Embedding (Int8 + per-row scales) ---
        embed_bytes  = vocab * dim
        embed_i8     = np.frombuffer(buf[cursor:cursor + embed_bytes], dtype=np.int8).reshape(vocab, dim)
        cursor      += embed_bytes
        embed_scales = np.frombuffer(buf[cursor:cursor + vocab * 4], dtype=np.float32)
        cursor      += vocab * 4

        embed_f32 = embed_i8.astype(np.float32) * embed_scales[:, None]
        with torch.no_grad():
            self.embed.weight.copy_(torch.tensor(embed_f32))

        print(f"  [embed] Loaded {vocab}x{dim} (dequant int8->f32)")

        # --- 2. Layers ---
        hidden   = self.cfg['hidden_dim']
        n_heads  = self.cfg['n_heads']
        n_kv     = self.cfg['n_kv_heads']
        head_dim = self.cfg['head_dim']
        n_exp    = self.cfg['n_experts']

        for layer_idx, block in enumerate(self.layers):
            # RMS attn
            rms_attn_bytes = dim * 4
            rms_attn = np.frombuffer(buf[cursor:cursor + rms_attn_bytes], dtype=np.float32)
            cursor  += rms_attn_bytes

            # QKV (Int8: 3*dim*dim bytes + 3*dim scales)
            qkv_bytes  = 3 * dim * dim
            qkv_i8     = np.frombuffer(buf[cursor:cursor + qkv_bytes], dtype=np.int8).reshape(3 * dim, dim)
            cursor    += qkv_bytes
            qkv_scales = np.frombuffer(buf[cursor:cursor + 3 * dim * 4], dtype=np.float32)
            cursor    += 3 * dim * 4

            qkv_f32 = qkv_i8.astype(np.float32) * qkv_scales[:, None]
            q_w = qkv_f32[0:dim]                   # [dim, dim] = [n_heads*head_dim, dim]
            k_w = qkv_f32[dim:2*dim]               # [dim, dim] — already expanded from n_kv heads
            v_w = qkv_f32[2*dim:3*dim]

            # Attention out (Int8: dim*dim bytes + dim scales)
            out_bytes  = dim * dim
            out_i8     = np.frombuffer(buf[cursor:cursor + out_bytes], dtype=np.int8).reshape(dim, dim)
            cursor    += out_bytes
            out_scales = np.frombuffer(buf[cursor:cursor + dim * 4], dtype=np.float32)
            cursor    += dim * 4

            out_f32 = out_i8.astype(np.float32) * out_scales[:, None]

            # RMS FFN
            rms_ffn = np.frombuffer(buf[cursor:cursor + rms_attn_bytes], dtype=np.float32)
            cursor += rms_attn_bytes

            # Copy vào PyTorch block
            with torch.no_grad():
                block.norm_attn.weight.copy_(torch.tensor(rms_attn))
                block.norm_ffn.weight.copy_(torch.tensor(rms_ffn))

                # Q: [n_heads*head_dim, dim] → Linear weight = [out, in]
                block.attn.q_proj.weight.copy_(torch.tensor(q_w))

                # K,V: myai lưu full n_heads dim (đã expand trong convert)
                # Nhưng student có n_kv heads → cần shrink lại
                # Lấy 1 head đại diện cho mỗi kv group (trung bình)
                k_full = torch.tensor(k_w)  # [n_heads*head_dim, dim]
                v_full = torch.tensor(v_w)

                k_reshaped = k_full.view(n_kv, n_heads // n_kv, head_dim, dim).mean(dim=1)  # [n_kv, head_dim, dim]
                v_reshaped = v_full.view(n_kv, n_heads // n_kv, head_dim, dim).mean(dim=1)

                block.attn.k_proj.weight.copy_(k_reshaped.view(n_kv * head_dim, dim))
                block.attn.v_proj.weight.copy_(v_reshaped.view(n_kv * head_dim, dim))
                block.attn.o_proj.weight.copy_(torch.tensor(out_f32))

            # Router (n_experts * dim f32 + n_experts f32 bias)
            if n_exp >= 2:
                router_w = np.frombuffer(buf[cursor:cursor + n_exp * dim * 4], dtype=np.float32).reshape(n_exp, dim)
                cursor  += n_exp * dim * 4
                router_b = np.frombuffer(buf[cursor:cursor + n_exp * 4], dtype=np.float32)
                cursor  += n_exp * 4

                with torch.no_grad():
                    block.moe.router.weight.copy_(torch.tensor(router_w))
                    block.moe.router.bias.copy_(torch.tensor(router_b))

            # Experts (Int4)
            for e_idx, expert in enumerate(block.moe.experts):
                n_groups_h   = (hidden + group - 1) // group
                n_groups_d_h = (dim   + group - 1) // group

                # Gate+Up packed: [2*hidden * dim / 2] bytes
                gate_up_packed_bytes  = hidden * dim  # = hidden * (dim/2) * 2 halves packed to bytes
                gate_up_packed_bytes  = (2 * hidden * dim + 1) // 2
                gate_up_scales_count  = 2 * hidden * n_groups_d_h

                gate_up_packed = buf[cursor:cursor + gate_up_packed_bytes]
                cursor        += gate_up_packed_bytes
                gate_up_scales = np.frombuffer(buf[cursor:cursor + gate_up_scales_count * 4], dtype=np.float32)
                cursor        += gate_up_scales_count * 4

                # Down packed: [dim * hidden / 2] bytes
                down_packed_bytes  = (dim * hidden + 1) // 2
                down_scales_count  = dim * n_groups_h

                down_packed = buf[cursor:cursor + down_packed_bytes]
                cursor     += down_packed_bytes
                down_scales = np.frombuffer(buf[cursor:cursor + down_scales_count * 4], dtype=np.float32)
                cursor     += down_scales_count * 4

                # Dequant int4
                gate_w = _dequant_int4(
                    gate_up_packed[:gate_up_packed_bytes // 2],
                    gate_up_scales[:hidden * n_groups_d_h],
                    hidden, dim, group
                )
                up_w = _dequant_int4(
                    gate_up_packed[gate_up_packed_bytes // 2:],
                    gate_up_scales[hidden * n_groups_d_h:],
                    hidden, dim, group
                )
                down_w = _dequant_int4(down_packed, down_scales, dim, hidden, group)

                with torch.no_grad():
                    expert.gate.weight.copy_(torch.tensor(gate_w))
                    expert.up.weight.copy_(torch.tensor(up_w))
                    expert.down.weight.copy_(torch.tensor(down_w))

            if layer_idx % 6 == 0:
                print(f"  [layer {layer_idx:2d}] loaded | cursor={cursor / 1024 / 1024:.1f} MB")

        # Final norm + output proj (shared with embed)
        final_norm = np.frombuffer(buf[cursor:cursor + dim * 4], dtype=np.float32)
        cursor    += dim * 4
        with torch.no_grad():
            self.norm.weight.copy_(torch.tensor(final_norm))

        print(f"[student] Done. Total read: {cursor / 1024 / 1024:.1f} MB")


def _dequant_int4(packed: np.ndarray, scales: np.ndarray, rows: int, cols: int, group: int) -> np.ndarray:
    """Unpack Int4 weights → float32  [rows, cols]"""
    n_groups = (cols + group - 1) // group
    total    = rows * cols

    # Unpack nibbles
    expanded = np.zeros(total, dtype=np.int8)
    for i in range(len(packed)):
        lo = packed[i] & 0xF
        hi = (packed[i] >> 4) & 0xF
        if 2 * i     < total: expanded[2 * i]     = int(lo) - 8
        if 2 * i + 1 < total: expanded[2 * i + 1] = int(hi) - 8

    out = expanded.reshape(rows, cols).astype(np.float32)
    for r in range(rows):
        for g in range(n_groups):
            s   = g * group
            e   = min(s + group, cols)
            out[r, s:e] *= scales[r * n_groups + g]
    return out


# =============================================================================
# Dataset
# =============================================================================
class TextDataset(torch.utils.data.Dataset):
    """Tokenize corpus và chia thành các chunk seq_len."""

    def __init__(self, corpus_path: str, tokenizer, seq_len: int, max_samples: int = 50_000):
        self.seq_len = seq_len
        self.samples: List[List[int]] = []

        print(f"[dataset] Loading corpus {corpus_path}...")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"[dataset] Tokenizing {len(text):,} chars...")
        # Tokenize in chunks (avoid OOM for huge files)
        chunk = 50_000
        token_ids = []
        for start in range(0, len(text), chunk):
            ids = tokenizer.encode(text[start:start + chunk])
            if isinstance(ids, list):
                token_ids.extend(ids)
            else:
                token_ids.extend(ids.tolist())

        print(f"[dataset] Total tokens: {len(token_ids):,}")

        # Build (seq_len + 1) windows
        for i in range(0, len(token_ids) - seq_len - 1, seq_len // 2):
            window = token_ids[i:i + seq_len + 1]
            if len(window) == seq_len + 1:
                self.samples.append(window)
                if len(self.samples) >= max_samples:
                    break

        print(f"[dataset] {len(self.samples):,} samples (seq_len={seq_len})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]
        x   = torch.tensor(ids[:-1], dtype=torch.long)
        y   = torch.tensor(ids[1:],  dtype=torch.long)
        return x, y


# =============================================================================
# Knowledge Distillation Loss
# =============================================================================
def kd_loss(student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            labels:         torch.Tensor,
            temperature:    float = 4.0,
            alpha:          float = 0.7) -> torch.Tensor:
    """
    Combined KD loss:
        L = alpha * KL(student || teacher) + (1 - alpha) * CE(student, labels)

    temperature: làm mềm phân phối teacher (higher = softer)
    alpha: weight giữa KD và CE (0.7 tốt cho distillation)
    """
    B, T, V = student_logits.shape

    # Soft KL divergence
    s_soft = F.log_softmax(student_logits / temperature, dim=-1)
    t_soft = F.softmax(teacher_logits   / temperature, dim=-1)
    kl     = F.kl_div(s_soft, t_soft, reduction='batchmean') * (temperature ** 2)

    # Hard cross-entropy
    ce = F.cross_entropy(
        student_logits.view(B * T, V),
        labels.view(B * T),
        ignore_index=-1
    )

    return alpha * kl + (1 - alpha) * ce


# =============================================================================
# Training Loop
# =============================================================================
def train(args):
    device = torch.device('cpu')
    print(f"[train] Device: {device}")
    print(f"[train] Epochs={args.epochs}, batch={args.batch_size}, seq={args.seq_len}")
    print(f"[train] LR={args.lr}, KD_temp={args.kd_temp}, KD_alpha={args.kd_alpha}")

    # --- Load Teacher (Qwen2.5-0.5B) ---
    print("\n[teacher] Loading Qwen2.5-0.5B-Instruct...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    teacher   = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"[teacher] Loaded: {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M params")

    # --- Build Student ---
    print("\n[student] Building MoE student...")
    student = StudentMoE(STUDENT_CONFIG).to(device)

    if args.student and Path(args.student).exists():
        student.load_from_myai(args.student)
    else:
        print("[student] No .myai provided — initializing from scratch")

    total_params = sum(p.numel() for p in student.parameters())
    train_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[student] {total_params / 1e6:.1f}M total params | {train_params / 1e6:.1f}M trainable")

    # --- Dataset ---
    if args.corpus and Path(args.corpus).exists():
        dataset = TextDataset(args.corpus, tokenizer, args.seq_len, max_samples=args.max_samples)
    else:
        print("[dataset] No corpus provided — generating synthetic data from teacher...")
        dataset = _generate_synthetic_dataset(teacher, tokenizer, args.seq_len, n_samples=500, device=device)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        eps=1e-8,
    )

    # Cosine LR scheduler
    total_steps = len(loader) * args.epochs
    warmup_steps = min(200, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    # --- Train ---
    print(f"\n[train] Starting distillation — {len(dataset):,} samples, {len(loader):,} steps/epoch")
    best_loss = float('inf')
    log_interval = max(1, len(loader) // 20)

    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            # Teacher forward (no grad)
            with torch.no_grad():
                teacher_out    = teacher(input_ids=x)
                teacher_logits = teacher_out.logits   # [B, T, V]

            # Student forward
            student_logits = student(x)               # [B, T, V]

            loss = kd_loss(
                student_logits, teacher_logits, y,
                temperature=args.kd_temp,
                alpha=args.kd_alpha
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % log_interval == 0:
                lr_now  = scheduler.get_last_lr()[0]
                elapsed = time.time() - epoch_start
                print(f"  epoch={epoch}/{args.epochs} step={step:4d}/{len(loader):4d} "
                      f"loss={loss.item():.4f} lr={lr_now:.2e} elapsed={elapsed:.0f}s")

        avg_loss = epoch_loss / len(loader)
        print(f"[epoch {epoch}] avg_loss={avg_loss:.4f} | time={time.time()-epoch_start:.0f}s")

        # Save checkpoint
        ckpt_path = f"checkpoint_epoch{epoch}.pt"
        torch.save({
            'epoch':        epoch,
            'model_state':  student.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'loss':         avg_loss,
            'config':       STUDENT_CONFIG,
        }, ckpt_path)
        print(f"[ckpt] Saved → {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(student.state_dict(), "best_student.pt")
            print(f"[ckpt] New best! loss={best_loss:.4f}")

    # --- Export ---
    print(f"\n[export] Exporting best checkpoint → {args.output}")
    student.load_state_dict(torch.load("best_student.pt", map_location='cpu'))
    export_to_myai(student, STUDENT_CONFIG, args.output)
    print(f"[done] Saved trained model → {args.output}")


def _generate_synthetic_dataset(teacher, tokenizer, seq_len, n_samples=500, device='cpu'):
    """Sinh dữ liệu tổng hợp khi không có corpus — dùng prompts đa dạng"""
    prompts = [
        "Giải thích ngắn gọn về trí tuệ nhân tạo:",
        "Python là gì và tại sao nó phổ biến?",
        "Hãy kể một câu chuyện ngắn:",
        "Toán học cơ bản: 2 + 2 = ",
        "Thủ đô của Việt Nam là ",
        "Lập trình hướng đối tượng có nghĩa là ",
        "Machine learning và deep learning khác nhau như thế nào?",
        "Hãy viết một đoạn code Python đơn giản:",
        "Ưu điểm của năng lượng tái tạo là:",
        "Ngôn ngữ Rust được thiết kế để:",
    ]

    samples = []
    print(f"[synthetic] Generating {n_samples} samples from teacher...")
    for i in range(n_samples):
        prompt = prompts[i % len(prompts)]
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            out = teacher.generate(
                **inputs,
                max_new_tokens=seq_len,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id,
            )
        ids = out[0].tolist()
        if len(ids) > seq_len + 1:
            ids = ids[:seq_len + 1]
        if len(ids) == seq_len + 1:
            samples.append(ids)
        if (i + 1) % 50 == 0:
            print(f"  generated {i+1}/{n_samples}")

    class _SyntheticDS(torch.utils.data.Dataset):
        def __init__(self, s): self.s = s
        def __len__(self): return len(self.s)
        def __getitem__(self, i):
            ids = self.s[i]
            return torch.tensor(ids[:-1], dtype=torch.long), torch.tensor(ids[1:], dtype=torch.long)

    return _SyntheticDS(samples)


# =============================================================================
# Export trained PyTorch → .myai format
# =============================================================================
def export_to_myai(model: StudentMoE, cfg: dict, output_path: str):
    """
    Export PyTorch weights → .myai binary format
    Quantize: attention Int8, experts Int4
    """
    import struct

    dim    = cfg['dim']
    hidden = cfg['hidden_dim']
    vocab  = cfg['vocab_size']
    n_lay  = cfg['n_layers']
    n_heads= cfg['n_heads']
    n_kv   = cfg['n_kv_heads']
    head_dim=cfg['head_dim']
    n_exp  = cfg['n_experts']
    top_k  = cfg['top_k']
    group  = cfg['int4_group_size']

    model.eval()

    with open(output_path, 'wb') as f:
        # --- Header (256 bytes) ---
        _write_header(f, cfg)

        # --- Embedding (Int8) ---
        embed_w = model.embed.weight.detach().numpy()   # [vocab, dim]
        e_q, e_s = _quant_int8(embed_w)
        f.write(e_q.tobytes())
        f.write(e_s.astype(np.float32).tobytes())
        print(f"[export] embed done | {f.tell()/1024/1024:.1f} MB")

        # --- Layers ---
        for li, block in enumerate(model.layers):
            # RMS norms
            f.write(block.norm_attn.weight.detach().numpy().astype(np.float32).tobytes())

            # QKV (expand K,V from n_kv → n_heads before writing)
            q_w = block.attn.q_proj.weight.detach().numpy()  # [n_heads*head_dim, dim]
            k_w = block.attn.k_proj.weight.detach().numpy()  # [n_kv*head_dim, dim]
            v_w = block.attn.v_proj.weight.detach().numpy()

            # Expand KV → full n_heads
            k_exp = np.repeat(k_w.reshape(n_kv, head_dim, dim), n_heads // n_kv, axis=0).reshape(dim, dim)
            v_exp = np.repeat(v_w.reshape(n_kv, head_dim, dim), n_heads // n_kv, axis=0).reshape(dim, dim)

            qkv = np.concatenate([q_w, k_exp, v_exp], axis=0)  # [3*dim, dim]
            qkv_q, qkv_s = _quant_int8(qkv)
            f.write(qkv_q.tobytes())
            f.write(qkv_s.astype(np.float32).tobytes())

            # Attention out
            out_w = block.attn.o_proj.weight.detach().numpy()
            out_q, out_s = _quant_int8(out_w)
            f.write(out_q.tobytes())
            f.write(out_s.astype(np.float32).tobytes())

            # RMS FFN
            f.write(block.norm_ffn.weight.detach().numpy().astype(np.float32).tobytes())

            # Router
            f.write(block.moe.router.weight.detach().numpy().astype(np.float32).tobytes())
            f.write(block.moe.router.bias.detach().numpy().astype(np.float32).tobytes())

            # Experts (Int4)
            for expert in block.moe.experts:
                gate_w = expert.gate.weight.detach().numpy()  # [hidden, dim]
                up_w   = expert.up.weight.detach().numpy()
                down_w = expert.down.weight.detach().numpy()  # [dim, hidden]

                gate_p, gate_s = _quant_int4(gate_w, group)
                up_p,   up_s   = _quant_int4(up_w,   group)
                down_p, down_s = _quant_int4(down_w, group)

                # gate+up packed together
                f.write(gate_p.tobytes())
                f.write(up_p.tobytes())
                f.write(gate_s.astype(np.float32).tobytes())
                f.write(up_s.astype(np.float32).tobytes())

                f.write(down_p.tobytes())
                f.write(down_s.astype(np.float32).tobytes())

            if li % 6 == 0:
                print(f"[export] layer {li:2d} done | {f.tell()/1024/1024:.1f} MB")

        # Final norm
        f.write(model.norm.weight.detach().numpy().astype(np.float32).tobytes())

        # Output projection (shared with embed, re-quantize)
        out_q, out_s = _quant_int8(model.lm_head.weight.detach().numpy())
        f.write(out_q.tobytes())
        f.write(out_s.astype(np.float32).tobytes())

        total = f.tell()
    print(f"[export] Done → {output_path} | {total / 1024 / 1024:.1f} MB")


def _quant_int8(w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols = w.shape
    scales = np.zeros(rows, dtype=np.float32)
    q      = np.zeros_like(w, dtype=np.int8)
    for i in range(rows):
        mx = np.abs(w[i]).max()
        s  = mx / 127.0 if mx > 0 else 1.0
        scales[i] = s
        q[i] = np.clip(w[i] / s, -128, 127).astype(np.int8)
    return q, scales


def _quant_int4(w: np.ndarray, group: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    rows, cols  = w.shape
    n_groups    = (cols + group - 1) // group
    scales      = np.zeros(rows * n_groups, dtype=np.float32)
    q_all       = []
    for r in range(rows):
        q_row = np.zeros(cols, dtype=np.int8)
        for g in range(n_groups):
            s  = g * group
            e  = min(s + group, cols)
            seg = w[r, s:e]
            mx  = np.abs(seg).max()
            sc  = mx / 7.0 if mx > 0 else 1.0
            scales[r * n_groups + g] = sc
            q_row[s:e] = np.clip(seg / sc, -8, 7).astype(np.int8)
        q_all.append(q_row)
    flat  = np.concatenate(q_all)
    uint4 = (flat + 8).astype(np.uint8)  # 0..15
    packed = np.zeros((len(uint4) + 1) // 2, dtype=np.uint8)
    packed[:len(uint4)//2] = (uint4[0::2][:len(uint4)//2] & 0xF) | ((uint4[1::2][:len(uint4)//2] & 0xF) << 4)
    if len(uint4) % 2 == 1:
        packed[-1] = uint4[-1] & 0xF
    return packed, scales


def _write_header(f, cfg: dict):
    """Write 256-byte header — phải khớp 100% với Rust FileHeader struct"""
    dim     = cfg['dim']
    hidden  = cfg['hidden_dim']
    vocab   = cfg['vocab_size']
    n_lay   = cfg['n_layers']
    n_heads = cfg['n_heads']
    n_exp   = cfg['n_experts']
    top_k   = cfg['top_k']
    group   = cfg['int4_group_size']
    max_seq = cfg['max_seq_len']

    # Convert rope_theta (float) to u32 bits for storage
    import struct as st
    rope_bits = st.unpack('I', st.pack('f', cfg['rope_theta']))[0]

    # Header layout (must match src/model/header.rs exactly)
    header  = st.pack('<I', 0x4D594149)  # magic = "MYAI"
    header += st.pack('<I', 2)           # version = 2 (MoE v2)
    header += st.pack('<I', dim)
    header += st.pack('<I', hidden)
    header += st.pack('<I', n_lay)
    header += st.pack('<I', n_heads)
    header += st.pack('<I', vocab)
    header += st.pack('<I', max_seq)
    header += st.pack('<I', n_exp)
    header += st.pack('<I', top_k)
    header += st.pack('<I', group)
    header += st.pack('<I', 1)           # flags: bit0=quantized
    header += st.pack('<I', 0)           # depth_router_layer = 0
    header += st.pack('<I', rope_bits)   # rope_theta as u32 bits

    # Padding to 256 bytes
    header += b'\x00' * (HEADER_SIZE - len(header))
    assert len(header) == HEADER_SIZE, f"Header size mismatch: {len(header)} != {HEADER_SIZE}"
    f.write(header)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Knowledge Distillation: Qwen → MoE Student')
    parser.add_argument('--teacher',     default='Qwen/Qwen2.5-0.5B-Instruct',
                        help='HuggingFace model ID hoặc path đến Qwen teacher')
    parser.add_argument('--student',     default='qwen_moe.myai',
                        help='Path đến .myai file đã convert (làm khởi tạo)')
    parser.add_argument('--corpus',      default='',
                        help='Path đến corpus text để distill (để trống = sinh tổng hợp)')
    parser.add_argument('--output',      default='trained_student.myai',
                        help='Output file .myai sau khi train')
    parser.add_argument('--epochs',      type=int,   default=3)
    parser.add_argument('--batch-size',  type=int,   default=2)
    parser.add_argument('--seq-len',     type=int,   default=128,
                        help='Sequence length (thấp hơn = ít RAM hơn)')
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--kd-temp',     type=float, default=4.0,
                        help='KD temperature (higher = softer targets)')
    parser.add_argument('--kd-alpha',    type=float, default=0.7,
                        help='KD weight (0=only CE, 1=only KD)')
    parser.add_argument('--max-samples', type=int,   default=10_000)

    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
