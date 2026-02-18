#!/usr/bin/env python3
"""
=============================================================================
train_tiny.py — Train model nhỏ ~20M params để test độ thông minh nhanh
=============================================================================

Model TINY_CONFIG:
  dim=256, hidden=512, 6 layers, 4 experts, top_k=2, vocab=32000
  Tổng: ~18.8M params

Pipeline:
  - KHÔNG cần .myai khởi tạo — train from scratch với random init
  - Teacher: Qwen2.5-0.5B để distill (hoặc bỏ qua, chỉ dùng CE loss)
  - Dataset: corpus.txt hoặc sinh tổng hợp tự động
  - Export: tiny_student.myai (tương thích Rust engine)

Ước lượng thời gian (Ryzen 5 2500, CPU):
  seq_len=64,  200 samples x3 epochs ≈  8–12 phút
  seq_len=64,  500 samples x3 epochs ≈ 20–30 phút
  seq_len=128, 200 samples x3 epochs ≈ 15–25 phút

Chạy nhanh (không cần corpus, không cần teacher):
    python train_tiny.py --no-teacher --epochs 3 --samples 200

Chạy đầy đủ với distillation:
    python train_tiny.py --corpus corpus.txt --epochs 3 --samples 500

Test sau khi train:
    python train_tiny.py --test --checkpoint tiny_student.myai
=============================================================================
"""

import argparse
import os
import time
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple

# Import từ train_distill.py — dùng lại toàn bộ architecture
from train_distill import (
    TINY_CONFIG,
    StudentMoE,
    export_to_myai,
    kd_loss,
    TextDataset,
    build_rope_cache,
)

# =============================================================================
# Cấu hình mặc định cho train tiny
# =============================================================================
DEFAULT_SEQ_LEN    = 64
DEFAULT_BATCH      = 1
DEFAULT_EPOCHS     = 3
DEFAULT_LR         = 5e-4      # LR cao hơn 1 chút cho model nhỏ
DEFAULT_SAMPLES    = 200
DEFAULT_OUTPUT     = "tiny_student.myai"
DEFAULT_TEACHER    = "Qwen/Qwen2.5-0.5B-Instruct"


# =============================================================================
# Synthetic dataset không cần teacher (dùng random prompts)
# =============================================================================
class SyntheticDataset(torch.utils.data.Dataset):
    """
    Dataset giả: random token IDs trong vocab_size.
    Dùng khi không có corpus và không muốn load teacher.
    Mục đích: kiểm tra pipeline hoạt động và loss giảm được không.
    """
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int):
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.n_samples  = n_samples
        torch.manual_seed(42)
        # Pre-generate tất cả để tránh lag runtime
        self.data = torch.randint(0, vocab_size, (n_samples, seq_len + 1))
        print(f"[synthetic] {n_samples} random samples (vocab={vocab_size}, seq={seq_len})")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        row = self.data[idx]
        return row[:-1], row[1:]


# =============================================================================
# Prompts tiếng Việt để test thông minh
# =============================================================================
TEST_PROMPTS = [
    "Xin chào! Bạn là ai?",
    "Thủ đô của Việt Nam là",
    "2 + 2 =",
    "Python là ngôn ngữ lập trình",
    "Hãy kể một câu chuyện ngắn về",
    "Trí tuệ nhân tạo có thể",
    "Rust là ngôn ngữ",
    "Học máy (machine learning) là",
]


# =============================================================================
# Train loop chính
# =============================================================================
def train(args):
    print("=" * 65)
    print("TRAIN TINY MODEL — ~18.8M params")
    print("=" * 65)
    print(f"Config: dim={TINY_CONFIG['dim']}, layers={TINY_CONFIG['n_layers']}, "
          f"experts={TINY_CONFIG['n_experts']}, vocab={TINY_CONFIG['vocab_size']}")

    device = torch.device('cpu')

    # --- Build student từ scratch ---
    print("\n[1/4] Building tiny student model...")
    student = StudentMoE(TINY_CONFIG).to(device)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"      {n_params / 1e6:.2f}M params | "
          f"{n_params * 4 / 1024 / 1024:.1f} MB (float32)")

    # --- Load teacher nếu cần ---
    teacher   = None
    tokenizer = None

    if not args.no_teacher:
        print(f"\n[2/4] Loading teacher: {args.teacher}")
        t0 = time.time()
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
            teacher   = AutoModelForCausalLM.from_pretrained(
                args.teacher, torch_dtype=torch.float32, trust_remote_code=True
            ).to(device)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            print(f"      Teacher loaded in {time.time()-t0:.1f}s | "
                  f"{sum(p.numel() for p in teacher.parameters())/1e6:.0f}M params")
        except Exception as e:
            print(f"      WARNING: Cannot load teacher ({e})")
            print("      Falling back to CE-only training...")
            teacher   = None
            tokenizer = None
    else:
        print("\n[2/4] Skipping teacher (--no-teacher mode)")

    # --- Dataset ---
    print(f"\n[3/4] Preparing dataset...")

    if args.corpus and Path(args.corpus).exists() and tokenizer is not None:
        dataset = TextDataset(args.corpus, tokenizer, args.seq_len, max_samples=args.samples)
    elif args.corpus and Path(args.corpus).exists():
        # Corpus có nhưng không có tokenizer → dùng simple char tokenizer
        dataset = _load_corpus_simple(args.corpus, TINY_CONFIG['vocab_size'],
                                      args.seq_len, args.samples)
    else:
        # Không có gì → random synthetic
        print("      No corpus found → using synthetic random data")
        dataset = SyntheticDataset(TINY_CONFIG['vocab_size'], args.seq_len, args.samples)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=0
    )
    print(f"      {len(dataset):,} samples → {len(loader):,} steps/epoch")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=0.01
    )
    total_steps  = len(loader) * args.epochs
    warmup_steps = max(1, min(50, total_steps // 10))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup_steps / total_steps,
    )

    # --- Estimate time ---
    print(f"\n[4/4] Starting training...")
    step_sec = 1.8 if teacher is None else 4.0   # ước lượng
    eta_h = total_steps * step_sec / 3600
    print(f"      Estimated time: {eta_h:.1f}h ({eta_h*60:.0f} phút)")
    print(f"      {args.epochs} epochs × {len(loader)} steps × ~{step_sec:.1f}s/step")
    print("-" * 65)

    t_train_start = time.time()
    best_loss     = float('inf')
    log_every     = max(1, len(loader) // 10)

    for epoch in range(1, args.epochs + 1):
        student.train()
        epoch_loss  = 0.0
        epoch_start = time.time()

        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            if teacher is not None:
                # Knowledge Distillation
                with torch.no_grad():
                    # Teacher vocab có thể lớn hơn tiny vocab → cắt
                    t_logits = teacher(input_ids=x).logits[:, :, :TINY_CONFIG['vocab_size']]
                s_logits = student(x)
                loss = kd_loss(s_logits, t_logits, y,
                               temperature=args.kd_temp, alpha=args.kd_alpha)
            else:
                # CE only — không cần teacher
                s_logits = student(x)
                B, T, V = s_logits.shape
                loss = F.cross_entropy(s_logits.view(B * T, V), y.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % log_every == 0:
                elapsed   = time.time() - epoch_start
                steps_done= step + 1
                eta_epoch = elapsed / steps_done * (len(loader) - steps_done)
                print(f"  epoch={epoch}/{args.epochs} "
                      f"step={step:3d}/{len(loader):3d} "
                      f"loss={loss.item():.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.1e} "
                      f"ETA={eta_epoch:.0f}s")

        avg = epoch_loss / len(loader)
        elapsed_epoch = time.time() - epoch_start
        print(f"[epoch {epoch}] avg_loss={avg:.4f} | {elapsed_epoch:.0f}s")

        # Checkpoint
        ckpt = f"tiny_ckpt_ep{epoch}.pt"
        torch.save({'epoch': epoch, 'model_state': student.state_dict(),
                    'loss': avg, 'config': TINY_CONFIG}, ckpt)
        print(f"  Saved → {ckpt}")

        if avg < best_loss:
            best_loss = avg
            torch.save(student.state_dict(), "tiny_best.pt")
            print(f"  ★ New best! loss={best_loss:.4f}")

    total_time = time.time() - t_train_start
    print(f"\n[done] Total training time: {total_time/60:.1f} phút")

    # --- Export .myai ---
    print(f"\n[export] Exporting → {args.output}")
    student.load_state_dict(torch.load("tiny_best.pt", map_location='cpu'))
    export_to_myai(student, TINY_CONFIG, args.output)
    print(f"[done] Saved → {args.output}")
    print(f"       File size: {Path(args.output).stat().st_size / 1024 / 1024:.1f} MB")

    # --- Quick test ---
    if not args.no_test:
        quick_test(student)


# =============================================================================
# Load corpus đơn giản không cần tokenizer (char-level fallback)
# =============================================================================
def _load_corpus_simple(corpus_path: str, vocab_size: int, seq_len: int, n_samples: int):
    print(f"      Loading corpus (char-level fallback)...")
    with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # Map chars → indices (modulo vocab_size)
    ids = [ord(c) % vocab_size for c in text]
    print(f"      {len(ids):,} chars → {len(ids):,} tokens")

    samples = []
    step = max(1, seq_len // 2)
    for i in range(0, len(ids) - seq_len - 1, step):
        window = ids[i:i + seq_len + 1]
        if len(window) == seq_len + 1:
            samples.append(window)
            if len(samples) >= n_samples:
                break

    class _DS(torch.utils.data.Dataset):
        def __init__(self, s): self.s = s
        def __len__(self): return len(self.s)
        def __getitem__(self, i):
            row = self.s[i]
            return torch.tensor(row[:-1], dtype=torch.long), torch.tensor(row[1:], dtype=torch.long)

    d = _DS(samples)
    print(f"      {len(d):,} samples")
    return d


# =============================================================================
# Quick test: kiểm tra model có "học" được gì không
# =============================================================================
def quick_test(model: StudentMoE):
    """
    Test đơn giản: cho model generate token và xem có coherent không.
    Không cần tokenizer — dùng token ID trực tiếp.
    """
    print("\n" + "=" * 65)
    print("QUICK TEST — Greedy generation (20 tokens)")
    print("=" * 65)
    model.eval()

    # Test với một vài seed token sequences
    test_inputs = [
        [1, 2, 3, 4, 5],           # Random short sequence
        [100, 200, 300],            # Mid-range tokens
        [0, 1, 0, 1, 0],           # Repeating pattern
    ]

    for i, input_ids in enumerate(test_inputs):
        ids = torch.tensor([input_ids], dtype=torch.long)
        generated = list(input_ids)

        with torch.no_grad():
            for _ in range(20):
                logits = model(ids)              # [1, T, vocab]
                next_token = logits[0, -1, :].argmax().item()
                generated.append(next_token)
                ids = torch.tensor([generated], dtype=torch.long)

        # Tính entropy của logits (đo độ tự tin)
        last_logits = logits[0, -1, :]
        probs = F.softmax(last_logits, dim=-1)
        entropy = -(probs * (probs + 1e-9).log()).sum().item()

        print(f"\nTest {i+1}: input={input_ids}")
        print(f"  Generated: {generated[len(input_ids):]}")
        print(f"  Entropy: {entropy:.2f} (thấp=tự tin, cao=lúng túng)")
        print(f"  Top-5 next tokens: {last_logits.topk(5).indices.tolist()}")

    # Kiểm tra có bị repetition collapse không
    unique_ratio = len(set(generated)) / len(generated)
    print(f"\nUnique token ratio: {unique_ratio:.2f} "
          f"({'OK' if unique_ratio > 0.3 else 'WARNING: repetition collapse!'})")
    print("=" * 65)


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train tiny ~20M MoE model để test intelligence'
    )
    parser.add_argument('--teacher',     default=DEFAULT_TEACHER,
                        help='HuggingFace teacher model')
    parser.add_argument('--no-teacher',  action='store_true',
                        help='Bỏ teacher, chỉ dùng CE loss (nhanh hơn ~2x)')
    parser.add_argument('--corpus',      default='',
                        help='Path corpus.txt (để trống = synthetic random data)')
    parser.add_argument('--output',      default=DEFAULT_OUTPUT)
    parser.add_argument('--epochs',      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument('--batch',       type=int,   default=DEFAULT_BATCH)
    parser.add_argument('--seq-len',     type=int,   default=DEFAULT_SEQ_LEN)
    parser.add_argument('--lr',          type=float, default=DEFAULT_LR)
    parser.add_argument('--samples',     type=int,   default=DEFAULT_SAMPLES)
    parser.add_argument('--kd-temp',     type=float, default=4.0)
    parser.add_argument('--kd-alpha',    type=float, default=0.7)
    parser.add_argument('--no-test',     action='store_true',
                        help='Bỏ qua quick test sau khi train')
    parser.add_argument('--test',        action='store_true',
                        help='Chỉ chạy test với model đã có')
    parser.add_argument('--checkpoint',  default='',
                        help='Path .pt hoặc .myai để test')

    args = parser.parse_args()
    args.seq_len = args.__dict__.get('seq_len', DEFAULT_SEQ_LEN)

    # Fix: argparse dùng - thay vì _
    if not hasattr(args, 'seq_len'):
        args.seq_len = DEFAULT_SEQ_LEN
    args.seq_len = vars(args).get('seq_len', DEFAULT_SEQ_LEN)

    if args.test:
        # Chỉ load và test
        print(f"[test] Loading checkpoint: {args.checkpoint}")
        model = StudentMoE(TINY_CONFIG)
        if args.checkpoint.endswith('.myai'):
            # Load from .myai — cần implement load_from_myai với TINY_CONFIG
            print("  (loading from .myai format)")
            model.load_from_myai(args.checkpoint)
        elif args.checkpoint.endswith('.pt') and Path(args.checkpoint).exists():
            state = torch.load(args.checkpoint, map_location='cpu')
            if isinstance(state, dict) and 'model_state' in state:
                model.load_state_dict(state['model_state'])
            else:
                model.load_state_dict(state)
        quick_test(model)
        return

    train(args)


if __name__ == '__main__':
    main()
