"""
Benchmark 1 bước train để ước lượng tổng thời gian trên CPU.
Chạy: python benchmark_train.py
"""
import time, torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

SEQ_LEN   = 64
BATCH     = 1
TEACHER   = "Qwen/Qwen2.5-0.5B-Instruct"

print("=" * 60)
print("BENCHMARK: 1 training step trên CPU")
print(f"SEQ_LEN={SEQ_LEN}, BATCH={BATCH}")
print("=" * 60)

t0 = time.time()
print(f"\n[1/4] Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(TEACHER, trust_remote_code=True)
print(f"      Done in {time.time()-t0:.1f}s")

t1 = time.time()
print(f"[2/4] Loading Qwen teacher (float32)...", flush=True)
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER, torch_dtype=torch.float32, trust_remote_code=True
)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
print(f"      Done in {time.time()-t1:.1f}s | "
      f"{sum(p.numel() for p in teacher.parameters())/1e6:.0f}M params")

print(f"\n[3/4] Loading student model...", flush=True)
from train_distill import StudentMoE, STUDENT_CONFIG
t2 = time.time()
student = StudentMoE(STUDENT_CONFIG)
student.load_from_myai("qwen_moe.myai")
student.train()
print(f"      Done in {time.time()-t2:.1f}s | "
      f"{sum(p.numel() for p in student.parameters())/1e6:.0f}M params")

optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)

# Dummy batch
ids = torch.randint(0, 151936, (BATCH, SEQ_LEN + 1))
x   = ids[:, :-1]
y   = ids[:, 1:]

print(f"\n[4/4] Benchmarking 1 forward+backward step...", flush=True)
t3 = time.time()

# Teacher forward
with torch.no_grad():
    teacher_logits = teacher(input_ids=x).logits

t_teacher = time.time() - t3
print(f"      Teacher forward: {t_teacher:.1f}s")

t4 = time.time()
# Student forward + backward
student_logits = student(x)
B, T, V = student_logits.shape
loss = F.kl_div(
    F.log_softmax(student_logits / 4.0, dim=-1),
    F.softmax(teacher_logits   / 4.0, dim=-1),
    reduction='batchmean'
) * 16.0 + F.cross_entropy(student_logits.view(B*T, V), y.view(B*T))
optimizer.zero_grad()
loss.backward()
optimizer.step()

t_student = time.time() - t4
t_step    = time.time() - t3

print(f"      Student fwd+bwd: {t_student:.1f}s")
print(f"      Total 1 step:    {t_step:.1f}s")
print(f"      Loss:            {loss.item():.4f}")

# --- Estimate ---
print("\n" + "=" * 60)
print("ƯỚC LƯỢNG THỜI GIAN TRAIN")
print("=" * 60)

for n_samples in [500, 2000, 10000]:
    steps_per_epoch = n_samples // BATCH
    secs_epoch      = steps_per_epoch * t_step
    print(f"\n  {n_samples:>6} samples, {3} epochs:")
    print(f"    {steps_per_epoch} steps/epoch × {t_step:.1f}s = {secs_epoch/3600:.1f}h/epoch")
    print(f"    Tổng 3 epochs ≈ {3*secs_epoch/3600:.1f}h ({3*secs_epoch/3600*60:.0f} phút)")

print(f"[TIP] Giam --seq-len va --max-samples de train nhanh hon:")
print("  seq_len=64,  500 samples, 3 epochs -> nhanh nhat co the tren CPU")
