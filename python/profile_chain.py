"""
验证 torch.mm 的完整调用链。
分三步跑: ① 只用 randn  → 看哪些 kernel
         ② 再加 matmul → 看多了哪些 kernel
         ③ TF32 toggle → 确认走 cuBLAS
"""
import torch

m = n = k = 1024

print("=" * 60)
print("STEP 1: randn only — 识别噪声 kernel")
print("=" * 60)
a = torch.randn(m, k, device="cuda", dtype=torch.float32)
b = torch.randn(k, n, device="cuda", dtype=torch.float32)
torch.cuda.synchronize()

print("\n" + "=" * 60)
print("STEP 2: matmul — 看多了什么 kernel")
print("=" * 60)
c = a @ b
torch.cuda.synchronize()

print("\n" + "=" * 60)
print("STEP 3: TF32 OFF — 证实走 cuBLAS")
print("=" * 60)
print("如果 TF32 toggle 改变性能 → kernel 走过 cuBLAS")
print("因为 allow_tf32 是 cuBLAS 的配置参数")
print()
for tf32 in [False, True]:
    torch.backends.cuda.matmul.allow_tf32 = tf32
    # fresh data
    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b = torch.randn(k, n, device="cuda", dtype=torch.float32)
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    for _ in range(5): a @ b
    torch.cuda.synchronize()
    s.record()
    for _ in range(10): a @ b
    e.record()
    torch.cuda.synchronize()
    t = s.elapsed_time(e) / 10
    tflops = 2 * m * n * k / (t / 1000) / 1e12
    print(f"  TF32={tf32}: {t:.3f} ms → {tflops:.3f} TFLOPS")
print(f"\n  2x speed change = cuBLAS in control")
print(f"  (CUTLASS kernel is INSIDE cuBLAS, not called directly by PyTorch)")

# Bonus: prove the call chain in Python
print(f"\n" + "=" * 60)
print("STEP 4: Python-level trace")
print("=" * 60)
import torch.autograd.profiler as prof
with prof.profile(use_cuda=True) as p:
    c = torch.randn(1024, 1024, device="cuda") @ torch.randn(1024, 1024, device="cuda")
    torch.cuda.synchronize()

# Filter for aten ops
aten_ops = [e for e in p.key_averages() if e.key.startswith("aten::")]
for e in aten_ops:
    print(f"  {e.key:40s} calls={e.count}")
print()
print("  aten::mm is the ATen op that dispatches to cuBLAS.")
print("  Source: PyTorch repo, aten/src/ATen/native/cuda/Blas.cpp")
