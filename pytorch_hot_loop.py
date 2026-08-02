"""PyTorch 标准实现 — Element-wise benchmark baseline"""
import torch
import statistics

N = 1_000_000
WARMUP = 3
RUNS = 10

x_cpu = torch.arange(N, dtype=torch.float32) * 0.001
x = x_cpu.cuda()

print("=" * 55)
print("  PyTorch Element-wise: sin^2 + cos^2 + sqrt(|x|)")
print("=" * 55)
print()


# ----- 版本 1: 原生 PyTorch (eager mode) -----
print("[1] Eager mode (native PyTorch, no fusion)")

def compute_eager(x):
    return torch.sin(x) ** 2 + torch.cos(x) ** 2 + torch.sqrt(torch.abs(x))

for _ in range(WARMUP):
    compute_eager(x)
torch.cuda.synchronize()

times = []
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
for _ in range(RUNS):
    start.record()
    y = compute_eager(x)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

t_eager = statistics.mean(times)
print(f"  avg: {t_eager:.4f} ms")
print()


# ----- 版本 2: torch.compile (Inductor fusion) -----
print("[2] torch.compile (Inductor kernel fusion)")

@torch.compile
def compute_compiled(x):
    return torch.sin(x) ** 2 + torch.cos(x) ** 2 + torch.sqrt(torch.abs(x))

# torch.compile 首次调用会编译 (更慢)，用更多 warmup
for _ in range(4):
    compute_compiled(x)
torch.cuda.synchronize()

times = []
for _ in range(RUNS):
    start.record()
    y = compute_compiled(x)
    end.record()
    torch.cuda.synchronize()
    times.append(start.elapsed_time(end))

t_compiled = statistics.mean(times)
print(f"  avg: {t_compiled:.4f} ms")
print()


# ----- 汇总对比 -----
# 数值验证
expected = torch.sin(x_cpu) ** 2 + torch.cos(x_cpu) ** 2 + torch.sqrt(torch.abs(x_cpu))
diff_eager = (y.cpu() - expected).abs().max().item()

print(f"  Numerical check: max_diff = {diff_eager:.2e} {'PASS' if diff_eager < 1e-4 else 'FAIL'}")
print()
print("=" * 55)
print("  Final Comparison: 1M elements, sin^2 + cos^2 + sqrt(|x|)")
print("=" * 55)
print(f"  PyTorch eager:      {t_eager:.4f} ms   (N kernels, no fusion)")
print(f"  PyTorch torch.compile: {t_compiled:.4f} ms   (Inductor fusion)")
print(f"  Numba CUDA:         0.0493 ms   (hand-rolled 1 kernel)")
print(f"  Triton:             0.0483 ms   (hand-rolled 1 kernel)")
print(f"  Raw CUDA C++:       0.0501 ms   (hand-rolled 1 kernel)")
print()
if t_compiled < t_eager:
    print(f"  torch.compile speedup: {t_eager / t_compiled:.1f}x over eager")
