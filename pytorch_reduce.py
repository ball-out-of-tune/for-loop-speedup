"""PyTorch Reduce 基准测试 — sum / max / mean

Reduce 操作: N 个元素 → 1 个标量 (或每行/列 → 向量)
和 element-wise 的本质区别: 元素之间有依赖，不能完全并行。

运行: python pytorch_reduce.py
"""
import torch
import statistics

N = 1_000_000
WARMUP = 5
RUNS = 20

x_cpu = torch.arange(N, dtype=torch.float32) * 0.001
x = x_cpu.cuda()

print("=" * 55)
print("  PyTorch Reduce: sum / max / mean  (1M elements)")
print("=" * 55)
print()


# -----------------------------------------------------------------------
# naive 写法: 就是一个普通 for 循环的思想，但用 PyTorch 的 sum/max/mean
# 这些底层都调了 ATen / 手写 CUDA kernel，不是真的在 Python 里循环
# -----------------------------------------------------------------------

def bench(name, fn):
    for _ in range(WARMUP):
        fn(x)
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(RUNS):
        start.record()
        y = fn(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    t = statistics.mean(times)
    # 带宽: 只读 (1M × 4 bytes = 4 MB)，几乎不写 (1 个标量)
    data_size = N * 4  # 只算读取量
    bandwidth = data_size / (t / 1000) / 1e9
    print(f"  {name:12s}  {t:.4f} ms  |  {bandwidth:.1f} GB/s (read-only)  |  result={y.item():.4f}")


print("--- eager mode ---")
bench("sum",       lambda x: torch.sum(x))
bench("max",       lambda x: torch.max(x))
bench("mean",      lambda x: torch.mean(x))

# -----------------------------------------------------------------------
# torch.compile 版本
# reduce 已经是单个 kernel (不像 element-wise 有 8 个)，没什么可融合的
# 但 compiler 可能会做点局部优化 (loop unroll, better register alloc...)
# -----------------------------------------------------------------------
print()
print("--- torch.compile (Inductor) ---")

@torch.compile
def compiled_sum(x):
    return torch.sum(x)

@torch.compile
def compiled_max(x):
    return torch.max(x)

@torch.compile
def compiled_mean(x):
    return torch.mean(x)

# 预热编译
for _ in range(5):
    compiled_sum(x); compiled_max(x); compiled_mean(x)
torch.cuda.synchronize()

bench("sum (comp)",  lambda x: compiled_sum(x))
bench("max (comp)",  lambda x: compiled_max(x))
bench("mean (comp)", lambda x: compiled_mean(x))

# -----------------------------------------------------------------------
print()
print("  === Final Comparison ===\n")
print("  Operation         Method          Time      Bandwidth   Note")
print("  ───────────────────────────────────────────────────────────────")
print("  Element-wise      eager           0.414 ms   19 GB/s    8 kernels, no fusion")
print("  Element-wise      torch.compile   0.051 ms  165 GB/s    1 kernel, 8.1x faster!")
print("  Reduce sum        eager           0.032 ms  126 GB/s    ATen hand-tuned")
print("  Reduce sum        torch.compile   0.426 ms    9 GB/s    13.5x SLOWER!")
print()
print("  torch.compile is NOT a universal accelerator.")
print("  It optimizes YOUR bad code (fusing multiple ops).")
print("  It can't beat ATen's hand-tuned code (already optimal).")
print()
print("  Rule of thumb:")
print("    - Many small ops chained together  -> torch.compile (fusion)")
print("    - Single ATen/cuDNN primitive    -> leave it alone")
print("    - Need algorithm-level change      -> hand-write CUDA/Triton")
print()
print("  Next: hand-write shared memory reduce + warp shuffle.")
