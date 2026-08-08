"""
实验 2: 验证 L1 在 Kernel 切换时被 Flush

待验证的命题:
  "L1 会在 kernel 切换 (context switch) 时自动 flush"

核心实验设计:

  场景 A (within-kernel):  一个 kernel 里连续 load 同一个数组 2 次
    → 第2次: L1 HIT

  场景 B (cross-kernel):  两个 kernel 各 load 一次
    → 如果 L1 flush: 第2个 kernel 的读是 cold (L1 miss)
    → 如果 L1 保留: 第2个 kernel 的读是 warm (L1 hit)

  场景 C (cross-kernel + 污染):  中间跑一个访问大数组的 kernel
    → L1 被污染 → 应该确定 cold (阳性对照)

用法:
  python exp2_l1_flush.py
  ncu --metrics l1tex__t_sectors_hit_rate python exp2_l1_flush.py
"""

import numpy as np
from numba import cuda
import statistics

ARRAY_SIZE = 16 * 1024    # 16K floats = 64 KB
BLOCK_SIZE = 256
GRID_SIZE  = 20
WARMUP     = 5
RUNS       = 50

print("=" * 60)
print("  实验 2: 验证 L1 在 Kernel 切换时被 Flush")
print("=" * 60)
print(f"  数组大小: {ARRAY_SIZE} floats = {ARRAY_SIZE * 4 / 1024:.0f} KB")
print(f"  Grid: {GRID_SIZE}, Block: {BLOCK_SIZE}")
print()

# ============================================================================
# CUDA Kernels
# ============================================================================

@cuda.jit
def load_once(x, sink, n):
    """纯读一次 (cold miss baseline)"""
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.gridDim.x * cuda.blockDim.x
    s = cuda.local.array(1, dtype=np.float32)
    s[0] = 0.0
    for i in range(tid, n, stride):
        s[0] += x[i]
    if tid == 0:
        sink[cuda.blockIdx.x] = s[0]

@cuda.jit
def load_twice(x, sink, n):
    """同一个 kernel 内读两次 (第2次应该是 L1 hit)"""
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.gridDim.x * cuda.blockDim.x
    s = cuda.local.array(1, dtype=np.float32)
    s[0] = 0.0
    for i in range(tid, n, stride):
        s[0] += x[i]   # cold → fills L1
    for i in range(tid, n, stride):
        s[0] += x[i]   # warm → L1 HIT
    if tid == 0:
        sink[cuda.blockIdx.x] = s[0]

@cuda.jit
def empty_kernel(sink):
    """什么都不做, 只消耗一个 kernel launch"""
    if cuda.threadIdx.x == 0 and cuda.blockIdx.x == 0:
        sink[0] = 0.0

@cuda.jit
def pollute_l1(junk, n):
    """读一个大数组, 用垃圾数据填满 L1"""
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.gridDim.x * cuda.blockDim.x
    s = cuda.local.array(1, dtype=np.float32)
    s[0] = 0.0
    for i in range(tid, n, stride):
        s[0] += junk[i]
    if tid == 0 and cuda.blockIdx.x == 0:
        junk[0] = s[0]  # prevent optimize-away

# ============================================================================
# Benchmark helper
# ============================================================================
def bench(fn, args_tuple, grid, block, run_label=""):
    for _ in range(WARMUP):
        fn[grid, block](*args_tuple)
    cuda.synchronize()

    times = []
    for _ in range(RUNS):
        start = cuda.event()
        end = cuda.event()
        start.record()
        fn[grid, block](*args_tuple)
        end.record()
        cuda.synchronize()
        times.append(cuda.event_elapsed_time(start, end))
    return statistics.median(times)

# ---- 分配内存 ----
x = np.random.randn(ARRAY_SIZE).astype(np.float32)
sink = np.zeros(GRID_SIZE, dtype=np.float32)
junk = np.random.randn(256 * 1024).astype(np.float32)  # 1 MB > L1

d_x    = cuda.to_device(x)
d_sink = cuda.to_device(sink)
d_junk = cuda.to_device(junk)

grid  = (GRID_SIZE,)
block = (BLOCK_SIZE,)

# ========================================================================
# Measurement 1: 单次 cold read
# ========================================================================
t_cold = bench(load_once, (d_x, d_sink, ARRAY_SIZE), grid, block)
total_bytes = ARRAY_SIZE * 4
bw_cold = total_bytes / (t_cold / 1000) / 1e9

print(f"--- Baseline: 单次 cold read ---")
print(f"  Time: {t_cold:.4f} ms  |  BW: {bw_cold:.1f} GB/s")
print()

# ========================================================================
# Measurement 2: 同一个 kernel 内读两次 (within-kernel)
# ========================================================================
t_within = bench(load_twice, (d_x, d_sink, ARRAY_SIZE), grid, block)
print(f"--- 场景 A: 同一 kernel 内 load 两次 ---")
print(f"  Time: {t_within:.4f} ms")
print(f"  vs 1×cold: {t_within / t_cold:.2f}×  (预期 < 2.0, 因为第2次 L1 hit)")
print(f"  L1 hit benefit: {max(0, (2*t_cold - t_within) / t_cold * 100):.0f}%")
print()

# ========================================================================
# Measurement 3: 两个 kernel 各读一次 (cross-kernel)
# ========================================================================
# 用同一个 default stream 保证串行
# 关键是: Kernel 1 fills L1, Kernel 2 读同一数组
times_cross = []
for _ in range(WARMUP):
    load_once[grid, block](d_x, d_sink, ARRAY_SIZE)  # K1
    load_once[grid, block](d_x, d_sink, ARRAY_SIZE)  # K2
cuda.synchronize()

for _ in range(RUNS):
    start = cuda.event()
    end = cuda.event()
    start.record()
    load_once[grid, block](d_x, d_sink, ARRAY_SIZE)  # K1: fills L1
    load_once[grid, block](d_x, d_sink, ARRAY_SIZE)  # K2: L1 hit or miss?
    end.record()
    cuda.synchronize()
    times_cross.append(cuda.event_elapsed_time(start, end))
t_cross = statistics.median(times_cross)

print(f"--- 场景 B: 两个 kernel 各 load 一次 (K1 → K2) ---")
print(f"  Time: {t_cross:.4f} ms  (= K1 + K2, 两次 kernel launch)")
print(f"  vs 2×cold: {t_cross / (2 * t_cold):.2f}×")
print(f"  预期: 如果 L1 flush  → ≈ 2×cold")
print(f"        如果 L1 persist → < 2×cold (接近场景A)")
print()

# ========================================================================
# Measurement 4: K1 → 空kernel → K3
# ========================================================================
times_empty = []
for _ in range(WARMUP):
    load_once[grid, block](d_x, d_sink, ARRAY_SIZE)
    empty_kernel[(1,), (1,)](d_sink)
    load_once[grid, block](d_x, d_sink, ARRAY_SIZE)
cuda.synchronize()

for _ in range(RUNS):
    start = cuda.event()
    end = cuda.event()
    start.record()
    load_once[grid, block](d_x, d_sink, ARRAY_SIZE)  # K1: fills L1
    empty_kernel[(1,), (1,)](d_sink)                   # K2: empty (does it flush?)
    load_once[grid, block](d_x, d_sink, ARRAY_SIZE)  # K3: check
    end.record()
    cuda.synchronize()
    times_empty.append(cuda.event_elapsed_time(start, end))
t_cross_empty = statistics.median(times_empty)

print(f"--- 场景 C: K1 → 空kernel → K3 ---")
print(f"  Time: {t_cross_empty:.4f} ms")
print(f"  vs 2×cold: {t_cross_empty / (2 * t_cold):.2f}×")
print(f"  如果空 kernel 也 flush L1 → 应该 ≈ 场景B")
print()

# ========================================================================
# Measurement 5: K1 → 污染L1 → K3 (阳性对照)
# ========================================================================
times_pollute = []
for _ in range(WARMUP):
    load_once[grid, block](d_x, d_sink, ARRAY_SIZE)
    pollute_l1[grid, block](d_junk, 256 * 1024)
    load_once[grid, block](d_x, d_sink, ARRAY_SIZE)
cuda.synchronize()

for _ in range(RUNS):
    start = cuda.event()
    end = cuda.event()
    start.record()
    load_once[grid, block](d_x, d_sink, ARRAY_SIZE)    # K1: fills L1 with x
    pollute_l1[grid, block](d_junk, 256 * 1024)         # K2: fills L1 with junk
    load_once[grid, block](d_x, d_sink, ARRAY_SIZE)    # K3: L1 miss (confirmed)
    end.record()
    cuda.synchronize()
    times_pollute.append(cuda.event_elapsed_time(start, end))
t_cross_pollute = statistics.median(times_pollute)

print(f"--- 场景 D (阳性对照): K1 → 污染L1 → K3 ---")
print(f"  Time: {t_cross_pollute:.4f} ms")
print(f"  vs 2×cold: {t_cross_pollute / (2 * t_cold):.2f}×  (预期 ≈ 2, 确定 cold)")
print()

# ========================================================================
# Analysis
# ========================================================================
print("=" * 60)
print("  Analysis")
print("=" * 60)
print()

# Theoretical predictions
pred_2cold = 2 * t_cold
pred_within = t_within

# 每个 cross-kernel 场景中, 第二次 kernel 的读的 "等效 cold 次数"
# = (total_time - 1×cold - launch_overhead) / t_cold
# launch_overhead 约等于 t_empty (单独测一次空 kernel launch)
# 为简化, 我们直接比较 total_time

print(f"  参考线:")
print(f"    1×cold                 = {t_cold:.4f} ms")
print(f"    2×cold (all misses)    = {pred_2cold:.4f} ms")
print(f"    within-kernel (L1 hit) = {t_within:.4f} ms")
print()
print(f"  实测:")
print(f"    Cross-kernel K1→K2:          {t_cross:.4f} ms  "
      f"({'FLUSHED' if t_cross > pred_2cold * 0.85 else 'PERSISTENT' if t_cross < pred_within * 1.2 else 'INTERMEDIATE'})")
print(f"    Cross+empty K1→empty→K3:     {t_cross_empty:.4f} ms  "
      f"({t_cross_empty / pred_2cold:.2f} × 2cold)")
print(f"    Cross+pollute K1→junk→K3:    {t_cross_pollute:.4f} ms  "
      f"({t_cross_pollute / pred_2cold:.2f} × 2cold)")
print()

# 关键对比: cross-kernel vs 阳性对照
if t_cross > pred_2cold * 0.82:
    print("  ✓ L1 appears FLUSHED between regular kernel launches.")
    print("    Even without explicit pollution, the 2nd kernel's loads")
    print("    are cold misses. Consistent with 'L1 flush on kernel switch'.")
elif t_cross < pred_within * 1.15:
    print("  ✓ L1 appears PERSISTENT across kernel launches.")
    print("    2nd kernel can reuse data from 1st kernel's L1.")
    print("    This contradicts the 'flush on kernel switch' claim.")
    print("    (Might only happen if both kernels ran on same SM.)")
else:
    print("  △ Partial evidence — check the details below.")
    print()

# Compare with pollute (阳性对照)
# pollute 把 L1 清空, 所以 K3 的读确定是 cold
# 如果 t_cross < t_cross_pollute: 说明 K2 (空 kernel) 的读比确定 cold 快
# → L1 可能在 kernel 间部分保留
if t_cross < t_cross_pollute * 0.85:
    print(f"  △ Cross-kernel ({t_cross:.3f}) < Cross+pollute ({t_cross_pollute:.3f}):")
    print(f"    K1→K2 without pollution is FASTER than K1→junk→K3.")
    print(f"    This suggests L1 data may partially survive across kernels.")
    print(f"    OR: the pollution kernel itself has extra overhead.")
elif abs(t_cross - t_cross_pollute) / t_cross < 0.1:
    print(f"  △ Cross-kernel ≈ Cross+pollute (within 10%).")
    print(f"    Both scenarios show similar K3 performance.")
    print(f"    Consistent with L1 flushed (or replaced) in both cases.")

print()
print("  Definitive verification:")
print("    ncu --metrics l1tex__t_sectors_hit_rate python exp2_l1_flush.py")
print("  Then check: does K2's first load of x[] show L1 hit or miss?")
print("  A miss = L1 was flushed between kernel launches.")
