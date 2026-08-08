"""
实验 1: 验证 L1 Write-Through + Write-Invalidate

核心思想:
  Kernel B 和 Kernel C 都有完全相同的操作数 (2 reads + 1 write),
  唯一区别是:
    B: store 到 x (和 read 相同的地址) → 如果 write-invalidate, L1 被清掉
    C: store 到 y (和 read 不同的地址) → L1 中的 x 不受影响

  所以直接比较 t_B vs t_C 即可, 不需要减去 store 时间!

用法:
  python exp1_l1_write_invalidate.py              # 纯 timing
  python exp1_l1_write_invalidate.py --large       # 大数组
  ncu --metrics l1tex__t_sectors_hit_rate python exp1_l1_write_invalidate.py  # 硬件计数器
"""

import numpy as np
from numba import cuda
import statistics
import sys

# ============================================================================
# 参数
# ============================================================================
LARGE_MODE = "--large" in sys.argv

if LARGE_MODE:
    # 每个 block 256 KB, 刚好是 L1 的两倍 → 一定会有 eviction
    WS_PER_BLOCK = 256 * 1024 // 4  # 256K floats = 1 MB, but we use 256 KB
    WS_PER_BLOCK = 64 * 1024        # 64K floats = 256 KB per block
    ITERS = 32
else:
    # 默认: 每个 block 64 KB, 能放进 L1
    WS_PER_BLOCK = 16 * 1024        # 16K floats = 64 KB per block (fits in L1)
    ITERS = 64                       # 更多迭代来放大差异

BLOCK_SIZE = 256
GRID_SIZE  = 20           # 20 SMs on GA107
WARMUP     = 10
RUNS       = 100

print("=" * 60)
print("  实验 1: 验证 L1 Write-Through + Write-Invalidate")
print("=" * 60)
print(f"  每 block: {WS_PER_BLOCK} floats = {WS_PER_BLOCK * 4 / 1024:.0f} KB")
print(f"  L1/SM (GA107): 128 KB, fits in L1: {WS_PER_BLOCK * 4 <= 128 * 1024}")
print(f"  Grid: {GRID_SIZE}, Block: {BLOCK_SIZE}")
print(f"  Mode: {'LARGE (exceeds L1)' if LARGE_MODE else 'small (fits in L1)'}")
print()

# ============================================================================
# Kernels
# ============================================================================

@cuda.jit
def kernel_b_store_same(x, sink, n):
    """
    load → store(SAME addr) → load
    预测: 如果 write-invalidate 成立, 第三次 load 是 L1 miss
    """
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.gridDim.x * cuda.blockDim.x
    s1 = 0.0
    s2 = 0.0

    # Pass 1: read x → fills L1 with x
    for _ in range(ITERS):
        for i in range(tid, n, stride):
            s1 += x[i]

    # Pass 2: write to x → if write-invalidate, invalidates L1 for x!
    for i in range(tid, n, stride):
        x[i] = s1  # store to SAME array we just read

    # Pass 3: read x again → L1 miss if invalidated, L1 hit if not
    for _ in range(ITERS):
        for i in range(tid, n, stride):
            s2 += x[i]

    if tid == 0:
        sink[cuda.blockIdx.x] = s1 + s2

@cuda.jit
def kernel_c_store_other(x, y, sink, n):
    """
    load → store(DIFFERENT array) → load
    预测: store 到不同地址不 invalidate x 的 L1 → 第三次 load 是 L1 hit
    """
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.gridDim.x * cuda.blockDim.x
    s1 = 0.0
    s2 = 0.0

    # Pass 1: read x → fills L1 with x
    for _ in range(ITERS):
        for i in range(tid, n, stride):
            s1 += x[i]

    # Pass 2: write to y (DIFFERENT from x) → should NOT invalidate x in L1
    for i in range(tid, n, stride):
        y[i] = s1  # store to DIFFERENT array

    # Pass 3: read x again → should still be in L1 (if write-invalidate is precise)
    for _ in range(ITERS):
        for i in range(tid, n, stride):
            s2 += x[i]

    if tid == 0:
        sink[cuda.blockIdx.x] = s1 + s2

@cuda.jit
def kernel_a_read_twice(x, sink, n):
    """
    Control: read twice, no stores. Pass 2 should be all L1 hits.
    """
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.gridDim.x * cuda.blockDim.x
    s1 = 0.0
    s2 = 0.0

    for _ in range(ITERS):
        for i in range(tid, n, stride):
            s1 += x[i]

    # No stores! Just read again
    for _ in range(ITERS):
        for i in range(tid, n, stride):
            s2 += x[i]

    if tid == 0:
        sink[cuda.blockIdx.x] = s1 + s2

# ============================================================================
# Benchmark
# ============================================================================
def bench(fn, args_tuple, grid, block):
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

# ---- 分配 ----
x = np.random.randn(WS_PER_BLOCK * GRID_SIZE).astype(np.float32)
y = np.zeros(WS_PER_BLOCK * GRID_SIZE, dtype=np.float32)
sink = np.zeros(GRID_SIZE, dtype=np.float32)

d_x = cuda.to_device(x)
d_y = cuda.to_device(y)
d_sink = cuda.to_device(sink)

grid  = (GRID_SIZE,)
block = (BLOCK_SIZE,)

total_bytes_per_read = WS_PER_BLOCK * GRID_SIZE * 4  # bytes per full read

# ---- 运行 ----
t_a = bench(kernel_a_read_twice, (d_x, d_sink, WS_PER_BLOCK * GRID_SIZE), grid, block)
t_b = bench(kernel_b_store_same, (d_x, d_sink, WS_PER_BLOCK * GRID_SIZE), grid, block)
t_c = bench(kernel_c_store_other, (d_x, d_y, d_sink, WS_PER_BLOCK * GRID_SIZE), grid, block)

# ============================================================================
# Analysis
# ============================================================================
print(f"--- Results ---")
print(f"  Kernel A (read 2×, no store):          {t_a:.4f} ms")
print(f"  Kernel B (read → storeSAME → read):    {t_b:.4f} ms")
print(f"  Kernel C (read → storeOTHER → read):   {t_c:.4f} ms")
print()

# 关键: B 和 C 的操作数完全相同 (ITERS×read + 1×write + ITERS×read)
# 唯一的区别是 store 的目标地址是否和 read 重叠
#
# 如果 write-invalidate 成立:
#   B 的第三个 pass 的 ITERS 次读全部 miss L1 (因为被 store 清掉了)
#   C 的第三个 pass 的 ITERS 次读全部 hit L1 (L1 中的 x 还在)
#   → t_B > t_C (B 必须从 L2 重读, C 从 L1 读)
#
# 如果 write-invalidate 不成立:
#   B 和 C 的第三个 pass 都是 L1 hit
#   → t_B ≈ t_C

ratio_bc = t_b / t_c
ratio_ba = t_b / t_a
ratio_ca = t_c / t_a

print(f"  Key ratios:")
print(f"    B/C = {ratio_bc:.3f}  (如果 >1.0: B 的 store-to-same 伤到了 B 自己的重读)")
print(f"    B/A = {ratio_ba:.3f}  (B vs pure-read control)")
print(f"    C/A = {ratio_ca:.3f}  (C vs pure-read control)")
print()

# 解释
if ratio_bc > 1.05:
    # B > C > 5%: B 的重读比 C 的重读慢
    print("  ✓✓ CONSISTENT with L1 Write-Invalidate:")
    print(f"    B (store to same) is {((ratio_bc - 1) * 100):.1f}% slower than C (store to other).")
    print("    → Storing to the same address invalidated L1 for x,")
    print("      forcing B's 2nd read to go to L2 instead of L1.")
    print("    → C's 2nd read still hits L1 because store went elsewhere.")
elif ratio_bc > 1.02:
    print("  ✓ Weakly CONSISTENT with L1 Write-Invalidate (2-5% effect).")
    print("    Effect exists but small — L2 bandwidth is very close to L1.")
    print("    → Nsight Compute would show clearer L1 hit rate difference.")
else:
    print("  △ No clear timing difference between B and C.")
    print("    Possible explanations:")
    print("    1) L2 effectively masks L1 misses at this data size")
    print("    2) The compiler reordered instructions in unexpected ways")
    print("    3) Store didn't invalidate L1 (unlikely per NVIDIA docs)")
    print(f"    → Use Nsight Compute to check L1 hit rate directly.")

print()
print(f"  Nsight Compute verification:")
print(f"    ncu --metrics l1tex__t_sectors_hit_rate \\")
print(f"        python exp1_l1_write_invalidate.py")
print(f"  Expected: Kernel B L1 hit rate < Kernel C L1 hit rate")
print()
print(f"  Tip: use --large flag for data that exceeds L1 (more visible effect)")
