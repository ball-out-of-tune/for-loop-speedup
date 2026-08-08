"""
实验 3: 探测 L1 Cache 行为 (容量 + Sector Cache)

两个子实验:
  3A: Working Set Size Scan → 找到 L1 容量拐点
  3B: Dense vs Sparse Access → 验证 Sector Cache (128B line = 4×32B sector)

用法:
  python exp3_l1_eviction.py
  ncu --metrics l1tex__t_sectors_hit_rate python exp3_l1_eviction.py
"""

import numpy as np
from numba import cuda
import statistics

BLOCK_SIZE = 256
GRID_SIZE  = 20           # 20 SMs on GA107
WARMUP     = 5
RUNS       = 30

print("=" * 60)
print("  实验 3: L1 Eviction 行为探测")
print("=" * 60)
print(f"  Grid: {GRID_SIZE}, Block: {BLOCK_SIZE}")
print(f"  L1/SM (GA107): 128 KB, L2 (GA107): 2 MB")
print()

# ============================================================================
# 3A: Working Set Size Scan
# ============================================================================
@cuda.jit
def working_set_scan(x, sink, working_set_elems, iters):
    """每个 block 只访问自己的 working set (不跨 block 共享)"""
    tid = cuda.threadIdx.x
    block_offset = cuda.blockIdx.x * working_set_elems
    s = 0.0
    for _ in range(iters):
        for i in range(tid, working_set_elems, cuda.blockDim.x):
            s += x[block_offset + i]
    if tid == 0:
        sink[cuda.blockIdx.x] = s

print("=" * 60)
print("  3A: Working Set Size Scan (L1 Capacity Detection)")
print("=" * 60)
print()

# 不同 working set 大小
ws_kb_list = [4, 8, 16, 32, 64, 128, 192, 256, 384, 512]
iters_3a   = 64  # 多次迭代放大差异

# 预分配 (最大 working set × GRID_SIZE)
max_ws = 512 * 1024 // 4  # 512K floats per block
total_floats = GRID_SIZE * max_ws
huge_x = np.random.randn(total_floats).astype(np.float32)
d_huge_x = cuda.to_device(huge_x)
sink = np.zeros(GRID_SIZE, dtype=np.float32)
d_sink = cuda.to_device(sink)

grid  = (GRID_SIZE,)
block = (BLOCK_SIZE,)

def bench(fn, args_tuple):
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

print(f"  {'WS(KB)':<10} {'Time(ms)':<12} {'BW(GB/s)':<14} {'Note'}")
print(f"  {'-' * 48}")

results = []
for ws_kb in ws_kb_list:
    elems = ws_kb * 1024 // 4
    t = bench(working_set_scan, (d_huge_x, d_sink, elems, iters_3a))

    total_access = elems * GRID_SIZE * iters_3a * 4
    bw = total_access / (t / 1000) / 1e9

    if ws_kb <= 128:
        note = "≤ L1 (expect HITS)"
    elif ws_kb <= 256:
        note = "1-2× L1 (mild thrash)"
    else:
        note = "> 2× L1 (heavy thrash)"

    print(f"  {ws_kb:<10} {t:<12.4f} {bw:<14.1f} {note}")
    results.append((ws_kb, t, bw))

print()
print("  解读:")
print("    - When WS ≤ L1 (128 KB): 大部分访问命中 L1, BW 低")
print("    - When WS > L1: 频繁 L1 miss, BW 上升")
print("    - BW 突然上升的点 ≈ L1 容量")
print("    - 如果没看到明显拐点: L2 (2 MB) 足够大, 掩盖了 L1 效应")
print()

# ============================================================================
# 3B: Dense vs Sparse (Sector Cache)
# ============================================================================
print("=" * 60)
print("  3B: Dense vs Sparse Access (Sector Cache Detection)")
print("=" * 60)
print()

@cuda.jit
def dense_access(x, sink, n, iters):
    """连续访问, 每个 128B cache line 中 4 个 sector 全部使用"""
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.gridDim.x * cuda.blockDim.x
    s = 0.0
    for _ in range(iters):
        for i in range(tid, n, stride):
            s += x[i]
    if tid == 0:
        sink[cuda.blockIdx.x] = s

@cuda.jit
def sparse_access(x, sink, n_floats, iters):
    """
    跳跃访问: 每 32 floats (=128 bytes = 1 cache line) 只访问 1 个 float
    → 只用到了 cache line 的 1 个 sector (32 bytes), 浪费其他 3 个
    → 有效 L1 容量只有 1/4
    """
    tid = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    tt = cuda.gridDim.x * cuda.blockDim.x
    s = 0.0

    # 步长 = tt * 32 个 float = tt * 128 bytes
    # 保证每个 warp 的相邻两次访问间隔 >= 1 个 cache line
    # 且只 touch 一个 sector
    stride = tt * 32
    for _ in range(iters):
        for i in range(tid * 32, n_floats, stride):
            s += x[i]

    if tid == 0:
        sink[cuda.blockIdx.x] = s

# 使用不同的数组大小
test_sizes_kb = [32, 64, 128, 256, 512]
iters_3b = 32

print(f"  Both kernels access the same NUMBER of floats.")
print(f"  But sparse only uses 1/4 of each cache line (1 sector of 4).")
print()
print(f"  {'Data(KB)':<10} {'Dense(ms)':<12} {'Sparse(ms)':<12} "
      f"{'Ratio':<8} {'Interpretation'}")
print(f"  {'-' * 60}")

for size_kb in test_sizes_kb:
    n_floats = size_kb * 1024 // 4

    # dense: 数组大小 = 要访问的 float 数
    arr_dense = np.random.randn(n_floats).astype(np.float32)
    # sparse: 需要 n_floats * 32 个 float 的空间
    # (因为我们每 32 个 float 才访问 1 个)
    arr_sparse = np.random.randn(n_floats * 32).astype(np.float32)

    d_dense  = cuda.to_device(arr_dense)
    d_sparse = cuda.to_device(arr_sparse)

    t_dense  = bench(dense_access,  (d_dense,  d_sink, n_floats, iters_3b))
    t_sparse = bench(sparse_access, (d_sparse, d_sink, n_floats * 32, iters_3b))

    ratio = t_sparse / t_dense
    if ratio < 1.2:
        interp = "≈same (L2 absorption)"
    elif ratio < 2.0:
        interp = "sparse > dense"
    elif ratio < 4.0:
        interp = "strong sector effect"
    else:
        interp = "extreme sector effect"

    print(f"  {size_kb:<10} {t_dense:<12.4f} {t_sparse:<12.4f} "
          f"{ratio:<8.2f} {interp}")

print()
print("  解读:")
print("    Sector cache 理论上使 sparse 的有效 L1 容量变成 1/4.")
print("    如果 sparse/dense ratio 在较大数据时显著 > 1:")
print("      → sector cache 确认, 且 L1 无法同时缓存那么多 sparse cache lines")
print("    如果 ratio ≈ 1:")
print("      → L2 掩盖了差异 (L2 也是 sector cache, 但也够大)")
print("      → 需要 Nsight Compute 直接看 L1 hit rate")
print()

# ============================================================================
# Summary
# ============================================================================
print("=" * 60)
print("  How to Verify DEFINITIVELY with Nsight Compute")
print("=" * 60)
print()
print("  这些 timing 实验只能给你 '暗示', 因为在 3050 Ti 上 L2 (2 MB)")
print("  足以吸收很多 L1 miss。真正的证据来自硬件计数器:")
print()
print("  # 实验 3A 的 Nsight 验证:")
print("  ncu --metrics l1tex__t_sectors_hit_rate \\")
print("       python exp3_l1_eviction.py")
print()
print("  预期:")
print("    3A: hit rate >> 80% at WS < 128 KB, drops at WS > 128 KB")
print("    3B: sparse hit rate < dense hit rate (same data, more misses)")
print()
print("  Key metrics:")
print("    l1tex__t_sectors_hit_rate         — sector-level L1 hit rate")
print("    l1tex__t_sectors_lookup_hit       — total L1 sector hits")
print("    l1tex__t_sectors_lookup_miss      — total L1 sector misses")
print("    l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit_rate — load-specific hit rate")
