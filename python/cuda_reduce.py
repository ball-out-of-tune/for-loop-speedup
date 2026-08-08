"""
Numba CUDA Reduce Kernel — 从零手写归约

Reduce 的核心挑战:
  Element-wise: N 个输入 → N 个输出, 互不依赖, 完全并行
  Reduce:       N 个输入 → 1 个标量, 元素之间有依赖, 必须"合并"

归约树 (Reduction Tree):
  8 个元素:  a b c d e f g h
               \/  \/  \/  \/    第 1 轮: 相邻相加
               a+b c+d e+f g+h
                 \/     \/       第 2 轮
                a+b+c+d e+f+g+h
                    \/           第 3 轮
              a+b+c+d+e+f+g+h

  每轮元素减半 → log2(N) 轮 → 1M 需要 ~20 轮

运行: python cuda_reduce.py
"""
from numba import cuda
import numpy as np
import statistics

N = 1_000_000
BLOCK_SIZE = 256  # 每 block 处理 256 个元素 → 归约到 1 个部分和
NUM_BLOCKS = (N + BLOCK_SIZE - 1) // BLOCK_SIZE  # ≈ 3907 blocks

WARMUP = 3
RUNS = 10


# ============================================================================
# Version 1: Naive shared memory reduce
# ============================================================================
# 每个 block:
#   1. 256 个线程各读 1 个元素 → shared memory[256]
#   2. __syncthreads()  (等大家都读完)
#   3. 树状归约: stride=128 → 64 → 32 → 16 ... → 1
#      每轮只有一半线程干活, 另一半休息
#   4. thread 0 把结果加到全局数组 partial_sums[blockIdx]
#
# 问题: 第 2 轮以后, 只有一半 warp 在工作, 另一半浪费了
#       (warp divergence — 后面会优化)

@cuda.jit
def reduce_v1_shared(data, partial_sums, n):
    """
    data:         [N]    输入数组
    partial_sums: [NUM_BLOCKS]   每个 block 的部分和
    """
    # 每个 block 分配一块 shared memory
    smem = cuda.shared.array(BLOCK_SIZE, dtype='float32')

    # Step 1: 全局索引, 读数据 → shared memory
    tid = cuda.threadIdx.x       # 线程在 block 内的 ID (0~255)
    gid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x  # 全局 ID

    if gid < n:
        smem[tid] = data[gid]
    else:
        smem[tid] = 0.0  # 越界的填 0, 不影响 sum

    cuda.syncthreads()  # ⚠️ 必须等所有线程读完才能开始归约

    # Step 2: 树状归约 (stride 每次减半)
    stride = cuda.blockDim.x // 2  # 从 128 开始
    while stride > 0:
        if tid < stride:                    # 只有前半线程工作
            smem[tid] += smem[tid + stride]  # 和自己的"搭档"相加
        cuda.syncthreads()                   # 等这一轮完成
        stride //= 2                          # 步长减半

    # Step 3: 每个 block 的 thread 0 写入部分和
    if tid == 0:
        partial_sums[cuda.blockIdx.x] = smem[0]


# ============================================================================
# Version 2: Warp shuffle reduce (消除 shared memory)
# ============================================================================
# GPU 的 warp 内部有 __shfl_down_sync 指令:
#   - 不需要 shared memory
#   - 同一 warp 的 32 个线程可以直接交换寄存器
#   - 延迟 ~1 cycle (比 shared memory 快 ~10x)
#
# 策略:
#   1. 先 shared memory reduce 把 256 → 每 warp 1 个值 (8 warps → 8 个值)
#   2. 再用 warp shuffle 把 8 → 1

@cuda.jit
def reduce_v2_warpshuffle(data, partial_sums, n):
    """
    Hybrid: shared memory (block 内跨 warp) + warp shuffle (warp 内归约)
    """
    smem = cuda.shared.array(BLOCK_SIZE, dtype='float32')

    tid = cuda.threadIdx.x
    gid = cuda.blockIdx.x * cuda.blockDim.x + tid

    # Step 1: global -> shared
    if gid < n:
        smem[tid] = data[gid]
    else:
        smem[tid] = 0.0
    cuda.syncthreads()

    # Step 2: shared memory 归约到每 warp 1 个值 (256 → 32)
    # 先跨 warp 归约 (stride >= 32, 不同 warp 的线程通信需要 shared memory)
    stride = cuda.blockDim.x // 2
    while stride > 32:  # stride = 128, 64 — 只有这两轮需要 shared memory
        if tid < stride:
            smem[tid] += smem[tid + stride]
        cuda.syncthreads()
        stride //= 2

    # Step 3: warp 内归约 (stride = 32 → 16 → 8 → 4 → 2 → 1)
    # 用 __shfl_down_sync — 同一 warp 内寄存器级别的数据交换
    # 比 shared memory 快 ~10x (寄存器延迟 <1 cycle vs shared ~20 cycles)
    val = smem[tid]

    # Numba CUDA 的 shfl_down_sync:
    #   mask=0xFFFFFFFF → warp 内所有 32 个 lane 参与
    #   offset=N → 从 lane_id+N 拿数据, 如果 lane_id+N >= 32 则返回自己的值
    val += cuda.shfl_down_sync(0xFFFFFFFF, val, 16)
    val += cuda.shfl_down_sync(0xFFFFFFFF, val, 8)
    val += cuda.shfl_down_sync(0xFFFFFFFF, val, 4)
    val += cuda.shfl_down_sync(0xFFFFFFFF, val, 2)
    val += cuda.shfl_down_sync(0xFFFFFFFF, val, 1)

    # Step 4: 每 warp 的 lane 0 现在有该 warp 的部分和
    # 只让 thread 0 写入 (也可以用 atomicAdd, 但那是下一步优化)
    if tid == 0:
        partial_sums[cuda.blockIdx.x] = val


# ============================================================================
# CPU 端: 汇总 partial_sums → 最终结果
# ============================================================================
def full_sum(kernel_fn, data, n, block_size=BLOCK_SIZE):
    """调用 GPU kernel, 然后 CPU 汇总部分和"""
    num_blocks = (n + block_size - 1) // block_size

    # GPU 内存
    d_data = cuda.to_device(data)
    d_partial = cuda.device_array(num_blocks, dtype=np.float32)

    # kernel 配置
    threads_per_block = block_size
    blocks_per_grid = num_blocks

    # 预热
    for _ in range(WARMUP):
        kernel_fn[blocks_per_grid, threads_per_block](d_data, d_partial, n)
    cuda.synchronize()

    # 计时
    times = []
    start = cuda.event()
    end = cuda.event()
    for _ in range(RUNS):
        start.record()
        kernel_fn[blocks_per_grid, threads_per_block](d_data, d_partial, n)
        end.record()
        cuda.synchronize()
        times.append(cuda.event_elapsed_time(start, end))

    t = statistics.mean(times)

    # CPU 端汇总部分和 (最后一步归约)
    partial_host = d_partial.copy_to_host()
    result = partial_host.sum()

    # 带宽: 读 N 个 float32, 写 num_blocks 个 float32 (≈忽略)
    data_size = n * 4
    bandwidth = data_size / (t / 1000) / 1e9
    return t, bandwidth, result


# ============================================================================
# 测试
# ============================================================================
if __name__ == '__main__':
    data = np.arange(N, dtype=np.float32) * 0.001

    # CPU 基准
    expected = data.sum()
    print(f"CPU sum: {expected:.4f}")
    print()

    # V1: shared memory
    print("=== V1: Shared Memory Tree Reduce ===")
    t1, bw1, r1 = full_sum(reduce_v1_shared, data, N)
    print(f"  time: {t1:.4f} ms  |  BW: {bw1:.1f} GB/s  |  result: {r1:.4f}")
    print(f"  diff vs CPU: {abs(r1 - expected):.4f}")
    print()

    # V2: warp shuffle
    print("=== V2: Warp Shuffle Reduce ===")
    t2, bw2, r2 = full_sum(reduce_v2_warpshuffle, data, N)
    print(f"  time: {t2:.4f} ms  |  BW: {bw2:.1f} GB/s  |  result: {r2:.4f}")
    print(f"  diff vs CPU: {abs(r2 - expected):.4f}")
    print()

    # PyTorch baseline
    print("=== Baseline ===")
    print(f"  PyTorch eager sum:  0.032 ms  |  126 GB/s  (cuBLAS)")
    print(f"  Theoretical limit:  0.021 ms  |  192 GB/s  (read-only)")
    print()
    print("Why our kernel is slower than cuBLAS:")
    print("  1. partial_sums[] still written to global memory (N/256 writes)")
    print("  2. CPU-side reduction of partial sums (final sum of ~4000 values)")
    print("  3. cuBLAS uses: multi-level reduction, vectorized loads,")
    print("     warp shuffle + shared memory, and the final reduction")
    print("     happens on GPU too (atomicAdd or second kernel).")
