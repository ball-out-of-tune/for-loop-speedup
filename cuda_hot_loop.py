"""
CUDA 加速 hot_loop — 从入门到写出自己的内核
=============================================

GPU 的思维模型：把 100 万个元素分给几千个小工人（线程）同时算。

CPU:  工人1 → [元素0] [元素1] [元素2] ...       (串行，一个接一个)
GPU:  工人1 → [元素0]
      工人2 → [元素1]
      工人3 → [元素2]
      ...                                       (并行，全部同时)
      工人2048 → [元素2047]

CUDA 编程只有一个核心概念：每个线程算一下"我是谁"，然后去处理对应的元素。

    idx = threadIdx.x + blockIdx.x * blockDim.x
    #   ↑ 我在当前块里的位置   ↑ 我在哪个块   ↑ 每块有多少线程
    #
    #   翻译成人话：我在第 idx 号，所以我去处理 data[idx]

"""

import math
import time
import numpy as np
import statistics

# ============================================================
#  准备工作：统一数据
# ============================================================
N = 1_000_000
data_np = np.arange(N, dtype=np.float64) * 0.001  # NumPy 数组（GPU 需要）


# ============================================================
#  方法 0：纯 Python 循环（基线，跑在 CPU 上）
# ============================================================
def hot_loop_python(data):
    ans = []
    for x in data:
        ans.append(math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x)))
    return ans


# ============================================================
#  方法 1：CuPy — 最简单！改一行 import 就 GPU 了
#  pip install cupy-cuda12x
# ============================================================
def bench_cupy():
    """CuPy：API 和 NumPy 几乎一样，但数据在 GPU 上算"""
    import cupy as cp

    d = cp.asarray(data_np)  # 把数据搬到 GPU 显存
    #    ↓ 这行和 NumPy 完全一样！
    result = cp.sin(d) ** 2 + cp.cos(d) ** 2 + cp.sqrt(cp.abs(d))
    cp.cuda.Stream.null.synchronize()  # 等 GPU 完成
    return result


# ============================================================
#  方法 2：Numba CUDA — 自己写内核，但继续用 Python 语法
#  pip install numba  (已经在用了)
# ============================================================
from numba import cuda

@cuda.jit
def cuda_kernel(data, ans, n):
    """
    这就是 CUDA 内核。GPU 会启动 n 个线程，每个线程执行这个函数一次。

    cuda.grid(1) 返回"我是第几个线程"，和 Python 的 `for i in range(n)` 里的 i 完全一样。
    唯一的区别：这里所有线程同时执行，而不是依次执行。
    """
    i = cuda.grid(1)        # ← 唯一的新概念！等于 for 循环里的那个 i
    if i < n:               #   边界检查（线程数可能超过 n）
        x = data[i]
        ans[i] = math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x))
        #          ↑ 注意：内核里只能用 math.sin，不能用 np.sin

@cuda.jit
def numba_cuda_hot_loop_my_own(data, ans, N):
    i = cuda.grid(1)
    if i < N:
        x = data[i]
        ans[i] = math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x))
    

def bench_numba_cuda():
    """用 Numba 的 @cuda.jit 写 CUDA 内核"""
    d_data = cuda.to_device(data_np)          # 数据搬到 GPU
    d_ans = cuda.device_array(N, dtype=np.float64)  # 在 GPU 上分配输出数组

    # 配置线程：每个块 256 个线程，需要 N/256 = 3907 个块（向上取整）
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block

    cuda_kernel[blocks_per_grid, threads_per_block](d_data, d_ans, N)
    cuda.synchronize()

    return d_ans.copy_to_host()  # 结果搬回 CPU 内存


# ============================================================
#  方法 3：纯 CUDA C++ — 需要 nvcc 编译器，最难但最灵活
#
#  kernel.cu 文件（已写好在同目录下），通过 Python 的 subprocess 编译 + 加载
#  实际项目中用 pybind11 或 ctypes 会更方便
# ============================================================


# ============================================================
#  基准测试
# ============================================================
if __name__ == '__main__':
    print("=" * 55)
    print("  GPU 加速对比 (1,000,000 float64)")
    print("=" * 55)
    print()

    data_py = data_np.tolist()

    # --- 纯 Python ---
    t0 = time.perf_counter()
    hot_loop_python(data_py)
    baseline = time.perf_counter() - t0
    print(f"pure Python:            {baseline:.6f} s  (baseline)")

    # --- CuPy (全流程：含数据传输) ---
    print("\n--- CuPy ---")
    try:
        bench_cupy()  # 预热
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            bench_cupy()       # CPU→GPU + 计算 + GPU→CPU 全包
            times.append(time.perf_counter() - t0)
        print(f"  end-to-end (含传输):  {statistics.mean(times):.6f} s")
    except ImportError:
        print("  CuPy not installed. Run: pip install cupy-cuda12x")

    # --- Numba CUDA (拆开看每一段) ---
    print("\n--- Numba CUDA (拆开看) ---")

    # 预热
    d_data = cuda.to_device(data_np)
    d_ans = cuda.device_array(N, dtype=np.float64)
    threads = 256
    blocks = (N + threads - 1) // threads
    cuda_kernel[blocks, threads](d_data, d_ans, N)
    cuda.synchronize()

    # 1. 纯数据传输：CPU → GPU
    times_h2d = []
    for _ in range(10):
        t0 = time.perf_counter()
        d = cuda.to_device(data_np)
        cuda.synchronize()
        times_h2d.append(time.perf_counter() - t0)

    # 2. 纯 kernel 计算（数据已就位）
    times_kernel = []
    for _ in range(10):
        t0 = time.perf_counter()
        cuda_kernel[blocks, threads](d_data, d_ans, N)
        cuda.synchronize()
        times_kernel.append(time.perf_counter() - t0)

    # 3. 纯数据传输：GPU → CPU
    times_d2h = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = d_ans.copy_to_host()
        times_d2h.append(time.perf_counter() - t0)

    # 4. 端到端（全包）
    times_e2e = []
    for _ in range(10):
        t0 = time.perf_counter()
        d = cuda.to_device(data_np)
        a = cuda.device_array(N, dtype=np.float64)
        cuda_kernel[blocks, threads](d, a, N)
        cuda.synchronize()
        _ = a.copy_to_host()
        times_e2e.append(time.perf_counter() - t0)

    print(f"  CPU→GPU 传输:         {statistics.mean(times_h2d) * 1000:.3f} ms")
    print(f"  kernel 纯计算:        {statistics.mean(times_kernel) * 1000:.3f} ms")
    print(f"  GPU→CPU 传输:         {statistics.mean(times_d2h) * 1000:.3f} ms")
    print(f"  ─────────────────────────────")
    print(f"  end-to-end (含传输):  {statistics.mean(times_e2e) * 1000:.3f} ms")
    print(f"  加速比 (vs Python):   {baseline / statistics.mean(times_e2e):.0f}x")

    # ============================================================
    #  总结
    # ============================================================
    print(f"""
    ╔══════════════════════════════════════════════════════════╗
    ║  关键认知                                               ║
    ╠══════════════════════════════════════════════════════════╣
    ║  GPU 计算本身极快（~1ms），但来回路费（数据传输）        ║
    ║  占了总时间的 80%。                                      ║
    ║                                                        ║
    ║  → 数据已经在 GPU 上时：快 300x+                        ║
    ║  → 每次都要搬数据：    快 50x 左右                      ║
    ║                                                        ║
    ║  所以 GPU 适合：数据一次搬上去，反复算很多次              ║
    ║  不适合：    算一次就扔，下次换一批新数据                ║
    ╚══════════════════════════════════════════════════════════╝
    """)
