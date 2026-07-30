"""
============================================================
  速度阶梯：比 NumPy 更快的方法
============================================================
从最慢到最快，同样一个计算任务，看看每个层级能快多少。
"""

import math
import time
import numpy as np
from numba import njit, prange

# -----------------------------------------------------------
# 准备数据：50万个点（用 NumPy 生成，避免 Python 列表内存爆炸）
# -----------------------------------------------------------
N = 500_000
data_np = np.arange(N, dtype=np.float64) * 0.001
data_py = data_np.tolist()  # 转成 Python 列表供纯 Python 测试用


# ===========================================================
#  方法 1：纯 Python for 循环
# ===========================================================
def level1_python(data):
    """基线：热循环用纯 Python 写"""
    ans = [0.0] * len(data)
    for i, x in enumerate(data):
        ans[i] = math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x))
    return ans


# ===========================================================
#  方法 2：NumPy 向量化
# ===========================================================
def level2_numpy(data):
    """NumPy：循环推到 C 层，CPU SIMD 指令加速"""
    x = np.asarray(data)
    return np.sin(x) ** 2 + np.cos(x) ** 2 + np.sqrt(np.abs(x))


# ===========================================================
#  方法 3：Numba JIT（即时编译为机器码）
# ===========================================================
@njit
def level3_numba(data):
    """
    Numba：代码看起来是 Python for 循环，
    但 @njit 把它实时编译成 LLVM IR → 机器码。
    没了 Python 解释器开销，接近 C 速度。
    """
    n = len(data)
    ans = np.empty(n)
    for i in range(n):
        x = data[i]
        ans[i] = math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x))
    return ans


# ===========================================================
#  方法 4：Numba + 并行（多线程，利用多核 CPU）
# ===========================================================
@njit(parallel=True)
def level4_numba_parallel(data):
    """
    Numba 并行：在 level3 的基础上，
    用 prange 把循环拆成多份，多个 CPU 核心同时算。
    配合 parallel=True，自动分到所有核心。
    """
    n = len(data)
    ans = np.empty(n)
    for i in prange(n):         # prange = parallel range
        x = data[i]
        ans[i] = math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x))
    return ans


# ===========================================================
#  基准测试
# ===========================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  同一个计算：sin^2 + cos^2 + sqrt(abs(x)) x 50万个点")
    print("=" * 60)
    print()

    results = {}

    # --- 方法 1：纯 Python ---
    print("[1/4] 纯 Python for 循环 ...")
    t0 = time.perf_counter()
    level1_python(data_py)
    results['1. 纯 Python for'] = time.perf_counter() - t0

    # --- 方法 2：NumPy ---
    print("[2/4] NumPy 向量化 ...")
    # 跑两次：第一次可能有冷启动，取第二次
    _ = level2_numpy(data_py)
    t0 = time.perf_counter()
    level2_numpy(data_py)
    results['2. NumPy 向量化'] = time.perf_counter() - t0

    # --- 方法 3：Numba ---
    print("[3/4] Numba @njit ...")
    # 第一次调用包含编译时间，先预热
    _ = level3_numba(data_np)
    t0 = time.perf_counter()
    level3_numba(data_np)
    results['3. Numba @njit'] = time.perf_counter() - t0

    # --- 方法 4：Numba 并行 ---
    print("[4/4] Numba @njit(parallel=True) ...")
    _ = level4_numba_parallel(data_np)
    t0 = time.perf_counter()
    level4_numba_parallel(data_np)
    results['4. Numba 并行'] = time.perf_counter() - t0

    # --- 结果 ---
    baseline = results['1. 纯 Python for']
    print()
    print("=" * 62)
    print(f"{'方法':<28} {'耗时':>9}  {'加速比':>8}  {"备注":>10}")
    print("-" * 62)

    notes = {
        '1. 纯 Python for':   "基线",
        '2. NumPy 向量化':    "C 循环 + SIMD",
        '3. Numba @njit':     "编译成机器码",
        '4. Numba 并行':      "多核同时跑",
    }

    for name, elapsed in results.items():
        ratio = baseline / elapsed
        note = notes.get(name, "")
        print(f"{name:<28} {elapsed:8.4f}s  {ratio:6.1f}x   {note}")

    print("=" * 62)
    print()

    # --- 再往上：不写 Python 代码的方案 ---
    print("=" * 60)
    print("  方法 5~8：连 Python 的'调度层'都不要了")
    print("=" * 60)
    print("""
  速度层级一览（同一计算，相对耗时）：

  Python for 循环      ████████████████████████████████████  1x     (基线)
  NumPy 向量化          ████████                                ~0.2x  (C + SIMD)
  Numba @njit            ███████                                ~0.15x (编译成机器码)
  Numba parallel          ████                                  ~0.08x (多核并行)

  ─────────── Python 地盘的边界 ───────────

  CuPy (GPU)             ███                                    ~0.05x (几千个核同时算)
  C/C++ 原生             ██                                     ~0.03x (零抽象开销)
  Rust (PyO3)            ██                                     ~0.03x (和 C 同级)
  hand-tuned CUDA        █                                      <0.01x (手写 GPU 汇编级)

  一句话指南：
  ┌─────────────────────────────────────────────────────────┐
  │ 热循环加速路线图：                                      │
  │                                                        │
  │ Python for → NumPy 向量化 → Numba @njit → Numba并行    │
  │                                  ↓                      │
  │                    还不够？离开 Python！                │
  │                                  ↓                      │
  │              C/C++/Rust/CUDA 写核心，Python 调          │
  │                                                        │
  │ 95% 的情况 NumPy 或 Numba 已经够了                      │
  │ 那 5% 是：游戏引擎、高频交易、实时音视频处理            │
  └─────────────────────────────────────────────────────────┘
""")
