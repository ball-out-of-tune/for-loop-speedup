"""
============================================================
  Numba 深度解析：JIT 是什么？为什么能快 141 倍？
============================================================
"""

import math
import time
import numpy as np
from numba import njit, prange

N = 300_000
data = np.arange(N, dtype=np.float64) * 0.001


# ============================================================
# 第 1 部分：JIT 到底是什么？
# ============================================================
print("=" * 65)
print("  第 1 部分：JIT（即时编译）到底做了什么？")
print("=" * 65)

print("""
先理解"正常 Python 怎么执行代码"：

  你写的 .py 文件
      |
      v
  ① Python 编译器 -> 字节码 (.pyc)  ← 这是"中间语言"，不是机器码
      |
      v
  ② Python 解释器 -> 逐条"解释"字节码
      |              每执行一条字节码，都要：
      |              - 查这是什么操作
      |              - 查操作数是什么类型
      |              - 找对应的 C 函数
      |              - 调用、创建结果对象、管理引用计数
      |              - 取下一条字节码... 重复
      |
      v
  程序运行（慢，因为中间多了一层"翻译官"）

对比：C 语言没有"翻译官"，编译器直接把代码翻译成 CPU 认得的机器码。
""")

print("-- 实验 1：第一次调用 vs 第二次调用 --")
print()


@njit
def numba_version(data):
    n = len(data)
    ans = np.empty(n)
    for i in range(n):
        x = data[i]
        ans[i] = math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x))
    return ans


print("第一次调用（含编译时间）：")
t0 = time.perf_counter()
numba_version(data)
first = time.perf_counter() - t0
print(f"  耗时: {first:.4f} 秒")
print(f"  这期间 Numba 做了：")
print(f"    ① 分析你的 Python 函数 -> 理解类型和循环结构")
print(f"    ② 翻译成 LLVM IR（一种中间表示）")
print(f"    ③ LLVM 优化并生成目标机器的原生机器码")
print(f"    ④ 把机器码缓存起来，下次直接用")

print()
print("第二次调用（直接用编译好的机器码）：")
t0 = time.perf_counter()
numba_version(data)
second = time.perf_counter() - t0
print(f"  耗时: {second:.4f} 秒")
print(f"  这次没有编译！直接跳进机器码执行，跟 C 程序一个速度。")
print(f"  第一次/第二次 = {first / second:.1f} 倍（编译开销 vs 纯执行）")
print()

print("""
一句话理解 JIT：
+----------------------------------------------------------+
|                                                          |
|  Python 正常执行：代码 -> 字节码 -> 解释器逐条翻译 -> 慢    |
|                                                          |
|  Numba JIT：    代码 -> 字节码 -> [JIT编译成机器码] -> 快   |
|                                  ↑                       |
|                          编译一次，以后直接跑             |
|                                                          |
|  就像你去外国----                                          |
|    普通 Python = 每说一句，翻译官现场翻一句              |
|    Numba JIT   = 提前把你的整篇演讲稿翻译成当地语言，     |
|                  上台直接念，不需要翻译官了               |
+----------------------------------------------------------+
""")


# ============================================================
# 第 2 部分：JIT 编译生成了什么？（用 inspect_types 查看）
# ============================================================
print("=" * 65)
print("  第 2 部分：Numba 对你的代码做了什么优化？")
print("=" * 65)

print("""
Numba 不只是"翻译"，还会做激进优化。以你的循环为例：

你写的代码：
    for i in range(n):
        x = data[i]
        ans[i] = math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x))

Numba 优化后（概念上）：
    - 知道 data 和 ans 都是 float64 数组 -> 省略所有类型检查
    - math.sin/cos/sqrt 直接调用 C 的 sin()/cos()/sqrt() -> 零开销 FFI
    - x >= 0 永远成立（因为 abs），所以 abs(x) 优化掉 -> 少算一步
    - sin^2 + cos^2 = 1（三角恒等式！）-> 如果有 fastmath 会直接优化成 1
    - 循环展开 (loop unrolling)：一次迭代处理 4 个元素
    - SIMD 向量化：一条 CPU 指令同时做 4 个 float64 的运算

对比 Python 逐元素循环：
+---------------------------------------------------------+
|                                                         |
| Python 每做一次 math.sin(x)：                            |
|   查 math 模块 -> 查 .sin 属性 -> 类型检查 -> 创建 float   |
|   -> 调用 C sin -> 返回 PyObject -> 引用计数+1             |
|   -> ~200 纳秒（其中 90% 是 Python 开销）                |
|                                                         |
| Numba 每做一次 math.sin(x)：                             |
|   取 x 的值 -> 调 C sin() -> 存结果                       |
|   -> ~5 纳秒（全是实际计算，零 Python 开销）             |
|                                                         |
+---------------------------------------------------------+
""")


# ============================================================
# 第 3 部分：JIT 的代价 -- 什么不能加速？
# ============================================================
print("=" * 65)
print("  第 3 部分：Numba 的代价和限制")
print("=" * 65)

print("""
Numba 不是万能的。以下东西在 @njit 函数里不能随便用：

  [NO] Python 对象 (str, dict, 自定义类)
  [NO] 动态类型变来变去 (x 一会是 int 一会是 str)
  [NO] 大部分 Python 标准库 (用 NumPy 函数代替)
  [NO] 生成器 yield / 异常处理 / with 语句
  [NO] 有 GIL 的 Python 第三方库

  [OK] NumPy 数组运算
  [OK] 数值数学 (math 模块)
  [OK] for / while 循环
  [OK] if / else 分支
  [OK] 嵌套循环

  最佳适用场景：数值计算的 for 循环，且循环体内是纯数学运算。
""")


# ============================================================
# 第 4 部分：并行 -- 为什么能再快 5 倍？
# ============================================================
print("=" * 65)
print("  第 4 部分：Numba 并行 -- 为什么再快 5 倍？")
print("=" * 65)

print("""
关键概念：GIL（全局解释器锁）

  纯 Python 多线程：
    +-------------------------------------+
    |         GIL（一把大锁）              |
    |  线程1 ##......####......####        |
    |  线程2 ..##........####......        |
    |  线程3 ....####........####..        |
    |         ↑ 任何时候只有一个线程在跑！   |
    |         多线程在 Python 里不能并行计算 |
    +-------------------------------------+

  Numba @njit(parallel=True) + prange：
    +-------------------------------------+
    |         没有 GIL！                    |
    |  核心1 ##########################    |
    |  核心2 ##########################    |  ← 所有核心同时跑！
    |  核心3 ##########################    |
    |  核心4 ##########################    |
    |         ↑ 全是机器码，不经过 Python   |
    |         所以绕过了 GIL               |
    +-------------------------------------+

prange 做了什么：
  prange(n) 会自动把 [0, 1, 2, 3, ..., n-1] 拆成 N 份
  （N = CPU 核心数），每份给一个线程独立算：

  for i in prange(12):     # 12 个元素，4 核
  +--------+--------+--------+--------+
  | 核1: 0,1,2 | 核2: 3,4,5 | 核3: 6,7,8 | 核4: 9,10,11|
  +--------+--------+--------+--------+
     同时进行，各算各的！

  重要前提：每个迭代必须是"独立的"----后面的迭代不依赖前面的结果。
  如果 ans[i] 依赖 ans[i-1]（数据依赖），那就不能并行。
""")


# ============================================================
# 第 5 部分：实测并行的威力
# ============================================================
print("=" * 65)
print("  第 5 部分：实测 -- 看看核心数影响")
print("=" * 65)

import os
# 设置不同线程数来对比


@njit
def single_core(data):
    n = len(data)
    ans = np.empty(n)
    for i in range(n):
        x = data[i]
        ans[i] = math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x))
    return ans


@njit(parallel=True)
def multi_core(data):
    n = len(data)
    ans = np.empty(n)
    for i in prange(n):
        x = data[i]
        ans[i] = math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x))
    return ans


# 预热
_ = single_core(data)
_ = multi_core(data)

# 测量
t0 = time.perf_counter()
single_core(data)
t_single = time.perf_counter() - t0

t0 = time.perf_counter()
multi_core(data)
t_multi = time.perf_counter() - t0

print(f"单核 (range):        {t_single:.4f} 秒")
print(f"多核 (prange):       {t_multi:.4f} 秒")
print(f"并行加速:            {t_single / t_multi:.1f} 倍")

# 检测 CPU 核心数
cpu_count = os.cpu_count()
print(f"\n你的 CPU 核心数: {cpu_count}")
print(f"理论加速上限: ~{cpu_count}x（实际受限于内存带宽和任务调度开销）")


# ============================================================
# 总结
# ============================================================
print()
print("=" * 65)
print("  总结：Numba 的加速是怎么一层层叠上去的")
print("=" * 65)

print("""
  纯 Python for 循环         ############################  1x
      ↓ 去掉 Python 解释器开销
  Numba @njit (单核)         ####                         ~25x
      ↓ 用上所有 CPU 核心
  Numba @njit(parallel=True) #                            ~140x

  JIT 带来的 ~25x：
    - 去掉字节码解释开销
    - 去掉类型检查
    - 去掉 PyObject 创建/销毁
    - LLVM 自动做 SIMD 向量化

  并行额外带来的 ~5x：
    - 绕过 GIL
    - 所有 CPU 核心同时工作
    - 任务完美均匀分块

  为什么不是正好核心数倍的加速？
    -> 内存带宽成为瓶颈：CPU 核心再快，数据要从内存读，
       内存带宽有限，4 个核抢一条内存总线。
""")
