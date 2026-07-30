"""
============================================================
  [新手教程] 热循环（Hot Loop）到底是什么？怎么写？
============================================================

一句话理解：
  "热循环"就是你的程序里，被执行了 成千上万次（甚至上亿次）
  的那段循环代码。它是整个程序的"性能瓶颈所在"。

  就像做饭：
  - 切一次洋葱（冷代码 / Cold Code）         → 慢一点没关系
  - 炒 10000 粒米，每粒都要翻面（热循环）    → 必须快！每粒慢 0.01 秒就完蛋

目录：
  1. 热循环长什么样
  2. 为什么 Python 写得慢（逐条拆开看）
  3. 方案一：NumPy — 把循环推到 C 层
  4. 方案二：Numba — 给 Python 装个"发动机"
  5. 总结：什么时候必须离开 Python
  6. 动手跑一跑
"""

import time
import math


# ==========================================================
#  第 1 步：认识热循环 — 它长什么样
# ==========================================================
print("=" * 60)
print("第 1 步：一个典型的热循环")
print("=" * 60)

def hot_loop_python(data):
    """
    这是一个"热循环"：
    - 对 1000 万个数据点，每个做三角运算和平方根
    - 在科学计算、信号处理、游戏中，这种循环太常见了
    """
    result = []
    for x in data:
        # 每次循环都要：查类型 → 找函数 → 创建对象 → 管理内存
        y = math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x))
        result.append(y)
    return result

# 用一个较小的数据来演示
small_data = [i * 0.001 for i in range(1_000_000)]  # 100 万个点

print(f"数据量: {len(small_data):,} 个点")
print("开始运行 hot_loop_python ...")
t0 = time.perf_counter()
hot_loop_python(small_data)
elapsed = time.perf_counter() - t0
print(f"耗时: {elapsed:.3f} 秒")
print(f"每个数据点平均耗时: {elapsed / len(small_data) * 1e9:.1f} 纳秒")
print()


# ==========================================================
#  第 2 步：为什么 Python 慢？逐条拆开看
# ==========================================================
print("=" * 60)
print("第 2 步：为什么 Python 写热循环会慢？")
print("=" * 60)

print("""
假设你写:
    y = math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x))

C 语言执行这条语句：大概 5~10 条 CPU 指令（微秒级）

Python 执行同一条语句,内部要做:
┌─────────────────────────────────────────────────────────┐
│ ① 加载 math 模块（查字典）                               │
│ ② 查找 math.sin 属性（又一次字典查找）                    │
│ ③ 检查 x 是什么类型（int? float? 自定义类？）            │
│ ④ 根据类型找到对应的 __sin__ 实现                        │
│ ⑤ 调用底层 C 函数（唯一"快"的部分）                      │
│ ⑥ 创建一个新的 PyFloatObject 对象（堆分配！）             │
│ ⑦ 设置该对象的引用计数 = 1                                │
│ ⑧ 对 math.cos(x) 重复 ②-⑦                                │
│ ⑨ 找到 ** 运算符对应的 __pow__ 方法（又是类型检查）       │
│ ⑩ 创建第三个 PyFloatObject ...                           │
│  ... 还有 + 号、sqrt、abs ...                            │
│                                                          │
│ 结果：C 语言一条指令 → Python 几十甚至上百条解释器指令     │
│       加上每个数字都是"堆上的对象"（C 里就是寄存器值）    │
│       这就是为什么纯 Python 热循环比 C 慢 10-100 倍       │
└─────────────────────────────────────────────────────────┘
""")


# ==========================================================
#  第 3 步：方案一 — NumPy（最常用! 把循环推到 C 层）
# ==========================================================
print("=" * 60)
print("第 3 步：方案一 — 用 NumPy 把循环推到 C 层")
print("=" * 60)

print("""
核心理念：
  你不是"不用 Python"，而是"不让 Python 做循环本身"。
  Python 只当"导演"（调度），真正的循环在 NumPy（C/Fortran）里跑。

类比：
  纯 Python   = 你亲手翻 10000 粒米，一粒一粒翻
  NumPy       = 你对厨师（C 代码）喊"全翻一遍"，厨师一秒搞定
""")

try:
    import numpy as np

    def hot_loop_numpy(data):
        """
        和上面一模一样的数学运算，但用 NumPy"向量化"写法。
        没有 for 循环！运算发生在 C 层面。
        """
        x = np.asarray(data)  # 把数据交给 NumPy 管理
        # 下面这四个操作：全部在 C 层一口气完成，Python 没有逐元素循环！
        return np.sin(x) ** 2 + np.cos(x) ** 2 + np.sqrt(np.abs(x))

    # 验证结果一致
    result_py = hot_loop_python(small_data[:5])
    result_np = hot_loop_numpy(small_data[:5])
    print(f"纯 Python 前 5 个结果: {[f'{v:.6f}' for v in result_py]}")
    print(f"NumPy    前 5 个结果: {[f'{v:.6f}' for v in result_np]}")

    # 性能对比
    print(f"\n数据量: {len(small_data):,}")
    print("运行 NumPy 版本 ...")
    t0 = time.perf_counter()
    hot_loop_numpy(small_data)
    np_elapsed = time.perf_counter() - t0
    print(f"NumPy 耗时:    {np_elapsed:.4f} 秒")

    # 再跑一遍 Python 做对比
    print("运行纯 Python 版本 ...")
    t0 = time.perf_counter()
    hot_loop_python(small_data)
    py_elapsed = time.perf_counter() - t0
    print(f"纯 Python 耗时: {py_elapsed:.4f} 秒")

    speedup = py_elapsed / np_elapsed if np_elapsed > 0 else float('inf')
    print(f"\n[RESULT] NumPy 快了 {speedup:.1f} 倍！")
    print("   这就是\"用 Python 写逻辑, 用 C 跑热循环\"的含义.")

except ImportError:
    print("  (NumPy 未安装，跳过演示)")
    print("  安装命令: pip install numpy")

print()


# ==========================================================
#  第 4 步：方案二 — Numba（给 Python 装个"即时编译器"）
# ==========================================================
print("=" * 60)
print("第 4 步：方案二 — Numba JIT 编译器")
print("=" * 60)

print("""
核心理念：
  你仍然写 Python 语法的 for 循环，但 Numba 在运行时把它
  "翻译"（编译）成机器码，跑起来就接近 C 的速度。

  就像给 Python 装了个涡轮增压。

注意：
  - 不是所有 Python 都能被 Numba 加速（有子集限制）
  - 需要避免 Python 对象、列表等动态结构
  - 最适合：数值计算、数组操作
""")

try:
    from numba import njit

    @njit  # ← 这个装饰器就是"涡轮增压开关"
    def hot_loop_numba(data):
        """
        代码看起来是 Python，但 @njit 会让 Numba
        把它编译成 LLVM IR → 机器码，跑起来接近 C 速度！
        """
        n = len(data)
        result = [0.0] * n  # Numba 下用预分配列表
        for i in range(n):
            x = data[i]
            # 下面这些运算会被直接编译成 CPU 指令
            sin_x = math.sin(x)
            cos_x = math.cos(x)
            result[i] = sin_x * sin_x + cos_x * cos_x + math.sqrt(abs(x))
        return result

    # 先将数据转为 Numba 友好的格式
    import numpy as np
    data_np = np.array(small_data, dtype=np.float64)

    print("第一次调用 Numba 版本（含编译时间）...")
    t0 = time.perf_counter()
    hot_loop_numba(data_np)
    first_call = time.perf_counter() - t0
    print(f"耗时（含编译）: {first_call:.4f} 秒")

    print("第二次调用（已编译，纯运行速度）...")
    t0 = time.perf_counter()
    hot_loop_numba(data_np)
    second_call = time.perf_counter() - t0
    print(f"耗时（纯运行）: {second_call:.4f} 秒")
    print(f"[RESULT] 纯 Python 对比 Numba：快了 {py_elapsed / second_call:.1f} 倍！")

except ImportError:
    print("  (Numba 未安装，跳过演示)")
    print("  安装命令: pip install numba")

print()


# ==========================================================
#  第 5 步：什么时候必须离开 Python
# ==========================================================
print("=" * 60)
print("第 5 步：什么时候你必须离开 Python")
print("=" * 60)

print("""
┌──────────────────────────────────────────────────────────────┐
│  场景                            推荐方案                    │
├──────────────────────────────────────────────────────────────┤
│  数据处理、矩阵运算              NumPy / Pandas              │
│  科学计算、数值模拟              NumPy + Numba / SciPy       │
│  图像/音频逐像素/逐采样处理       Numba / Cython             │
│  游戏引擎主循环（每帧 16ms）     C++ / Rust  ← 必须离开Python│
│  嵌入式设备、内存受限             C / Rust                   │
│  高频交易（微秒级延迟）          C++ / Java                  │
│  深度学习训练循环                 PyTorch/TF（底层都是C++/CUDA│
│  操作系统内核                     C ← Python 根本进不去      │
└──────────────────────────────────────────────────────────────┘

经验法则：
  ① 先写纯 Python → 跑起来,保证正确
  ② 用 profiler 找到热循环（cProfile / line_profiler）
  ③ 优先试 NumPy 向量化（改动最小,收益最大）
  ④ 还不够 → Numba / Cython
  ⑤ 再不够 → 用 C/C++/Rust 写热循环,Python 调用（ctypes/cffi/PyO3）
""")


# ==========================================================
#  第 6 步：动手 — 跑一下完整对比
# ==========================================================
print("=" * 60)
print("第 6 步：动手！完整性能对比")
print("=" * 60)

def run_benchmark():
    """
    用不同方法跑同一个热循环，直观对比速度。
    """

    # 准备数据（不同规模）
    sizes = [10_000, 100_000, 1_000_000]
    methods = {}

    # --- 纯 Python ---
    def pure_python(data):
        result = [0.0] * len(data)
        for i, x in enumerate(data):
            result[i] = math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x))
        return result
    methods["纯 Python (for 循环)"] = pure_python

    # --- 列表推导式 ---
    def list_comp(data):
        return [math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x))
                for x in data]
    methods["纯 Python (列表推导式)"] = list_comp

    # --- NumPy ---
    try:
        import numpy as np
        def numpy_vec(data):
            x = np.asarray(data)
            return np.sin(x) ** 2 + np.cos(x) ** 2 + np.sqrt(np.abs(x))
        methods["NumPy 向量化"] = numpy_vec
    except ImportError:
        pass

    # --- Numba ---
    try:
        from numba import njit
        import numpy as np
        @njit
        def numba_loop(data):
            n = len(data)
            result = np.empty(n)
            for i in range(n):
                x = data[i]
                result[i] = math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x))
            return result
        methods["Numba @njit"] = numba_loop
    except ImportError:
        pass

    # 跑对比
    print()
    print(f"{'方法':<28} | {'1万点':>8} | {'10万点':>8} | {'100万点':>9}")
    print("-" * 62)

    for name, func in methods.items():
        times = []
        for size in sizes:
            data = [i * 0.001 for i in range(size)]
            if "Numba" in name or "NumPy" in name:
                data = np.array(data, dtype=np.float64)

            # 预热一次（对 Numba 尤其重要）
            _ = func(data)

            t0 = time.perf_counter()
            _ = func(data)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        print(f"{name:<28} | {times[0]:7.4f}s | {times[1]:7.4f}s | {times[2]:8.4f}s")


print()
run_benchmark()

print("""
总结一句话：
  热循环不能写纯 Python → 不是说你不用 Python，
  而是 Python 只当"导演"，NumPy/Numba/C 来当"演员"。

  "热循环不能用 Python 写" = "不要在性能瓶颈处使用 Python 的解释循环"
""")
