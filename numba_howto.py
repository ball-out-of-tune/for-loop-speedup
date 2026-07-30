"""
============================================================
  @njit 是什么意思？Numba 循环怎么写？
============================================================
"""

import math
import numpy as np
from numba import njit

# ========================================================
# 第 1 部分：@njit 到底是什么？
# ========================================================
print("=" * 55)
print("  @njit 是什么？")
print("=" * 55)

print("""
@njit 是一个"装饰器" (decorator)。装饰器的本质是：

  你写了一个函数  -->  @njit 把它"包装"成一个新版本
                    这个新版本 = 自动编译成机器码的版本

写法拆解：
""")

print("""
# 写法 1：用 @ 语法糖（最常用）
@njit
def my_func(data):
    ...

# 写法 2：不用 @，效果完全一样
def my_func(data):
    ...
my_func = njit(my_func)    # 手动包装

# 写法 3：@njit 是 @numba.jit(nopython=True) 的缩写
from numba import jit
@jit(nopython=True)        # 和 @njit 完全等价
def my_func(data):
    ...

三行重点：
  @ = 语法糖，就是"把下面这个函数交给 njit 处理"
  njit = nopython + jit   (强制"无 Python 模式"编译)
  nopython 模式 = 整个函数里不能有 Python 对象，全是机器码
""")

# 证明：@njit 返回的不是原来的函数
def normal_func(x):
    return x + 1

@njit
def numba_func(x):
    return x + 1

print("验证:")
print(f"  普通函数类型: {type(normal_func)}")
print(f"  @njit 后类型: {type(numba_func)}")
print(f"  普通函数名字: {normal_func.__name__}")
print(f"  @njit 后名字: {numba_func.__name__}")
print()
print("  @njit 把它变成了一个 'CPUDispatcher' 对象")
print("  你调用它时，Numba 会拦截调用，用机器码版本执行。")
print()


# ========================================================
# 第 2 部分：Numba 循环的编写逻辑
# ========================================================
print("=" * 55)
print("  Numba 循环编写逻辑：三条规则")
print("=" * 55)

N = 5  # 小数据，方便看清

# ---- 规则 1：输入和输出都用 NumPy 数组 ----
print("""
规则 1：输入和输出都用 NumPy 数组
  Numba 操作的是原始 C 内存，不是 Python 对象。
  所以数据必须封装在 np.ndarray 里传进去。
""")

print("--- 演示 ---")
data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"输入: {data}  (类型: {type(data).__name__})")
print()


# ---- 规则 2：用 range 写循环，不要用 Python 迭代器 ----
print("""
规则 2：用 for i in range(n)，不要用 for x in data

  Numba 能编译 range(n)，但不擅长处理 Python 的迭代器。
  用整数索引来访问数组元素。
""")

# 正确写法
@njit
def good_loop(data):
    n = len(data)
    result = np.empty(n)
    for i in range(n):        # range(n) -> Numba 直接编译
        x = data[i]           # 整数索引访问 -> 直接内存偏移
        result[i] = x ** 2
    return result

# 错误写法（在 @njit 里也能跑，但更慢且有限制）
@njit
def bad_loop(data):
    result = np.empty(len(data))
    i = 0
    for x in data:            # 迭代器 -> Numba 需要额外处理
        result[i] = x ** 2
        i += 1
    return result

print(f"good_loop (range): {good_loop(data)}")
print(f"bad_loop  (iter):  {bad_loop(data)}")
print("两者结果一样，但 range 版本的机器码更优。")
print()


# ---- 规则 3：循环体内用 math 模块的数学函数 ----
print("""
规则 3：数学运算用 math 模块，不要用 numpy.xxx

  在 @njit 函数里：
    math.sin(x)  -> 直接编译成 CPU 的 sin 指令（C 标准库）
    np.sin(x)    -> 也能用，但需要走 Numba 的 NumPy 兼容层

  单元素操作优先用 math，对整个数组操作才用 np.xxx。
""")

@njit
def use_math(data):
    """对单个元素操作时，用 math 函数"""
    n = len(data)
    result = np.empty(n)
    for i in range(n):
        x = data[i]
        result[i] = math.sin(x) + math.cos(x) + math.sqrt(abs(x))
    return result

print(f"use_math: {use_math(data)}")
print()


# ========================================================
# 第 3 部分：完整编写模板（记住这个就够了）
# ========================================================
print("=" * 55)
print("  模板：Numba 循环 = 记住这个就够了")
print("=" * 55)

print("""
@njit
def my_hot_loop(input_array):
    # 步骤 1：取长度
    n = len(input_array)

    # 步骤 2：预分配输出数组（关键！不要用 .append）
    result = np.empty(n)        # 或者 np.zeros(n)

    # 步骤 3：用 range(n) 循环
    for i in range(n):
        # 步骤 4：读入当前元素
        x = input_array[i]

        # 步骤 5：纯数学计算（用 math.xxx）
        # ... 你的计算逻辑 ...

        # 步骤 6：写入结果
        result[i] = ...   # 上面算出来的值

    # 步骤 7：返回整个数组
    return result

核心要点：
  +--------+----------------------------------+
  | 规则   | 原因                             |
  +--------+----------------------------------+
  | np.empty(n) | 一次性分配，不要 append       |
  | range(n)    | Numba 认识，直接编译成机器码  |
  | 索引访问    | arr[i] = 直接内存偏移，无开销  |
  | math.xxx    | 单个元素用 math，整个数组用 np |
  | 无 Python 对象 | str/dict/list 都不能出现      |
  +--------+----------------------------------+
""")


# ========================================================
# 第 4 部分：常见错误
# ========================================================
print("=" * 55)
print("  常见错误：新手容易踩的坑")
print("=" * 55)

# 错误 1：在 @njit 函数里用 list
print("""
错误 1：在 @njit 函数里用 Python list（包括 .append）

  [NO]
  @njit
  def wrong(data):
      result = []              # Python list!
      for i in range(len(data)):
          result.append(data[i] ** 2)   # append!
      return result

  报错：Numba 在 nopython 模式下无法处理 list.append
       或者能编译但退化成"对象模式"，速度骤降

  [OK]
  @njit
  def right(data):
      n = len(data)
      result = np.empty(n)     # NumPy 数组
      for i in range(n):
          result[i] = data[i] ** 2
      return result
""")

# 错误 2：忘记预分配
print("""
错误 2：忘记用 np.empty() 预分配输出数组

  你可能会想"我先建个空数组，再一个个填进去"，
  但 Numba 里没有 np.append（那会创建新数组，慢）。
  正确做法：先算出要多大，一次性用 np.empty(n) 分配好。
""")

# 错误 3：把 @njit 放在调用层
print("""
错误 3：@njit 应该加在"有 for 循环"的函数上，不是最外层

  [NO]
  @njit
  def outer(data):
      # 这个函数只是调用别人，没有循环
      return inner(data)     # 即使 inner 有 @njit，outer 的编译也浪费了

  [OK]
  def outer(data):
      return inner(data)     # outer 不加 @njit

  @njit
  def inner(data):           # 热循环所在的函数才加 @njit
      for i in range(len(data)):
          ...
""")


# ========================================================
# 第 5 部分：动手写一个
# ========================================================
print("=" * 55)
print("  动手：你的第一个完整的 Numba 循环")
print("=" * 55)

# 任务：计算数组的指数移动平均 (Exponential Moving Average)
# 这是一个有"数据依赖"的循环（result[i] 依赖 result[i-1]），
# NumPy 向量化不好写，但 Numba 非常适合！

@njit
def exponential_moving_average(data, alpha):
    """
    计算 EMA。注意：result[i] 依赖 result[i-1]，
    所以这里用不了 prange 并行（有数据依赖），
    但 @njit 单核仍然比纯 Python 快几十倍。
    """
    n = len(data)
    result = np.empty(n)
    result[0] = data[0]                         # 第一个值照抄
    for i in range(1, n):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result

# 生成测试数据
np.random.seed(42)
prices = np.random.randn(100).cumsum() + 100  # 模拟股价走势

ema = exponential_moving_average(prices, 0.3)

print(f"输入  (前5个): {prices[:5]}")
print(f"EMA   (前5个): {ema[:5]}")
print(f"输入长度: {len(prices)}, 输出长度: {len(ema)}")

# 验证纯 Python 版本一致
def ema_python(data, alpha):
    result = [0.0] * len(data)
    result[0] = data[0]
    for i in range(1, len(data)):
        result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
    return result

ema_py = ema_python(list(prices), 0.3)
print(f"纯 Python EMA  (前5个): {[f'{v:.4f}' for v in ema_py[:5]]}")
print(f"Numba   EMA   (前5个): {[f'{v:.4f}' for v in ema[:5]]}")
print("结果一致！")

print()
print("""
总结：@njit 的思维模型

  把 @njit 想象成"函数级别的加速开关"：

    def my_func(data):        @njit
        for ...               def my_func(data):
        ↑                         for ...
        这是 Python 代码              ↑
                                    这是机器码

    你写的还是 Python 语法，
    但 @njit 保证它会被编译成机器码执行。

  Numba 循环的编写逻辑 = NumPy 数组 + range(n) + math 函数
  记住这个公式就够了。
""")
