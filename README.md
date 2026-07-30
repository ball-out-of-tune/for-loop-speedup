# Numba Performance Benchmark

Python 热循环性能加速实验：纯 Python → NumPy → Numba → Numba 并行。

## 文件说明

| 文件 | 内容 | 运行方式 |
|---|---|---|
| `hot_loop.py` | **核心 benchmark**：4 个函数的预热、编译、10 次平均耗时对比。包含轻量计算和重量计算两个维度，验证内存瓶颈 vs 计算瓶颈 | `python hot_loop.py` |
| `speed_ladder.py` | 速度阶梯展示：4 种方法（纯 Python / NumPy / Numba / Numba 并行）的加速比对比 | `python speed_ladder.py` |
| `bench_roofline.py` | Roofline 模型实战：实测本机的内存带宽和计算峰值，判断瓶颈类型 | `python bench_roofline.py` |
| `numba_explained.py` | Numba 原理讲解：JIT 编译、LLVM、类型推断等概念 | 阅读用 |
| `numba_howto.py` | Numba 实操指南：`@njit`、`prange`、常见坑等 | 阅读用 |
| `hot_loop_tutorial.py` | hot_loop 教学版：从 Python 到 Numba 的逐步优化过程 | 阅读用 |
| `got_loop.py` | hot_loop 的早期版本 | 参考用 |

## 实验结论

在 i7-11800H (8 核) / DDR4 笔记本上，对 100 万个 float64 做 `sin² + cos² + sqrt`：

| 方法 | 轻量计算 (1×) | 重量计算 (100×) |
|---|---|---|
| 纯 Python | 0.35s | — |
| NumPy | 0.06s | — |
| Numba `@njit` | **0.048s** | 1.98s |
| Numba `parallel=True` | 0.06s（更慢） | **0.28s（快 7.2×）** |

核心教训：**并行只在计算密度足够高时才有用。用之前先判断瓶颈是内存还是 CPU。**
