# Python For-Loop Speedup

热循环性能加速实验：纯 Python → NumPy → Numba → GPU。
从 0.35s 到 0.00005s，**加速 6,250 倍**。

## 文件说明

| 文件 | 内容 | 运行方式 |
|---|---|---|
| `hot_loop.py` | CPU benchmark：6 种方法（Python/NumPy/Numba/并行/heavy） | `python hot_loop.py` |
| `cuda_hot_loop.py` | GPU 加速教程：CuPy + Numba CUDA，含数据传输/计算拆时 | `python cuda_hot_loop.py` |
| `speed_ladder.py` | 速度阶梯：4 种方法加速比对比 | `python speed_ladder.py` |
| `bench_roofline.py` | Roofline 模型：实测内存带宽和计算峰值 | `python bench_roofline.py` |
| `profile_kernel.py` | Nsight Compute 用：GPU 硬件计数器精确计时 (FP64) | `ncu --metrics ... python profile_kernel.py` |
| `profile_kernel_f32.py` | Nsight Compute 用：同上 (FP32) | `ncu --metrics ... python profile_kernel_f32.py` |
| `profile_cupy.py` | Nsight Compute 用：CuPy kernel 分析 | `nvprof python profile_cupy.py` |
| `numba_explained.py` | Numba 原理：JIT 编译、LLVM、类型推断 | 阅读 |
| `numba_howto.py` | Numba 实操：`@njit`、`prange`、常见坑 | 阅读 |
| `hot_loop_tutorial.py` | hot_loop 逐步优化教程 | 阅读 |
| `got_loop.py` | 早期版本 | 参考 |

## 实验结论

**测试环境：** i7-11800H (8C/16T) / RTX 3050 Ti Laptop 4GB (GA107, 2,560 CUDA cores) / DDR4
**测试数据：** 1,000,000 个 `float64` (Python/NumPy/Numba) 或 `float32` (GPU)，计算 `sin²(x) + cos²(x) + sqrt(|x|)`

### 完整性能对比

| 方法 | 设备 | 精度 | 耗时 | 加速比 |
|---|---|---|---|---|
| 纯 Python for 循环 | CPU | f64 | 0.35 s | 1× |
| NumPy 向量化 | CPU | f64 | 0.06 s | 5.8× |
| Numba `@njit` | CPU | f64 | 0.048 s | 7.3× |
| Numba `@njit(parallel=True)` | CPU | f64 | 0.068 s | 慢于非并行 → 内存瓶颈 |
| Numba `@njit` heavy (100× 计算量) | CPU | f64 | 1.98 s | — |
| Numba `@njit(parallel=True)` heavy | CPU | f64 | 0.28 s | 7.2× → 计算瓶颈时并行才有效 |
| CuPy | GPU | f64 | 0.0067 s | 52× |
| Numba `@cuda.jit` | GPU | f64 | 0.00089 s | 395× |
| Numba `@cuda.jit` | GPU | **f32** | **0.000056 s** | **6,250×** |

### Roofline 瓶颈分析

**GPU 理论峰值 (RTX 3050 Ti Laptop)**：
- FP32: 5.3 TFLOPS | FP64: 82.8 GFLOPS | 显存带宽: 192 GB/s
- Ridge point (f64): 0.43 FLOPs/Byte | Ridge point (f32): 27.6 FLOPs/Byte

**NVIDIA Nsight Compute 硬件计数器实测（`ncu --metrics`）：**

| 指标 | float64 | float32 |
|---|---|---|
| 显存读写 | 15.96 MB | 7.72 MB |
| 真实 FLOPs (硬件计数) | 66,000,000 | 40,000,000 |
| 实测 GFLOPS | 74.5 | 800 |
| 峰值 GFLOPS | 82.8 | 5,300 |
| **算力利用率 (MFU)** | **90%** ✅ | 15% |
| 实测带宽 | 18.0 GB/s | 154.4 GB/s |
| 带宽利用率 | 9.4% | **80%** |
| 瓶颈 | 算力 | **带宽** |

### 核心 Insight

1. **瓶颈永远在转移，不是固定的。**
   - f64 下算力利用率 90%，几乎打满 → 提高 f64 没意义
   - 换 f32 炸开算力天花板后，瓶颈跳到带宽 (80% 利用率)
   
2. **Roofline 模型是优化指南针。**
   - 每次优化后重算 AI 和 ridge point，判断落点，决定下一步方向
   - f64: AI=4.1 > ridge=0.43 → 算力瓶颈 → 换 f32
   - f32: AI=5.2 < ridge=27.6 → 带宽瓶颈 → 优化访存

3. **消费级 GPU 的 FP64 被故意砍了 64 倍。**
   很多科学计算应用因此用混合精度或纯 FP32

4. **`prange` 并行只在计算密度足够高时有价值。**
   - 轻量计算 → 线程开销大于收益 → 用 `range`
   - 重量计算 → 计算远大于线程开销 → 用 `prange`

5. **CuPy 一行代码 = 7 次 kernel 启动 + 6 个临时数组。**
   开发效率高，但多余的显存读写让它比手写 Numba CUDA kernel 慢 5×
