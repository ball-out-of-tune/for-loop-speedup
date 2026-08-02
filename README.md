# Python For-Loop Speedup

热循环性能加速实验：纯 Python → NumPy → Numba → GPU。
从 0.35s 到 0.049 ms，**加速 7,100 倍，摸到硬件天花板。**

## 文件说明

| 文件 | 内容 | 运行方式 |
|---|---|---|
| `hot_loop.py` | CPU benchmark：6 种方法（Python/NumPy/Numba/并行/heavy） | `python hot_loop.py` |
| `cuda_hot_loop.py` | GPU 加速教程：CuPy + Numba CUDA，含数据传输/计算拆时 | `python cuda_hot_loop.py` |
| `cuda_kernel_float4.py` | float4 向量化尝试 + block size 调优 | `python cuda_kernel_float4.py` |
| `speed_ladder.py` | 速度阶梯：4 种方法加速比对比 | `python speed_ladder.py` |
| `bench_roofline.py` | Roofline 模型：实测内存带宽和计算峰值 | `python bench_roofline.py` |
| `profile_kernel.py` | Nsight Compute 用：GPU 硬件计数器 (FP64) | `ncu --metrics ... python profile_kernel.py` |
| `profile_kernel_f32.py` | Nsight Compute 用：GPU 硬件计数器 (FP32) | `ncu --metrics ... python profile_kernel_f32.py` |
| `profile_cupy.py` | Nsight Compute 用：CuPy kernel 分析 | `ncu --metrics ... python profile_cupy.py` |
| `hot_loop.cu` | Raw CUDA C++ 实现（nvcc 编译） | `nvcc -o hot_loop_cuda hot_loop.cu && ./hot_loop_cuda` |
| `hot_loop_triton.py` | Triton kernel 实现（block 级编程） | `python hot_loop_triton.py` |
| `pytorch_hot_loop.py` | PyTorch eager vs torch.compile 对比 | `python pytorch_hot_loop.py` |
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
| Numba `@cuda.jit` (block=256) | GPU | f32 | 0.000049 s | 7,100× |
| Triton (block=512) | GPU | f32 | 0.000048 s | 7,200× |
| Raw CUDA C++ (nvcc) | GPU | f32 | 0.000050 s | 7,000× |
| PyTorch eager | GPU | f32 | 0.000414 s | 845× (8 kernel, 无融合) |
| PyTorch `torch.compile` | GPU | f32 | 0.000051 s | 6,800× (Inductor 融合) |
| **=== 理论最快 ===** | | | **0.000040 s** | **物理极限 (8 MB ÷ 192 GB/s)** |

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

5. **CuPy / PyTorch eager 一行代码 = N 次 kernel 启动 + (N-1) 个临时数组。**
   开发效率高，但多余的显存读写让它比手写 kernel 慢 8×。`torch.compile` 自动融合可追平手写——不改代码，性能拉满

### Block Size 调优

同一个 kernel，修改 `threads_per_block` 参数（128/256/512/1024），测量带宽利用率：

| threads | blocks | 耗时 | 带宽 | 利用率 |
|---|---|---|---|---|
| 128 | 7813 | 0.0506 ms | 158.2 GB/s | 82% |
| **256** | **3907** | **0.0493 ms** | **162.4 GB/s** | **85%** |
| 512 | 1954 | 0.0507 ms | 157.8 GB/s | 82% |
| 1024 | 977 | 0.0602 ms | 132.9 GB/s | 69% |

Block size 影响 SM 占用率（Occupancy）。SM 有固定资源上限（2048 线程 / 64 warp / 16 block / 65536 寄存器 / 100 KB shared memory），block size 决定了哪些资源先爆、决定了每个 SM 同时驻留多少 warp。warp 不够多 → 内存延迟藏不住 → 带宽利用率下降。256 恰好是这张卡最优值。

### GPU 编程三种写法：Numba CUDA vs Raw CUDA C++ vs Triton

同一个公式，三种 GPU 编程风格——底层硬件相同，天花板相同，但抽象层级和思维方式完全不同：

| 写法 | 语言 | 编译路径 | 你控制什么 | 适合场景 |
|---|---|---|---|---|
| `@cuda.jit` (Numba) | Python decorator | LLVM → PTX → SASS | 每个 thread | 学习、简单 kernel 快速迭代 |
| `__global__` (Raw CUDA) | C++ | nvcc → PTX → SASS | 每个 thread | 性能极致、工程化部署 |
| `@triton.jit` (Triton) | Python decorator | MLIR → PTX → SASS | 每个 block | 复杂 tensor 运算、自动调优 |

**实测——三者都在同一个天花板下：**

```
Numba CUDA:    0.0493 ms   (162.4 GB/s, 85.0%)
Raw CUDA C++:  0.0501 ms   (159.5 GB/s, 83.1%)
Triton:        0.0483 ms   (165.6 GB/s, 86.2%)
差异: ±2% — 测量噪声，硬件天花板无人突破
```

**核心差异：**

- **CUDA / Numba CUDA**: 你写 thread 级代码——`cuda.grid(1)` 手动算全局线程号，`if i < n` 手动边界检查
- **Triton**: 你写 block 级代码——`tl.program_id(0)` 获取 block ID，`tl.arange(0, BLOCK_SIZE)` 向量化处理 block 内所有元素，`mask=mask` 自动边界处理。编译器自动完成 thread 分配、float4 向量化、寄存器优化
- **Numba CUDA 的优势**: 和 Python 无缝集成，一个 decorator 搞定，适合快速验证想法
- **Raw CUDA C++ 的优势**: 零 Python 运行时开销，支持 shared memory 精细控制、warp shuffle、PTX 内联汇编、Tensor Core——当 kernel 变复杂时威力才显现

### PyTorch：Eager vs torch.compile — Kernel Fusion 的价值

PyTorch 标准写法（eager mode）和 CuPy 一样，每个运算都是独立的 kernel 启动：

```python
# Eager mode: 8 次 kernel 启动 + 6 个中间 tensor
y = torch.sin(x)**2 + torch.cos(x)**2 + torch.sqrt(torch.abs(x))
# → 0.414 ms — 每个 kernel 启动一次，中间结果全写回显存
```

**`torch.compile` (Inductor backend)** 自动完成 kernel fusion——和手写 CUDA 一样，看到全局后合并成 1 个 kernel：

```python
@torch.compile
def compute(x):
    return torch.sin(x)**2 + torch.cos(x)**2 + torch.sqrt(torch.abs(x))
# → 0.051 ms — Inductor 自动融合，追平手写 kernel
```

| 方法 | 耗时 | Kernel 数 | 显存流量 | 加速比 |
|---|---|---|---|---|
| PyTorch eager | 0.414 ms | 8 | 72 MB | 1× |
| PyTorch torch.compile | 0.051 ms | 1 | 8 MB | 8.1× |
| Numba CUDA | 0.049 ms | 1 | 8 MB | — |
| Triton | 0.048 ms | 1 | 8 MB | — |

**为什么 eager mode "不懂"融合？** Eager mode 看到一行立即执行一行——执行 `torch.sin(x)` 时不知道你接下来还要算 `cos` 和 `sqrt`，不敢省略中间写回。`torch.compile` / 手写 kernel 看到了整个表达式，中间结果留在寄存器里，不写回显存。这就是 **kernel fusion**——推理引擎最重要的优化之一。

### 三种 GPU 编程范式对比

```
                   抽象层级     融合方式         性能(简单kernel)   性能(复杂kernel)
PyTorch eager       最高      无融合, N个kernel     慢(0.41ms)        最慢
PyTorch + compile   高        Inductor自动融合      快(0.05ms)        快(自动调优)
Numba CUDA          中        手动写1个kernel        快(0.05ms)        中(缺warp级控制)
Triton              中低      block级编程,自动向量化  快(0.05ms)        快(自动调优)
Raw CUDA C++        最低      thread级完全控制       快(0.05ms)        最快(全手动)
```

**选型建议：**
- 原型验证 / 学习 → Numba CUDA（最快从想法到结果）
- 复杂 tensor 运算 / 需要自动调优 → Triton（编译器替你试 block size 和向量化）
- 生产部署 / 极致性能 → Raw CUDA C++（shared memory、warp shuffle、Tensor Core 全控制）
- 日常训练/推理 → `torch.compile`（不改代码，自动融合，实战首选）

### float4 向量化尝试

每个线程处理 4 个元素（`base = i * 4`），期望 LLVM 合并 4 次 32-bit 读取为一次 128-bit 加载。

**结果：0.0536 ms，反而慢于 baseline。** 原因：baseline 已经 151 GB/s（79%），和实测上限 162 GB/s 只差 7%。float4 省下的线程开销远小于内存带宽的物理上限，优化空间被硬件锁死了。

### 极限在哪？

```
理论最快 = 数据量 ÷ 带宽 = 7.72 MB ÷ 192 GB/s = 0.040 ms

实测 0.049 ms，达到物理极限的 82%（时间）/ 85%（带宽）。

剩下 15% 是 GPU 显存的固定物理开销：DRAM 页切换、刷新周期、地址/命令总线——软件改不了。
```

### 优化全景图

```
纯 Python              350,000 μs  ████████████████████████████████  1×
NumPy                   60,000 μs  ██████                            5.8×
Numba @njit             48,000 μs  █████                             7.3×
PyTorch eager (GPU)        414 μs  ▏                                 845× (8 kernel, 无融合)
Numba CUDA f64             886 μs  ▏                                 395×
PyTorch torch.compile       51 μs  ▏                                 6,800× (自动融合)
Raw CUDA C++                50 μs  ▏                                 7,000×
Numba CUDA f32              49 μs  ▏                                 7,100×
Triton f32                  48 μs  ▏                                 7,200×
─────────────────────────────────────────────────────────────────────
理论最快 (带宽)              40 μs  ▏                                 物理极限
```

**7,000+ 倍加速、三种 GPU 写法殊途同归、摸到硬件天花板 —— 这块卡、这个计算，已经到终点了。下一步：Reduce → Softmax → MatMul，进入真正的推理引擎世界。**
