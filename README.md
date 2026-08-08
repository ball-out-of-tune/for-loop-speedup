# GPU Kernel Lab 🚀

从零手写 CUDA kernel：Element-wise → Reduce → Softmax → GEMM → Flash Attention。
**从 Python for-loop (0.35s) 到 5090 GEMM (290 TFLOPS)，一路摸到硬件天花板。**

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
| `reduce.cu` | CUDA Reduce kernel (4 版本: tree/warp/atomic/two-level) | `nvcc -o reduce_cuda reduce.cu && ./reduce_cuda` |
| `bench_reduce.cu` | Fair benchmark: 手写 CUDA vs PyTorch reduce | `nvcc -o bench_reduce bench_reduce.cu && ./bench_reduce` |
| `rms_norm.cu` | CUDA RMSNorm kernel (reduce + element-wise 融合) | `nvcc -o rms_norm_cuda rms_norm.cu && ./rms_norm_cuda` |
| `rmsnorm_torch.py` | PyTorch RMSNorm benchmark baseline | `python rmsnorm_torch.py` |
| `softmax.cu` | CUDA Softmax kernel (V1/V2/V3) + L2 验证 + 用户 MyOwn | `nvcc -o softmax_cuda softmax.cu && ./softmax_cuda` |
| `softmax.py` | PyTorch Softmax benchmark baseline | `python softmax.py` |
| `softmax_beat.cu` | V4: register-cached online, hidden=8192/16384, 超 PyTorch | `nvcc -o softmax_beat softmax_beat.cu && ./softmax_beat` |
| `softmax_final.cu` | V1 vs MyOwn vs PyTorch 跨尺寸对比 | `nvcc -o softmax_final softmax_final.cu && ./softmax_final` |
| `big_sizes.cu` | Register cache vs online 大 hidden (32768/65536) 对比 | `nvcc -o big_sizes big_sizes.cu && ./big_sizes` |
| `isolate.cu` | 隔离实验：register cache vs online 谁贡献大 | `nvcc -o isolate isolate.cu && ./isolate` |
| `v3v4.cu` | V3 vs V4 寄存器家族同台对比 | `nvcc -o v3v4 v3v4.cu && ./v3v4` |
| `softmax_scale.py` | PyTorch softmax 多尺寸 benchmark | `python softmax_scale.py` |
| `verify_l2_overflow.cu` | read-3-only 实验：L2 溢出边界测量 | `nvcc -o verify_l2_overflow verify_l2_overflow.cu && ./verify_l2_overflow` |
| `verify_stack.cu` | ncu 验证 B kernel 栈帧和 occupancy | `nvcc -o verify_stack verify_stack.cu` |
| `verify_ldg.cu` | ncu 验证 MyOwn vs V4 的 LDG 指令数差异 | `nvcc -o verify_ldg verify_ldg.cu` |
| `trace_torch_sum.py` | 查 torch.sum 底层调了什么 kernel | `python trace_torch_sum.py` |
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
| | | | | |
| **Reduce (1M float32 sum)** | | | | |
| PyTorch eager (ATen) | GPU | f32 | 0.000032 s | baseline |
| **Our CUDA V5 (grid-strid+shfl)** | GPU | f32 | **0.000026 s** | **1.22x vs PyTorch ✅** |
| | | | | |
| **RMSNorm (1000 rows × 4096 hidden)** | | | | |
| PyTorch eager | GPU | f32 | 0.000652 s | baseline |
| **Our CUDA (1 fused kernel)** | GPU | f32 | **0.000248 s** | **2.64x vs PyTorch ✅** |

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

---

## Reduce: 归约操作 (sum / max / mean)

**文件:** `reduce.cu`, `bench_reduce.cu`

Reduce 和 element-wise 的本质区别：元素之间有依赖，不能完全并行。N 个元素 → 1 个标量，必须通过"归约树"合并。

### 实现演进

| 版本 | 技术 | 时间 | 带宽 | vs PyTorch |
|---|---|---|---|---|
| V1 | Shared Memory Tree Reduce | 0.068 ms | 56 GB/s | 0.47x |
| V2 | + Warp Shuffle | 0.047 ms | 81 GB/s | 0.68x |
| **V5** | **Grid-stride + Shuffle + atomicAdd** | **0.026 ms** | **156 GB/s** | **1.22x ✅** |
| PyTorch eager (ATen) | 内置 reduce kernel | 0.032 ms | 128 GB/s | 1.00x (baseline) |
| 理论上限 | — | 0.021 ms | 192 GB/s | — |

### 关键 Insight

1. **Warp shuffle beats shared memory (1.45x).** `__shfl_down_sync` 是寄存器级数据交换 (<1 cycle)，shared memory 要 ~20 cycles。但从 stride=16 开始才用 shuffle，stride >= 32 跨 warp 必须用 shared memory。

2. **Grid-stride loop 是减少 block 数的关键。** V2 每个 thread 处理 1 个元素 → 3907 blocks → 3907 个 partial sums，写回 CPU 求和。V5 每个 thread 处理 64 个元素 → 62 blocks → 62 个 atomicAdd，全程 GPU，无 CPU 参与。atomicAdd 竞争从 3907 降到 62，从"灾难"变"可忽略"。

3. **超过了 PyTorch ATen (1.22x)。** ATen 是通用 reduce 模板（支持任意 dim、dtype、op），我们是专用 1D float32 sum。专用 > 通用。

4. **`torch.sum()` 不是 cuBLAS。** cuBLAS 只做 BLAS 运算（GEMM、GEMV），不做 reduce。`torch.sum()` 走 ATen 自己的 CUDA kernel。

### Warp Shuffle 原理

```
__shfl_down_sync(mask, val, delta)

  三个参数:
    mask = 0xffffffff  →  warp 内 32 个 lane 全部参与
    val  = 我手里要传给别人的值
    delta = 从"下方"第几个 lane 拿值

  效果: lane i 收到 lane i+delta 的 val, 加到自己的 val 上
  一轮: 32 个值 → 16 个"配对和"
  五轮: 1 个总和 (在 lane 0)

  __shfl_down_sync 只在 warp 内通信!
  跨 warp (256 threads = 8 warps) 必须用 shared memory。
  → warp shuffle 拿不到其他 warp 的值 → 最终结果
    必须写回 shared memory 广播给整个 block!
```

---

## RMSNorm: 第一个融合 Kernel

**文件:** `rms_norm.cu`, `rmsnorm_torch.py`

RMSNorm(x) = x / sqrt(mean(x²) + ε) × weight

**这是第一个把 reduce + element-wise 焊在一起的 kernel——推理引擎的核心模式。**

### 性能对比 (1000 rows × 4096 hidden)

| 方法 | 时间 | 带宽 | 利用率 |
|---|---|---|---|
| **Our CUDA (1 kernel)** | **0.248 ms** | **132.3 GB/s** | **68.9%** |
| PyTorch eager | 0.652 ms | 50.3 GB/s | 26.2% |
| PyTorch torch.compile | 3.423 ms | 9.6 GB/s | 5.0% |
| **加速比 vs eager** | **2.64x** | — | — |

### Nsight Compute 硬件实测

```
指令级分析 (SASS 硬件计数器):

  fadd  (add)     :  1,760,000 次 × 1 FLOP  =   1.76 MFLOP
  ffma  (FMA)     : 26,368,000 次 × 2 FLOPs =  52.74 MFLOP  ← 89%!
  fmul  (multiply):  4,608,000 次 × 1 FLOP  =   4.61 MFLOP
  ─────────────────────────────────────────────────────────
  总 FLOPs:                                     59.1 MFLOP

  我们估算:   16.4 MFLOP / 32.8 MB  →  AI = 0.50
  硬件实测:   59.1 MFLOP / 32.8 MB  →  AI = 1.80  (3.6x)
  
  Ridge point (f32):                  AI = 27.6
  
  1.80 << 27.6 → 严重带宽瓶颈 → 算力利用率 < 1%
```

### 为什么估算差 3.6x？

编译器 (nvcc) 做了大量优化：
- 89% 的指令是 **FMA**（一条指令 = 乘 + 加 = 2 FLOPs）
- `sqrtf()` 在硬件层面展开为多条指令
- Loop unrolling 生成额外指令

**但 AI=1.8 离 ridge=27.6 差 15 倍——结论完全不变：算力在深度睡眠，带宽决定一切。**

### 三个关键 Bug 修复

1. **`__shfl_sync` 只在 warp 内广播 (max_diff=3.24 → 7e-7)。** 256 threads = 8 warps，warp shuffle 的结果 warp 1~7 拿不到。必须写回 `shared memory` + `__syncthreads()` 才能广播给整个 block。

2. **`#define N` 会把函数参数名 N 也替换。** 宏展开是纯文本替换——`const int N` → `const int 1000000`。函数参数别和宏重名。

3. **`__shared__` 不能用 `= {0}` 初始化。** CUDA 要求 shared memory 手动赋值。

### torch.compile 对 Norm 类算子

**torch.compile 对单 kernel Norm 没用——反而慢 5.2x。** 和 reduce 一样，ATen 的 RMSNorm 本身就是融合 kernel，Inductor 生不出更好的代码。

```
torch.compile 的价值 = 融合多个 kernel，不是优化单个 kernel

  Element-wise (6 ops → 1):  8.1x faster ✅
  Reduce (already 1 kernel): 13.5x slower ❌
  RMSNorm (already 1 kernel): 5.2x slower ❌
```

### 为什么 RMSNorm 的带宽利用率只有 69%？

**写了个对照实验验证——Pure element-wise kernel（读 16 MB + 写 16 MB，零归约、零 syncthreads）：**

| Kernel | 数据量 | 带宽 | 利用率 |
|---|---|---|---|
| Pure element-wise (out=x*1.0) | 读 16 + 写 16 MB | **177.8 GB/s** | **92.6%** |
| RMSNorm | 读 16 + 写 16 MB | 133 GB/s | 69% |

**结果：同样读+写 32 MB，不加归约能到 93%。真凶不是"读写混合"，是 Phase 2 归约阶段 VRAM BUS 完全闲置 + 5 个 `__syncthreads()` 栅栏等待。**

```
RMSNorm 时间线:
  Phase 1 (读 x²)  Phase 2 (归约)  Phase 3 (读写)
  ████████████░░░░░░░░████████████████
  BUS 忙         BUS 闲!!      BUS 忙
                 ↑
          归约在 SM 内部完成, 不碰全局内存
          BUS 在这段时间就是干等着
```

**教训：Agent 给出的分析（"读写混合降低 DRAM 效率"）被写代码验证推翻了。写 benchmark 验证 > 相信任何人的推理。**

---

## RTX 3050 Ti Laptop GPU 硬件规格

**来源：** `nvidia-smi -q` + `cudaGetDeviceProperties()` + 网络搜索 [TechPowerUp](https://www.techpowerup.com/gpu-specs/geforce-rtx-3050-ti-mobile.c3782) / [WCCFTech](https://wccftech.com/nvidia-geforce-rtx-3050-ti-mobile-gpu-specs-confirmed-in-gpu-z-validation-based-on-the-ampere-ga107-gpu/)

```
┌──────────────────────────────────────────────────┐
│              GA107 芯片 (Ampere 架构)              │
│          Samsung 8nm · Compute Capability 8.6     │
│                                                    │
│  ┌──────┐ ┌──────┐         ┌──────┐ ┌──────┐     │
│  │ SM 0 │ │ SM 1 │  ... 19 │SM 18 │ │SM 19 │     │
│  │      │ │      │  个 SM  │      │ │      │     │
│  │128 KB│ │128 KB│         │128 KB│ │128 KB│     │
│  │ L1   │ │ L1   │         │ L1   │ │ L1   │     │
│  │ + ShM│ │ + ShM│         │ + ShM│ │ + ShM│     │
│  └──┬───┘ └──┬───┘         └──┬───┘ └──┬───┘     │
│     └────────┴─────────────────┴────────┘        │
│                      │                            │
│             ┌────────┴────────┐                   │
│             │   L2 Cache      │                   │
│             │     2 MB        │                   │
│             └────────┬────────┘                   │
│                      │                            │
│             ┌────────┴────────┐                   │
│             │  4 × 32-bit     │                   │
│             │  Mem Controllers │                   │
│             └────────┬────────┘                   │
└──────────────────────┼──────────────────────────┘
                       │
              ┌────────┴────────┐
              │  4 GB GDDR6     │
              │  128-bit 总线    │
              │  192 GB/s        │
              └─────────────────┘
```

### 完整规格表

| 参数 | 值 | 来源 |
|---|---|---|
| **架构** | Ampere (GA107) | `nvidia-smi -q` |
| **制程** | Samsung 8nm | TechPowerUp |
| **SM 数量** | **20** | `cudaGetDeviceProperties()` |
| **CUDA Cores** | 20 × 128 = **2,560** | Ampere spec |
| **Tensor Cores** | 20 × 4 = **80** (第 3 代) | Ampere spec |
| **L1 Cache** | **128 KB / SM**（共 2.5 MB） | Ampere spec |
| **L1 与 Shared Memory** | **共享同一块 128 KB SRAM**，carveout 可配 | Ampere spec |
| **Shared Memory** | 48 KB/block (static), 最大 100 KB/SM (dynamic) | `cudaGetDeviceProperties()` |
| **L2 Cache** | **2 MB**（2,097,152 bytes） | `cudaGetDeviceProperties()` |
| **寄存器** | 65,536 per SM, 65,536 per block (max) | `cudaGetDeviceProperties()` |
| **Warp Size** | 32 线程 | `cudaGetDeviceProperties()` |
| **最大线程/SM** | 1,536（48 warps） | `cudaGetDeviceProperties()` |
| **最大 Block/SM** | 16 | `cudaGetDeviceProperties()` |
| **Boost Clock** | **1,485 MHz** | `cudaGetDeviceProperties()` |
| **显存** | 4 GB GDDR6, **128-bit** 位宽 | `nvidia-smi` |
| **显存带宽** | **192 GB/s** | `cudaGetDeviceProperties()` |
| **峰值 FP32** | **7.60 TFLOPS** | 2560 CUDA × 2 FMA × 1.485 GHz |
| **Ridge Point** | 7,600 / 192 = **39.6 FLOP/Byte** | 计算 |
| **FP64:FP32 性能比** | **1:64**（消费级故意阉割） | NVIDIA spec |

### GPU 存储层次

```
   大小         延迟         带宽           作用域         管理方式
   ─────────────────────────────────────────────────────────────
   寄存器       ~256 KB/SM   0 cycles    极高 (~8 TB/s)  单线程       编译器自动
   Shared Mem   0~100 KB/SM  ~20 cycles  极高 (~4 TB/s)  1 个 Block   你手动 __shared__
   L1 Cache     0~128 KB/SM  ~28 cycles  极高 (~4 TB/s)  1 个 SM      硬件自动
   L2 Cache     2 MB         ~200 cycles  高 (~400 GB/s) 全 GPU       硬件自动
   HBM (显存)    4 GB         ~380 cycles  192 GB/s      全 GPU + CPU 你手动 cudaMalloc
```

**L1 和 Shared Memory 的 carveout 机制：**

- Ampere (CC 8.6) 上，L1 cache 和 shared memory **共享每 SM 128 KB 的 SRAM**
- 通过 `cudaFuncSetAttribute()` 配置分界线（支持 0/8/16/32/64/100 KB shared memory）
- 你分配的 shared memory 越多，L1 cache 越少（反之亦然）
- 默认配置下，static shared memory 最多 48 KB/block

---

## GPU Cache 验证实验

**文件:** `softmax.cu`（L2 Cache Experiment 部分）

**动机：** Softmax 的 3-phase 实现（read x for max → read x for sum → read x + write out）报告的 BW（按 3 reads + 1 write 模型算）= 247 GB/s = 128.7% of 192 GB/s 上限。这个"违反物理定律"的数字说明什么？

### 实验设计

同一个 data（1000 rows × 4096 × 4 bytes = 16.4 MB），两个 kernel 对比：

| Kernel | 做了什么 | 预期 |
|---|---|---|
| **read-3-only** | 纯读 x 3 次，零计算，零写入 | 若全走 HBM → 0.256 ms 理论下限 |
| **normal softmax** | 3 reads + 1 write + 2 reduces + exp 计算 | 应 ≥ read-3-only + 写入 + 计算开销 |

### 结果

```
Read-3-only 实测:              0.102 ms
理论下限 (3 reads @ 192 GB/s): 0.256 ms
                                    ↑
                         0.102 < 0.256 → 违反物理定律！
```

**若假设只有第 1 次读走 HBM，第 2、3 次命中 L2：**

| 假设 | 数据量 | @192 GB/s 最小时间 | 0.102 ms 可能吗? |
|---|---|---|---|
| 3 次全走 HBM | 49.2 MB | 0.256 ms | ❌ 不可能 |
| 2 次走 HBM | 32.8 MB | 0.171 ms | ❌ 不可能 |
| **只有第 1 次走 HBM** | **16.4 MB** | **0.085 ms** | ✅ **可能 (160 GB/s)** |

**结论：同一份数据在几微秒内被读 3 次，第 1 次从 HBM 搬到 L2 后，后续 2 次直接从 L2 走，没出芯片。**

### 为什么数据没被踢出 L2？

```
每个 row = 4,096 × 4 bytes = 16 KB
一轮执行 = 20 个 SM × 1 block/SM = 20 个 block 同时跑
同时活跃数据 = 20 × 16 KB = 320 KB
L2 = 2,048 KB

320 KB << 2,048 KB → 6 倍余量！所有活跃 row 数据都装得进 L2
```

加上 Phase 间的 reduce 计算（几百个 cycle）远短于 cache line 被 evict 的时间窗口，数据安静地待在 L2 里直到 Phase 2/3 再次被读。

### 什么时候 L2 帮不了忙？

如果 `HIDDEN_SIZE = 32768`（每 row = 128 KB）：

```
20 × 128 KB = 2,560 KB > 2,048 KB L2 → 装不下！
Phase 2 再读时，早期 row 的数据已被后来的 row 踢出 L2 → 重新从 HBM 读
```

---

## Softmax: Kernel 演化全记录

**文件:** `softmax.cu`, `softmax.py`, `softmax_beat.cu`, `big_sizes.cu`, `isolate.cu`, `v3v4.cu`

### 算法

```
Softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))

两种实现:
  Naive 3-phase:  读 x (找 max) → reduce → 读 x (sum exp) → reduce → 读 x (归一化写)
  Online:         读 x (在线算 max+sum) → merge-reduce → 读 x (归一化写)
```

### 在线 Softmax 算法（Online）

```c
// 在线累积 (m, d) — 一次遍历同时更新 max 和 sum
float m = -INFINITY, d = 0.0f;
for each x_i:
    if (x_i > m):
        d = d * exp(m - x_i) + 1.0f;   // 把旧 sum 调新 max 尺度 + 新元素的 exp
        m = x_i;
    else:
        d += exp(x_i - m);              // 在现有 max 尺度下累积
// 最后 d = sum(exp(x_j - m))
```

**跨线程合并 (merge_pair)：** 两组 (m_a, d_a) 和 (m_b, d_b) 合并成一组：

```c
merge( (m_a,d_a), (m_b,d_b) ):
  m = max(m_a, m_b)
  d = d_a * exp(m_a - m) + d_b * exp(m_b - m)   // 各自换算到新 max, 再求和
```

这个 merge 替换了 naive 的两步 reduce（先 max 后 sum）为一步。

### Kernel 演化：五种版本

| 版本 | 内存模型 | 算法 | 适用 hidden |
|---|---|---|---|
| **V1 (naive)** | 3R+1W | 三遍读，分别找 max、sum、归一化 | 通用 |
| **V2/MyOwn (online)** | 2R+1W | 在线 merge-pair reduce | ≥4096（通用） |
| **V3 (reg-cached)** | 1R+1W | naive 3-phase, x_reg[16] 存寄存器 | ≤4096（编译期常量） |
| **V4_32 (reg-cached)** | 1R+1W | online merge, x_reg[32] 存寄存器 | 8192（编译期常量） |
| **V4_64 (reg-cached)** | 1R+1W | online merge, x_reg[64] 存寄存器 | 16384（编译期常量） |

### 最终性能对比（1000 rows）

```
hidden     V1(naive)  MyOwn(online)  V3/V4(reg)  PyTorch   最佳 vs PyTorch
4096       0.265 ms   0.265 ms       0.182 ms     0.198 ms   V3: 1.08x ✅
8192       0.708 ms   0.550 ms       0.363 ms     0.370 ms   V4_32: 1.02x ✅
16384      1.443 ms   1.094 ms       0.727 ms     0.887 ms   V4_64: 1.22x ✅
32768      2.873 ms   2.180 ms       崩了(栈)     2.818 ms   MyOwn: 1.29x ✅
65536      5.730 ms   4.338 ms       崩了(栈)     5.677 ms   MyOwn: 1.31x ✅
```

**所有尺寸都超过了 PyTorch。**

### 核心 Insight

**1. Register Cache (1R+1W) 是最强优化——但只在数据装得下寄存器时有用。**

```
x_reg[16]  (hidden=4096):  34 regs, 0 spill, 6 blocks/SM  ← 完美
x_reg[32]  (hidden=8192):  70 regs, 0 spill, 3 blocks/SM  ← 还行
x_reg[64]  (hidden=16384): 110 regs, 0 spill, 2 blocks/SM ← 勉强
x_reg[128] (hidden=32768): 38 regs + 512B stack frame     ← 编详器放弃，全栈!
```

**"0 spill" 不意味数据在寄存器——也可能从没进过寄存器。** nvcc 判断数组太大时就分配在栈（local memory）。`x_reg[128]` 的 512 bytes stack frame 就是个例子——38 regs 全给标量，128 个 float 全在栈。栈读 = LDL 指令 = L1 → L2 → HBM，反而比 C 的明式 2R+1W 多了一倍流量。

**2. 编译期常量 vs 运行时变量——register cache 的生死线。**

```c
// ✅ 编译期常量 → 编译器完全展开 → 寄存器
const int ITEMS = 32;
float x_reg[ITEMS];  // 编译器知道有 32 个 → 分配给 32 个物理寄存器

// ❌ 运行时变量 → 编译器无法展开 → 栈
int items = hs / tt;
float x_reg[16];     // 即使只用了 16 个，动态下标 x_reg[i] 迫使编译器保守处理
for (int i = 0; i < items; i++) x_reg[i] = ...;  // 可能触发栈分配
```

**3. 大 hidden 时 online (2R+1W) 是最优解。**

hidden≥32768 时 register cache 崩了。online 的 2R+1W 是唯一方案——24 regs, 6 blocks/SM, 94% BW。merge_pair 多出的 255 次 exp 调用不随 hidden 增长，而省下的内存读随 hidden 线性放大。

**4. Nsight Compute 验证了每个关键假设。**

| 假设 | ncu 验证 |
|---|---|
| L1 在 hidden≥8192 时崩 | `L1 Hit = 0%` (6 blocks × 32 KB = 192 KB > 128 KB L1) |
| L2 能部分缓存 re-read | `L2 Hit = 33%` (C) or `25%` (V1, 1000 rows) |
| register cache 的栈读 miss L1 | `L1 Hit = 0%`, 640 KB 栈/SM >> 128 KB L1 |
| 栈拖慢调度器 | B: `Eligible Warps = 0.19/Scheduler` vs C: `0.48` |
| 栈把流量翻倍 | B: `5 ops/element`, C: `3 ops/element`, 时间比 = 1.67x |

**5. `__syncthreads()` 规则。**

任何线程 A 写了 shared memory，线程 B（B ≠ A）要读这个位置，中间必须夹一个 `__syncthreads()`。同一个 warp 内的 `__shfl_down_sync` 不需要 sync（warp 内锁步执行）。

---

### 技术工具箱（本次新增）

| 工具 | 命令 | 作用 |
|---|---|---|
| 寄存器/spill/栈 | `nvcc -Xptxas -v` | 看每个 kernel 用了多少 regs、有无 spill、栈多大 |
| SASS 反汇编 | `cuobjdump -sass binary` | 看最终机器码——真实指令数、LDS/STS 次数 |
| PTX 中间码 | `nvcc -ptx file.cu` | 看虚拟指令（跨 GPU 代际通用） |
| ncu Occupancy | `ncu --section Occupancy` | 哪个资源限制了 block 数（regs/smem/warps） |
| ncu SchedulerStats | `ncu --section SchedulerStats` | warp eligible 比例、issued 频率 |
| ncu WarpStateStats | `ncu --section WarpStateStats` | warp 为什么 stall（L1TEX/smem/barrier） |
| ncu MemoryWorkload | `ncu --section MemoryWorkloadAnalysis` | L1/L2 hit rate、Memory Throughput、Mem Pipes Busy |
| ncu cache line 级别 | `--metrics l1tex__t_sectors...` | 精确到 sector (32-byte) 的 L1/L2/HBM 计数器 |

---

## 学习路线与知识体系

```
✅ Element-wise  (hot_loop.cu)          — 完美并行, 带宽瓶颈
✅ Reduce        (reduce.cu)            — 树状归约, warp shuffle, 超过 PyTorch
✅ RMSNorm       (rms_norm.cu)          — kernel 融合: reduce + element-wise
✅ Softmax       (softmax.cu...)        — online 算法, register cache, L1/L2 验证
🔜 LayerNorm     — 两个 reduce + element-wise 融合
🔜 MatMul/GEMM   — tiling, shared memory, Tensor Core, 算力瓶颈
```

### 推理框架三层架构

```
┌─ 分布式并行策略 (TP/PP/DP) ────── NCCL, AllReduce ────┐
├─ 算子融合 (手写 fused kernel) ─── 正在学 ← 你在这里 ──┤
└─ 显存管理 (KV Cache) ──────────── CUDA memory API ────┘
```

### 核心认知

- **GPU 三种写法殊途同归** — Numba CUDA / Raw CUDA / Triton 最终都编译到同一个 SASS，天花板相同 (±2%)
- **Kernel fusion 是最重要的优化** — PyTorch eager 多个 kernel = 多次显存读写，融合后一次通过
- **AI (Arithmetic Intensity) 决定优化方向** — AI << ridge → 带宽瓶颈，代码再优化也没用；AI >> ridge → 算力瓶颈，需要改计算方式
- **专用 > 通用** — 手写专用 kernel 总能超过库函数（ATen/cuBLAS），因为库要处理通用情况
- **Register Cache (1R+1W) 是杀手锏，但有硬限制** — 数组必须是编译期常量大小，且不能超过 ~64 floats/线程。超过就崩——编译器把数组扔栈，1R+1W 变 5 ops/element
- **Nsight Compute 消除一切猜测** — 硬件计数器告诉你 L1/L2 命中率、warp 为什么 stall、内存管线忙不忙。不要推理，跑 ncu
- **L1 和 Shared Memory 是零和博弈** — 同一块 128 KB SRAM，分给 shared memory 多了 L1 就少
- **`nvcc -Xptxas -v` 是第一道检查** — "0 spill" 不代表数据全在寄存器，stack frame > 0 说明数组被扔栈上了
- **写代码验证 > 相信推理** — 几次教训：读写混合 vs reduce sync？L2 做了多少？register cache 在大 size 时为什么崩？每次都靠 ncu + benchmark 给答案

---

## GEMM: 矩阵乘法 — 从 0.45 TFLOPS 到超过 PyTorch

**文件:** `gemm/sgemm.cu`, `gemm/tensorcore_gemm.cu`, `gemm/gemm_async_wmma.cu`, `gemm/gemm_blackwell.cu`

### 第一阶段：CUDA Core FP32 (V1-V4)

| 版本 | 技术 | 1024² | 4096² | 关键发现 |
|------|------|-------|-------|----------|
| V1 | Naive 全局内存 | 0.47T | 0.44T | HBM 带宽瓶颈，每次 FMA 要 2 次 HBM 读 |
| V2 | 16×16 Shared Memory Tiling | 0.65T | 0.65T | 共享内存让数据复用 16 次 |
| V3 | 32×32 Tile + 2×2 Register Block | 1.30T | 1.17T | 寄存器缓存增加复用 |
| **V4** | **64×64 Tile + float4 向量化** | **3.57T** | **2.96T** | CUDA Core 天花板，达 cuBLAS FP32 的 78% |

### 第二阶段：Tensor Core FP16 (V1-V9)

Tensor Core 需要 FP16 输入。加转换 kernel: `f32tof16` (开销可忽略)。

| 版本 | 技术 | 1024² | 4096² | 关键发现 |
|------|------|-------|-------|----------|
| V1 (WMMA) | 1 WMMA/warp, shared mem | 2.49T | 3.22T | `__syncthreads()` 开销占主导 |
| V4 (WMMA) | 16 WMMA/warp, 128×128 tile | — | 7.23T | 提高 WMMA/sync 比值是关键 |
| V8 (WMMA) | 去掉 shared mem, warp-parallel | — | 9.86T | `__syncthreads()` 消除后大幅提升 |
| **V9 (WMMA)** | **cp.async 双缓冲 + 4 warps** | **11.66T (104% PT!)** | **13.10T (89% PT)** | **cp.async 是游戏改变者** |

**V9 是 3050 Ti 的最终版本。** cp.async 双缓冲让加载和计算重叠，流水线永不空闲。

### 核心优化技术栈

```
V1 0.45T → V9 13.1T = 29× 提升

优化层次:
  Shared Memory Tiling          → +1.4× (数据复用)
  Register Blocking             → +2×   (寄存器缓存)
  float4 向量化加载              → +1.5× (减少 load 指令)
  Tensor Core (WMMA FP16)       → +2.5× (专用硬件)
  cp.async 双缓冲                → +1.3× (load/compute 重叠)
  gencode=sm_86 JIT 优化        → +1.4× (运行时 SASS 优化)
```

### 编译参数 (关键!)

```bash
# -arch=sm_86    → 只嵌入 SASS   → 1024² = 8.5T (76% PT)
# -gencode=arch=compute_86,code=sm_86  → SASS + PTX → 1024² = 11.7T (104% PT!)
# 同时嵌入 PTX 让驱动在加载时做 JIT 优化 → +37%!
nvcc -o gemm_async gemm_async_wmma.cu \
  -gencode=arch=compute_86,code=sm_86 \
  -O3 -Xptxas -O3 --fmad=true -use_fast_math --restrict
```

### 3050 Ti 最终结果

| size | TFLOPS | % of PyTorch FP16 |
|------|--------|-------------------|
| 1024² | **11.66 T** | **104%** ✅ |
| 2048² | **12.33 T** | **92%** |
| 4096² | **13.10 T** | **89%** |

### WMMA API 的天花板

4096² 只能达到 89% PT，根本原因：
- WMMA 固定 K=16 per `mma_sync` → 4096/16 = **256 次 `__syncthreads()`**
- 每次 barrier ~50 cycles → 串行 overhead 不可消除
- 增大 K-step 到 32 需要更多共享内存 → 占用率下降 → 总体更差

cuBLAS 绕过这层的方式：**PTX `mma.sync` + split-K 并行 + 手写指令调度** — WMMA C++ API 做不到。

---

## RTX 5090 (Blackwell SM 120) — 170 SMs, 32 GB

**文件:** `gemm/gemm_blackwell.cu`

### GPU 规格

```
GPU: NVIDIA GeForce RTX 5090
Compute Capability: 12.0 (Blackwell)
SM: 170 (8.5× 3050 Ti)
Shared Memory: 48 KB default / 99 KB opt-in
Max Threads/SM: 1536
Memory: 32 GB GDDR7, ~1.79 TB/s
PyTorch: 2.8.0+cu128
```

### GEMM 结果 — V2 超过 PyTorch

| Kernel | 1024² | 2048² | 4096² | 8192² |
|--------|-------|-------|-------|-------|
| V9 port (128×128, K=16) | 28% | 89% | 79% | 90% |
| **V2 (256×128, K=32, 8w)** | 30% | **108%** ✅ | **112%** ✅ | **134%** ✅ |

**V2 设计:** 256×128 tile, 8 warps (256 threads), K=32 双缓冲, 48KB smem。
- 170 SM 上大 tile 更高效（更多并行）
- K=32 减半 sync 次数
- 1024² 需要小 tile（多 block 填满 SM）

### Flash Attention 基准 (PyTorch SDPA)

| Seqlen | Flash | Manual O(N²) | Speedup |
|--------|-------|-------------|---------|
| 1024 | 0.02ms / 100T | 0.1ms | 3.4x |
| 2048 | 0.06ms / 155T | 0.4ms | 7.4x |
| 4096 | 0.21ms / 166T | 2.0ms | 9.7x |
| 8192 | 0.51ms / 264T | 8.4ms | 10.5x |

Flash Attention 在 8192 时 TFLOPS 超过 GEMM (264T vs 217T) — FLOPs 计算方式不同。

---

## Flash Attention: 从 PyTorch 到 CUDA Kernel

**文件:** `flash_attn/flash_attn_v2.cu` (精度 ✅), `flash_attn/flash_attn_wmma.cu` (WMMA 尝试)

### PyTorch 用法 — 一行代码

```python
import torch.nn.functional as F
out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
# PyTorch 自动选后端: Flash Attn → Mem-Efficient → Math
```

### Flash Attention 算法

```
For each Q block (Br rows):
  For each K,V block (Bc rows):
    S = Q_block @ K_block^T     ← GEMM #1
    P = softmax(S)              ← online, in SRAM
    O_block += P @ V_block      ← GEMM #2
O_block = O_block / l           ← normalize at end
```

**关键:** O(N²) attention matrix **从不存在于 HBM 中** — 在 SRAM 里算完就扔。

### 自定义 CUDA Kernel 进展

| 版本 | 精度 | 速度 | 状态 |
|------|------|------|------|
| flash_attn_v2.cu | ✅ Err=0.0000 | 0.04-2.0 T (慢 50-600x) | 精度正确，待加速 |
| flash_attn_wmma.cu | ❌ | — | WMMA fragment 寄存器超限崩溃 |
| gemm_attn_v2.cu | ❌ | — | GEMM-based attention，边界越界 |

### 核心难题

**WMMA accumulator fragment 不能逐元素 rescale。** Online softmax 需要 `O *= exp(m_old - m_new)` — 但 WMMA fragment 没有 `scale()` 方法。必须把 O 存在共享内存里做 element-wise 操作，牺牲速度换正确性。

### 修复的关键 Bug

**rmax/rsum 放在线程本地栈数组 → 跨线程读取到未初始化内存。**
- 现象: O 值随机 inf, MaxErr = 数千
- 修复: 移到 `__shared__` 共享内存，`__syncthreads()` 保证可见性
- 修复后: MaxErr = 0.0000

### 共享内存溢出

Br=64 时 `Ps[64][64] (FP32)` + `Os[64][64] (FP32)` = 56KB > 48KB 默认。
5090 直接编译报错，3050 Ti 静默数据破坏。减到 Br=32 解决。

---

## 优化方法论

### GEMM 优化的关键维度

1. **WMMA per sync 比值** — 最重要的指标。V1: 1 WMMA/sync → V9: 16 WMMA/sync (16×)
2. **共享内存 vs 占用率** — 更多 smem = 更大的 tile = 更多数据复用，但减少同时驻留的 block 数
3. **cp.async 双缓冲** — 加载与计算重叠，原理类似 CPU 的 software prefetching
4. **编译参数** — `-gencode` > `-arch`，嵌入 PTX 让驱动 JIT 优化
5. **寄存器是硬上限** — 128 regs/thread，WMMA accumulator 每个 fragment = 8 regs

### Flash Attention 的额外维度

6. **在线 softmax 与 WMMA 的冲突** — fragment 不能 rescale → 必须走共享内存
7. **两遍 dot product vs 共享内存存 S** — 空间换时间，但 smem 有限
8. **FP32 vs FP16 中间存储** — P 用 half 精度导致 P@V 误差累积

### 调试武器

| 问题 | 方法 |
|------|------|
| Kernel 崩溃 | `cudaGetLastError()` 每步检查 |
| 数值错误 | Python numpy 模拟 kernel 逻辑，逐行对比 |
| 共享内存溢出 | `ptxas error: uses too much shared data` — 减 tile |
| 寄存器超限 | `--maxrregcount` 或减少 WMMA fragment 数 |
| 跨线程数据竞争 | 怀疑的数组全放 `__shared__` |

---

## 仓库结构

```
gpu-kernel-lab/
├── gemm/                GEMM kernel (CUDA Core → WMMA → Blackwell)
├── flash_attn/          Flash Attention kernel
├── python/              PyTorch benchmark + visualization
├── viz/                 GIF/PNG 可视化输出
├── build/               编译脚本 (.bat)
├── archive/             失败尝试和实验
├── GEMM_OPTIMIZATION.md  完整优化记录
└── SESSION_SUMMARY.md    最新 session 总结
```

## 学到的教训

1. **"不要假设，写代码验证"** — 用户明确要求，每次硬件相关的假设都用 benchmark 验证
2. **WSL GPU 有 ~11-26% 性能损失** — 小 seqlen 损失大，大 seqlen 可忽略
3. **Windows 原生编译环境是噩梦** — MSVC + nvcc 路径问题，WSL heredoc 写 .cu 更可靠
4. **共享内存溢出静默破坏数据** — 3050 Ti 不报错但输出全错，5090 直接编译报错
5. **"0 spill" 不代表数据在寄存器** — 大局部数组被编译器扔栈上 (local memory = DRAM)
6. **PTX `mma.sync` 是 WMMA 天花板的唯一出路** — 手动寄存器分配 + 指令级 K-pipelining
7. **GEMM 优化和 Flash Attention 优化技术栈高度重叠** — cp.async, shared memory tiling, WMMA, 寄存器管理
