# GEMM 优化全记录：从 0.45 TFLOPS 到 11.66 TFLOPS

## 硬件环境
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop (GA107, 20 SMs, 2560 CUDA Cores, 80 Tensor Cores)
- FP32 峰值: ~8.2 TFLOPS (CUDA Core), FP16 峰值: ~30 TFLOPS (Tensor Core)
- 显存带宽: 192 GB/s, Shared Memory: 48 KB/SM (默认), L1: 128 KB/SM

---

## 第一阶段：CUDA Core (FP32)

### V1: Naive 全局内存
```c
// 每个线程一个输出元素，直接从 HBM 读 A 和 B
for (int k = 0; k < K; k++)
    sum += A[row * K + k] * B[k * N + col];
```
| size | TFLOPS | % of cuBLAS FP32 |
|------|--------|-------------------|
| 512² | 0.45 T | 13% |
| 1024² | 0.47 T | 10% |
| 2048² | 0.49 T | 10% |
| 4096² | 0.44 T | 11% |

**瓶颈**: HBM 带宽。每个 FMA 需要 2 次 HBM 读取, 算术强度 ≈ 0.25 FLOPs/byte。

### V2: 16×16 Shared Memory Tiling
```c
// As[16][16], Bs[16][16] 缓存到 shared memory
// 沿 K 方向滑动 tile, 每个元素复用 16 次
```
| size | TFLOPS | vs V1 |
|------|--------|-------|
| 512² | 0.60 T | 1.3× |
| 1024² | 0.65 T | 1.4× |
| 2048² | 0.67 T | 1.4× |
| 4096² | 0.65 T | 1.5× |

### V3: 32×32 Tile + 2×2 Register Blocking
```c
// 16×16 线程, 每线程算 2×2=4 个输出（4 个累加器）
// 每个共享内存元素被复用更多次
```
| size | TFLOPS | vs V1 |
|------|--------|-------|
| 1024² | 1.30 T | 2.8× |
| 4096² | 1.17 T | 2.7× |

### V4: 64×64 Tile + float4 向量化加载
```c
// float4 一次加载 4 个 float, 减少 load 指令数
// 64×64 tile: 每个 shared memory 元素复用 64 次
```
| size | TFLOPS | vs V1 |
|------|--------|-------|
| 1024² | 3.57 T | 7.6× |
| 2048² | 3.60 T | 7.3× |
| 4096² | 2.96 T | 6.7× |

**V4 是 CUDA Core 的天花板** —— 达到 cuBLAS FP32 的 70-78%。

### 尝试过但失败的优化:
- **As 转置** (减少 smem 读取): 2.4T —— scatter 存储开销 > 收益
- **转置 + padding** (消除 bank conflict): 2.4T —— 没改善
- **32×32 高 occupancy**: 2.0T —— 数据复用比 occupancy 更重要
- **__ldg() 只读缓存**: 无变化

---

## 第二阶段：Tensor Core (FP16)

Tensor Core 需要 FP16 输入。加一个转换 kernel 把 FP32 → FP16（耗时可忽略）。

### V1 (WMMA): Naive — 1 WMMA/warp/K-step
```c
// 1 个 warp 算一个 16×16 输出
// wmma::load_matrix_sync → mma_sync → store
// 每 K-step: 1 次 WMMA = 4096 FMA
```
| size | TFLOPS | % of PyTorch FP16 |
|------|--------|-------------------|
| 1024² | 2.49 T | 22% |
| 4096² | 3.22 T | 22% |

**瓶颈**: __syncthreads() 开销。每个 K-step 2 次 sync，只有 1 个 WMMA 在中间。

### V2: 2 WMMA/warp/K-step (64×64 tile)
```c
// 8 warps, 每个 warp 覆盖 16×32 输出 = 2 个 WMMA
```
| size | TFLOPS |
|------|--------|
| 1024² | 3.21 T |
| 4096² | 4.05 T |

### V3: 4 WMMA/warp/K-step (128×64 tile)
```c
// 8 warps, 每个 warp 覆盖 16×64 输出 = 4 个 WMMA
```
| size | TFLOPS |
|------|--------|
| 1024² | 3.91 T |
| 4096² | 5.08 T |

### V4: 16 WMMA/warp/K-step (128×128 tile + smem)
```c
// 4 warps, 每个 warp 覆盖 64×64 输出 = 16 个 WMMA
// Shared memory: As[128][16] + Bs[16][128] = 8KB
```
| size | TFLOPS |
|------|--------|
| 2048² | 6.50 T |
| 4096² | 7.23 T |

### V8: 去掉 shared memory — warp-parallel
```c
// 4 warps, 无 smem, 无 __syncthreads()
// 每个 warp 直接从 HBM 加载 WMMA fragment
// L1/L2 cache 自动处理数据复用
```
| size | TFLOPS |
|------|--------|
| 2048² | 8.81 T |
| 4096² | 9.86 T |

**突破**: 去掉 __syncthreads() 后 2048² 从 6.5T → 8.8T。但 HBM 带宽浪费（每个 warp 独立加载重叠数据）。

### V9: cp.async 双缓冲 (最终版本)
```c
// 16 字节 cp.async 异步拷贝 → 双缓冲 shared memory
// 加载下一个 tile 的同时 WMMA 计算当前 tile
// 4 warps, 128×128 tile, 16 WMMA/warp, 8KB smem × 2
//
// 流水线:
//   cp.async tile N+1 → commit_group
//   WMMA compute tile N (同时 cp.async 在后台传输)
//   wait_group 0
//   __syncthreads()
//   交换缓冲区
```
| size | TFLOPS | % of PyTorch FP16 |
|------|--------|-------------------|
| 1024² | **11.66 T** | **104% ← 超过 PyTorch!** |
| 2048² | **12.33 T** | **92%** |
| 4096² | **13.10 T** | **89%** |

### 编译参数 (关键!)
```bash
nvcc -o gemm_async.exe gemm_async_wmma.cu \
  -gencode=arch=compute_86,code=sm_86 \  # 同时嵌入 SASS + PTX, 启用运行时 JIT
  -O3 -Xptxas -O3 --fmad=true -use_fast_math --restrict
```
`-gencode` 比 `-arch` 更快 —— 驱动在加载时做 PTX→SASS JIT 优化。

---

## 最终代码: gemm_async_wmma.cu

```c
// V9: async double-buffered WMMA, 4 warps (128 threads), 128x128 tile
__global__ void gemm_async_wmma(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64;
    int c0 = blockIdx.x * 128 + wx * 64;

    __shared__ half A_buf[2][128][16];
    __shared__ half B_buf[2][16][128];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    // Prefetch buf 0 (synchronous)
    // ... short2* loads ...

    __syncthreads();
    int read_buf = 0;

    for (int kb = 16; kb < K; kb += 16) {   // K 方向滑动
        int write_buf = 1 - read_buf;

        // === 异步加载下一个 tile (16 字节 cp.async) ===
        for (int chunk = threadIdx.x; chunk < 128*16/8; chunk += 128) {
            // ... cp.async.ca.shared.global 16 字节 ...
        }
        for (int chunk = threadIdx.x; chunk < 16*128/8; chunk += 128) {
            // ... cp.async.ca.shared.global 16 字节 ...
        }
        asm volatile("cp.async.commit_group;\n" ::);

        // === WMMA 计算当前 tile (同时异步加载在后台跑) ===
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy*64 + mi*16][0], 16);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx*64 + ni*16], 128);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);

        // === 等异步加载完成, 交换缓冲区 ===
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        read_buf = write_buf;
    }
    // Last tile + store ...
}
```

---

## 关键技术洞察

### 1. WMMA per sync 是最重要的指标
```
V1: 1 WMMA / sync  → 2.5T
V2: 2 WMMA / sync  → 3.2T
V3: 4 WMMA / sync  → 5.0T
V4: 16 WMMA / sync → 7.2T
```
每个 `__syncthreads()` 有 ~50 cycle 开销。提高 WMMA/sync 比值是核心优化方向。

### 2. cp.async 消除了加载延迟
```
无 cp.async:  load → sync → compute → sync  (串行)
有 cp.async:  load ──→ sync → compute ──→ sync
              └─ async ────────────→ wait ─┘  (加载和计算重叠)
```

### 3. 寄存器是硬上限
- 16 个 WMMA 累加器 fragment = 128 寄存器 —— V9 刚好不 spill
- 20 个 fragment (V11) → ~160 寄存器, 编译通过但启动失败 (LaunchOutOfResources)
- 25 个 fragment (V10) → ~200 寄存器, 编译通过但启动失败

### 4. 编译参数很重要
- `-arch=sm_86`: 只嵌入 SASS, 无 PTX → 无 JIT 优化
- `-gencode=arch=compute_86,code=sm_86`: 同时嵌入 SASS + PTX → 驱动 JIT 优化 → 1024² 从 8.5T → 11.7T

### 5. CUDA C++ 的天花板
- FP32 CUDA Core: V4 = 3.6T = 78% of cuBLAS
- FP16 Tensor Core: V9 = 11.7T = 104% of PyTorch (1024²), 89-92% (2048², 4096²)
- 剩余差距来自 cuBLAS 手写 SASS 汇编: 完美指令调度 + L1 预取 + 架构特化 tile 尺寸

---

## 第三阶段：V9 继续优化 (2026-08-08)

### 目标
在 V9 基础上继续优化，目标在 1024²、2048²、4096² 三个测试点全部超过 PyTorch。

### 关键发现

#### 1. WSL GPU 穿透有 ~11-26% 性能损失
WSL GPU passthrough 在小 seqlen 开销大 (26%)，大 seqlen 开销小 (~2%)：
| size | Windows V9 | WSL V9 | 开销 |
|------|-----------|--------|------|
| 1024² | 11.66T | 9.27T | 26% |
| 2048² | 12.33T | 10.99T | 12% |
| 4096² | 13.10T | 12.89T | 2% |

**结论：需要在 Windows 原生环境才能看到真实性能。** WSL 可用于相对比较。

#### 2. cp.async 双缓冲是不可替代的
V11 (K=32 单缓冲，不用 cp.async) 性能暴跌到 50%：
- V9 (cp.async 双缓冲): 12.89T at 4096²
- V11 (无 cp.async): 7.34T at 4096² → 仅 57%
- cp.async 的 load/compute overlap 绝对值 ~40% 性能提升

#### 3. WMMA load + MMA 交错调度 (V14_interleave)
将原来的"加载所有 A → 加载所有 B → 所有 MMA" 改为"加载 A → 每个 B：加载 + MMA 交错"：
| size | V9 | V14_interleave | 提升 |
|------|-----|----------------|------|
| 2048² | 82% | 92% | +10% |
| 4096² | 87% | 86% | -1% |

在 2048² 表现好（ILP 提升），4096² 基本不变。

#### 4. 三重缓冲 + 交错调度 (V15_hybrid)
24KB 共享内存，3 阶段 cp.async pipeline + 交错 WMMA：
| size | V9 | V15_hybrid | 提升 |
|------|-----|-----------|------|
| 2048² | 82% | 95% | +13% |
| 4096² | 87% | 85% | -2% |

在 WSL 中达到 2048² 的 95%。**估算 Windows 原生：95% × (1+12%) ≈ 106%，应该超过 PyTorch！**

#### 5. 4096² 的根本瓶颈
4096² 有 K=4096，WMMA 的固定 K=16 意味着 256 个 K-step，每个都有 `__syncthreads()` 开销（~50 cycles × 256 = 12,800 cycles per block）。

尝试的方案：
- **K=32 双缓冲** (V10): +3.6% at 2048², -6% at 4096²（32KB smem → 1 block/SM，占用率下降）
- **三重缓冲** (V13): +2% at 4096²（24KB smem → 2 blocks/SM，比 K=32 好但还不够）
- **交错调度** (V14): 无显著差异
- **混合方案** (V15): -2% at 4096²

**4096² 天花板：WMMA API 层面。** cuBLAS 使用手动 PTX `mma.sync` + 完美指令调度 + split-K 并行才能达到更高利用率。

### 最终结果 (WSL 测量，Windows 原生会更高)

| size | 最佳 Kernel | WSL %PT | 估算 Windows |
|------|------------|---------|-------------|
| 1024² | V9 | 82% | **104%** ✓ |
| 2048² | V15_hybrid | 95% | **~106%** ← 待 Windows 验证 |
| 4096² | V9 | 87% | **~89%** |

### Flash Attention CUDA Kernel
基于 GEMM 优化经验编写了 Flash Attention V2 Forward kernel (`flash_attn_fwd.cu`)：
- 使用 WMMA FP16 Tensor Core 计算 S=Q@K^T 和 O+=P@V
- cp.async 双缓冲 K,V tile 加载
- 2 warps, Br=64, Bc=64, d=64

**已知限制：** WMMA accumulator fragment 无法进行逐元素 rescaling（online softmax 需要）。要正确实现需要将 O 存到共享内存中逐元素操作。这对于生产级 Flash Attention kernel 是正确的做法但需要更多开发时间。

---

## 文件清单
| 文件 | 内容 |
|------|------|
| `sgemm.cu` | V1-V4: CUDA Core FP32 全系列 |
| `gemm_interview.cu` | 面试版 SGEMM (32×32 tile, 2×2 register blocking) |
| `tensorcore_gemm.cu` | V1-V8: FP16 WMMA 全系列 |
| `gemm_async_wmma.cu` | **V9: 最终版本**, cp.async + WMMA |
| `gemm_v10.cu` | V10: 160×160 tile (寄存器溢出, 启动失败) |
| `gemm_torch.py` | PyTorch GEMM baseline |
| `GEMM_OPTIMIZATION.md` | 本文档 |
