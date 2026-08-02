/**
 * CUDA Reduce — 追平 cuBLAS 的优化版本
 *
 * 编译: nvcc -o reduce_cuda reduce.cu && ./reduce_cuda
 *
 * 优化路径:
 *   V2: 每个 thread 处理 1 个元素, 3907 blocks → 大量全局写 + CPU 归约
 *   V5: 每个 thread 处理 N 个元素 (grid-stride loop), fewer blocks
 *   V6: + float4 向量化加载 (128-bit 事务)
 *   V7: block size sweep (128/256/512)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N            1000000
#define BLOCK_SIZE   256
#define WARMUP       10
#define RUNS         30

// ============================================================================
// V2 (baseline, 之前的实现)
// ============================================================================
__global__ void reduce_v2_warpshuffle(const float *data, float *partial_sums, int n) {
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? data[gid] : 0.0f;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    float val = smem[tid];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    if (tid == 0) partial_sums[blockIdx.x] = val;
}


// ============================================================================
// V5: Grid-stride loop — 每 thread 多处理几个元素, 减少 block 数
// ============================================================================
// 核心思想:
//   之前: 1M 元素 = 3907 blocks × 256 threads × 1 elem/thread
//   现在: 1M 元素 = 245  blocks × 256 threads × 16 elem/thread
//
//   好处 1: 只有 245 个 partial_sums (不是 3907)
//   好处 2: 每个 thread 的寄存器累加代替 shared memory
//   好处 3: 245 个 atomicAdd 竞争可控 (不是 3907)
//
//   grid-stride loop 的热循环模式:
//     for (i = gid; i < N; i += total_threads)
//       累加 data[i]
//
//   这是 GPU 上最经典的"对任意 N 都能跑"的模式

__global__ void reduce_v5_gridstride(const float *data, float *result, int n) {
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // ---- 每个 thread 用寄存器累加多个元素 ----
    float my_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += total_threads) {
        my_sum += data[i];  // 寄存器累加, 不写 shared memory
    }

    // ---- 现在把 256 个寄存器值归约到 1 个 ----
    smem[tid] = my_sum;
    __syncthreads();

    // cross-warp reduce
    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    // in-warp reduce
    float val = smem[tid];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    // 最终写入: 245 个 block 竞争同一个地址 → atomicAdd
    if (tid == 0) atomicAdd(result, val);
}

__global__ void reduce_my_own(const float* data, float* ans, int n) {
    int tid = threadIdx.x;
    int global_id = blockDim.x * blockIdx.x + tid;

    __shared__  float block_level_sum[256];
    float my_sum = 0.0f;
    for (int i = global_id; i < n; i += blockDim.x * gridDim.x) {
        my_sum += data[i];
    }
    block_level_sum[tid] = my_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 32; stride = stride / 2) {
        if (tid < stride) {
            block_level_sum[tid] += block_level_sum[tid + stride];
        }
        __syncthreads();
    }

    float val = block_level_sum[tid];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    if (tid == 0) {
        atomicAdd(ans, val);
    }
}

// ============================================================================
// V6: + float4 向量化加载
// ============================================================================
// GPU 内存总线是 128-bit 宽的 (从 Kepler 开始)
//   单独读 1 个 float (32-bit): 用掉 128-bit 总线中的 25%, 浪费 75%
//   float4 (128-bit): 一次事务, 满载
//
// 注意: 对 reduce, 我们不需要 float4 来"合并访问" —
//       相邻线程读相邻地址天然会合并 (coalesced).
//       float4 的价值是让每个线程多干 4 倍的活, 进一步减少 block 数.
//
// reinterpret_cast<const float4*>(data)[i4]:
//   把 data 指针强转为 float4*, 一次读 4 个 float

__global__ void reduce_v6_float4(const float *data, float *result, int n) {
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;
    int n4 = n / 4;  // float4 的元素数

    float my_sum = 0.0f;

    // ---- float4 向量化 grid-stride loop ----
    int i4 = blockIdx.x * blockDim.x + tid;
    for (; i4 < n4; i4 += total_threads) {
        float4 v = reinterpret_cast<const float4*>(data)[i4];
        my_sum += v.x + v.y + v.z + v.w;
    }

    // ---- tail: 处理不足 float4 的剩余元素 ----
    for (int i = i4 * 4; i < n; i++) {
        my_sum += data[i];
    }

    // ---- reduce (same as V5) ----
    smem[tid] = my_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    float val = smem[tid];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    if (tid == 0) atomicAdd(result, val);
}


// ============================================================================
// Bench tools
// ============================================================================
float millis(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}
void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

float bench_kernel(
    void (*kernel)(const float*, float*, int),
    const float *d_data, float *d_result, int n,
    int blocks, int threads,
    const char *label
) {
    float zero = 0.0f;

    // warmup
    for (int i = 0; i < WARMUP; i++) {
        check(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice), "reset");
        kernel<<<blocks, threads>>>(d_data, d_result, n);
    }
    check(cudaDeviceSynchronize(), "warmup");

    // benchmark
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");

    float best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        check(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice), "reset");
        cudaEventRecord(start, 0);
        kernel<<<blocks, threads>>>(d_data, d_result, n);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }
    (void)zero;  // suppress unused warning

    // read result
    float result;
    check(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost), "D2H");

    // BW calculation (decimal GB):
    //   data_size_GB = N * 4 bytes / 1e9 bytes/GB
    //   t_s = best / 1000 ms/s
    //   BW = data_size_GB / t_s
    float data_size_gb = (float)n * sizeof(float) / 1e9f;
    float bw = data_size_gb / (best / 1000.0f);
    printf("  %-25s  %.4f ms  |  %6.1f GB/s  (%5.1f%%)  |  result=%.1f\n",
           label, best, bw, bw / 192.0f * 100, result);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return best;
}


int main() {
    printf("===== CUDA Reduce: chasing cuBLAS =====\n");
    printf("  Target:  0.032 ms  (125 GB/s, cuBLAS)\n");
    printf("  Ceiling: 0.021 ms  (192 GB/s, physics)\n\n");

    // CPU data
    float *h_data = (float *)malloc(N * sizeof(float));
    float expected = 0.0f;
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i * 0.001f;
        expected += h_data[i];
    }
    printf("  CPU reference: %.1f\n\n", expected);

    // GPU memory
    float *d_data, *d_result, *d_partial;
    check(cudaMalloc(&d_data, N * sizeof(float)), "d_data");
    check(cudaMalloc(&d_result, sizeof(float)), "d_result");
    check(cudaMalloc(&d_partial, ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * sizeof(float)), "d_partial");
    check(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice), "H2D");

    // ---- V2 (baseline: 3907 blocks, CPU final reduce) ----
    printf("--- V2 baseline (1 elem/thread, CPU final reduce) ---\n");
    {
        int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        float zero = 0.0f;
        for (int i = 0; i < WARMUP; i++) {
            reduce_v2_warpshuffle<<<blocks, BLOCK_SIZE>>>(d_data, d_partial, N);
        }
        check(cudaDeviceSynchronize(), "v2 warmup");

        cudaEvent_t start, stop;
        check(cudaEventCreate(&start), "event");
        check(cudaEventCreate(&stop), "event");
        float best = 1e9;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            reduce_v2_warpshuffle<<<blocks, BLOCK_SIZE>>>(d_data, d_partial, N);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best) best = t;
        }
        float *h_partial = (float *)malloc(blocks * sizeof(float));
        check(cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost), "D2H");
        float sum = 0;
        for (int i = 0; i < blocks; i++) sum += h_partial[i];
        float data_size_gb = (float)N * sizeof(float) / 1e9f;
        float bw = data_size_gb / (best / 1000.0f);
        printf("  V2 shared+warp+CPU        %.4f ms  |  %6.1f GB/s  (%5.1f%%)  |  result=%.1f\n",
               best, bw, bw / 192.0f * 100, sum);
        free(h_partial);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // ---- V5: grid-stride, less blocks ----
    printf("\n--- V5 grid-stride loop (16 elem/thread -> 245 blocks) ---\n");
    {
        int items_per_thread = 16;
        int blocks = (N + BLOCK_SIZE * items_per_thread - 1) / (BLOCK_SIZE * items_per_thread);
        printf("  grid: %d blocks x %d threads, ~%d elem/thread\n", blocks, BLOCK_SIZE, items_per_thread);
        bench_kernel(reduce_v5_gridstride, d_data, d_result, N, blocks, BLOCK_SIZE, "V5 grid-stride+shfl");
    }

    // ---- V5 sweep: try different items_per_thread ----
    printf("\n--- V5 sweep: tuning items_per_thread ---\n");
    for (int items = 4; items <= 64; items *= 2) {
        int blocks = (N + BLOCK_SIZE * items - 1) / (BLOCK_SIZE * items);
        char label[64];
        snprintf(label, sizeof(label), "V5 items=%d (%d blocks)", items, blocks);
        bench_kernel(reduce_v5_gridstride, d_data, d_result, N, blocks, BLOCK_SIZE, label);
    }

    // ---- V6: float4 ----
    printf("\n--- V6 float4 vectorized loads ---\n");
    {
        int items_per_thread = 16;
        int blocks = (N + BLOCK_SIZE * items_per_thread - 1) / (BLOCK_SIZE * items_per_thread);
        printf("  grid: %d blocks x %d threads\n", blocks, BLOCK_SIZE);
        bench_kernel(reduce_v6_float4, d_data, d_result, N, blocks, BLOCK_SIZE, "V6 float4 grid-stride");
    }

    // ---- Block size sweep ----
    printf("\n--- Block size sweep (V5, items=16) ---\n");
    for (int bs = 128; bs <= 512; bs *= 2) {  // skip 1024 (hits register/shared mem limit on some GPUs)
        int items = 16;
        int blocks = (N + bs * items - 1) / (bs * items);
        char label[64];
        snprintf(label, sizeof(label), "BLOCK=%d (%d blocks)", bs, blocks);
        bench_kernel(reduce_v5_gridstride, d_data, d_result, N, blocks, bs, label);
    }

    // ---- My own: reduce_my_own ----
    printf("\n--- reduce_my_own ---\n");
    {
        int items_per_thread = 64;
        int blocks = (N + BLOCK_SIZE * items_per_thread - 1) / (BLOCK_SIZE * items_per_thread);
        bench_kernel(reduce_my_own, d_data, d_result, N, blocks, BLOCK_SIZE, "reduce_my_own");
    }

    // ---- Best vs target ----
    printf("\n===========================================\n");
    printf("  Target:  cuBLAS (PyTorch eager)  0.032 ms  |  125 GB/s\n");
    printf("  Ceiling: physics                 0.021 ms  |  192 GB/s\n");
    printf("===========================================\n");

    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_partial);
    free(h_data);
    return 0;
}
