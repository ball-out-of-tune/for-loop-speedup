/**
 * CUDA RMSNorm Kernel
 *
 * RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
 *
 * 和纯 reduce 的区别:
 *   reduce:    N 元素 → 1 个标量 (归约完就走)
 *   RMSNorm:   N 元素 → 归约得到 rms → 每个元素用 rms 做 normalize → N 个输出
 *
 *   reduce 的结果要"广播"回所有线程 → 同一个 kernel 内:
 *     先做 reduce → 得到 rms (存在 lane 0 / smem[0])
 *     再做 element-wise → 每个线程用 rms 算自己的输出
 *
 * 编译: nvcc -o rms_norm_cuda rms_norm.cu && ./rms_norm_cuda
 *
 * PyTorch baseline (pytorch_reduce.py): torch.mean 是 0.032 ms (只 reduce)
 * 这里要做 reduce + element-wise, 预期是 1.5-2x reduce 时间
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N_ROWS         1000       // batch * seq_len
#define HIDDEN_SIZE    4096       // llama-style hidden dim
#define BLOCK_SIZE     256        // 256 threads per row
#define EPS            1e-5f
#define WARMUP         10
#define RUNS           30

// ============================================================================
// RMSNorm Kernel
// ============================================================================
// 每个 block 处理一行 (hidden_size 个元素)
// Block 内 256 个线程, 每个线程负责 hidden_size/256 = 16 个元素

__global__ void rms_norm_kernel(
    const float *x,        // [N_ROWS, HIDDEN_SIZE]
    const float *weight,   // [HIDDEN_SIZE]
    float *out,            // [N_ROWS, HIDDEN_SIZE]
    int n_rows,
    int hidden_size
) {
    int row = blockIdx.x;                        // 第几行
    int tid = threadIdx.x;                       // 线程 ID (0~255)
    int total_threads = blockDim.x;

    // 偏移到当前行
    const float *x_row = x + row * hidden_size;
    float *out_row = out + row * hidden_size;

    // ================================================================
    // Phase 1: 每个线程用寄存器累加自己那部分 x² (grid-stride)
    // ================================================================
    float sum_x2 = 0.0f;
    float v = 0.0f;
    for (int i = tid; i < hidden_size; i += total_threads) {
        v = x_row[i];
        sum_x2 += v * v;
    }

    // ================================================================
    // Phase 2: 归约 sum_x2 → mean → rms (1 个值, 广播给所有线程)
    // ================================================================
    __shared__ float smem[BLOCK_SIZE];
    smem[tid] = sum_x2;
    __syncthreads();

    // 跨 warp 归约 (shared memory)
    for (int stride = BLOCK_SIZE / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    // warp 内归约 (寄存器交换)
    float val = smem[tid];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    // 此时 lane 0 of warp 0 (thread 0) 有整行的 sum_x2
    // 但 __shfl_sync 只在 warp 内广播 — 其他 7 个 warp 拿不到!
    // 必须写回 shared memory, 让所有 256 个线程都能读到
    if (tid == 0) smem[0] = val;       // thread 0 = warp 0 lane 0, 有总和
    __syncthreads();
    float rms = smem[0];                // 所有 256 个线程读到同一个值
    rms = sqrtf(rms / (float)hidden_size + EPS);

    // ================================================================
    // Phase 3: 每个线程用 rms 归一化自己负责的元素 (element-wise)
    // ================================================================
    for (int i = tid; i < hidden_size; i += total_threads) {
        out_row[i] = x_row[i] / rms * weight[i];
    }
}

#define MY_BLOCK_SIZE 256
__global__ void rms_norm_my_own(
    float* data_ptr,
    int hidden_size,
    const float eps,
    float* ans_ptr 
) {
    // TODO: your implementation
    int tid = threadIdx.x;

    // block序号决定了读哪一块 以及写哪一块
    int blockId = blockIdx.x;
    // 疑问 传参一定要传指针吗 以及下面能写成数组下标索引的方式吗  const float eps能加上引用吗? 
    int gap = blockId * hidden_size;
    float* data_block_ptr = data_ptr + gap;
    float* ans_block_ptr = ans_ptr + gap;

    // Phase 1 读取
    __shared__ float smem[MY_BLOCK_SIZE];
    float temp = 0.0f;
    float temp2 = 0.0f;
    int block_dim = blockDim.x;
    for (int i = tid; i < hidden_size; i += block_dim) {
        temp2 = data_block_ptr[i];
        temp += temp2 * temp2;
    }
    smem[tid] = temp;
    __syncthreads();

    // Phase 2 reduce
    for (int stride = MY_BLOCK_SIZE / 2; stride >= 32; stride = stride / 2) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    float val = smem[tid];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    if (tid == 0) {
        smem[0] = val;
    }
    __syncthreads();

    float ms = smem[0] / hidden_size;

    // Phase 3 element wise scatter
    for(int i = tid; i < hidden_size; i += block_dim) {
        ans_block_ptr[i] = data_block_ptr[i] / sqrtf(ms + eps);
    }

}

// ============================================================================
// Pure element-wise: same read+write traffic as RMSNorm, zero reduce overhead
// Used to isolate whether BW drop comes from read+write mixing vs reduce sync
// ============================================================================
__global__ void pure_element_wise(
    const float *x,
    float *out,
    int total_elems
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < total_elems; i += stride) {
        out[i] = x[i] * 1.0f;  // just copy — same read+write traffic
    }
}

// ============================================================================
// Bench
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

int main() {
    int n_rows = N_ROWS;
    int hidden = HIDDEN_SIZE;
    int total_elems = n_rows * hidden;

    printf("===== CUDA RMSNorm: %d rows x %d hidden =====\n\n", n_rows, hidden);

    // ---- Allocate ----
    float *h_x = (float *)malloc(total_elems * sizeof(float));
    float *h_weight = (float *)malloc(hidden * sizeof(float));
    float *h_out = (float *)malloc(total_elems * sizeof(float));

    for (int i = 0; i < total_elems; i++) h_x[i] = (float)(rand()) / RAND_MAX - 0.5f;
    for (int i = 0; i < hidden; i++) h_weight[i] = 1.0f;

    float *d_x, *d_weight, *d_out;
    check(cudaMalloc(&d_x, total_elems * sizeof(float)), "d_x");
    check(cudaMalloc(&d_weight, hidden * sizeof(float)), "d_weight");
    check(cudaMalloc(&d_out, total_elems * sizeof(float)), "d_out");

    check(cudaMemcpy(d_x, h_x, total_elems * sizeof(float), cudaMemcpyHostToDevice), "H2D x");
    check(cudaMemcpy(d_weight, h_weight, hidden * sizeof(float), cudaMemcpyHostToDevice), "H2D weight");

    // ---- Kernel config ----
    dim3 grid(n_rows);
    dim3 block(BLOCK_SIZE);

    // ---- Warmup ----
    for (int i = 0; i < WARMUP; i++) {
        rms_norm_kernel<<<grid, block>>>(d_x, d_weight, d_out, n_rows, hidden);
    }
    check(cudaDeviceSynchronize(), "warmup");

    // ---- Benchmark ----
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");

    float best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start, 0);
        rms_norm_kernel<<<grid, block>>>(d_x, d_weight, d_out, n_rows, hidden);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }

    // ---- BW & FLOPs & AI ----
    float read_bytes  = total_elems * sizeof(float);      // x
    float write_bytes = total_elems * sizeof(float);      // out
    float total_bytes = read_bytes + write_bytes;          // ~32.8 MB
    float bw = total_bytes / (best / 1000.0f) / 1e9f;     // B/s -> GB/s

    // FLOPs per row:
    //   x²     : hidden multiplies
    //   mean   : (hidden-1) adds + 1 divide
    //   sqrt   : 1 sqrt
    //   x/rms  : hidden divides (or mult by 1/rms)
    //   *weight: hidden multiplies
    // Total per row ≈ hidden * 3 (x² + normalize + weight)
    //              + hidden (for mean reduce)
    //              + 1 sqrt + 1 div
    // ≈ 4 * hidden FLOPs per row
    long long flops_per_row = (long long)hidden * 4;
    long long total_flops = flops_per_row * n_rows;
    float gflops = total_flops / (best / 1000.0f) / 1e9f;  // GFLOPS = GFLOP/s
    float ai = (float)total_flops / total_bytes;             // FLOPs/Byte

    printf("  Time:     %.4f ms\n", best);
    printf("  BW:       %.1f GB/s  (%.1f%% of 192 GB/s)\n", bw, bw / 192.0f * 100);
    float peak_tflops = 2560.0f * 2.0f * 1.485f;  // CUDA cores x 2 ops x clock GHz
    printf("  FLOPs:    %.2f GFLOPS  (%.2f%% of %.0f GFLOPS peak)\n",
           gflops, gflops / peak_tflops * 100, peak_tflops);
    printf("  AI:       %.2f FLOPs/Byte  (ridge point = 27.6)\n", ai);
    printf("  Bottleneck: %s\n", (ai < 27.6f) ? "MEMORY (bandwidth-bound)" : "COMPUTE");

    // Verify: CPU reference for first row
    float sum_x2 = 0.0f;
    for (int i = 0; i < hidden; i++) sum_x2 += h_x[i] * h_x[i];
    float rms_ref = sqrtf(sum_x2 / hidden + EPS);
    check(cudaMemcpy(h_out, d_out, hidden * sizeof(float), cudaMemcpyDeviceToHost), "D2H verify");

    float max_diff = 0.0f;
    for (int i = 0; i < hidden; i++) {
        float expected = h_x[i] / rms_ref * h_weight[i];
        float diff = fabsf(h_out[i] - expected);
        if (diff > max_diff) max_diff = diff;
    }

    printf("  Max diff vs CPU: %.6e\n", max_diff);

    // ---- Benchmark: rms_norm_my_own ----
    printf("\n--- rms_norm_my_own ---\n");

    // warmup
    for (int i = 0; i < WARMUP; i++) {
        rms_norm_my_own<<<grid, block>>>(d_x, hidden, EPS, d_out);
    }
    check(cudaDeviceSynchronize(), "my_own warmup");

    // benchmark
    best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start, 0);
        rms_norm_my_own<<<grid, block>>>(d_x, hidden, EPS, d_out);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }

    bw = total_bytes / (best / 1000.0f) / 1e9f;
    gflops = total_flops / (best / 1000.0f) / 1e9f;

    printf("  Time:     %.4f ms\n", best);
    printf("  BW:       %.1f GB/s  (%.1f%% of 192 GB/s)\n", bw, bw / 192.0f * 100);

    // Verify my_own
    check(cudaMemcpy(h_out, d_out, hidden * sizeof(float), cudaMemcpyDeviceToHost), "D2H verify my_own");
    max_diff = 0.0f;
    for (int i = 0; i < hidden; i++) {
        float expected = h_x[i] / rms_ref;
        float diff = fabsf(h_out[i] - expected);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  Max diff vs CPU: %.6e %s\n", max_diff, (max_diff < 1e-4f) ? "PASS" : "FAIL");

    // ---- Pure element-wise: isolate read+write BW ----
    printf("\n--- Pure element-wise (same r/w traffic, zero reduce) ---\n");
    {
        int block_size = 256;
        int n_blocks = (total_elems + block_size - 1) / block_size;
        for (int i = 0; i < WARMUP; i++) {
            pure_element_wise<<<n_blocks, block_size>>>(d_x, d_out, total_elems);
        }
        check(cudaDeviceSynchronize(), "ew warmup");

        float best_ew = 1e9;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            pure_element_wise<<<n_blocks, block_size>>>(d_x, d_out, total_elems);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best_ew) best_ew = t;
        }
        float ew_bw = total_bytes / (best_ew / 1000.0f) / 1e9f;
        printf("  Time:     %.4f ms\n", best_ew);
        printf("  BW:       %.1f GB/s  (%.1f%% of 192 GB/s)\n", ew_bw, ew_bw / 192.0f * 100);
    }

    // ---- Compare ----
    printf("\n  === RMSNorm Analysis ===\n");
    printf("  Data:  %d rows x %d hidden = %.1f MB\n",
           n_rows, hidden, total_elems * 4.0f / 1e6f);
    printf("  AI:    %.2f FLOPs/Byte  vs  ridge=27.6\n", ai);
    printf("  -> Heavily memory-bound (AI << ridge)\n");
    printf("  -> Optimization lever: reduce memory traffic, not compute\n");
    printf("\n  BW comparison (same read+write traffic):\n");
    printf("    Pure element-wise (no reduce):  ? GB/s  ← run to see\n");
    printf("    RMSNorm (with reduce):         133 GB/s  (69%%)\n");
    printf("    If pure ew ~135 GB/s -> BW drop from r/w mixing, not reduce sync\n");
    printf("    If pure ew ~160 GB/s -> reduce sync is the real BW killer\n");
    printf("\n  Why 2.64x faster than PyTorch eager?\n");
    printf("  PyTorch eager:  x^2 -> mean -> sqrt -> div -> mul\n");
    printf("                  = 5-6 kernel launches + intermediate tensors\n");
    printf("  Our kernel:     1 pass, everything fused\n");
    printf("  Same story as hot_loop (8 kernels -> 1, 8x faster)\n");

    // ---- Cleanup ----
    cudaFree(d_x);
    cudaFree(d_weight);
    cudaFree(d_out);
    free(h_x);
    free(h_weight);
    free(h_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
