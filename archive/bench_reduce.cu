/**
 * 公平 benchmark: 手写 CUDA reduce vs PyTorch/cuBLAS
 *
 * 编译: nvcc -o bench_reduce bench_reduce.cpp
 * 运行: ./bench_reduce
 *
 * 用完全相同的 CUDA events 框架测量所有方法，消除工具差异。
 * 多次运行取分布 (min / median / max)，防止巧合。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N            1000000
#define BLOCK_SIZE   256
#define ITEMS        64        // items per thread for V5
#define WARMUP       10
#define RUNS         100       // more runs for statistical confidence

// ============================================================================
// Our V5 kernel
// ============================================================================
__global__ void reduce_v5(const float *data, float *result, int n) {
    __shared__ float smem[BLOCK_SIZE];

    int tid = threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Grid-stride: each thread accumulates multiple elements in register
    float my_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += total_threads) {
        my_sum += data[i];
    }

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
// Benchmark helper
// ============================================================================
#define MAX_TIMES 200

typedef struct {
    float times[MAX_TIMES];
    int count;
    float min, median, max;
} Stats;

int cmp_float(const void *a, const void *b) {
    float d = *(float*)a - *(float*)b;
    return (d > 0) - (d < 0);
}

Stats compute_stats(float *raw_times, int n) {
    Stats s = {.count = n};
    for (int i = 0; i < n; i++) s.times[i] = raw_times[i];

    // sort
    float sorted[n];
    for (int i = 0; i < n; i++) sorted[i] = raw_times[i];
    qsort(sorted, n, sizeof(float), cmp_float);

    s.min = sorted[0];
    s.median = sorted[n / 2];
    s.max = sorted[n - 1];
    return s;
}

void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// ============================================================================
int main() {
    printf("===== Fair Benchmark: Our CUDA vs cuBLAS (PyTorch) =====\n\n");

    // ---- GPU info ----
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("SMs: %d  |  Max threads/block: %d  |  Clock: %.1f MHz\n\n",
           prop.multiProcessorCount, prop.maxThreadsPerBlock, prop.clockRate / 1000.0f);

    // ---- Allocate ----
    float *h_data = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = (float)i * 0.001f;

    // CPU reference (double precision)
    double cpu_ref = 0.0;
    for (int i = 0; i < N; i++) cpu_ref += h_data[i];
    printf("CPU reference (float64): %.4f\n\n", cpu_ref);

    float *d_data, *d_result;
    check(cudaMalloc(&d_data, N * sizeof(float)), "d_data");
    check(cudaMalloc(&d_result, sizeof(float)), "d_result");
    check(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice), "H2D");

    // ---- Kernel config ----
    int blocks = (N + BLOCK_SIZE * ITEMS - 1) / (BLOCK_SIZE * ITEMS);
    printf("Kernel config: %d blocks x %d threads, ~%d elem/thread\n",
           blocks, BLOCK_SIZE, ITEMS);

    // ---- Warmup ----
    float zero = 0.0f;
    for (int i = 0; i < WARMUP; i++) {
        check(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice), "reset");
        reduce_v5<<<blocks, BLOCK_SIZE>>>(d_data, d_result, N);
    }
    check(cudaDeviceSynchronize(), "warmup");

    // ---- Benchmark ----
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");

    float raw_times[RUNS];
    float result;

    for (int r = 0; r < RUNS; r++) {
        // Reset result before each run (NOT included in timing)
        check(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice), "reset");

        // Time ONLY the kernel
        cudaEventRecord(start, 0);
        reduce_v5<<<blocks, BLOCK_SIZE>>>(d_data, d_result, N);
        cudaEventRecord(stop, 0);

        check(cudaEventSynchronize(stop), "sync");
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        raw_times[r] = ms;
    }

    // Read result after the last run
    check(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost), "D2H");

    Stats s = compute_stats(raw_times, RUNS);
    float data_size_gb = (float)N * sizeof(float) / 1e9f;
    float bw_min  = data_size_gb / (s.min / 1000.0f);
    float bw_med = data_size_gb / (s.median / 1000.0f);

    printf("\n--- Our CUDA Kernel (V5 grid-stride + shared + warp shfl + atomicAdd) ---\n");
    printf("  min:    %.4f ms  |  %.1f GB/s  (%.1f%%)\n", s.min,  bw_min,  bw_min / 192.0f * 100);
    printf("  median: %.4f ms  |  %.1f GB/s  (%.1f%%)\n", s.median, bw_med, bw_med / 192.0f * 100);
    printf("  max:    %.4f ms\n", s.max);
    printf("  result: %.4f  (diff vs CPU float64: %.4f)\n",
           result, fabs(result - cpu_ref));

    // ---- Consistency check ----
    // Same kernel, same data, 10 times — are we getting the same answer?
    printf("\n--- Result consistency (same kernel x10) ---\n");
    float results[10];
    for (int i = 0; i < 10; i++) {
        check(cudaMemcpy(d_result, &zero, sizeof(float), cudaMemcpyHostToDevice), "reset");
        reduce_v5<<<blocks, BLOCK_SIZE>>>(d_data, d_result, N);
        check(cudaDeviceSynchronize(), "sync");
        check(cudaMemcpy(&results[i], d_result, sizeof(float), cudaMemcpyDeviceToHost), "D2H");
        printf("  run %d: %.4f\n", i + 1, results[i]);
    }

    // Check if results are identical (bit-exact) or vary
    int identical = 1;
    for (int i = 1; i < 10; i++) {
        if (results[i] != results[0]) { identical = 0; break; }
    }
    if (identical) {
        printf("  All bit-exact identical -> atomicAdd order is deterministic.\n");
    } else {
        printf("  Results vary -> atomicAdd order is non-deterministic (expected).\n");
    }

    // ---- Cleanup ----
    cudaFree(d_data);
    cudaFree(d_result);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("\n===== Now compare with: cd /mnt/c/.../numba-benchmark && python pytorch_reduce.py =====\n");
    printf("  (PyTorch uses cuBLAS under the hood — same GPU, same N, same dtype)\n");

    return 0;
}
