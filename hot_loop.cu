/**
 * 纯 CUDA C++ 实现
 * 编译: nvcc -o hot_loop_cuda.exe hot_loop.cu
 * 运行: ./hot_loop_cuda.exe
 *
 * 和 Numba CUDA 的关系:
 *   Numba 在 Python 运行时把 @cuda.jit 编译成 PTX (GPU 汇编的中间表示)
 *   然后交给 CUDA driver 执行。
 *   nvcc 是提前编译: .cu → PTX → SASS (GPU 机器码)，没有 Python 运行时开销。
 *
 * 结果应该和 Numba 几乎一样 (同一个硬件，编译器的区别很小)。
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N 1000000
#define BLOCK_SIZE 256
#define WARMUP 3
#define RUNS 10

// ---------------------------------------------------------------------------
// GPU kernel — 完全等价于 Python 里的 @cuda.jit def kernel(...)
// ---------------------------------------------------------------------------
__global__ void hot_loop_kernel(const float *data, float *ans, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // == cuda.grid(1)
    if (i < n) {
        float x = data[i];
        // sinf / cosf / sqrtf / fabsf = float 版本的数学函数
        ans[i] = sinf(x) * sinf(x) + cosf(x) * cosf(x) + sqrtf(fabsf(x));
    }
}

// ---------------------------------------------------------------------------
// CPU 端计时 & 验证工具
// ---------------------------------------------------------------------------
float milliseconds(cudaEvent_t start, cudaEvent_t stop) {
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
    printf("===== Raw CUDA C++: hot_loop =====\n\n");

    // ---- 分配 host 内存 (CPU) ----
    float *h_data = (float *)malloc(N * sizeof(float));
    float *h_ans  = (float *)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) h_data[i] = (float)(i * 0.001);

    // ---- 分配 device 内存 (GPU) ----
    float *d_data, *d_ans;
    check(cudaMalloc(&d_data, N * sizeof(float)), "cudaMalloc d_data");
    check(cudaMalloc(&d_ans,  N * sizeof(float)), "cudaMalloc d_ans");

    // ---- 数据搬运 CPU → GPU ----
    cudaEvent_t t_start, t_stop;
    check(cudaEventCreate(&t_start), "event create");
    check(cudaEventCreate(&t_stop),  "event create");

    cudaEventRecord(t_start, 0);
    check(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice), "H2D");
    cudaEventRecord(t_stop, 0);
    cudaEventSynchronize(t_stop);
    printf("CPU → GPU : %.4f ms\n", milliseconds(t_start, t_stop));

    // ---- Kernel 配置 ----
    int blocks  = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;  // 向上取整
    int threads = BLOCK_SIZE;
    printf("Grid: %d blocks × %d threads\n", blocks, threads);

    // ---- 预热 ----
    for (int i = 0; i < WARMUP; i++) {
        hot_loop_kernel<<<blocks, threads>>>(d_data, d_ans, N);
    }
    check(cudaDeviceSynchronize(), "warmup sync");

    // ---- 正式计时 ----
    float best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(t_start, 0);
        hot_loop_kernel<<<blocks, threads>>>(d_data, d_ans, N);
        cudaEventRecord(t_stop, 0);
        cudaEventSynchronize(t_stop);
        float t = milliseconds(t_start, t_stop);
        if (t < best) best = t;
    }
    check(cudaDeviceSynchronize(), "bench sync");

    // ---- 结果搬运 GPU → CPU ----
    check(cudaMemcpy(h_ans, d_ans, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H");

    // ---- 计算带宽 ----
    float data_size = 2.0f * N * sizeof(float);  // 读 4MB + 写 4MB = 8MB
    float bandwidth = (data_size / (best / 1000.0f)) / 1e9f;  // GB/s

    printf("\nKernel 耗时 (best of %d) : %.4f ms\n", RUNS, best);
    printf("显存带宽              : %.1f GB/s\n", bandwidth);
    printf("利用率 (vs 192 GB/s)  : %.1f%%\n", bandwidth / 192.0f * 100);

    // ---- 数值验证 ----
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float x = h_data[i];
        float expected = sinf(x) * sinf(x) + cosf(x) * cosf(x) + sqrtf(fabsf(x));
        if (fabsf(h_ans[i] - expected) > 1e-5f) {
            if (errors < 5)  // 只打印前 5 个错误
                printf("Mismatch at [%d]: GPU=%.6f  CPU=%.6f\n", i, h_ans[i], expected);
            errors++;
        }
    }
    printf("数值验证: %s (%d 错误)\n", errors == 0 ? "PASS ✅" : "FAIL ❌", errors);

    // ---- 清理 ----
    cudaFree(d_data); cudaFree(d_ans);
    free(h_data);     free(h_ans);
    cudaEventDestroy(t_start); cudaEventDestroy(t_stop);

    return 0;
}

__global__ void my_own_loop_cu(const float* data, float* ans, const int N) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        float x = data[i];
        ans[i] = sinf(x) * sinf(x) + cosf(x) * cosf(x) + sqrt(fabsf(x));
    }
}