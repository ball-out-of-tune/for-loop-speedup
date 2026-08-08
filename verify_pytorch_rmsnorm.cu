/**
 * 验证: PyTorch eager RMSNorm 是不是真的启动了 5 个独立 kernel?
 *
 * 模拟 PyTorch eager 执行:
 *   Kernel 1: tmp1 = x²        (element-wise)
 *   Kernel 2: tmp2 = mean(tmp1) (reduce, 1000 行各自求平均)
 *   Kernel 3: tmp3 = sqrt(tmp2 + eps)  (element-wise, 极小)
 *   Kernel 4: tmp4 = x / tmp3  (element-wise, 广播 tmp3)
 *   Kernel 5: out  = tmp4 * weight  (element-wise)
 *
 * 如果总时间 ≈ PyTorch eager 的 0.652 ms → 确认 PyTorch 就这么干的
 * 如果总时间 << 0.652 ms → PyTorch 可能没拆这么多 kernel
 *
 * 编译: nvcc -o verify_pytorch_rmsnorm verify_pytorch_rmsnorm.cu
 * 运行: ./verify_pytorch_rmsnorm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N_ROWS         1000
#define HIDDEN_SIZE    4096
#define BLOCK_SIZE     256
#define EPS            1e-5f
#define WARMUP         10
#define RUNS           30

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

// ---------------------------------------------------------------------------
// Kernel 1: tmp1 = x²  (element-wise, 读 x 写 tmp1)
// ---------------------------------------------------------------------------
__global__ void k1_square(const float *x, float *tmp1, int total_elems) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < total_elems; i += stride) {
        float v = x[i];
        tmp1[i] = v * v;
    }
}

// ---------------------------------------------------------------------------
// Kernel 2: tmp2 = mean(tmp1) along last dim  (reduce, 1000 rows)
//           每个 block 处理一行, 输出 1000 个标量到 tmp2
// ---------------------------------------------------------------------------
__global__ void k2_reduce_mean(const float *tmp1, float *tmp2,
                                int n_rows, int hidden_size) {
    __shared__ float smem[BLOCK_SIZE];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = row * hidden_size + tid; i < (row + 1) * hidden_size; i += blockDim.x) {
        sum += tmp1[i];
    }

    smem[tid] = sum;
    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    float val = smem[tid];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    if (tid == 0) tmp2[row] = val / (float)hidden_size;
}

// ---------------------------------------------------------------------------
// Kernel 3: tmp3 = sqrt(tmp2 + eps)  (element-wise, 1000 个元素, 极小)
// ---------------------------------------------------------------------------
__global__ void k3_sqrt_eps(const float *tmp2, float *tmp3, int n_rows, float eps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_rows) {
        tmp3[i] = sqrtf(tmp2[i] + eps);
    }
}

// ---------------------------------------------------------------------------
// Kernel 4: tmp4 = x / tmp3  (element-wise, broadcast tmp3 to each row)
// ---------------------------------------------------------------------------
__global__ void k4_normalize(const float *x, const float *tmp3,
                              float *tmp4, int n_rows, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total = n_rows * hidden_size;
    for (; i < total; i += stride) {
        int row = i / hidden_size;
        tmp4[i] = x[i] / tmp3[row];
    }
}

// ---------------------------------------------------------------------------
// Kernel 5: out = tmp4 * weight  (element-wise)
// ---------------------------------------------------------------------------
__global__ void k5_scale(const float *tmp4, const float *weight,
                          float *out, int total_elems, int hidden_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (; i < total_elems; i += stride) {
        int col = i % hidden_size;
        out[i] = tmp4[i] * weight[col];
    }
}


// ============================================================================
int main() {
    int n_rows = N_ROWS;
    int hidden = HIDDEN_SIZE;
    int total_elems = n_rows * hidden;

    printf("===== Verify: PyTorch eager RMSNorm = 5 kernels? =====\n\n");

    // ---- Allocate ----
    float *h_x = (float *)malloc(total_elems * sizeof(float));
    float *h_weight = (float *)malloc(hidden * sizeof(float));
    for (int i = 0; i < total_elems; i++) h_x[i] = (float)(rand()) / RAND_MAX - 0.5f;
    for (int i = 0; i < hidden; i++) h_weight[i] = 1.0f;

    float *d_x, *d_tmp1, *d_tmp2, *d_tmp3, *d_tmp4, *d_out, *d_weight;
    check(cudaMalloc(&d_x, total_elems * sizeof(float)), "d_x");
    check(cudaMalloc(&d_tmp1, total_elems * sizeof(float)), "d_tmp1");
    check(cudaMalloc(&d_tmp2, n_rows * sizeof(float)), "d_tmp2");
    check(cudaMalloc(&d_tmp3, n_rows * sizeof(float)), "d_tmp3");
    check(cudaMalloc(&d_tmp4, total_elems * sizeof(float)), "d_tmp4");
    check(cudaMalloc(&d_out, total_elems * sizeof(float)), "d_out");
    check(cudaMalloc(&d_weight, hidden * sizeof(float)), "d_weight");

    check(cudaMemcpy(d_x, h_x, total_elems * sizeof(float), cudaMemcpyHostToDevice), "H2D x");
    check(cudaMemcpy(d_weight, h_weight, hidden * sizeof(float), cudaMemcpyHostToDevice), "H2D w");

    // Common grid configs
    int n_blocks_ew = (total_elems + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid_reduce(n_rows);

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");

    // ---- Warmup all kernels ----
    for (int i = 0; i < WARMUP; i++) {
        k1_square<<<n_blocks_ew, BLOCK_SIZE>>>(d_x, d_tmp1, total_elems);
        k2_reduce_mean<<<grid_reduce, BLOCK_SIZE>>>(d_tmp1, d_tmp2, n_rows, hidden);
        k3_sqrt_eps<<<1, 32>>>(d_tmp2, d_tmp3, n_rows, EPS);
        k4_normalize<<<n_blocks_ew, BLOCK_SIZE>>>(d_x, d_tmp3, d_tmp4, n_rows, hidden);
        k5_scale<<<n_blocks_ew, BLOCK_SIZE>>>(d_tmp4, d_weight, d_out, total_elems, hidden);
    }
    check(cudaDeviceSynchronize(), "warmup");

    // ---- Benchmark each kernel individually ----
    float t_total = 0.0f, best;

    printf("Individual kernel timings:\n");

    // K1: x^2
    best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start, 0);
        k1_square<<<n_blocks_ew, BLOCK_SIZE>>>(d_x, d_tmp1, total_elems);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }
    printf("  %-30s  %.4f ms\n", "K1: x^2 (ew)", best);
    t_total += best;

    // K2: mean reduce
    best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start, 0);
        k2_reduce_mean<<<dim3(n_rows), BLOCK_SIZE>>>(d_tmp1, d_tmp2, n_rows, hidden);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }
    printf("  %-30s  %.4f ms\n", "K2: mean reduce", best);
    t_total += best;

    // K3: sqrt(+eps)
    best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start, 0);
        k3_sqrt_eps<<<1, 32>>>(d_tmp2, d_tmp3, n_rows, EPS);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }
    printf("  %-30s  %.4f ms\n", "K3: sqrt(+eps) (ew)", best);
    t_total += best;

    // K4: x / rms
    best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start, 0);
        k4_normalize<<<n_blocks_ew, BLOCK_SIZE>>>(d_x, d_tmp3, d_tmp4, n_rows, hidden);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }
    printf("  %-30s  %.4f ms\n", "K4: x / rms (ew)", best);
    t_total += best;

    // K5: * weight
    best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start, 0);
        k5_scale<<<n_blocks_ew, BLOCK_SIZE>>>(d_tmp4, d_weight, d_out, total_elems, hidden);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }
    printf("  %-30s  %.4f ms\n", "K5: * weight (ew)", best);
    t_total += best;

    printf("  ─────────────────────────────────────────\n");
    printf("  Sum of 5 kernels:                    %.4f ms\n", t_total);
    printf("  PyTorch eager (measured):            0.652 ms\n");

    // ---- Verify correctness against CPU ----
    float sum_x2 = 0;
    for (int i = 0; i < hidden; i++) sum_x2 += h_x[i] * h_x[i];
    float rms_ref = sqrtf(sum_x2 / hidden + EPS);

    float *h_out = (float *)malloc(hidden * sizeof(float));
    check(cudaMemcpy(h_out, d_out, hidden * sizeof(float), cudaMemcpyDeviceToHost), "D2H");

    float max_diff = 0;
    for (int i = 0; i < hidden; i++) {
        float expected = h_x[i] / rms_ref * h_weight[i];
        float diff = fabsf(h_out[i] - expected);
        if (diff > max_diff) max_diff = diff;
    }
    printf("\n  Verification max_diff: %.6e\n", max_diff);

    // ---- Summary ----
    printf("\n===== Conclusion =====\n");
    printf("  If 5-kernel sum ≈ 0.65 ms: PyTorch likely does 5 separate kernels\n");
    printf("  If 5-kernel sum << 0.65 ms: PyTorch uses fewer kernels (some fused)\n");
    printf("  Our fused kernel:     0.248 ms  (2.6x faster)\n");

    // Cleanup
    cudaFree(d_x); cudaFree(d_tmp1); cudaFree(d_tmp2);
    cudaFree(d_tmp3); cudaFree(d_tmp4); cudaFree(d_out); cudaFree(d_weight);
    free(h_x); free(h_weight); free(h_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
