/**
 * 验证 L2 是否溢出：在不同 hidden_size 下，re-read 走 L2 还是 HBM？
 *
 * 方法: read-3-only kernel (纯读 x 3 次，零计算零写入)
 *   - 若实测 << 理论 HBM 下限 (3R @ 192 GB/s) → re-reads 命中 L2
 *   - 若实测 ≈ 理论 HBM 下限 → re-reads 走 HBM (L2 溢出)
 *
 * 每个 row 大小: hidden * 4 bytes
 * 20 个 SM 同时跑 → 20 rows 同时在 L2
 * L2 = 2 MB → 20 * hidden * 4 < 2 MB 时能全缓存，否则溢出
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N_ROWS         1000
#define BLOCK_SIZE     256
#define WARMUP         10
#define RUNS           30

// Pure read x 3 times.  Zero compute, zero write.  Only memory traffic.
__global__ void read_three_times(const float *x, float *sink, int n_rows, int hidden_size) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int tt = blockDim.x;
    int hs = hidden_size;
    const float *xr = x + row * hs;

    float dummy = 0.0f;
    // Read 1
    for (int i = tid; i < hs; i += tt) dummy += xr[i];
    // Read 2
    for (int i = tid; i < hs; i += tt) dummy += xr[i];
    // Read 3
    for (int i = tid; i < hs; i += tt) dummy += xr[i];

    if (tid == 0) sink[row] = dummy;
}

float millis(cudaEvent_t start, cudaEvent_t stop) {
    float ms; cudaEventElapsedTime(&ms, start, stop); return ms;
}
void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) { fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err)); exit(1); }
}

int main() {
    int test_sizes[] = {4096, 8192, 16384, 32768, 65536};
    int n_tests = 5;

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");

    printf("===== 验证: re-read 走 L2 还是 HBM？ =====\n\n");
    printf("  L2 = 2 MB.  20 SMs, each 1 block, 1 row per block.\n");
    printf("  When 20 * row_size > 2 MB, L2 overflows.\n\n");

    printf("%-10s %-12s %-18s %-16s %-12s %s\n",
           "HIDDEN", "row_size", "20 rows in L2",
           "3R@192GB/s(min)",
           "read-3实测", "结论");

    for (int t = 0; t < n_tests; t++) {
        int hidden = test_sizes[t];
        long long total_elems = (long long)N_ROWS * hidden;
        long long total_bytes = total_elems * sizeof(float);

        float row_bytes = hidden * 4.0f;
        float rows20_kb = 20.0f * row_bytes / 1024.0f;
        float l2_mb = 2.0f * 1024.0f;  // L2 in KB

        // theoretical minimum: 3 reads × data_size / 192 GB/s
        float data_mb = total_bytes / 1e6f;
        float theoretical_ms = data_mb * 3.0f / 192e3f * 1000.0f;

        float *hx = (float*)malloc(total_bytes);
        for (long long i = 0; i < total_elems; i++)
            hx[i] = (float)(rand())/RAND_MAX - 0.5f;

        float *dx, *dsink;
        check(cudaMalloc(&dx, total_bytes), "dx");
        check(cudaMalloc(&dsink, N_ROWS * sizeof(float)), "dsink");

        check(cudaMemcpy(dx, hx, total_bytes, cudaMemcpyHostToDevice), "H2D");

        dim3 grid(N_ROWS), block(BLOCK_SIZE);

        // warmup
        for (int i = 0; i < WARMUP; i++)
            read_three_times<<<grid, block>>>(dx, dsink, N_ROWS, hidden);
        check(cudaDeviceSynchronize(), "wu");

        float best = 1e9;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            read_three_times<<<grid, block>>>(dx, dsink, N_ROWS, hidden);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best) best = t;
        }

        // Effective BW for 3 reads
        float bw = (data_mb * 3.0f) / (best / 1000.0f) * 1e3f;

        // verdict
        const char *verdict;
        float ratio = best / theoretical_ms;
        if (ratio < 0.6f)
            verdict = "L2 HIT (re-reads cached)";
        else if (ratio < 0.9f)
            verdict = "L2 PARTIAL";
        else
            verdict = "HBM (L2 overflow)";

        printf("%-10d %-8.0f KB    %-6.0f KB / 2048 KB   %-8.4f ms       %-8.4f ms    %.2fx → %s\n",
               hidden, row_bytes/1024, rows20_kb,
               theoretical_ms, best, ratio, verdict);

        cudaFree(dx); cudaFree(dsink); free(hx);
    }

    printf("\n===== 所以 =====\n");
    printf("L2 在 row 超过 ~100 KB (hidden > 25600) 时明显溢出。\n");
    printf("溢出前: V1 的 re-read = L2 hit ≈ 200 cycles → V2 省不了多少\n");
    printf("溢出后: V1 的 re-read = HBM 读取 ≈ 380 cycles → V2 一次读代替两次读，赚了\n");
    printf("这正是 V2 在 hidden=4096 没用 (1.04x) 但在 hidden=32768 有用 (1.32x) 的原因\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
