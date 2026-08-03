/**
 * Beat PyTorch at hidden=8192 and 16384.
 * Strategy: register-cached online softmax → 1R+1W (same as our V3, but for larger sizes)
 *
 * Key: at hidden=8192, each thread has 32 elements. 32 floats fit in ~32 registers.
 *      At hidden=16384, each thread has 64 elements. Still fits but tighter.
 *      1R+1W ≈ 1.5x less HBM traffic than 2R+1W online.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N_ROWS         1000
#define BLOCK_SIZE     256
#define WARMUP         10
#define RUNS           30

// ============================================================================
// V4: register-cached online softmax — 1R+1W
// ============================================================================
// The compiler needs a compile-time constant to NOT spill x_reg[] to local mem.
// We use __launch_bounds__ to tell it how many blocks to target.

__global__ void __launch_bounds__(256, 5)
softmax_v4_32(const float *x, float *out, int n_rows, int hidden_size) {
    const int ITEMS = 32;  // hidden=8192 / 256
    int row = blockIdx.x, tid = threadIdx.x, tt = blockDim.x;
    const float *xr = x + row * hidden_size;
    float *out_r = out + row * hidden_size;
    __shared__ float sm_m[BLOCK_SIZE], sm_d[BLOCK_SIZE];

    // --- Read all 32 elements into registers ---
    float x_reg[ITEMS];
    #pragma unroll
    for (int i = 0; i < ITEMS; i++) x_reg[i] = xr[tid + i * tt];

    // --- Online (m, d) from registers ---
    float m = -INFINITY, d = 0.0f;
    #pragma unroll
    for (int i = 0; i < ITEMS; i++) {
        float v = x_reg[i];
        if (v > m) { d = d * expf(m - v) + 1.0f; m = v; }
        else       { d += expf(v - m); }
    }

    // --- Merge-reduce across threads ---
    sm_m[tid] = m; sm_d[tid] = d; __syncthreads();
    for (int s = BLOCK_SIZE/2; s >= 32; s >>= 1) {
        if (tid < s) {
            float a_m = sm_m[tid], a_d = sm_d[tid];
            float b_m = sm_m[tid+s], b_d = sm_d[tid+s];
            if (a_m >= b_m) { sm_m[tid] = a_m; sm_d[tid] = a_d + b_d * expf(b_m - a_m); }
            else            { sm_m[tid] = b_m; sm_d[tid] = b_d + a_d * expf(a_m - b_m); }
        }
        __syncthreads();
    }
    m = sm_m[tid]; d = sm_d[tid];
    for (int delta = 16; delta >= 1; delta >>= 1) {
        float om = __shfl_down_sync(0xffffffff, m, delta);
        float od = __shfl_down_sync(0xffffffff, d, delta);
        if (m >= om) { d = d + od * expf(om - m); }
        else         { d = od + d * expf(m - om); m = om; }
    }
    if (tid == 0) { sm_m[0] = m; sm_d[0] = d; } __syncthreads();
    float rmax = sm_m[0], rsum = sm_d[0];

    // --- Write output from registers (no re-read!) ---
    #pragma unroll
    for (int i = 0; i < ITEMS; i++) {
        out_r[tid + i * tt] = expf(x_reg[i] - rmax) / rsum;
    }
}

__global__ void __launch_bounds__(256, 3)
softmax_v4_64(const float *x, float *out, int n_rows, int hidden_size) {
    const int ITEMS = 64;  // hidden=16384 / 256
    int row = blockIdx.x, tid = threadIdx.x, tt = blockDim.x;
    const float *xr = x + row * hidden_size;
    float *out_r = out + row * hidden_size;
    __shared__ float sm_m[BLOCK_SIZE], sm_d[BLOCK_SIZE];

    float x_reg[ITEMS];
    #pragma unroll
    for (int i = 0; i < ITEMS; i++) x_reg[i] = xr[tid + i * tt];

    float m = -INFINITY, d = 0.0f;
    #pragma unroll
    for (int i = 0; i < ITEMS; i++) {
        float v = x_reg[i];
        if (v > m) { d = d * expf(m - v) + 1.0f; m = v; }
        else       { d += expf(v - m); }
    }

    sm_m[tid] = m; sm_d[tid] = d; __syncthreads();
    for (int s = BLOCK_SIZE/2; s >= 32; s >>= 1) {
        if (tid < s) {
            float a_m = sm_m[tid], a_d = sm_d[tid];
            float b_m = sm_m[tid+s], b_d = sm_d[tid+s];
            if (a_m >= b_m) { sm_m[tid] = a_m; sm_d[tid] = a_d + b_d * expf(b_m - a_m); }
            else            { sm_m[tid] = b_m; sm_d[tid] = b_d + a_d * expf(a_m - b_m); }
        }
        __syncthreads();
    }
    m = sm_m[tid]; d = sm_d[tid];
    for (int delta = 16; delta >= 1; delta >>= 1) {
        float om = __shfl_down_sync(0xffffffff, m, delta);
        float od = __shfl_down_sync(0xffffffff, d, delta);
        if (m >= om) { d = d + od * expf(om - m); }
        else         { d = od + d * expf(m - om); m = om; }
    }
    if (tid == 0) { sm_m[0] = m; sm_d[0] = d; } __syncthreads();
    float rmax = sm_m[0], rsum = sm_d[0];

    #pragma unroll
    for (int i = 0; i < ITEMS; i++) {
        out_r[tid + i * tt] = expf(x_reg[i] - rmax) / rsum;
    }
}


// ============================================================================
// Bench
// ============================================================================
float millis(cudaEvent_t s, cudaEvent_t e) {
    float ms; cudaEventElapsedTime(&ms, s, e); return ms;
}
void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) { fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err)); exit(1); }
}

typedef void (*kfn)(const float*, float*, int, int);

float bench_one(kfn k, const float *dx, float *dout, int rows, int hidden,
                dim3 grid, dim3 block, cudaEvent_t start, cudaEvent_t stop) {
    for (int i = 0; i < WARMUP; i++) k<<<grid, block>>>(dx, dout, rows, hidden);
    check(cudaDeviceSynchronize(), "wu");
    float best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start, 0);
        k<<<grid, block>>>(dx, dout, rows, hidden);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }
    return best;
}

int main() {
    printf("===== V4: Register-Cached Online Softmax =====\n\n");

    int sizes[] = {8192, 16384};
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "ev"); check(cudaEventCreate(&stop), "ev");

    for (int s = 0; s < 2; s++) {
        int hidden = sizes[s];
        long long total_elems = (long long)N_ROWS * hidden;
        long long total_bytes = total_elems * sizeof(float);
        int items = hidden / BLOCK_SIZE;

        printf("--- HIDDEN=%d | items/thread=%d | data=%.1f MB ---\n",
               hidden, items, total_bytes / 1e6f);

        float *hx = (float*)malloc(total_bytes);
        for (long long i = 0; i < total_elems; i++)
            hx[i] = (float)(rand())/RAND_MAX - 0.5f;

        float *dx, *dout;
        check(cudaMalloc(&dx, total_bytes), "dx");
        check(cudaMalloc(&dout, total_bytes), "dout");
        check(cudaMemcpy(dx, hx, total_bytes, cudaMemcpyHostToDevice), "H2D");

        dim3 grid(N_ROWS), block(BLOCK_SIZE);

        // Pick the right kernel for this hidden size
        kfn v4 = (items == 32) ? softmax_v4_32 : softmax_v4_64;
        const char *v4name = (items == 32) ? "V4 (reg=32)" : "V4 (reg=64)";

        float t_v4 = bench_one(v4, dx, dout, N_ROWS, hidden, grid, block, start, stop);

        // BW for 1R+1W model
        float traffic_1r1w = total_bytes * 2.0f;
        float bw_1r1w = traffic_1r1w / (t_v4 / 1000.0f) / 1e9f;
        printf("  %s  %.4f ms  |  BW %.1f GB/s (%.1f%% of 192)\n",
               v4name, t_v4, bw_1r1w, bw_1r1w/192.0f*100);

        // Verify
        float *hout = (float*)malloc(hidden * sizeof(float));
        check(cudaMemcpy(hout, dout, hidden * sizeof(float), cudaMemcpyDeviceToHost), "D2H");

        float cpu_max = -INFINITY;
        for (int i = 0; i < hidden; i++) if (hx[i] > cpu_max) cpu_max = hx[i];
        float cpu_sum = 0.0f;
        for (int i = 0; i < hidden; i++) cpu_sum += expf(hx[i] - cpu_max);
        float max_diff = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float expected = expf(hx[i] - cpu_max) / cpu_sum;
            float diff = fabsf(hout[i] - expected);
            if (diff > max_diff) max_diff = diff;
        }
        printf("  Verify: max_diff=%.2e %s\n", max_diff, (max_diff<1e-4f)?"PASS":"FAIL");

        printf("  PyTorch:  %.4f ms\n",
               hidden == 8192 ? 0.370f : 0.887f);

        cudaFree(dx); cudaFree(dout); free(hx); free(hout);
    }

    printf("\n===== Summary =====\n");
    printf("  V4 strategy: register-cached (1R+1W) like V3, but for larger hidden.\n");
    printf("  If regs don't spill → should beat PyTorch (same 1R+1W, our BW is >90%%).\n");
    printf("  Check nvcc --resource-usage to see if x_reg[] stays in regs.\n");

    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
