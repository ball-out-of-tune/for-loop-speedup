/**
 * Softmax N_ROWS scaling experiment: does V2 advantage change with batch size?
 *
 * Compile: nvcc -o softmax_rows_cuda softmax_rows.cu && ./softmax_rows_cuda
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define HIDDEN_SIZE    4096
#define BLOCK_SIZE     256
#define WARMUP         5
#define RUNS           20

__device__ inline void merge_pair(
    float m_a, float d_a, float m_b, float d_b,
    float *m_out, float *d_out
) {
    if (m_a >= m_b) {
        *m_out = m_a;
        *d_out = d_a + d_b * expf(m_b - m_a);
    } else {
        *m_out = m_b;
        *d_out = d_b + d_a * expf(m_a - m_b);
    }
}

__global__ void v1_naive(const float *x, float *out, int n_rows, int hidden_size) {
    int row = blockIdx.x, tid = threadIdx.x, tt = blockDim.x;
    const float *xr = x + row * hidden_size;
    float *out_r = out + row * hidden_size;
    __shared__ float sm[BLOCK_SIZE];

    float lmax = -INFINITY;
    for (int i = tid; i < hidden_size; i += tt) {
        float v = xr[i]; if (v > lmax) lmax = v;
    }
    sm[tid] = lmax; __syncthreads();
    for (int s = BLOCK_SIZE/2; s >= 32; s >>= 1) {
        if (tid < s && sm[tid+s] > sm[tid]) sm[tid] = sm[tid+s];
        __syncthreads();
    }
    float val = sm[tid], o;
    o = __shfl_down_sync(0xffffffff, val, 16); if (o > val) val = o;
    o = __shfl_down_sync(0xffffffff, val, 8);  if (o > val) val = o;
    o = __shfl_down_sync(0xffffffff, val, 4);  if (o > val) val = o;
    o = __shfl_down_sync(0xffffffff, val, 2);  if (o > val) val = o;
    o = __shfl_down_sync(0xffffffff, val, 1);  if (o > val) val = o;
    if (tid == 0) sm[0] = val; __syncthreads();
    float rmax = sm[0];

    float lsum = 0.0f;
    for (int i = tid; i < hidden_size; i += tt) lsum += expf(xr[i] - rmax);
    sm[tid] = lsum; __syncthreads();
    for (int s = BLOCK_SIZE/2; s >= 32; s >>= 1) {
        if (tid < s) sm[tid] += sm[tid+s]; __syncthreads();
    }
    val = sm[tid];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (tid == 0) sm[0] = val; __syncthreads();
    float rsum = sm[0];

    for (int i = tid; i < hidden_size; i += tt) out_r[i] = expf(xr[i] - rmax) / rsum;
}

__global__ void v2_online(const float *x, float *out, int n_rows, int hidden_size) {
    int row = blockIdx.x, tid = threadIdx.x, tt = blockDim.x;
    const float *xr = x + row * hidden_size;
    float *out_r = out + row * hidden_size;
    __shared__ float sm_m[BLOCK_SIZE], sm_d[BLOCK_SIZE];

    float m = -INFINITY, d = 0.0f;
    for (int i = tid; i < hidden_size; i += tt) {
        float v = xr[i];
        if (v > m) { d = d * expf(m - v) + 1.0f; m = v; }
        else       { d += expf(v - m); }
    }
    sm_m[tid] = m; sm_d[tid] = d; __syncthreads();
    for (int s = BLOCK_SIZE/2; s >= 32; s >>= 1) {
        if (tid < s) {
            float nm, nd;
            merge_pair(sm_m[tid], sm_d[tid], sm_m[tid+s], sm_d[tid+s], &nm, &nd);
            sm_m[tid] = nm; sm_d[tid] = nd;
        }
        __syncthreads();
    }
    m = sm_m[tid]; d = sm_d[tid];
    for (int delta = 16; delta >= 1; delta >>= 1) {
        float om = __shfl_down_sync(0xffffffff, m, delta);
        float od = __shfl_down_sync(0xffffffff, d, delta);
        float nm, nd;
        merge_pair(m, d, om, od, &nm, &nd);
        m = nm; d = nd;
    }
    if (tid == 0) { sm_m[0] = m; sm_d[0] = d; } __syncthreads();
    float rmax = sm_m[0], rsum = sm_d[0];

    for (int i = tid; i < hidden_size; i += tt) out_r[i] = expf(xr[i] - rmax) / rsum;
}

float millis(cudaEvent_t start, cudaEvent_t stop) {
    float ms; cudaEventElapsedTime(&ms, start, stop); return ms;
}
void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) { fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err)); exit(1); }
}

int main() {
    int n_rows_tests[] = {100, 500, 1000, 5000, 10000};
    int n_tests = 5;
    int hidden = HIDDEN_SIZE;

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");

    printf("===== Fixed HIDDEN=%d, vary N_ROWS =====\n\n", hidden);
    printf("  Each block = 1 row = %d bytes = %.1f KB\n",
           hidden * 4, hidden * 4.0f / 1024);

    for (int t = 0; t < n_tests; t++) {
        int n_rows = n_rows_tests[t];
        long long total_elems = (long long)n_rows * hidden;
        long long total_bytes = total_elems * sizeof(float);
        int total_blocks = n_rows;

        printf("--- N_ROWS=%d | blocks=%d | data=%.1f MB | waves=%.0f ---\n",
               n_rows, total_blocks, total_bytes / 1e6f,
               (float)total_blocks / (20 * 6));  // 20 SMs × 6 blocks/SM

        float *hx = (float*)malloc(total_bytes);
        for (long long i = 0; i < total_elems; i++)
            hx[i] = (float)(rand())/RAND_MAX - 0.5f;

        float *dx, *dout;
        check(cudaMalloc(&dx, total_bytes), "dx");
        check(cudaMalloc(&dout, total_bytes), "dout");
        check(cudaMemcpy(dx, hx, total_bytes, cudaMemcpyHostToDevice), "H2D");

        dim3 grid(n_rows), block(BLOCK_SIZE);

        float t_v1, t_v2;

        // V1
        for (int i = 0; i < WARMUP; i++) v1_naive<<<grid, block>>>(dx, dout, n_rows, hidden);
        check(cudaDeviceSynchronize(), "v1 wu");
        {
            float best = 1e9;
            for (int r = 0; r < RUNS; r++) {
                cudaEventRecord(start, 0);
                v1_naive<<<grid, block>>>(dx, dout, n_rows, hidden);
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                float t = millis(start, stop);
                if (t < best) best = t;
            }
            t_v1 = best;
        }

        // V2
        for (int i = 0; i < WARMUP; i++) v2_online<<<grid, block>>>(dx, dout, n_rows, hidden);
        check(cudaDeviceSynchronize(), "v2 wu");
        {
            float best = 1e9;
            for (int r = 0; r < RUNS; r++) {
                cudaEventRecord(start, 0);
                v2_online<<<grid, block>>>(dx, dout, n_rows, hidden);
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                float t = millis(start, stop);
                if (t < best) best = t;
            }
            t_v2 = best;
        }

        // Time per row (us)
        float us_per_row_v1 = t_v1 / n_rows * 1000.0f;
        float us_per_row_v2 = t_v2 / n_rows * 1000.0f;

        printf("  V1:  %.4f ms  (%.2f us/row)\n", t_v1, us_per_row_v1);
        printf("  V2:  %.4f ms  (%.2f us/row)\n", t_v2, us_per_row_v2);
        printf("  V2/V1: %.3fx\n", t_v1 / t_v2);

        cudaFree(dx); cudaFree(dout); free(hx);
    }

    printf("\n===== Key point =====\n");
    printf("  N_ROWS increases = more blocks, same per-block work.\n");
    printf("  If V2 saves X per row, total savings = N_ROWS × X.\n");
    printf("  The V2/V1 ratio should be CONSTANT regardless of N_ROWS.\n");
    printf("  (Unless the L2 cache hit rate changes with wave count)\n");

    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
