/**
 * Softmax scaling experiment: V1 vs V2 vs V3 across hidden sizes.
 * Tests whether Online softmax (V2) beats register-cached (V3) at large sizes.
 *
 * Compile: nvcc -o softmax_scale_cuda softmax_scale.cu
 * Run:     ./softmax_scale_cuda
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N_ROWS         1000
#define BLOCK_SIZE     256
#define WARMUP         10
#define RUNS           30

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

// V1: naive 3-pass
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

// V2: online softmax
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

// V3: register-cached (works when items/thread <= REG_MAX)
#define REG_MAX 16

__global__ void v3_regcache(const float *x, float *out, int n_rows, int hidden_size) {
    int row = blockIdx.x, tid = threadIdx.x, tt = blockDim.x;
    const float *xr = x + row * hidden_size;
    float *out_r = out + row * hidden_size;
    __shared__ float sm[BLOCK_SIZE];

    int items = hidden_size / tt;
    float x_reg[REG_MAX];
    for (int i = 0; i < items; i++) x_reg[i] = xr[tid + i * tt];

    float lmax = -INFINITY;
    for (int i = 0; i < items; i++) { if (x_reg[i] > lmax) lmax = x_reg[i]; }
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
    for (int i = 0; i < items; i++) lsum += expf(x_reg[i] - rmax);
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

    for (int i = 0; i < items; i++) out_r[tid + i * tt] = expf(x_reg[i] - rmax) / rsum;
}


float millis(cudaEvent_t start, cudaEvent_t stop) {
    float ms; cudaEventElapsedTime(&ms, start, stop); return ms;
}
void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) { fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err)); exit(1); }
}

typedef void (*kernel_fn)(const float*, float*, int, int);

void bench_one(const char *label, kernel_fn k,
               const float *dx, float *dout, int rows, int hidden,
               dim3 grid, dim3 block, float traffic,
               cudaEvent_t start, cudaEvent_t stop) {
    for (int i = 0; i < WARMUP; i++) k<<<grid, block>>>(dx, dout, rows, hidden);
    check(cudaDeviceSynchronize(), "warmup");
    float best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start, 0);
        k<<<grid, block>>>(dx, dout, rows, hidden);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }
    float bw = traffic / (best / 1000.0f) / 1e9f;
    long long l2_kb = (long long)rows * hidden * sizeof(float) / 1024;
    printf("  %s  %.4f ms   BW %.1f GB/s (%5.1f%%)   row=%.1f KB  total=%.1f MB L2=2 MB\n",
           label, best, bw, bw/192.0f*100,
           hidden*4.0f/1024, l2_kb/1024.0);
}


int main() {
    int test_sizes[] = {4096, 8192, 16384, 32768, 65536};
    int n_sizes = 5;

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");

    for (int s = 0; s < n_sizes; s++) {
        int hidden = test_sizes[s];
        long long total_elems = (long long)N_ROWS * hidden;
        long long total_bytes = total_elems * sizeof(float);

        int items = hidden / BLOCK_SIZE;

        printf("\n===== HIDDEN=%d | items/thread=%d | data=%.1f MB =====",
               hidden, items, total_bytes / 1e6f);
        // Check if all rows fit in L2
        float rows_in_l2 = 20.0f * hidden * 4.0f / 1024.0f;
        printf("   (20 rows in flight = %.0f KB)\n", rows_in_l2);

        float *hx = (float*)malloc(total_bytes);
        for (long long i = 0; i < total_elems; i++)
            hx[i] = (float)(rand())/RAND_MAX - 0.5f;

        float *dx, *dout;
        check(cudaMalloc(&dx, total_bytes), "dx");
        check(cudaMalloc(&dout, total_bytes), "dout");
        check(cudaMemcpy(dx, hx, total_bytes, cudaMemcpyHostToDevice), "H2D");

        dim3 grid(N_ROWS), block(BLOCK_SIZE);

        // Traffic models for each kernel
        float v1_traffic = total_bytes * 4.0f;   // 3R+1W
        float v2_traffic = total_bytes * 3.0f;   // 2R+1W
        float v3_traffic = total_bytes * 2.0f;   // 1R+1W

        bench_one("V1", v1_naive,     dx, dout, N_ROWS, hidden, grid, block, v1_traffic, start, stop);
        bench_one("V2", v2_online,    dx, dout, N_ROWS, hidden, grid, block, v2_traffic, start, stop);
        if (items <= REG_MAX && hidden % BLOCK_SIZE == 0)
            bench_one("V3", v3_regcache, dx, dout, N_ROWS, hidden, grid, block, v3_traffic, start, stop);
        else
            printf("  V3     SKIP (items/thread=%d > REG_MAX=%d)\n", items, REG_MAX);

        cudaFree(dx); cudaFree(dout); free(hx);
    }

    printf("\n===== Summary =====\n");
    printf("  L2 = 2 MB.  When 20 rows > L2, re-reads must go to HBM.\n");
    printf("  Hidden=4096:  20 x 16  KB = 320 KB << 2 MB → L2 covers all re-reads\n");
    printf("  Hidden=8192:  20 x 32  KB = 640 KB <  2 MB → L2 still covers\n");
    printf("  Hidden=16384: 20 x 64  KB = 1.28 MB < 2 MB → L2 borderline\n");
    printf("  Hidden=32768: 20 x 128 KB = 2.56 MB > 2 MB → L2 overflow!\n");

    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
