/**
 * Nsight Compute instruction-level profile: V1 vs V2 softmax.
 * Run:
 *   nvcc -o softmax_profile softmax_profile.cu
 *   ncu --metrics \
 *     sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
 *     sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
 *     sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
 *     sm__sass_thread_inst_executed_op_ldg_pred_on.sum,\
 *     sm__sass_thread_inst_executed_op_stg_pred_on.sum,\
 *     sm__sass_thread_inst_executed_op_mufu_pred_on.sum,\
 *     sm__sass_thread_inst_executed_op_barrier_pred_on.sum,\
 *     sm__sass_thread_inst_executed_op_sel_pred_on.sum,\
 *     sm__sass_thread_inst_executed_op_br_pred_on.sum,\
 *     sm__sass_thread_inst_executed_op_shfl_pred_on.sum,\
 *     sm__inst_executed.avg.pct_of_peak_sustained_elapsed \
 *     --kernel-name regex:^softmax ./softmax_profile
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N_ROWS         10
#define HIDDEN_SIZE    32768
#define BLOCK_SIZE     256
#define EPS            1e-5f

__device__ inline void merge_pair(
    float m_a, float d_a, float m_b, float d_b,
    float *m_out, float *d_out
) {
    if (m_a >= m_b) { *m_out = m_a; *d_out = d_a + d_b * expf(m_b - m_a); }
    else            { *m_out = m_b; *d_out = d_b + d_a * expf(m_a - m_b); }
}

__global__ void softmax_v1(const float *x, float *out, int n_rows, int hidden_size) {
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

__global__ void softmax_v2(const float *x, float *out, int n_rows, int hidden_size) {
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

void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) { fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err)); exit(1); }
}

int main() {
    int n_rows = N_ROWS, hidden = HIDDEN_SIZE;
    long long total_elems = (long long)n_rows * hidden;
    long long total_bytes = total_elems * sizeof(float);
    printf("N_ROWS=%d HIDDEN=%d data=%.1f MB\n", n_rows, hidden, total_bytes / 1e6f);

    float *hx = (float*)malloc(total_bytes);
    for (long long i = 0; i < total_elems; i++) hx[i] = (float)(rand())/RAND_MAX - 0.5f;

    float *dx, *dout;
    check(cudaMalloc(&dx, total_bytes), "dx");
    check(cudaMalloc(&dout, total_bytes), "dout");
    check(cudaMemcpy(dx, hx, total_bytes, cudaMemcpyHostToDevice), "H2D");

    dim3 grid(n_rows), block(BLOCK_SIZE);

    printf("Launching V1...\n");
    softmax_v1<<<grid, block>>>(dx, dout, n_rows, hidden);
    check(cudaDeviceSynchronize(), "V1");

    printf("Launching V2...\n");
    softmax_v2<<<grid, block>>>(dx, dout, n_rows, hidden);
    check(cudaDeviceSynchronize(), "V2");

    printf("Done.\n");
    cudaFree(dx); cudaFree(dout); free(hx);
    return 0;
}
