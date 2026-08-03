/**
 * Softmax final comparison: V1 vs V2 vs MyOwn vs PyTorch
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N_ROWS         1000
#define BLOCK_SIZE     256
#define WARMUP         10
#define RUNS           30

// --- V1: naive 3-pass ---
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

// --- My Own: online softmax ---
#define MY_BLOCK_SIZE 256
__global__ void softmax_my_own(const float *x, float *out, int n_rows, int hidden_size) {
    int tid = threadIdx.x;
    int gap = blockIdx.x * hidden_size;
    const float* x_cur_ptr = x + gap;
    float* out_cur_ptr = out + gap;

    __shared__ float smem_max[MY_BLOCK_SIZE];
    __shared__ float smem_exp[MY_BLOCK_SIZE];
    float _max = -INFINITY;
    float exp_sum = 0.0f;
    float temp1 = 0.0f;

    for (int i = tid; i < hidden_size; i += MY_BLOCK_SIZE) {
        temp1 = x_cur_ptr[i];
        if (temp1 > _max) {
            exp_sum = exp_sum * expf(_max - temp1) + 1.0f;
            _max = temp1;
        } else {
            exp_sum += expf(temp1 - _max);
        }
    }
    smem_max[tid] = _max;
    smem_exp[tid] = exp_sum;
    __syncthreads();

    for (int stride = MY_BLOCK_SIZE / 2; stride >= 32; stride = stride / 2) {
        if (tid < stride) {
            if (smem_max[tid] < smem_max[tid + stride]) {
                smem_exp[tid] = smem_exp[tid] * expf(smem_max[tid] - smem_max[tid + stride]) + smem_exp[tid + stride];
                smem_max[tid] = smem_max[tid + stride];
            } else {
                smem_exp[tid] = smem_exp[tid] + smem_exp[tid + stride] * expf(smem_max[tid + stride] - smem_max[tid]);
            }
        }
        __syncthreads();
    }

    float val = smem_exp[tid];
    float _max2 = smem_max[tid];

    float temp = __shfl_down_sync(0xffffffff, _max2, 16);
    if (_max2 < temp) { val = val * expf(_max2 - temp) + __shfl_down_sync(0xffffffff, val, 16); _max2 = temp; }
    else              { val = val + __shfl_down_sync(0xffffffff, val, 16) * expf(temp - _max2); }
    temp = __shfl_down_sync(0xffffffff, _max2, 8);
    if (_max2 < temp) { val = val * expf(_max2 - temp) + __shfl_down_sync(0xffffffff, val, 8); _max2 = temp; }
    else              { val = val + __shfl_down_sync(0xffffffff, val, 8) * expf(temp - _max2); }
    temp = __shfl_down_sync(0xffffffff, _max2, 4);
    if (_max2 < temp) { val = val * expf(_max2 - temp) + __shfl_down_sync(0xffffffff, val, 4); _max2 = temp; }
    else              { val = val + __shfl_down_sync(0xffffffff, val, 4) * expf(temp - _max2); }
    temp = __shfl_down_sync(0xffffffff, _max2, 2);
    if (_max2 < temp) { val = val * expf(_max2 - temp) + __shfl_down_sync(0xffffffff, val, 2); _max2 = temp; }
    else              { val = val + __shfl_down_sync(0xffffffff, val, 2) * expf(temp - _max2); }
    temp = __shfl_down_sync(0xffffffff, _max2, 1);
    if (_max2 < temp) { val = val * expf(_max2 - temp) + __shfl_down_sync(0xffffffff, val, 1); _max2 = temp; }
    else              { val = val + __shfl_down_sync(0xffffffff, val, 1) * expf(temp - _max2); }

    if (tid == 0) {
        smem_max[tid] = _max2;
        smem_exp[tid] = val;
    }
    __syncthreads();
    float exp_final_sum = smem_exp[0];
    float max_final = smem_max[0];

    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        out_cur_ptr[i] = expf(x_cur_ptr[i] - max_final) / exp_final_sum;
    }
}

float millis(cudaEvent_t s, cudaEvent_t e) {
    float ms; cudaEventElapsedTime(&ms, s, e); return ms;
}
void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) { fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err)); exit(1); }
}

typedef void (*kfn)(const float*, float*, int, int);

void bench_one(const char *label, kfn k, const float *dx, float *dout,
               int rows, int hidden, dim3 grid, dim3 block,
               cudaEvent_t start, cudaEvent_t stop) {
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
    printf("  %-12s %8.4f ms\n", label, best);
}

int main() {
    int sizes[] = {4096, 8192, 16384, 32768};
    int n_sizes = 4;

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "ev");
    check(cudaEventCreate(&stop), "ev");

    printf("===== Softmax: CUDA vs PyTorch =====");
    printf("\n  %d rows, hidden = {4096, 8192, 16384, 32768}\n\n", N_ROWS);

    for (int s = 0; s < n_sizes; s++) {
        int hidden = sizes[s];
        long long total_elems = (long long)N_ROWS * hidden;
        long long total_bytes = total_elems * sizeof(float);

        printf("--- HIDDEN=%d (data=%.1f MB) ---\n", hidden, total_bytes / 1e6f);

        float *hx = (float*)malloc(total_bytes);
        for (long long i = 0; i < total_elems; i++)
            hx[i] = (float)(rand())/RAND_MAX - 0.5f;

        float *dx, *dout;
        check(cudaMalloc(&dx, total_bytes), "dx");
        check(cudaMalloc(&dout, total_bytes), "dout");
        check(cudaMemcpy(dx, hx, total_bytes, cudaMemcpyHostToDevice), "H2D");

        dim3 grid(N_ROWS), block(BLOCK_SIZE);

        bench_one("V1 (naive)",  softmax_v1,      dx, dout, N_ROWS, hidden, grid, block, start, stop);
        bench_one("MyOwn",       softmax_my_own,   dx, dout, N_ROWS, hidden, grid, block, start, stop);

        // Verify MyOwn
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
        printf("  Verify MyOwn: max_diff=%.2e %s\n", max_diff, (max_diff<1e-4f)?"PASS":"FAIL");

        cudaFree(dx); cudaFree(dout); free(hx); free(hout);
    }

    printf("\n===== 对照：PyTorch (run 'python softmax_scale.py' for these) =====\n");
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
