/**
 * Micro benchmark: warp shuffle vs shared memory
 * compile: nvcc -o micro micro.cu -arch=sm_80 && ./micro
 */
#include <stdio.h>
#include <cuda_runtime.h>

const int N = 1 << 20;
const int WU = 5, RN = 10;

float millis(cudaEvent_t s, cudaEvent_t e) {
    float ms; cudaEventElapsedTime(&ms, s, e); return ms;
}

// WARP-level (32 threads): smem vs shuffle
__global__ void warp32_smem(const float *x, float *y, int n) {
    __shared__ float smem[32];
    int tid = threadIdx.x, gid = blockIdx.x * 32 + tid;
    float v = (gid < n) ? x[gid] : 0;
    smem[tid] = v; __syncthreads();
    for (int s = 16; s >= 1; s >>= 1) smem[tid] += __shfl_down_sync(0xffffffff, smem[tid], s);
    if (gid < n) y[gid] = smem[tid];
}
__global__ void warp32_shuffle(const float *x, float *y, int n) {
    int tid = threadIdx.x, gid = blockIdx.x * 32 + tid;
    float v = (gid < n) ? x[gid] : 0;
    for (int s = 16; s >= 1; s >>= 1) v += __shfl_down_sync(0xffffffff, v, s);
    if (gid < n) y[gid] = v;
}

// BLOCK-level (256 threads): all-smem vs hybrid
__global__ void block256_smem(const float *x, float *y, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x, gid = blockIdx.x * 256 + tid;
    float v = (gid < n) ? x[gid] : 0;
    smem[tid] = v; __syncthreads();
    for (int s = 128; s >= 32; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    v = smem[tid];
    for (int d = 16; d >= 1; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    if (gid < n) y[gid] = v;
}
__global__ void block256_hybrid(const float *x, float *y, int n) {
    __shared__ float ws[8];  // only 8 floats for cross-warp
    int tid = threadIdx.x, gid = blockIdx.x * 256 + tid;
    float v = (gid < n) ? x[gid] : 0;
    for (int d = 16; d >= 1; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    if ((tid & 31) == 0) ws[tid >> 5] = v;
    __syncthreads();
    v = (tid < 8) ? ws[tid] : 0;
    if (tid < 8) for (int d = 4; d >= 1; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    v = __shfl_sync(0xffffffff, v, 0);
    if (gid < n) y[gid] = v;
}

// REPEATED: amplify sync cost (4x repeats)
__global__ void repeat_smem(const float *x, float *y, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x, gid = blockIdx.x * 256 + tid;
    float v = (gid < n) ? x[gid] : 0;
    for (int r = 0; r < 4; r++) {
        smem[tid] = v; __syncthreads();
        for (int s = 128; s >= 32; s >>= 1) {
            if (tid < s) smem[tid] += smem[tid + s];
            __syncthreads();
        }
        v = smem[tid];
        for (int d = 16; d >= 1; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    }
    if (gid < n) y[gid] = v;
}
__global__ void repeat_hybrid(const float *x, float *y, int n) {
    __shared__ float ws[8];
    int tid = threadIdx.x, gid = blockIdx.x * 256 + tid;
    float v = (gid < n) ? x[gid] : 0;
    for (int r = 0; r < 4; r++) {
        for (int d = 16; d >= 1; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
        if ((tid & 31) == 0) ws[tid >> 5] = v;
        __syncthreads();
        v = (tid < 8) ? ws[tid] : 0;
        if (tid < 8) for (int d = 4; d >= 1; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
        v = __shfl_sync(0xffffffff, v, 0);
    }
    if (gid < n) y[gid] = v;
}

float bench(void (*k)(const float*,float*,int), const float *dx, float *dy, int n,
            dim3 g, dim3 b, cudaEvent_t s, cudaEvent_t e) {
    for (int w = 0; w < WU; w++) k<<<g, b>>>(dx, dy, n);
    cudaDeviceSynchronize();
    float best = 1e9;
    for (int r = 0; r < RN; r++) {
        cudaEventRecord(s, 0);
        k<<<g, b>>>(dx, dy, n);
        cudaEventRecord(e, 0);
        cudaEventSynchronize(e);
        float t = millis(s, e);
        if (t < best) best = t;
    }
    return best;
}

int main() {
    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s | SM: %d | smem/SM: %dKB | maxThr/SM: %d\n\n",
           p.name, p.multiProcessorCount, p.sharedMemPerMultiprocessor/1024,
           p.maxThreadsPerMultiProcessor);

    float *dx, *dy;
    cudaMalloc(&dx, N*sizeof(float));
    cudaMalloc(&dy, N*sizeof(float));

    // === Test 1: WARP level ===
    printf("=== Test 1: WARP-level (blockDim=32, 1 warp) ===\n");
    printf("    __shfl covers ALL 32 threads. smem = pure overhead.\n");
    int b32 = (N + 31) / 32;
    float t1 = bench(warp32_smem,   dx, dy, N, dim3(b32), dim3(32), s, e);
    float t2 = bench(warp32_shuffle, dx, dy, N, dim3(b32), dim3(32), s, e);
    printf("  smem (wasted):    %.4f ms\n", t1);
    printf("  __shfl (correct): %.4f ms\n", t2);
    printf("  => __shfl %.2fx faster  (no smem, no __syncthreads)\n\n", t1/t2);

    // === Test 2: BLOCK level ===
    printf("=== Test 2: BLOCK-level (blockDim=256, 8 warps) ===\n");
    printf("    Must cross warps. But hybrid uses ONLY 1 sync vs 3.\n");
    int b256 = (N + 255) / 256;
    float t3 = bench(block256_smem,  dx, dy, N, dim3(b256), dim3(256), s, e);
    float t4 = bench(block256_hybrid, dx, dy, N, dim3(b256), dim3(256), s, e);
    printf("  all-smem (3 syncs/block): %.4f ms\n", t3);
    printf("  hybrid   (1 sync/block):  %.4f ms\n", t4);
    printf("  => hybrid %.2fx faster  (4x fewer __syncthreads)\n", t3/t4);
    printf("  => smem still NEEDED for cross-warp, but MINIMIZE it!\n\n");

    // === Test 3: REPEATED ===
    printf("=== Test 3: REPEATED 4x (amplified sync cost) ===\n");
    int sc_smem = 4 * 4;   // 4 repeats * 4 syncs each
    int sc_hyb  = 4 * 1;   // 4 repeats * 1 sync each
    float t5 = bench(repeat_smem,  dx, dy, N, dim3(b256), dim3(256), s, e);
    float t6 = bench(repeat_hybrid, dx, dy, N, dim3(b256), dim3(256), s, e);
    printf("  all-smem (%d syncs/block): %.4f ms\n", sc_smem, t5);
    printf("  hybrid  (%d syncs/block):  %.4f ms\n", sc_hyb, t6);
    printf("  => hybrid %.2fx faster\n\n", t5/t6);

    // === SUMMARY ===
    printf("============================================================\n");
    printf("  Communication scope  | Best tool            | smem needed?\n");
    printf("  ---------------------+----------------------+------------\n");
    printf("  Within 1 warp (<=32) | __shfl_down_sync     | NO\n");
    printf("  Cross-warp (>32)     | smem (minimize!)     | YES\n");
    printf("  Cross-block          | global atomics       | NO\n");
    printf("\n");
    printf("  Takeaway from your smem question:\n");
    printf("  1. WARP-level ops: skip smem entirely, use __shfl\n");
    printf("  2. BLOCK-level ops: warp-reduce FIRST, smem only for\n");
    printf("     warp results (8 floats, not 256)\n");
    printf("  3. Every __syncthreads() you eliminate = ~90ns saved\n");
    printf("     per block. Adds up when repeated.\n");
    printf("============================================================\n");

    cudaFree(dx); cudaFree(dy);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return 0;
}
