
#include <stdio.h>
#include <cuda_runtime.h>

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

__global__ void block256_all_smem(const float *x, float *y, int n) {
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
    __shared__ float warp_sums[8];
    int tid = threadIdx.x, gid = blockIdx.x * 256 + tid;
    float v = (gid < n) ? x[gid] : 0;
    for (int d = 16; d >= 1; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    if ((tid & 31) == 0) warp_sums[tid >> 5] = v;
    __syncthreads();
    v = (tid < 8) ? warp_sums[tid] : 0;
    if (tid < 8) {
        for (int d = 4; d >= 1; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    }
    v = __shfl_sync(0xffffffff, v, 0);
    if (gid < n) y[gid] = v;
}

// Repeated versions (only 8 repeats for speed)
__global__ void repeat_all_smem(const float *x, float *y, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x, gid = blockIdx.x * 256 + tid;
    float v = (gid < n) ? x[gid] : 0;
    for (int r = 0; r < 8; r++) {
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
    __shared__ float warp_sums[8];
    int tid = threadIdx.x, gid = blockIdx.x * 256 + tid;
    float v = (gid < n) ? x[gid] : 0;
    for (int r = 0; r < 8; r++) {
        for (int d = 16; d >= 1; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
        if ((tid & 31) == 0) warp_sums[tid >> 5] = v;
        __syncthreads();
        v = (tid < 8) ? warp_sums[tid] : 0;
        if (tid < 8) {
            for (int d = 4; d >= 1; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
        }
        v = __shfl_sync(0xffffffff, v, 0);
    }
    if (gid < n) y[gid] = v;
}

int main() {
    const int N = 1<<20;
    const int WU = 5, RN = 10;
    float *dx, *dy;
    cudaMalloc(&dx, N*sizeof(float));
    cudaMalloc(&dy, N*sizeof(float));

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);

    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s | SM: %d | smem/SM: %dKB

", p.name, p.multiProcessorCount, p.sharedMemPerMultiprocessor/1024);

    auto bench = [&](const char *label, auto kernel, dim3 grid, dim3 block) {
        for (int w = 0; w < WU; w++) kernel<<<grid, block>>>(dx, dy, N);
        cudaDeviceSynchronize();
        float best = 1e9;
        for (int r = 0; r < RN; r++) {
            cudaEventRecord(s, 0);
            kernel<<<grid, block>>>(dx, dy, N);
            cudaEventRecord(e, 0);
            cudaEventSynchronize(e);
            float t; cudaEventElapsedTime(&t, s, e);
            if (t < best) best = t;
        }
        printf("  %-30s  %.4f ms
", label, best);
        return best;
    };

    printf("Test 1: WARP-level (32 threads = 1 warp)
");
    printf("  __shfl covers ALL threads. smem = waste.
");
    int b32 = (N+31)/32;
    float t1 = bench("smem (unnecessary)", warp32_smem, dim3(b32), dim3(32));
    float t2 = bench("__shfl only", warp32_shuffle, dim3(b32), dim3(32));
    printf("  => __shfl %.2fx faster

", t1/t2);

    printf("Test 2: BLOCK-level (256 threads = 8 warps)
");
    printf("  Must cross warps. But hybrid uses 4x fewer syncs.
");
    int b256 = (N+255)/256;
    float t3 = bench("all-smem (3 syncs)", block256_all_smem, dim3(b256), dim3(256));
    float t4 = bench("hybrid (1 sync)", block256_hybrid, dim3(b256), dim3(256));
    printf("  => hybrid %.2fx faster

", t3/t4);

    printf("Test 3: REPEATED (8x) ¡ª sync cost AMPLIFIED
");
    float t5 = bench("all-smem (32 syncs)", repeat_all_smem, dim3(b256), dim3(256));
    float t6 = bench("hybrid (8 syncs)", repeat_hybrid, dim3(b256), dim3(256));
    printf("  => hybrid %.2fx faster

", t5/t6);

    printf("SUMMARY:
");
    printf("  1. WARP-level: NEVER use smem, __shfl is %.1fx faster
", t1/t2);
    printf("  2. BLOCK-level: smem unavoidable for cross-warp, but minimize it (%.1fx)
", t3/t4);
    printf("  3. Repeated: fewer syncs = huge win (%.1fx)
", t5/t6);

    cudaFree(dx); cudaFree(dy);
    cudaEventDestroy(s); cudaEventDestroy(e);
    return 0;
}
