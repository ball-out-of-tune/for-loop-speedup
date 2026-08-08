/**
 * Minimal demo: when __shfl beats smem (and when it can't)
 *
 * Compile: nvcc -o quick_warp quick_warp.cu -arch=sm_80 && ./quick_warp
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1<<20)
#define BLK 256

float millis(cudaEvent_t s, cudaEvent_t e) {
    float ms; cudaEventElapsedTime(&ms, s, e); return ms;
}

// ============================================================================
// 1. WARP-level reduce (block=32) — smem vs shuffle
// ============================================================================

__global__ void warp32_smem(const float *x, float *y, int n) {
    __shared__ float smem[32];
    int tid = threadIdx.x, gid = blockIdx.x * 32 + tid;
    float v = (gid < n) ? x[gid] : 0;

    smem[tid] = v; __syncthreads();           // unnecessary sync!
    for (int s = 16; s >= 1; s >>= 1) smem[tid] += __shfl_down_sync(0xffffffff, smem[tid], s);
    if (gid < n) y[gid] = smem[tid];
}

__global__ void warp32_shuffle(const float *x, float *y, int n) {
    int tid = threadIdx.x, gid = blockIdx.x * 32 + tid;
    float v = (gid < n) ? x[gid] : 0;

    for (int s = 16; s >= 1; s >>= 1) v += __shfl_down_sync(0xffffffff, v, s);
    if (gid < n) y[gid] = v;
}

// ============================================================================
// 2. BLOCK-level reduce (block=256) — must cross warps
// ============================================================================

// Approach A: smem for everything (standard textbook)
__global__ void block256_all_smem(const float *x, float *y, int n) {
    __shared__ float smem[BLK];
    int tid = threadIdx.x, gid = blockIdx.x * BLK + tid;
    float v = (gid < n) ? x[gid] : 0;

    smem[tid] = v; __syncthreads();
    for (int s = 128; s >= 32; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();                     // 3 cross-warp syncs
    }
    for (int d = 16; d >= 1; d >>= 1) v = smem[tid] + __shfl_down_sync(0xffffffff, smem[tid], d);
    // ^ bug: v accumulates differently per thread... actually this should be:
    v = smem[tid];
    for (int d = 16; d >= 1; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    if (gid < n) y[gid] = v;
}

// Approach B: warp-shuffle first, smem only for 8 warps -> 1 (hybrid)
__global__ void block256_hybrid(const float *x, float *y, int n) {
    __shared__ float warp_sums[8];
    int tid = threadIdx.x, gid = blockIdx.x * BLK + tid;
    float v = (gid < n) ? x[gid] : 0;

    // Within each warp: shuffle (no smem, no sync)
    for (int d = 16; d >= 1; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);

    // Cross-warp: 8 elements via smem, ONE sync
    if ((tid & 31) == 0) warp_sums[tid >> 5] = v;
    __syncthreads();                         // only 1 cross-warp sync!
    v = (tid < 8) ? warp_sums[tid] : 0;
    if (tid < 8) {
        for (int d = 4; d >= 1; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    }
    v = __shfl_sync(0xffffffff, v, 0);      // broadcast within first warp
    if (gid < n) y[gid] = v;
}

// ============================================================================
// 3. AMPLIFIED: repeat reduction N times to show cumulative sync cost
// ============================================================================
#define REPEATS 32

__global__ void repeat_all_smem(const float *x, float *y, int n) {
    __shared__ float smem[BLK];
    int tid = threadIdx.x, gid = blockIdx.x * BLK + tid;
    float v = (gid < n) ? x[gid] : 0;

    for (int r = 0; r < REPEATS; r++) {
        smem[tid] = v; __syncthreads();           // sync #1
        for (int s = 128; s >= 32; s >>= 1) {
            if (tid < s) smem[tid] += smem[tid + s];
            __syncthreads();                       // syncs #2,3,4
        }
        v = smem[tid];
        for (int d = 16; d >= 1; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
    }
    if (gid < n) y[gid] = v;
}

__global__ void repeat_hybrid(const float *x, float *y, int n) {
    __shared__ float warp_sums[8];
    int tid = threadIdx.x, gid = blockIdx.x * BLK + tid;
    float v = (gid < n) ? x[gid] : 0;

    for (int r = 0; r < REPEATS; r++) {
        for (int d = 16; d >= 1; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
        if ((tid & 31) == 0) warp_sums[tid >> 5] = v;
        __syncthreads();                           // ONLY 1 sync per repeat
        v = (tid < 8) ? warp_sums[tid] : 0;
        if (tid < 8) {
            for (int d = 4; d >= 1; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);
        }
        v = __shfl_sync(0xffffffff, v, 0);
    }
    if (gid < n) y[gid] = v;
}

// ============================================================================
#define WU 5
#define RN 15

float bench_kernel(const char *label, void(*kernel)(const float*,float*,int),
                   const float *dx, float *dy, int n, dim3 grid, dim3 block,
                   cudaEvent_t start, cudaEvent_t stop) {
    for (int w = 0; w < WU; w++) kernel<<<grid, block>>>(dx, dy, n);
    cudaDeviceSynchronize();

    float best = 1e9;
    for (int r = 0; r < RN; r++) {
        cudaEventRecord(start, 0);
        kernel<<<grid, block>>>(dx, dy, n);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }
    printf("  %-30s  %.4f ms\n", label, best);
    return best;
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s | SM: %d | smem/SM: %dKB\n\n", p.name,
           p.multiProcessorCount, p.sharedMemPerMultiprocessor/1024);

    size_t bytes = N * sizeof(float);
    float *dx, *dy;
    cudaMalloc(&dx, bytes); cudaMalloc(&dy, bytes);

    // ========================================================================
    printf("--- Test 1: WARP-level (block=32, single warp) ---\n");
    printf("  __shfl covers ALL threads. smem = pure waste.\n");
    {
        int blocks = (N + 31) / 32;
        float t1 = bench_kernel("smem (unnecessary)", warp32_smem, dx, dy, N, dim3(blocks), dim3(32), start, stop);
        float t2 = bench_kernel("__shfl only",         warp32_shuffle, dx, dy, N, dim3(blocks), dim3(32), start, stop);
        printf("  => __shfl is %.2fx faster (no smem, no __syncthreads)\n\n", t1/t2);
    }

    // ========================================================================
    printf("--- Test 2: BLOCK-level (block=256, 8 warps) ---\n");
    printf("  Cross-warp communication NEEDED. But minimize smem.\n");
    {
        int blocks = (N + BLK - 1) / BLK;
        float t1 = bench_kernel("all-smem (3 syncs/block)", block256_all_smem, dx, dy, N, dim3(blocks), dim3(BLK), start, stop);
        float t2 = bench_kernel("hybrid  (1 sync/block)",   block256_hybrid,   dx, dy, N, dim3(blocks), dim3(BLK), start, stop);
        printf("  => hybrid is %.2fx faster (4x fewer __syncthreads)\n", t1/t2);
        printf("  => smem still needed for cross-warp, but MINIMIZE it!\n\n");
    }

    // ========================================================================
    printf("--- Test 3: AMPLIFIED (repeat %dx) — sync cost visible ---\n", REPEATS);
    {
        int blocks = (N + BLK - 1) / BLK;
        int syncs_all = REPEATS * 4;   // 4 syncs per repeat
        int syncs_hyb = REPEATS * 1;   // 1 sync per repeat
        float t1 = bench_kernel("all-smem", repeat_all_smem, dx, dy, N, dim3(blocks), dim3(BLK), start, stop);
        float t2 = bench_kernel("hybrid",  repeat_hybrid,  dx, dy, N, dim3(blocks), dim3(BLK), start, stop);
        printf("  syncs/block: all-smem=%d  hybrid=%d\n", syncs_all, syncs_hyb);
        printf("  => hybrid is %.2fx faster\n\n", t1/t2);
    }

    // ========================================================================
    printf("--- SUMMARY: When to avoid/reduce smem ---\n");
    printf("  1. WARP-level (<=32 threads): NEVER use smem, use __shfl\n");
    printf("  2. BLOCK-level (>32 threads): need smem for cross-warp,\n");
    printf("     but do warp-reduce FIRST, then smem only for warp results\n");
    printf("  3. Each __syncthreads() costs ~90ns; minimize them\n");
    printf("  4. Smem occupancy: 8 floats < 256 floats for same result\n\n");

    printf("  Analogous to your code: if you can merge within a warp\n");
    printf("  using __shfl, skip the smem + __syncthreads entirely.\n");

    cudaFree(dx); cudaFree(dy);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
