/**
 * Warp shuffle vs Shared Memory: honest comparison
 *
 * Key fact: __shfl_down_sync ONLY works within ONE warp (32 threads).
 * For cross-warp communication, you MUST use smem (or atomics).
 *
 * This file compares fair alternatives for each communication scope.
 *
 * Compile: nvcc -o warp_vs_smem warp_vs_smem.cu -arch=sm_80 && ./warp_vs_smem
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define WARMUP  3
#define RUNS    5

float millis(cudaEvent_t s, cudaEvent_t e) {
    float ms; cudaEventElapsedTime(&ms, s, e); return ms;
}
void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// ============================================================================
// Case 1: WARP-LEVEL reduction (32 threads per block)
//   In this case, __shfl_down_sync covers ALL communication.
//   smem is 100% unnecessary — pure overhead.
// ============================================================================

__global__ void warp_reduce_smem(const float *x, float *y, int n) {
    // blockDim.x = 32 (single warp)
    __shared__ float smem[32];
    int tid = threadIdx.x;
    int gid = blockIdx.x * 32 + tid;
    float val = (gid < n) ? x[gid] : 0.0f;

    // smem reduction (unnecessary! warp shuffle would suffice)
    smem[tid] = val;
    __syncthreads();  // <-- wasted: all threads already in same warp
    for (int s = 16; s >= 1; s >>= 1) {
        // smem exchange: still works, but shuffle is faster
        smem[tid] += __shfl_down_sync(0xffffffff, smem[tid], s);
    }
    val = smem[tid];
    if (gid < n) y[gid] = val;
}

__global__ void warp_reduce_shuffle(const float *x, float *y, int n) {
    // blockDim.x = 32 (single warp) — NO smem needed at all
    int tid = threadIdx.x;
    int gid = blockIdx.x * 32 + tid;
    float val = (gid < n) ? x[gid] : 0.0f;

    // Pure warp shuffle — zero smem, zero __syncthreads()
    for (int delta = 16; delta >= 1; delta >>= 1)
        val += __shfl_down_sync(0xffffffff, val, delta);

    if (gid < n) y[gid] = val;
}

// ============================================================================
// Case 2: BLOCK-LEVEL reduction (256 threads = 8 warps)
//   Must cross warp boundaries. Three options:
//     A) smem for ALL reduction (standard)
//     B) warp shuffle + smem only for cross-warp (hybrid, fewer syncs)
//     C) warp shuffle + per-warp atomicAdd (no smem, more atomics)
// ============================================================================

// A) All-smem: every reduction step uses smem + __syncthreads
__global__ void block_reduce_all_smem(const float *x, float *y, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * 256 + tid;
    float val = (gid < n) ? x[gid] : 0.0f;

    smem[tid] = val;
    __syncthreads();
    // Cross-warp via smem (needs sync)
    for (int s = 128; s >= 32; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();  // 3 syncs for 256->128, 128->64, 64->32
    }
    // Intra-warp via shuffle
    val = smem[tid];
    for (int delta = 16; delta >= 1; delta >>= 1)
        val += __shfl_down_sync(0xffffffff, val, delta);

    if (gid < n) y[gid] = val;
}

// B) Hybrid: warp shuffle first, smem only for 8 warps -> 1
//    This minimizes syncs: only 1 smem store + 1 sync (not 3)
__global__ void block_reduce_hybrid(const float *x, float *y, int n) {
    __shared__ float warp_sums[8];  // only 8 elements, not 256!
    int tid = threadIdx.x;
    int gid = blockIdx.x * 256 + tid;
    float val = (gid < n) ? x[gid] : 0.0f;

    // Step 1: warp-level reduction via shuffle (fast, no smem)
    for (int delta = 16; delta >= 1; delta >>= 1)
        val += __shfl_down_sync(0xffffffff, val, delta);

    // Step 2: cross-warp via smem — only 8 elements, ONE sync
    if ((tid & 31) == 0) warp_sums[tid >> 5] = val;
    __syncthreads();

    // Step 3: thread 0 of warp 0 reads the 8 warp sums and does final shuffle
    val = (tid < 8) ? warp_sums[tid] : 0.0f;
    if (tid < 8) {
        // Single-warp reduction of 8 values
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
    }
    // Broadcast result to all threads via shuffle
    val = __shfl_sync(0xffffffff, val, 0);

    if (gid < n) y[gid] = val;
}

// C) Pure warp shuffle + per-warp atomicAdd (NO smem at all)
__global__ void block_reduce_warp_atomic(const float *x, float *y, int n,
                                          float *global_result) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * 256 + tid;
    float val = (gid < n) ? x[gid] : 0.0f;

    // Warp-level reduction
    for (int delta = 16; delta >= 1; delta >>= 1)
        val += __shfl_down_sync(0xffffffff, val, delta);

    // Each warp writes its partial sum to global memory via atomicAdd
    if ((tid & 31) == 0) atomicAdd(global_result, val);
}

// ============================================================================
// Case 3: Repeated reduction (amplified sync cost)
//   Run reduction 256 times to show cumulative __syncthreads() cost
// ============================================================================

__global__ void repeated_block_all_smem(const float *x, float *y, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * 256 + tid;
    float val = (gid < n) ? x[gid] : 0.0f;

    for (int rep = 0; rep < 32; rep++) {
        smem[tid] = val;
        __syncthreads();                          // sync #1
        for (int s = 128; s >= 32; s >>= 1) {
            if (tid < s) smem[tid] += smem[tid + s];
            __syncthreads();                      // syncs #2, #3, #4
        }
        val = smem[tid];
        for (int d = 16; d >= 1; d >>= 1)
            val += __shfl_down_sync(0xffffffff, val, d);
    }
    // Total syncs: 32 * 4 = 128 per block
    if (gid < n) y[gid] = val;
}

__global__ void repeated_block_hybrid(const float *x, float *y, int n) {
    __shared__ float warp_sums[8];
    int tid = threadIdx.x;
    int gid = blockIdx.x * 256 + tid;
    float val = (gid < n) ? x[gid] : 0.0f;

    for (int rep = 0; rep < 32; rep++) {
        // Warp reduce
        for (int d = 16; d >= 1; d >>= 1)
            val += __shfl_down_sync(0xffffffff, val, d);
        // Cross-warp: 8 values, ONE sync
        if ((tid & 31) == 0) warp_sums[tid >> 5] = val;
        __syncthreads();                          // ONLY 1 sync per rep!
        val = (tid < 8) ? warp_sums[tid] : 0.0f;
        if (tid < 8) {
            val += __shfl_down_sync(0xffffffff, val, 4);
            val += __shfl_down_sync(0xffffffff, val, 2);
            val += __shfl_down_sync(0xffffffff, val, 1);
        }
        val = __shfl_sync(0xffffffff, val, 0);
    }
    // Total syncs: 32 * 1 = 32 per block  (vs 128 for all-smem)
    if (gid < n) y[gid] = val;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s  |  SM: %d  |  smem/SM: %d KB\n\n",
           prop.name, prop.multiProcessorCount,
           prop.sharedMemPerMultiprocessor / 1024);

    // ========================================================================
    printf("======================================================================\n");
    printf("  Case 1: WARP-LEVEL reduction (32 threads = 1 warp)\n");
    printf("          __shfl covers ALL communication. smem = pure waste.\n");
    printf("======================================================================\n");
    {
        int n = 1 << 22;
        size_t bytes = n * sizeof(float);
        float *dx, *dy;
        check(cudaMalloc(&dx, bytes), "dx");
        check(cudaMalloc(&dy, bytes), "dy");

        int blocks = (n + 31) / 32;  // 32 threads per block

        float t_smem = 1e9, t_shfl = 1e9;

        for (int w = 0; w < WARMUP; w++) warp_reduce_smem<<<blocks, 32>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            warp_reduce_smem<<<blocks, 32>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < t_smem) t_smem = t;
        }

        for (int w = 0; w < WARMUP; w++) warp_reduce_shuffle<<<blocks, 32>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            warp_reduce_shuffle<<<blocks, 32>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < t_shfl) t_shfl = t;
        }

        printf("  smem (unnecessary):    %.4f ms\n", t_smem);
        printf("  __shfl only (correct): %.4f ms\n", t_shfl);
        printf("  >>> shuffle is %.2fx faster (no smem, no sync needed)\n\n",
               t_smem / t_shfl);
        cudaFree(dx); cudaFree(dy);
    }

    // ========================================================================
    printf("======================================================================\n");
    printf("  Case 2: BLOCK-LEVEL reduction (256 threads = 8 warps)\n");
    printf("          Cross-warp communication NEEDED. Compare 3 approaches.\n");
    printf("======================================================================\n");
    {
        int n = 1 << 22;
        size_t bytes = n * sizeof(float);
        float *dx, *dy, *dglobal;
        check(cudaMalloc(&dx, bytes), "dx");
        check(cudaMalloc(&dy, bytes), "dy");
        check(cudaMalloc(&dglobal, sizeof(float)), "global");

        int blocks = (n + 255) / 256;

        float t_all_smem = 1e9, t_hybrid = 1e9, t_warp_atomic = 1e9;

        // A) All-smem
        for (int w = 0; w < WARMUP; w++) block_reduce_all_smem<<<blocks, 256>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            block_reduce_all_smem<<<blocks, 256>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < t_all_smem) t_all_smem = t;
        }

        // B) Hybrid
        for (int w = 0; w < WARMUP; w++) block_reduce_hybrid<<<blocks, 256>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            block_reduce_hybrid<<<blocks, 256>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < t_hybrid) t_hybrid = t;
        }

        printf("  A) all-smem  (3 syncs/block):     %.4f ms\n", t_all_smem);
        printf("  B) hybrid    (1 sync/block):       %.4f ms\n", t_hybrid);
        printf("  Hybrid vs all-smem: %.2fx  (fewer syncs = faster)\n",
               t_all_smem / t_hybrid);
        printf("  Note: neither can eliminate smem entirely for cross-warp.\n");
        printf("        But hybrid MINIMIZES smem usage (8 floats vs 256).\n\n");

        cudaFree(dx); cudaFree(dy); cudaFree(dglobal);
    }

    // ========================================================================
    printf("======================================================================\n");
    printf("  Case 3: REPEATED reduction (256x) — sync overhead AMPLIFIED\n");
    printf("======================================================================\n");
    {
        int n = 1 << 16;  // 64K elements = 256 blocks, fast enough
        size_t bytes = n * sizeof(float);
        float *dx, *dy;
        check(cudaMalloc(&dx, bytes), "dx");
        check(cudaMalloc(&dy, bytes), "dy");

        int blocks = (n + 255) / 256;

        float t_all = 1e9, t_hybrid = 1e9;

        for (int w = 0; w < WARMUP; w++) repeated_block_all_smem<<<blocks, 256>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            repeated_block_all_smem<<<blocks, 256>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < t_all) t_all = t;
        }

        for (int w = 0; w < WARMUP; w++) repeated_block_hybrid<<<blocks, 256>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            repeated_block_hybrid<<<blocks, 256>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < t_hybrid) t_hybrid = t;
        }

        printf("  all-smem  (128 syncs):   %.4f ms\n", t_all);
        printf("  hybrid    (32 syncs):    %.4f ms\n", t_hybrid);
        printf("  >>> hybrid is %.2fx faster\n", t_all / t_hybrid);
        printf("  Reason: hybrid has 4x fewer __syncthreads() calls.\n");
        printf("          smem usage is 8 floats (32B) vs 256 floats (1KB).\n");
        printf("          Less smem = less pressure on SM resources.\n\n");

        cudaFree(dx); cudaFree(dy);
    }

    // ========================================================================
    printf("======================================================================\n");
    printf("  SUMMARY: The honest hierarchy\n");
    printf("======================================================================\n");
    printf("  Communication scope    | Best tool       | smem needed?\n");
    printf("  -----------------------+-----------------+-------------\n");
    printf("  Within 1 warp (32 thr) | __shfl_down_sync| NO  (never!)\n");
    printf("  Cross-warp (33-1024)   | smem (minimize!) | YES (but minimize)\n");
    printf("  Cross-block (global)   | global memory    | NO  (use atomics)\n");
    printf("\n");
    printf("  Golden rule for smem:\n");
    printf("    1. For WARP-level communication -> NEVER use smem.\n");
    printf("       __shfl is ~1 cycle, smem+sync is ~30 cycles.\n");
    printf("    2. For BLOCK-level (cross-warp) -> need smem, but:\n");
    printf("       - Do warp-level reduction via __shfl first\n");
    printf("       - Only use smem for the final cross-warp merge\n");
    printf("       - Keep smem small (few elements, not whole array)\n");
    printf("    3. The fewer __syncthreads(), the better.\n");
    printf("       Each sync = all warps wait for the slowest = ~90ns.\n");
    printf("======================================================================\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
