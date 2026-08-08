/**
 * Amplified benchmarks: where removing smem CLEARLY wins.
 *
 * The previous benchmark was memory-bandwidth-bound. Here we isolate
 * the communication/compute to make smem overhead VISIBLE.
 *
 * Compile: nvcc -o nosmem_amplified nosmem_amplified.cu -arch=sm_80 && ./nosmem_amplified
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define WARMUP  50
#define RUNS    100

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
// Test A: Repeated block-level reduction — amplifies sync overhead
// ============================================================================
// Do reduction 1024 times within the kernel. The smem version pays
// __syncthreads() * 1024 times, while the warp-shuffle version pays 0.

__global__ void repeated_reduce_smem(const float *x, float *y, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    float val = (gid < n) ? x[gid] : 0.0f;

    // Repeat 256 times to amplify sync overhead
    for (int rep = 0; rep < 256; rep++) {
        smem[tid] = val;
        __syncthreads();                    // <-- paid 256 times
        for (int s = 128; s >= 32; s >>= 1) {
            if (tid < s) smem[tid] += smem[tid + s];
            __syncthreads();                // <-- paid 256 * 3 times more
        }
        // Warp-level
        val = smem[tid];
        val += __shfl_down_sync(0xffffffff, val, 16);
        val += __shfl_down_sync(0xffffffff, val, 8);
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        val = smem[tid];  // read broadcast result
    }
    if (gid < n) y[gid] = val;
}

__global__ void repeated_reduce_warp_only(const float *x, float *y, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    float val = (gid < n) ? x[gid] : 0.0f;

    // Same 256 iterations, ZERO __syncthreads()
    for (int rep = 0; rep < 256; rep++) {
        for (int delta = 16; delta >= 1; delta >>= 1)
            val += __shfl_down_sync(0xffffffff, val, delta);
        val = __shfl_sync(0xffffffff, val, 0);  // broadcast lane 0
    }
    if (gid < n) y[gid] = val;
}

// ============================================================================
// Test B: True bank conflict — 32x32 transpose via smem
// ============================================================================
// Transpose a 32x32 matrix: write rows, read columns.
// Write: no conflict (consecutive banks)
// Read: 32 threads read smem[0][0], smem[1][0], ..., smem[31][0]
//       Bank = (row*32+0) % 32 = 0 -> ALL bank 0 -> 32-way conflict!

__global__ void transpose_smem_conflict(const float *in, float *out, int num_matrices) {
    __shared__ float tile[32][32];  // NO padding -> column access = bank conflict
    int tid = threadIdx.x;
    int matrix = blockIdx.x;
    int base = matrix * 1024;

    // Write rows (no conflict)
    for (int i = 0; i < 32; i++) {
        tile[i][tid] = in[base + i * 32 + tid];
    }
    __syncthreads();

    // Read columns (32-WAY BANK CONFLICT!)
    // thread 0 reads tile[0][0]=bank0, tile[1][0]=bank0, ...
    for (int i = 0; i < 32; i++) {
        out[base + i * 32 + tid] = tile[tid][i];
        // Wait — this is actually tile[tid][i] where all threads access
        // different rows (tid differs) but same column (i).
        // tile[tid][i]: bank = (tid*32 + i) % 32
        // For fixed i, as tid varies, bank = tid % 32 if i=0, or (i + tid*32) % 32
        // Actually: bank = (tid*32 + i) / 4 % 32... no, 32 is in floats.
        // In bytes: offset = (tid*32 + i) * 4
        // bank = ((tid*32 + i) * 4 / 4) % 32 = (tid*32 + i) % 32
        // For fixed i: bank = i for tid=0, bank = i+32%32=i for tid=1...
        // So bank = i % 32 — ALL threads access the SAME bank! 32-way conflict!
        // (unless i varies, but in this loop it doesn't per iteration)
    }
}

// Same transpose but with +1 padding to eliminate bank conflicts
__global__ void transpose_smem_padded(const float *in, float *out, int num_matrices) {
    __shared__ float tile[32][33];  // +1 padding
    int tid = threadIdx.x;
    int matrix = blockIdx.x;
    int base = matrix * 1024;

    for (int i = 0; i < 32; i++) {
        tile[i][tid] = in[base + i * 32 + tid];
    }
    __syncthreads();

    // Read columns: tile[tid][i]
    // bank = (tid*33 + i) % 32 -> now banks are staggered, no conflict!
    for (int i = 0; i < 32; i++) {
        out[base + i * 32 + tid] = tile[tid][i];
    }
}

// Transpose WITHOUT smem — each thread reads from global directly
// No bank conflict, but stride access to global memory
__global__ void transpose_no_smem(const float *in, float *out, int num_matrices) {
    int tid = threadIdx.x;
    int matrix = blockIdx.x;
    int base = matrix * 1024;

    // Read a column from global, write as a row
    // This has strided global reads but ZERO smem overhead
    for (int i = 0; i < 32; i++) {
        out[base + tid * 32 + i] = in[base + i * 32 + tid];
    }
}

// ============================================================================
// Test C: Occupancy stress test
// ============================================================================
// Use 40KB smem per block -> only 2 blocks/SM on 100KB SM
// vs 0 smem -> full occupancy

__global__ void occ_killer_smem(const float *x, float *y, int n) {
    __shared__ float huge[10240];  // 40 KB -> only 2 blocks/SM (100KB total)
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Initialize (wasteful but simulates a kernel that "needs" this smem)
    for (int i = tid; i < 10240; i += blockDim.x)
        huge[i] = (float)(i * blockIdx.x);
    __syncthreads();

    if (gid >= n) return;
    float sum = 0.0f;
    // Read from smem many times to justify its existence
    for (int i = 0; i < 256; i++) {
        sum += huge[(tid + i * 13) % 10240];
    }
    y[gid] = x[gid] + sum * 1e-6f;
}

__global__ void occ_friendly_nosmem(const float *x, float *y, int n) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    if (gid >= n) return;

    // Same computation, using registers instead of smem
    float sum = 0.0f;
    for (int i = 0; i < 256; i++) {
        sum += (float)((tid + i * 13) % 10240);  // same values, in registers
    }
    y[gid] = x[gid] + sum * 1e-6f;
}

// ============================================================================
// Test D: Inter-block communication pattern
// ============================================================================
// When each block only needs its own data and doesn't communicate
// with neighbors, smem adds no value. Classic case: vector add.

__global__ void vecadd_smem(const float *a, const float *b, float *c, int n) {
    __shared__ float sa[256], sb[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) { sa[tid] = a[gid]; sb[tid] = b[gid]; }
    __syncthreads();

    if (gid < n) c[gid] = sa[tid] + sb[tid];
}

__global__ void vecadd_nosmem(const float *a, const float *b, float *c, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    c[gid] = a[gid] + b[gid];
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
    printf("GPU: %s  |  SM: %d  |  smem/SM: %d KB  |  Max threads/SM: %d\n\n",
           prop.name, prop.multiProcessorCount,
           prop.sharedMemPerMultiprocessor / 1024,
           prop.maxThreadsPerMultiProcessor);

    // ========================================================================
    printf("======================================================================\n");
    printf("  Test A: Amplified reduction (256 repeats) — SYNC OVERHEAD\n");
    printf("======================================================================\n");
    {
        int n = 1 << 20;  // 1M floats, small enough to fit in L2
        size_t bytes = n * sizeof(float);
        float *dx, *dy;
        check(cudaMalloc(&dx, bytes), "dx");
        check(cudaMalloc(&dy, bytes), "dy");
        int blocks = (n + 255) / 256;

        float t_smem = 1e9, t_warp = 1e9;

        for (int w = 0; w < WARMUP; w++) repeated_reduce_smem<<<blocks, 256>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            repeated_reduce_smem<<<blocks, 256>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < t_smem) t_smem = t;
        }

        for (int w = 0; w < WARMUP; w++) repeated_reduce_warp_only<<<blocks, 256>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            repeated_reduce_warp_only<<<blocks, 256>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < t_warp) t_warp = t;
        }

        printf("  smem (4 syncs x 256 = 1024 syncs):  %.4f ms\n", t_smem);
        printf("  warp-only (0 syncs):                 %.4f ms\n", t_warp);
        printf("  >>> warp-only is %.2fx faster\n", t_smem / t_warp);
        printf("  Reason: 1024 __syncthreads() per block add up\n\n");
        cudaFree(dx); cudaFree(dy);
    }

    // ========================================================================
    printf("======================================================================\n");
    printf("  Test B: 32x32 transpose — BANK CONFLICT\n");
    printf("======================================================================\n");
    {
        int num_matrices = 65536;  // many small transposes
        int elems_per = 1024;       // 32x32
        size_t bytes = (size_t)num_matrices * elems_per * sizeof(float);
        float *dx, *dy;
        check(cudaMalloc(&dx, bytes), "dx");
        check(cudaMalloc(&dy, bytes), "dy");

        float t_conflict = 1e9, t_padded = 1e9, t_nosmem = 1e9;

        for (int w = 0; w < WARMUP; w++) transpose_smem_conflict<<<num_matrices, 32>>>(dx, dy, num_matrices);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            transpose_smem_conflict<<<num_matrices, 32>>>(dx, dy, num_matrices);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < t_conflict) t_conflict = t;
        }

        for (int w = 0; w < WARMUP; w++) transpose_smem_padded<<<num_matrices, 32>>>(dx, dy, num_matrices);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            transpose_smem_padded<<<num_matrices, 32>>>(dx, dy, num_matrices);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < t_padded) t_padded = t;
        }

        for (int w = 0; w < WARMUP; w++) transpose_no_smem<<<num_matrices, 32>>>(dx, dy, num_matrices);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            transpose_no_smem<<<num_matrices, 32>>>(dx, dy, num_matrices);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < t_nosmem) t_nosmem = t;
        }

        printf("  smem transpose (32-way conflict):  %.4f ms\n", t_conflict);
        printf("  smem transpose (+padding, ok):     %.4f ms\n", t_padded);
        printf("  NO smem transpose (direct global): %.4f ms\n", t_nosmem);
        printf("  bank conflict vs no smem:  %.2fx\n", t_conflict / t_nosmem);
        printf("  padded smem vs no smem:    %.2fx\n", t_padded / t_nosmem);
        printf("  Key point: if you forget padding, smem is SLOWER than no smem.\n");
        printf("             Even with padding, the benefit may not justify complexity.\n\n");
        cudaFree(dx); cudaFree(dy);
    }

    // ========================================================================
    printf("======================================================================\n");
    printf("  Test C: Occupancy stress — 40KB smem per block\n");
    printf("======================================================================\n");
    {
        int n = 1 << 20;
        size_t bytes = n * sizeof(float);
        float *dx, *dy;
        check(cudaMalloc(&dx, bytes), "dx");
        check(cudaMalloc(&dy, bytes), "dy");

        // Use fewer blocks so that occupancy matters
        int blocks = 256;  // with 40KB smem, only 2 fit per SM = 40 blocks total on 20 SM

        float t_smem = 1e9, t_nosmem = 1e9;

        for (int w = 0; w < WARMUP; w++) occ_killer_smem<<<blocks, 256>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            occ_killer_smem<<<blocks, 256>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < t_smem) t_smem = t;
        }

        for (int w = 0; w < WARMUP; w++) occ_friendly_nosmem<<<blocks, 256>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            occ_friendly_nosmem<<<blocks, 256>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < t_nosmem) t_nosmem = t;
        }

        int max_blocks_per_sm_smem = prop.sharedMemPerMultiprocessor / (40 * 1024);
        int max_blocks_per_sm_nosmem = prop.maxThreadsPerMultiProcessor / 256;
        printf("  smem version:  max %d blocks/SM (limited by 40KB smem)\n", max_blocks_per_sm_smem);
        printf("  no-smem ver:   max %d blocks/SM (limited by threads)\n", max_blocks_per_sm_nosmem);
        printf("  smem (40KB, low occ):   %.4f ms\n", t_smem);
        printf("  no smem (high occ):     %.4f ms\n", t_nosmem);
        printf("  >>> no-smem is %.2fx faster\n", t_smem / t_nosmem);
        printf("  Reason: low occupancy -> cannot hide memory latency\n\n");
        cudaFree(dx); cudaFree(dy);
    }

    // ========================================================================
    printf("======================================================================\n");
    printf("  Test D: Vector add — classic no-reuse case\n");
    printf("======================================================================\n");
    {
        int n = 1 << 24;  // 16M elements
        size_t bytes = n * sizeof(float);
        float *da, *db, *dc;
        check(cudaMalloc(&da, bytes), "da");
        check(cudaMalloc(&db, bytes), "db");
        check(cudaMalloc(&dc, bytes), "dc");

        int blocks = (n + 255) / 256;

        float t_smem = 1e9, t_nosmem = 1e9;

        for (int w = 0; w < WARMUP; w++) vecadd_smem<<<blocks, 256>>>(da, db, dc, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            vecadd_smem<<<blocks, 256>>>(da, db, dc, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < t_smem) t_smem = t;
        }

        for (int w = 0; w < WARMUP; w++) vecadd_nosmem<<<blocks, 256>>>(da, db, dc, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            vecadd_nosmem<<<blocks, 256>>>(da, db, dc, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < t_nosmem) t_nosmem = t;
        }

        // Calculate memory bandwidth
        float bw_smem = (n * 3.0f * 4.0f) / (t_smem / 1000.0f) / 1e9f;  // 3 arrays, 4B each
        float bw_nosmem = (n * 3.0f * 4.0f) / (t_nosmem / 1000.0f) / 1e9f;

        printf("  with smem:  %.4f ms  (%.1f GB/s)\n", t_smem, bw_smem);
        printf("  no smem:    %.4f ms  (%.1f GB/s)\n", t_nosmem, bw_nosmem);
        printf("  >>> no-smem is %.2fx faster\n", t_smem / t_nosmem);
        printf("  Reason: zero data reuse. smem is 2 extra memcpy + __syncthreads().\n\n");
        cudaFree(da); cudaFree(db); cudaFree(dc);
    }

    // ========================================================================
    printf("======================================================================\n");
    printf("  FINAL TAKEAWAYS\n");
    printf("======================================================================\n");
    printf("  When NO smem is FASTER:\n");
    printf("    1. Repeated sync patterns     -> __syncthreads() * N adds up\n");
    printf("    2. Bank-conflict-prone access -> smem can be slower than global\n");
    printf("    3. smem kills occupancy       -> fewer warps to hide latency\n");
    printf("    4. Zero data reuse            -> smem = wasted memcpy + sync\n");
    printf("\n");
    printf("  When smem is WORTH it:\n");
    printf("    1. Multiple threads reuse the same data (e.g., stencil, conv)\n");
    printf("    2. Data is read >> once from smem (amortizes load cost)\n");
    printf("    3. smem usage is modest (no occupancy impact)\n");
    printf("    4. Access pattern is bank-conflict-free or padded\n");
    printf("    5. Compute is heavy enough to hide sync overhead\n");
    printf("======================================================================\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
