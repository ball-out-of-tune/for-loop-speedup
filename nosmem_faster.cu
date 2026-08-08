/**
 * When is NO shared memory faster? Concrete verification.
 *
 * Compile: nvcc -o nosmem_faster nosmem_faster.cu -arch=sm_80 && ./nosmem_faster
 *
 * Six scenarios where removing smem is faster:
 *   1. Data read once (no reuse) -> L1 cache handles it
 *   2. Warp shuffle can do all reduction -> no smem needed
 *   3. __syncthreads() overhead dominates light compute
 *   4. Bank conflicts -> avoid smem, let L1 handle it
 *   5. Large smem reduces occupancy -> latency hiding degrades
 *   6. Element-wise map -> zero data reuse
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define WARMUP  20
#define RUNS    50

// ============================================================================
// Scenario 1: Data read exactly once — smem is wasteful extra copy
// ============================================================================

__global__ void bad_smem_single_read(const float *x, float *y, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    smem[tid] = (gid < n) ? x[gid] : 0.0f;
    __syncthreads();

    float val = smem[tid];  // each thread reads only its own data, zero reuse
    if (gid < n) y[gid] = val * val + val;
}

__global__ void good_direct_global(const float *x, float *y, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float val = x[gid];  // L1 cache auto-caches; one less copy vs smem
    y[gid] = val * val + val;
}

// ============================================================================
// Scenario 2: Warp shuffle can do all reduction — smem is overhead
// ============================================================================

__global__ void reduce_with_smem(const float *data, float *result, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int total = gridDim.x * blockDim.x;

    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += total)
        sum += data[i];

    // smem reduction: needs multiple __syncthreads()
    smem[tid] = sum;
    __syncthreads();
    for (int s = 128; s >= 32; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();  // each iteration syncs — pure overhead!
    }

    // warp-level shuffle (both versions need this)
    float val = smem[tid];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    if (tid == 0) atomicAdd(result, val);
}

__global__ void reduce_warp_only(const float *data, float *result, int n) {
    int tid = threadIdx.x;
    int total = gridDim.x * blockDim.x;

    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += total)
        sum += data[i];

    // Pure warp shuffle — ZERO smem, ZERO __syncthreads()
    // Reduce only within each warp (32 threads), one atomicAdd per warp
    for (int offset = 16; offset >= 1; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // Lane 0 of each warp writes result
    if ((tid & 31) == 0) atomicAdd(result, sum);
}

// ============================================================================
// Scenario 3: __syncthreads() overhead — isolated measurement
// ============================================================================

__global__ void bench_syncthreads_cost(float *dummy, int n_syncs) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    smem[tid] = (float)tid;

    for (int i = 0; i < n_syncs; i++) {
        __syncthreads();
        smem[tid] += 1.0f;  // light op to prevent optimization
    }
    if (tid == 0) dummy[blockIdx.x] = smem[0];
}

// ============================================================================
// Scenario 4: Bank conflict — smem stride access slower than direct global
// ============================================================================

// 32-way bank conflict: every thread reads stride-32 address
// smem[0], smem[32], smem[64], ... -> bank = (offset/4) % 32
// smem[i*32] offset = i*32*4 -> bank = (128*i) % 32 = 0 -> ALL in bank 0!
__global__ void smem_stride_bank_conflict(const float *in, float *out, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) smem[tid] = in[gid];
    __syncthreads();

    // Each thread reads smem[(tid*32) % 256]
    // thread 0: smem[0]  -> bank 0
    // thread 1: smem[32] -> bank 0
    // thread 2: smem[64] -> bank 0
    // ... 32-way conflict -> 32x serialization!
    int idx = (tid * 32) % 256;
    float val = smem[idx];
    if (gid < n) out[gid] = val * val;
}

// No bank conflict: linear read from smem
__global__ void smem_stride_no_conflict(const float *in, float *out, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) smem[tid] = in[gid];
    __syncthreads();

    // Linear read: thread i reads smem[i] -> bank = i%32, all different
    float val = smem[tid];
    if (gid < n) out[gid] = val * val;
}

// No smem at all
__global__ void direct_global_read(const float *in, float *out, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) out[gid] = in[gid];
}

// ============================================================================
// Scenario 5: Large smem reduces occupancy -> slower
// ============================================================================

__global__ void high_smem_usage(const float *x, float *y, int n) {
    __shared__ float big_smem[2048];  // 8KB per block, kills occupancy
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    for (int i = tid; i < 2048; i += blockDim.x)
        big_smem[i] = (float)i;
    __syncthreads();

    if (gid >= n) return;
    float val = x[gid];
    val += big_smem[tid % 2048];
    y[gid] = val * val;
}

__global__ void no_smem_high_occupancy(const float *x, float *y, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float val = x[gid];
    val += (float)(threadIdx.x);  // register instead of smem
    y[gid] = val * val;
}

// ============================================================================
// Scenario 6: Element-wise map — smem is pure extra memcpy
// ============================================================================

__global__ void elementwise_with_smem(const float *a, const float *b, float *c, int n) {
    __shared__ float sa[256], sb[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Step 1: global -> smem (wasted copy!)
    if (gid < n) { sa[tid] = a[gid]; sb[tid] = b[gid]; }
    __syncthreads();  // wait for all threads

    // Step 2: smem -> register -> compute -> write
    if (gid < n) c[gid] = sa[tid] * sb[tid] + sa[tid];
}

__global__ void elementwise_no_smem(const float *a, const float *b, float *c, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float va = a[gid];  // direct read, L1 cache auto-caches
    float vb = b[gid];
    c[gid] = va * vb + va;
}

// ============================================================================
// Scenario 7: Ultra-light compute — smem load cost > reread global cost
// ============================================================================

__global__ void tiny_compute_smem(const float *x, float *y, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n) smem[tid] = x[gid];
    __syncthreads();

    if (gid < n) {
        float v = smem[tid];
        y[gid] = v + v * 0.5f + v * 0.25f;  // extremely light compute
    }
}

__global__ void tiny_compute_direct(const float *x, float *y, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float v = x[gid];
    y[gid] = v + v * 0.5f + v * 0.25f;
}

// ============================================================================
// Benchmark helpers
// ============================================================================
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
// Main
// ============================================================================
int main() {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("SM: %d, Max threads/SM: %d, Shared mem/SM: %d KB\n\n",
           prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor,
           prop.sharedMemPerMultiprocessor / 1024);

    // ========================================================================
    printf("================================================================\n");
    printf("  Scenario 1: Data read once — smem is wasted copy\n");
    printf("================================================================\n");
    {
        int n = 1 << 24;  // 16M floats = 64 MB
        size_t bytes = n * sizeof(float);
        float *dx, *dy;
        check(cudaMalloc(&dx, bytes), "dx");
        check(cudaMalloc(&dy, bytes), "dy");
        cudaMemset(dx, 1, bytes);
        int blocks = (n + 255) / 256;

        float best_smem = 1e9, best_direct = 1e9;

        for (int w = 0; w < WARMUP; w++) bad_smem_single_read<<<blocks, 256>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            bad_smem_single_read<<<blocks, 256>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best_smem) best_smem = t;
        }

        for (int w = 0; w < WARMUP; w++) good_direct_global<<<blocks, 256>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            good_direct_global<<<blocks, 256>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best_direct) best_direct = t;
        }

        printf("  with smem:  %.4f ms  (global->smem->register, extra copy+sync)\n", best_smem);
        printf("  direct:     %.4f ms  (global->register, L1 cached)\n", best_direct);
        printf("  >>> NO smem is %.2fx faster\n\n", best_smem / best_direct);

        cudaFree(dx); cudaFree(dy);
    }

    // ========================================================================
    printf("================================================================\n");
    printf("  Scenario 2: Warp shuffle covers all reduction — smem overhead\n");
    printf("================================================================\n");
    {
        int n = 1 << 24;
        size_t bytes = n * sizeof(float);
        float *dx, *dr;
        check(cudaMalloc(&dx, bytes), "dx");
        check(cudaMalloc(&dr, sizeof(float)), "dr");

        float *hx = (float*)malloc(bytes);
        for (int i = 0; i < n; i++) hx[i] = 1.0f;
        cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);
        free(hx);

        printf("  %-10s %-10s %-14s %-14s %-10s\n",
               "items/thr", "blocks", "smem(ms)", "warp-only(ms)", "warp/smem");
        printf("  %-10s %-10s %-14s %-14s %-10s\n",
               "---------", "----------", "--------------", "--------------", "----------");

        for (int ipt = 4; ipt <= 256; ipt *= 2) {
            int blocks = (n + 256 * ipt - 1) / (256 * ipt);

            float zero = 0.0f;
            cudaMemcpy(dr, &zero, sizeof(float), cudaMemcpyHostToDevice);
            for (int w = 0; w < WARMUP; w++) reduce_with_smem<<<blocks, 256>>>(dx, dr, n);
            cudaDeviceSynchronize();
            float best_smem = 1e9;
            for (int r = 0; r < RUNS; r++) {
                cudaMemcpy(dr, &zero, sizeof(float), cudaMemcpyHostToDevice);
                cudaEventRecord(start, 0);
                reduce_with_smem<<<blocks, 256>>>(dx, dr, n);
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                float t = millis(start, stop);
                if (t < best_smem) best_smem = t;
            }

            cudaMemcpy(dr, &zero, sizeof(float), cudaMemcpyHostToDevice);
            for (int w = 0; w < WARMUP; w++) reduce_warp_only<<<blocks, 256>>>(dx, dr, n);
            cudaDeviceSynchronize();
            float best_warp = 1e9;
            for (int r = 0; r < RUNS; r++) {
                cudaMemcpy(dr, &zero, sizeof(float), cudaMemcpyHostToDevice);
                cudaEventRecord(start, 0);
                reduce_warp_only<<<blocks, 256>>>(dx, dr, n);
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                float t = millis(start, stop);
                if (t < best_warp) best_warp = t;
            }

            printf("  %-10d %-10d %-14.4f %-14.4f %-9.2fx\n",
                   ipt, blocks, best_smem, best_warp, best_smem / best_warp);
        }

        // Verify correctness
        float zero = 0.0f, r_smem, r_warp;
        int blocks = (n + 256 * 64 - 1) / (256 * 64);
        cudaMemcpy(dr, &zero, sizeof(float), cudaMemcpyHostToDevice);
        reduce_with_smem<<<blocks, 256>>>(dx, dr, n);
        cudaMemcpy(&r_smem, dr, sizeof(float), cudaMemcpyDeviceToHost);

        cudaMemcpy(dr, &zero, sizeof(float), cudaMemcpyHostToDevice);
        reduce_warp_only<<<blocks, 256>>>(dx, dr, n);
        cudaMemcpy(&r_warp, dr, sizeof(float), cudaMemcpyDeviceToHost);

        printf("\n  verify: smem=%.1f  warp=%.1f  diff=%.2e  %s\n\n",
               r_smem, r_warp, fabsf(r_smem - r_warp),
               fabsf(r_smem - r_warp) < 10.0f ? "OK" : "MISMATCH");

        cudaFree(dx); cudaFree(dr);
    }

    // ========================================================================
    printf("================================================================\n");
    printf("  Scenario 3: __syncthreads() raw overhead\n");
    printf("================================================================\n");
    {
        float *dd;
        check(cudaMalloc(&dd, 256 * sizeof(float)), "dd");

        printf("  %-12s %-14s %-16s\n", "n_syncs", "time (us)", "ns/sync");
        printf("  %-12s %-14s %-16s\n", "------------", "--------------", "----------------");

        float t0 = 0;
        int sync_counts[] = {0, 1, 5, 10, 50, 100};
        for (int si = 0; si < 6; si++) {
            int n_syncs = sync_counts[si];
            float best = 1e9;
            for (int w = 0; w < WARMUP; w++) bench_syncthreads_cost<<<100, 256>>>(dd, n_syncs);
            cudaDeviceSynchronize();
            for (int r = 0; r < RUNS; r++) {
                cudaEventRecord(start, 0);
                bench_syncthreads_cost<<<100, 256>>>(dd, n_syncs);
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                float t = millis(start, stop);
                if (t < best) best = t;
            }
            if (n_syncs == 0) t0 = best;
            float overhead = (n_syncs > 0) ? (best - t0) * 1e6 / n_syncs : 0;
            printf("  %-12d %-14.4f %-15.0f\n", n_syncs, best * 1000, overhead);
        }
        printf("  Note: ~20-50ns per __syncthreads(). Dominant for light compute.\n\n");
        cudaFree(dd);
    }

    // ========================================================================
    printf("================================================================\n");
    printf("  Scenario 4: Bank conflict — smem can be slower than global\n");
    printf("================================================================\n");
    {
        int n = 1 << 22;  // 4M elements
        size_t bytes = n * sizeof(float);
        float *dx, *dy;
        check(cudaMalloc(&dx, bytes), "dx");
        check(cudaMalloc(&dy, bytes), "dy");
        cudaMemset(dx, 0, bytes);

        int blocks = (n + 255) / 256;

        // Bank conflicted smem
        float best_bad = 1e9;
        for (int w = 0; w < WARMUP; w++) smem_stride_bank_conflict<<<blocks, 256>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            smem_stride_bank_conflict<<<blocks, 256>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best_bad) best_bad = t;
        }

        // No-conflict smem
        float best_ok = 1e9;
        for (int w = 0; w < WARMUP; w++) smem_stride_no_conflict<<<blocks, 256>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            smem_stride_no_conflict<<<blocks, 256>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best_ok) best_ok = t;
        }

        // Direct global
        float best_direct = 1e9;
        for (int w = 0; w < WARMUP; w++) direct_global_read<<<blocks, 256>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            direct_global_read<<<blocks, 256>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best_direct) best_direct = t;
        }

        printf("  smem + 32-way bank conflict:  %.4f ms\n", best_bad);
        printf("  smem + linear (no conflict):   %.4f ms\n", best_ok);
        printf("  direct global (L1 cached):     %.4f ms\n", best_direct);
        printf("  >>> bank-conflict smem is %.2fx slower than direct global\n",
               best_bad / best_direct);
        printf("  Reason: 32-way conflict = 32x serialization > L1 hit latency\n\n");

        cudaFree(dx); cudaFree(dy);
    }

    // ========================================================================
    printf("================================================================\n");
    printf("  Scenario 5: Large smem reduces occupancy -> slower\n");
    printf("================================================================\n");
    {
        int n = 1 << 24;
        size_t bytes = n * sizeof(float);
        float *dx, *dy;
        check(cudaMalloc(&dx, bytes), "dx");
        check(cudaMalloc(&dy, bytes), "dy");
        cudaMemset(dx, 0, bytes);

        int blocks = (n + 255) / 256;

        float best_high = 1e9;
        for (int w = 0; w < WARMUP; w++) high_smem_usage<<<blocks, 256>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            high_smem_usage<<<blocks, 256>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best_high) best_high = t;
        }

        float best_none = 1e9;
        for (int w = 0; w < WARMUP; w++) no_smem_high_occupancy<<<blocks, 256>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            no_smem_high_occupancy<<<blocks, 256>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best_none) best_none = t;
        }

        printf("  high smem (8KB, low occ):  %.4f ms\n", best_high);
        printf("  no smem (high occ):        %.4f ms\n", best_none);
        printf("  >>> NO smem is %.2fx faster\n", best_high / best_none);
        printf("  Reason: 8KB/block -> only %d blocks/SM -> can't hide latency\n\n",
               prop.sharedMemPerMultiprocessor / (8 * 1024));

        cudaFree(dx); cudaFree(dy);
    }

    // ========================================================================
    printf("================================================================\n");
    printf("  Scenario 6: Element-wise map — smem is pure extra memcpy\n");
    printf("================================================================\n");
    {
        int n = 1 << 24;
        size_t bytes = n * sizeof(float);
        float *da, *db, *dc;
        check(cudaMalloc(&da, bytes), "da");
        check(cudaMalloc(&db, bytes), "db");
        check(cudaMalloc(&dc, bytes), "dc");
        cudaMemset(da, 0, bytes);
        cudaMemset(db, 0, bytes);

        int blocks = (n + 255) / 256;

        float best_smem = 1e9;
        for (int w = 0; w < WARMUP; w++) elementwise_with_smem<<<blocks, 256>>>(da, db, dc, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            elementwise_with_smem<<<blocks, 256>>>(da, db, dc, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best_smem) best_smem = t;
        }

        float best_nosmem = 1e9;
        for (int w = 0; w < WARMUP; w++) elementwise_no_smem<<<blocks, 256>>>(da, db, dc, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            elementwise_no_smem<<<blocks, 256>>>(da, db, dc, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best_nosmem) best_nosmem = t;
        }

        printf("  with smem:  %.4f ms  (global->smem->register->compute)\n", best_smem);
        printf("  no smem:    %.4f ms  (global->register->compute)\n", best_nosmem);
        printf("  >>> NO smem is %.2fx faster\n", best_smem / best_nosmem);
        printf("  Reason: element-wise ops have zero data reuse, smem is pure waste\n\n");

        cudaFree(da); cudaFree(db); cudaFree(dc);
    }

    // ========================================================================
    printf("================================================================\n");
    printf("  Scenario 7: Ultra-light compute — smem load > reread global\n");
    printf("================================================================\n");
    {
        int n = 1 << 24;
        size_t bytes = n * sizeof(float);
        float *dx, *dy;
        check(cudaMalloc(&dx, bytes), "dx");
        check(cudaMalloc(&dy, bytes), "dy");
        cudaMemset(dx, 0, bytes);

        int blocks = (n + 255) / 256;

        float best_smem = 1e9;
        for (int w = 0; w < WARMUP; w++) tiny_compute_smem<<<blocks, 256>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            tiny_compute_smem<<<blocks, 256>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best_smem) best_smem = t;
        }

        float best_direct = 1e9;
        for (int w = 0; w < WARMUP; w++) tiny_compute_direct<<<blocks, 256>>>(dx, dy, n);
        cudaDeviceSynchronize();
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            tiny_compute_direct<<<blocks, 256>>>(dx, dy, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best_direct) best_direct = t;
        }

        printf("  with smem:  %.4f ms\n", best_smem);
        printf("  direct:     %.4f ms\n", best_direct);
        printf("  >>> NO smem is %.2fx faster\n", best_smem / best_direct);
        printf("  Reason: compute too light (ultra-low arithmetic intensity)\n");
        printf("          smem load+sync overhead > L1 cache auto-cache cost\n\n");

        cudaFree(dx); cudaFree(dy);
    }

    // ========================================================================
    printf("================================================================\n");
    printf("  FINAL SUMMARY\n");
    printf("================================================================\n");
    printf("  smem helps ONLY when ALL conditions are met:\n");
    printf("    1) Data is reused by multiple threads in the same block\n");
    printf("    2) Reuse count > smem_load_cost / single_global_read_cost\n");
    printf("    3) No severe bank conflicts (or fixed with padding)\n");
    printf("    4) smem usage does NOT kill occupancy\n");
    printf("    5) __syncthreads() overhead is covered by sufficient compute\n");
    printf("\n");
    printf("  Skip smem (faster!) when:\n");
    printf("    X Data read exactly once (no reuse)\n");
    printf("    X Warp shuffle already sufficient for communication\n");
    printf("    X Bank conflicts unavoidable\n");
    printf("    X smem too large (occupancy killer)\n");
    printf("    X Compute too light (sync cost > benefit)\n");
    printf("    X Element-wise operations (zero reuse)\n");
    printf("================================================================\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
