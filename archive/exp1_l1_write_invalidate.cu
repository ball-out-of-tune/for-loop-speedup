/**
 * 实验 1: 验证 L1 Write-Through + Write-Invalidate
 *
 * 待验证的命题:
 *   (a) Global store 不会在 L1 中缓存 (Write-Through)
 *   (b) Store 会 Invalidate L1 中对应地址的 cache line (Write-Invalidate)
 *
 * 核心预测:
 *   如果你先 load 地址 X (X 被缓存到 L1)，然后 store 到 X，
 *   再 load X 一次 —— 第三次 load 必须是 L1 miss（因为 store 把 L1 的 X 失效了）。
 *
 * 实验策略:
 *   我们用两个 kernel 对比:
 *
 *   Kernel A (control — load 2 次，之间没有 store):
 *     for i: val = x[i]         // 第1次 load, L1 miss (cold), 然后 L1 缓存
 *     for i: val = x[i]         // 第2次 load, L1 HIT  (warm)
 *     总 HBM 流量: 1× (只第1次读从 HBM 进, 第2次从 L1 读)
 *
 *   Kernel B (experiment — load → store → load, same address):
 *     for i: val = x[i]         // 第1次 load, L1 miss (cold), 然后 L1 缓存
 *     for i: x[i] = val * 2     // Store → write-through to L2, invalidate L1!
 *     for i: val = x[i]         // 第3次 load, 如果 write-invalidate 成立: L1 MISS
 *     总 HBM 流量: 2× (第1次 read + 第3次 read) + 1× write
 *
 *   如果命题成立:
 *     - B 的第3次 load 耗时 > A 的第2次 load 耗时 (都减去 store 开销后)
 *     - B 的 L1 hit rate < A 的 L1 hit rate (可用 Nsight Compute 直接看)
 *
 *   额外对照 Kernel C (load → store other → load):
 *     for i: val = x[i]         // L1 缓存 x
 *     for i: other[i] = val     // Store to DIFFERENT address
 *     for i: val = x[i]         // 如果 L1 还在: HIT. 如果 store 把 L1 evict 了: MISS
 *     这区分了 "write-invalidate (只失效匹配地址)" vs "store evicts L1 indiscriminately"
 *
 * 编译:
 *   nvcc -o exp1_l1 exp1_l1_write_invalidate.cu
 * 运行:
 *   ./exp1_l1
 * 用 Nsight Compute 验证:
 *   ncu --metrics l1tex__t_sectors_hit_rate,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit_rate \
 *       ./exp1_l1
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE    (16 * 1024)    // 16K floats = 64 KB (fits in L1: GA107 has 128 KB/SM)
#define BLOCK_SIZE    256
#define GRID_SIZE     20            // match SM count, 1 block per SM
#define WARMUP        5
#define RUNS          30

// ============================================================================
// Kernel A: 同一个 array 读两次 (no stores in between)
// 期望: 第二次读应该是 L1 hit
// ============================================================================
__global__ void load_twice_no_store(const float * __restrict__ x,
                                     float * __restrict__ sink1,
                                     float * __restrict__ sink2,
                                     int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    float sum1 = 0.0f, sum2 = 0.0f;

    // Pass 1: load x (cold → fills L1)
    for (int i = tid; i < n; i += stride) {
        sum1 += x[i];
    }

    // Pass 2: load x AGAIN (should be L1 HIT)
    for (int i = tid; i < n; i += stride) {
        sum2 += x[i];
    }

    if (tid == 0) {
        sink1[blockIdx.x] = sum1;
        sink2[blockIdx.x] = sum2;
    }
}

// ============================================================================
// Kernel B: load → store TO SAME ADDRESS → load
// 预测: 第三次 load 是 L1 miss (store 把 L1 invalidate 了)
// ============================================================================
__global__ void load_store_same_load(float * __restrict__ x,
                                      float * __restrict__ sink1,
                                      float * __restrict__ sink2,
                                      int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    float sum1 = 0.0f, sum2 = 0.0f;

    // Pass 1: load x (cold → fills L1)
    for (int i = tid; i < n; i += stride) {
        sum1 += x[i];
    }

    // Pass 2: store TO THE SAME x (write-through → invalidates L1 for x)
    //         我们写回相同的值, 不改变语义但触发 store
    for (int i = tid; i < n; i += stride) {
        x[i] = x[i];   // store to same address → invalidates L1 entry for x[i]
    }

    // Pass 3: load x AGAIN
    // 如果 write-invalidate 成立: L1 里 x 的数据被 Pass 2 失效了 → L1 MISS
    // 如果 write-invalidate 不成立: L1 里 x 的数据还在 → L1 HIT
    for (int i = tid; i < n; i += stride) {
        sum2 += x[i];
    }

    if (tid == 0) {
        sink1[blockIdx.x] = sum1;
        sink2[blockIdx.x] = sum2;
    }
}

// ============================================================================
// Kernel C: load → store TO DIFFERENT ARRAY → load
// 预测: 第三次 load 可能是 L1 hit (store 到不同地址不 invalidate x 的 L1)
//       但如果 store 的地址和 x 共享 L1 set, 可能被 evict (capacity eviction)
// ============================================================================
__global__ void load_store_other_load(const float * __restrict__ x,
                                       float * __restrict__ y,    // different array!
                                       float * __restrict__ sink1,
                                       float * __restrict__ sink2,
                                       int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    float sum1 = 0.0f, sum2 = 0.0f;

    // Pass 1: load x (cold → fills L1 with x)
    for (int i = tid; i < n; i += stride) {
        sum1 += x[i];
    }

    // Pass 2: store to DIFFERENT array y (not x!)
    // 这不会 invalidate x 在 L1 中的数据 (y 不在 L1 中)
    // 但如果 L1 满了, y 的 store 也不应该影响 x (x 更老, LRU 会更优先保留)
    for (int i = tid; i < n; i += stride) {
        y[i] = sum1;   // store to different address than x
    }

    // Pass 3: load x AGAIN
    // 如果 write-invalidate 是精确的 (只失效匹配地址): x 还在 L1 → HIT
    // 如果 store 会 indiscriminately evict: x 可能被 y 挤出 → MISS
    for (int i = tid; i < n; i += stride) {
        sum2 += x[i];
    }

    if (tid == 0) {
        sink1[blockIdx.x] = sum1;
        sink2[blockIdx.x] = sum2;
    }
}

// ============================================================================
// Kernel D: 纯 store 带宽测量 (用于归一化)
// 测量单独写一遍 y 的耗时，用于从 Kernel B 的时间中减掉 store 贡献
// ============================================================================
__global__ void pure_store(float * __restrict__ y, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += stride) {
        y[i] = 1.0f;
    }
}

// ============================================================================
// Kernel E: load → store → load 但 store 使用 streaming (cs) hint
// 用 volatile + 特定 pattern 希望编译器生成 st.global.cs
// 这测试: streaming store 是否也触发 L1 invalidate?
// (实际上 st.cs 主要影响 L2 eviction priority, L1 行为应该不变)
// ============================================================================

// ============================================================================
// Benchmark helpers
// ============================================================================
float millis(cudaEvent_t start, cudaEvent_t stop) {
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

float bench_kernel(void (*kernel)(), dim3 grid, dim3 block,
                   cudaEvent_t start, cudaEvent_t stop) {
    float best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start, 0);
        kernel<<<grid, block>>>();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }
    return best;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    int n = ARRAY_SIZE;
    int total_bytes = n * sizeof(float);
    int total_threads = GRID_SIZE * BLOCK_SIZE;

    printf("============================================================\n");
    printf("  实验 1: 验证 L1 Write-Through + Write-Invalidate\n");
    printf("============================================================\n\n");
    printf("  数组大小: %d floats = %.1f KB\n", n, total_bytes / 1024.0f);
    printf("  每个 SM 的 L1: 128 KB (GA107)\n");
    printf("  数据 fits in L1: %s\n\n", (total_bytes <= 128*1024) ? "YES" : "NO");

    // ---- Allocate ----
    float *h_x = (float*)malloc(total_bytes);
    float *h_y = (float*)malloc(total_bytes);
    for (int i = 0; i < n; i++) {
        h_x[i] = (float)(rand()) / RAND_MAX;
        h_y[i] = 0.0f;
    }

    float *d_x, *d_y, *d_sink1, *d_sink2;
    check(cudaMalloc(&d_x, total_bytes), "d_x");
    check(cudaMalloc(&d_y, total_bytes), "d_y");
    check(cudaMalloc(&d_sink1, GRID_SIZE * sizeof(float)), "d_sink1");
    check(cudaMalloc(&d_sink2, GRID_SIZE * sizeof(float)), "d_sink2");

    check(cudaMemcpy(d_x, h_x, total_bytes, cudaMemcpyHostToDevice), "H2D x");
    check(cudaMemcpy(d_y, h_y, total_bytes, cudaMemcpyHostToDevice), "H2D y");

    dim3 grid(GRID_SIZE), block(BLOCK_SIZE);
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");

    // ---- Warmup all kernels ----
    for (int i = 0; i < WARMUP; i++) {
        load_twice_no_store<<<grid, block>>>(d_x, d_sink1, d_sink2, n);
        load_store_same_load<<<grid, block>>>(d_x, d_sink1, d_sink2, n);
        load_store_other_load<<<grid, block>>>(d_x, d_y, d_sink1, d_sink2, n);
        pure_store<<<grid, block>>>(d_y, n);
    }
    check(cudaDeviceSynchronize(), "warmup");

    // ---- Measure pure store BW (for normalization) ----
    float t_store;
    {
        float best = 1e9;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            pure_store<<<grid, block>>>(d_y, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best) best = t;
        }
        t_store = best;
    }
    float store_bw = total_bytes / (t_store / 1000.0f) / 1e9f;
    printf("--- Baseline: pure store ---\n");
    printf("  Time:    %.4f ms\n", t_store);
    printf("  BW:      %.1f GB/s  (write-only, no L1 involved)\n\n", store_bw);

    // ========================================================================
    // Kernel A: load twice, no store between
    // ========================================================================
    // total operations: 2 reads
    // expected: read 1 from HBM → L1, read 2 from L1 (HIT)
    float t_a;
    {
        float best = 1e9;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            load_twice_no_store<<<grid, block>>>(d_x, d_sink1, d_sink2, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best) best = t;
        }
        t_a = best;
    }
    // 2 reads total. Effective BW based on HBM traffic:
    // Read 1: HBM → L2 → L1
    // Read 2: L1 hit (no HBM involved)
    // So effective HBM reads = 1× total_bytes
    float bw_a_hbm = total_bytes / (t_a / 1000.0f) / 1e9f;       // 1-read model
    float bw_a_2read = (total_bytes * 2.0f) / (t_a / 1000.0f) / 1e9f; // 2-read model

    printf("--- Kernel A: load → load (no store in between) ---\n");
    printf("  Time:    %.4f ms\n", t_a);
    printf("  BW (1-read model, expect ≤192 GB/s):  %.1f GB/s\n", bw_a_hbm);
    printf("  BW (2-read model, HBM+L1):            %.1f GB/s\n", bw_a_2read);
    printf("  Prediction: if 2nd load hits L1, 1-read-BW ≈ HBM peak (~192 GB/s).\n");
    printf("  If >192 GB/s → L1 is serving 2nd load (proves L1 caches loads).\n\n");

    // ========================================================================
    // Kernel B: load → store (same addr) → load
    // ========================================================================
    // total operations: 2 reads + 1 write
    // expected: read 1 from HBM, store goes to L2 (write-through, invalidates L1),
    //           read 2 from L2 (L1 MISS, because L1 was invalidated)
    float t_b;
    {
        float best = 1e9;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            load_store_same_load<<<grid, block>>>(d_x, d_sink1, d_sink2, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best) best = t;
        }
        t_b = best;
    }
    // HBM traffic model:
    // Read 1: HBM → L2 → L1 (~1× n)
    // Write:  L1 → L2 (write-through) → eventually HBM (~1× n)
    // Read 3: L2 → ... (if L1 invalidated: L2 supplies; if L1 cached: L1 supplies)
    float time_without_store = t_b - t_store;  // rough substraction
    float bw_b_2read = (total_bytes * 2.0f) / (t_b / 1000.0f) / 1e9f;
    float bw_b_nostore = (total_bytes * 2.0f) / (time_without_store / 1000.0f) / 1e9f;

    printf("--- Kernel B: load → store(same) → load ---\n");
    printf("  Time:    %.4f ms  (includes 1 write)\n", t_b);
    printf("  Est. time minus write:  %.4f ms\n", time_without_store);
    printf("  BW (2-read model, no adj):           %.1f GB/s\n", bw_b_2read);
    printf("  BW (2-read model, minus store time):  %.1f GB/s\n", bw_b_nostore);
    printf("  Prediction: if write-invalidate, read 3 misses L1 → slower than Kernel A.\n");
    printf("  If time > Kernel A by more than store overhead: L1 was invalidated.\n\n");

    // ========================================================================
    // Kernel C: load → store (DIFFERENT array) → load
    // ========================================================================
    float t_c;
    {
        float best = 1e9;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            load_store_other_load<<<grid, block>>>(d_x, d_y, d_sink1, d_sink2, n);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best) best = t;
        }
        t_c = best;
    }
    float time_c_nostore = t_c - t_store;
    float bw_c_nostore = (total_bytes * 2.0f) / (time_c_nostore / 1000.0f) / 1e9f;

    printf("--- Kernel C: load → store(different array) → load ---\n");
    printf("  Time:    %.4f ms  (includes 1 write)\n", t_c);
    printf("  Est. time minus write:  %.4f ms\n", time_c_nostore);
    printf("  BW (2-read model, minus store):       %.1f GB/s\n", bw_c_nostore);
    printf("  Prediction: L1 still has x → read 3 ≈ L1 hit.\n");
    printf("  If C < B (after store correction): write-invalidate is ADDRESS-SPECIFIC.\n\n");

    // ========================================================================
    // Analysis
    // ========================================================================
    printf("============================================================\n");
    printf("  Analysis\n");
    printf("============================================================\n\n");

    printf("  Key metrics for 2-read portion (after subtracting store time):\n");
    printf("    Kernel A (no store):              %.4f ms  (reference)\n", t_a);
    printf("    Kernel B (store same, no-store):  %.4f ms  (if > A: L1 invalidated)\n",
           time_without_store);
    printf("    Kernel C (store other, no-store): %.4f ms  (should ≈ A)\n",
           time_c_nostore);
    printf("\n");

    float ratio_b_a = time_without_store / t_a;
    float ratio_c_a = time_c_nostore / t_a;

    printf("  B/A ratio: %.2f  (>1.0 = B slower, L1 invalidated by store)\n", ratio_b_a);
    printf("  C/A ratio: %.2f  (≈1.0 = store to other addr doesn't hurt)\n", ratio_c_a);
    printf("\n");

    if (ratio_b_a > 1.05f && ratio_c_a < 1.05f) {
        printf("  ✓ RESULT CONSISTENT with L1 Write-Invalidate:\n");
        printf("    Store to same address invalidated L1 → re-read was slower.\n");
        printf("    Store to different address did NOT invalidate → re-read fast.\n");
    } else if (ratio_b_a > 1.05f && ratio_c_a > 1.05f) {
        printf("  △ Both B and C slower than A.\n");
        printf("    Possible explanations:\n");
        printf("    1) L1 capacity: stores to y evicted x from L1 (same cache sets)\n");
        printf("    2) Store bandwidth not fully subtracted\n");
        printf("    → Try running with Nsight Compute to check L1 hit rate directly.\n");
    } else if (ratio_b_a < 1.05f) {
        printf("  ✗ B not slower than A.\n");
        printf("    Possible explanations:\n");
        printf("    1) L1 is NOT invalidated by stores (contradicts NVIDIA docs)\n");
        printf("    2) L2 is fast enough to mask the difference (small array fits entirely in L2)\n");
        printf("    → Try larger array or Nsight Compute metrics.\n");
    }

    printf("\n");
    printf("  To verify with hardware counters:\n");
    printf("    ncu --metrics l1tex__t_sectors_hit_rate \\\n");
    printf("        --kernel-name regex:load ./exp1_l1\n");
    printf("\n");
    printf("  Expected under Write-Invalidate:\n");
    printf("    Kernel A L1 hit rate >> Kernel B L1 hit rate\n");
    printf("    Kernel C L1 hit rate ≈ Kernel A L1 hit rate\n");

    // ---- Cleanup ----
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_sink1); cudaFree(d_sink2);
    free(h_x); free(h_y);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
