/**
 * 实验 2: 验证 L1 Cache 在 Kernel 切换时被 Flush
 *
 * 待验证的命题:
 *   "L1 会在 kernel 切换 (context switch) 时自动 flush"
 *
 * 核心预测:
 *   如果 Kernel 1 把数据 A 加载到了 SM 的 L1 中,
 *   Kernel 2 (在同一个 SM 上运行) 再加载数据 A 时:
 *     - 如果 L1 在 kernel 切换时被 flush → Kernel 2 的 load 是 L1 miss (cold)
 *     - 如果 L1 在 kernel 切换后保留 → Kernel 2 的 load 是 L1 hit
 *
 *   但我们如何知道 Kernel 2 是不是跑在同一个 SM 上?
 *   → 用 CUDA stream + 小 grid 保证顺序执行, 大概率复用同一 SM
 *   → 或者更简单的: 用 grid size = 1 block, 只测一个 SM
 *
 * 实验设计:
 *
 *   设置: grid=(1,1,1), block=(256,1,1)  → 只用 1 个 SM
 *   数组: 16K floats = 64 KB (fits in GA107 128 KB L1)
 *
 *   场景 A (within-kernel):  一个 kernel 里 load 同一数组两次
 *     Pass 1: 第一次 load  → L1 miss (cold), 数据进入 L1
 *     Pass 2: 第二次 load  → L1 HIT
 *     时间 T_A ≈ 1×HBM_read + 1×L1_read
 *
 *   场景 B (cross-kernel):  两个 kernel, 各 load 一次
 *     Kernel 1: load 数组 → L1 miss → 数据进入 L1
 *     Kernel 2: load 同一数组 → ???
 *       如果 L1 flush: L1 miss (cold) → 必须从 L2 读
 *       如果 L1 保留: L1 HIT
 *     时间 T_B ≈ T_K1 + T_K2
 *       如果 flush:    T_B ≈ 2 × HBM_read (每个 kernel 都是 cold read)
 *       如果不 flush:  T_B ≈ 1 × HBM_read + 1 × L1_read ≈ T_A
 *
 *   预测:
 *     如果 T_B >> T_A: L1 在 kernel 间被 flush 了
 *     如果 T_B ≈ T_A:  L1 数据在 kernel 间保留
 *
 *   干扰因素:
 *     1) Kernel launch overhead (~5-10 us) — T_B 比 T_A 多一次 launch
 *     2) L2 可能缓存了数据 — 即使 L1 flush, L2 仍然 hit
 *     3) SM 分配 — 第二次 kernel 不保证分配到同一个 SM
 *
 *   缓解策略:
 *     - 用 cudaStream 保证串行执行
 *     - 用较大的 grid (比如 20 blocks = 20 SMs) 增加统计效应
 *       → 如果 L1 flush: 所有 kernel-2 的 blocks 都是 cold
 *       → 如果不 flush 但 SM 重分配: 部分 blocks 是 warm, 部分是 cold
 *     - 直接对比 within-kernel 第2次读 vs cross-kernel 读的延迟
 *     - 最可靠: 用 Nsight Compute 看 L1 hit rate
 *
 * 编译:
 *   nvcc -o exp2_l1 exp2_l1_flush.cu
 * 运行:
 *   ./exp2_l1
 * Nsight Compute:
 *   ncu --metrics l1tex__t_sectors_hit_rate ./exp2_l1
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE    (16 * 1024)    // 16K floats = 64 KB
#define BLOCK_SIZE    256
#define WARMUP        5
#define RUNS          50

// ============================================================================
// Scenario A: Within-kernel — 同一个 kernel 里 load 数组两次
// ============================================================================
__global__ void within_kernel_load_twice(
    const float * __restrict__ x,
    float * __restrict__ sink,
    int n
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    float sum = 0.0f;

    // Pass 1: cold → fills L1
    for (int i = tid; i < n; i += stride) {
        sum += x[i];
    }

    // Pass 2: should be L1 HIT
    for (int i = tid; i < n; i += stride) {
        sum += x[i];
    }

    if (tid == 0) sink[blockIdx.x] = sum;
}

// ============================================================================
// Scenario B: Cross-kernel — 两个 kernel 各 load 一次
// Kernel B1: 只 Load 一次 (fills L1 if L1 persists)
// Kernel B2: 只 Load 一次 (如果 L1 flush 了, 这是 cold; 否则 warm)
//
// 注意: 我们不能在 kernel B1 结束时检查 "L1 里有什么"。
// 只能通过 B2 的运行时间来推断 B2 的 load 是否 hit L1。
//
// 所以我们还需要一个 reference: "单次 load 的纯 cold 时间"。
// ============================================================================

// 单次 load (用作 cold-miss baseline)
__global__ void load_once(const float * __restrict__ x,
                           float * __restrict__ sink,
                           int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (int i = tid; i < n; i += stride) {
        sum += x[i];
    }
    if (tid == 0) sink[blockIdx.x] = sum;
}

// 这个 kernel 什么都不做, 只是用来 "消耗" 一个 kernel launch,
// 看空 kernel launch 是否也会 flush L1 (如果是 context-switch 触发的话)
__global__ void empty_kernel(float * __restrict__ sink) {
    if (threadIdx.x == 0 && blockIdx.x == 0) sink[0] = 0.0f;
}

// ============================================================================
// Scenario C: Fill L1, launch empty kernel, then check
// 如果空 kernel 也会 flush L1, 说明 flush 是任何 kernel launch 都触发的
// 如果空 kernel 不 flush, 说明 flush 只在 "需要 L1 空间" 时才发生
// ============================================================================

// ============================================================================
// Bench helpers
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

// ============================================================================
// Main
// ============================================================================
int main() {
    int n = ARRAY_SIZE;
    int total_bytes = n * sizeof(float);

    printf("============================================================\n");
    printf("  实验 2: 验证 L1 在 Kernel 切换时被 Flush\n");
    printf("============================================================\n\n");

    // Test multiple grid sizes to see the effect
    int grid_sizes[] = {1, 20};  // 1 SM, all SMs
    int n_grids = 2;

    for (int g = 0; g < n_grids; g++) {
        int ngrid = grid_sizes[g];
        int total_threads = ngrid * BLOCK_SIZE;

        printf("--- Grid = %d block(s) (~%d SM%s) ---\n",
               ngrid, ngrid, ngrid > 1 ? "s" : "");

        // Allocate
        float *h_x = (float*)malloc(total_bytes);
        for (int i = 0; i < n; i++) h_x[i] = (float)(rand()) / RAND_MAX;

        float *d_x, *d_sink;
        check(cudaMalloc(&d_x, total_bytes), "d_x");
        check(cudaMalloc(&d_sink, ngrid * sizeof(float)), "d_sink");
        check(cudaMemcpy(d_x, h_x, total_bytes, cudaMemcpyHostToDevice), "H2D");

        dim3 grid(ngrid), block(BLOCK_SIZE);
        cudaEvent_t start, stop;
        check(cudaEventCreate(&start), "event");
        check(cudaEventCreate(&stop), "event");

        // ---- Warmup ----
        for (int i = 0; i < WARMUP; i++) {
            within_kernel_load_twice<<<grid, block>>>(d_x, d_sink, n);
            load_once<<<grid, block>>>(d_x, d_sink, n);
            empty_kernel<<<1, 1>>>(d_sink);
        }
        check(cudaDeviceSynchronize(), "warmup");

        // ================================================================
        // Measurement 1: single cold read (baseline)
        // ================================================================
        float t_cold;
        {
            float best = 1e9;
            for (int r = 0; r < RUNS; r++) {
                cudaEventRecord(start, 0);
                load_once<<<grid, block>>>(d_x, d_sink, n);
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                float t = millis(start, stop);
                if (t < best) best = t;
            }
            t_cold = best;
        }
        float bw_cold = total_bytes / (t_cold / 1000.0f) / 1e9f;

        // ================================================================
        // Measurement 2: within-kernel load twice
        // ================================================================
        float t_within;
        {
            float best = 1e9;
            for (int r = 0; r < RUNS; r++) {
                cudaEventRecord(start, 0);
                within_kernel_load_twice<<<grid, block>>>(d_x, d_sink, n);
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                float t = millis(start, stop);
                if (t < best) best = t;
            }
            t_within = best;
        }

        // ================================================================
        // Measurement 3: cross-kernel (K1 reads, K2 reads same data)
        // 两个 kernel 串行, 用 default stream 保证顺序
        // ================================================================
        float t_cross;
        {
            float best = 1e9;
            for (int r = 0; r < RUNS; r++) {
                cudaEventRecord(start, 0);
                load_once<<<grid, block>>>(d_x, d_sink, n);         // K1: fills L1
                load_once<<<grid, block>>>(d_x, d_sink, n);         // K2: does L1 persist?
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                float t = millis(start, stop);
                if (t < best) best = t;
            }
            t_cross = best;
        }

        // ================================================================
        // Measurement 4: K1 reads, EMPTY kernel, K3 reads
        // 空 kernel 是否也触发 flush?
        // ================================================================
        float t_cross_empty;
        {
            float best = 1e9;
            for (int r = 0; r < RUNS; r++) {
                cudaEventRecord(start, 0);
                load_once<<<grid, block>>>(d_x, d_sink, n);         // K1: fills L1
                empty_kernel<<<1, 1>>>(d_sink);                      // K2: empty!
                load_once<<<grid, block>>>(d_x, d_sink, n);         // K3: check
                cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);
                float t = millis(start, stop);
                if (t < best) best = t;
            }
            t_cross_empty = best;
        }

        // ================================================================
        // Analysis
        // ================================================================
        printf("  Cold 1×read (baseline):     %.4f ms  (BW=%.1f GB/s)\n",
               t_cold, bw_cold);
        printf("  Within-kernel 2×read:       %.4f ms\n", t_within);
        printf("  Cross-kernel 2×read:        %.4f ms  (K1 → K2)\n", t_cross);
        printf("  Cross-kernel+empty:         %.4f ms  (K1 → empty → K3)\n",
               t_cross_empty);
        printf("\n");

        // Predicted times under different models
        float predicted_flush = t_cold * 2.0f;  // both reads cold
        float predicted_noflush = t_cold + (t_within - t_cold); // warm 2nd read

        printf("  Predicted if L1 FLUSHED:    %.4f ms  (2×cold)\n", predicted_flush);
        printf("  Predicted if L1 PERSISTS:   %.4f ms  (cold + warm)\n", predicted_noflush);
        printf("\n");

        float l1_hit_speedup = (t_cold * 2.0f - t_within) / t_cold;
        printf("  Within-kernel L1 hit benefit: %.2f × cold-read\n", l1_hit_speedup);
        printf("  Cross-kernel effective reads: %.2f × cold-read\n",
               t_cross / t_cold);
        printf("\n");

        if (t_cross > t_cold * 1.8f) {
            printf("  ✓ L1 appears FLUSHED across kernels (t_cross ≈ 2×cold).\n");
        } else if (t_cross < t_cold * 1.3f) {
            printf("  ✓ L1 appears PERSISTENT across kernels (t_cross ≈ 1×cold).\n");
        } else {
            printf("  △ Intermediate result: partial persistence or other effects.\n");
        }

        if (t_cross_empty < t_cross * 0.9f) {
            printf("  △ Empty kernel between reads seems to HELP (??).\n");
            printf("    Maybe empty kernel doesn't disturb L1.\n");
        }

        printf("\n");

        // Cleanup
        cudaFree(d_x); cudaFree(d_sink);
        free(h_x);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    printf("============================================================\n");
    printf("  Interpretation Guide\n");
    printf("============================================================\n\n");
    printf("  If t_cross ≈ 2 × t_cold:\n");
    printf("    → L1 flushed between kernels. Both kernel reads are cold.\n");
    printf("    → Confirms the \"L1 flush on context switch\" claim.\n\n");

    printf("  If t_cross ≈ t_cold + (t_within - t_cold):\n");
    printf("    → L1 persists. Second kernel's read hits L1.\n");
    printf("    → Contradicts the flush claim.\n\n");

    printf("  If t_cross is somewhere in between:\n");
    printf("    → Partial effect (some SMs flushed, or L2 helps).\n");
    printf("    → Use Nsight Compute to check per-SM L1 hit rates:\n");
    printf("      ncu --metrics l1tex__t_sectors_hit_rate ./exp2_l1\n\n");

    return 0;
}
