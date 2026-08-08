/**
 * 实验 3: 探测 L1 Cache 的 Eviction 行为 (容量 + LRU 特征)
 *
 * 待验证的命题:
 *   "L1 默认使用 LRU 替换策略"
 *
 * 实验策略:
 *   GPU 上直接证明 "是 LRU 不是 FIFO 不是 Random" 极其困难,
 *   因为 (a) 我们不知道 L1 set mapping function,
 *        (b) 无法用单次 load 延迟区分 L1 hit vs L2 hit (pipelining 掩盖了).
 *
 *   所以我们采用两层实验: 宏观 (吞吐量拐点) + 微观 (Nsight Compute 计数器).
 *
 * 实验 3A: L1 容量拐点 (Working Set Size Scan)
 *   每个 block 处理一个固定大小的 working set.
 *   当 working set < L1 per SM → 大部分 hits → 高吞吐
 *   当 working set > L1 per SM → 频繁 eviction → 低吞吐
 *   拐点 ≈ L1 容量 per SM
 *
 * 实验 3B: Re-access 间隔 vs 命中率
 *   固定 working set size, 改变 re-access 间隔 (stride).
 *   如果 L1 是 LRU: 只要 working set < associativity × sets, 循环访问就全命中.
 *   从命中→缺失的过渡点可以推断 associativity 的近似范围.
 *
 * 编译:
 *   nvcc -o exp3_l1 exp3_l1_eviction.cu
 * 运行:
 *   ./exp3_l1
 * Nsight Compute (推荐):
 *   ncu --metrics l1tex__t_sectors_hit_rate,l1tex__t_sectors_lookup_hit, \
 *        l1tex__t_sectors_lookup_miss ./exp3_l1
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE  256
#define WARMUP      5
#define RUNS        30

// ============================================================================
// 实验 3A: Working Set Size Scan
// ============================================================================
// 每个 block 只处理自己的 working set (不和别的 block 共享)
// 所以每个 SM 的 L1 只需要缓存 1 个 block 的 working set
// (因为 block 之间不共享 L1 — 每个 SM 有独立的 L1)
//
// 当 working set size > L1/SM 时, 出现 L1 thrashing → 吞吐下降

__global__ void working_set_scan(
    const float * __restrict__ x,   // large array (one region per block)
    float * __restrict__ sink,
    int working_set_elems,          // elements accessed per block
    int stride                      // access stride (controls re-access pattern)
) {
    int tid = threadIdx.x;
    int block_offset = blockIdx.x * working_set_elems;
    const float *my_x = x + block_offset;

    float sum = 0.0f;

    // Access my working set ITEMS_PER_THREAD times (cycling through)
    int iters = 128;  // enough iterations to amplify hit/miss difference
    for (int iter = 0; iter < iters; iter++) {
        for (int i = tid; i < working_set_elems; i += BLOCK_SIZE) {
            sum += my_x[i];
        }
    }

    if (tid == 0) sink[blockIdx.x] = sum;
}

// ============================================================================
// 实验 3B: LRU vs Other Eviction — Re-access Pattern Test
// ============================================================================
// 思路: 在一个已知会 evict 的场景下, 对比两种访问模式:
//
//   模式 1 (LRU-friendly, "cyclic"):
//     访问顺序: A → B → C → D → A → B → C → D → ...
//     在 LRU 下: 如果 ABCD 总大小 fits in cache → 全命中
//     在 FIFO 下: 同上 → 也全命中 (因为恰好是 round-robin)
//     → 这个模式不能区分 LRU 和 FIFO
//
//   模式 2 (LRU-friendly, "re-access oldest", 专为区分设计):
//     Step 1: 访问 A, B, C, D, E (比 cache associativity 多一个)
//     Step 2: 立即再访问 A (A 是 LRU, 应该已被 E 挤出)
//     Step 3: 访问 F (新的)
//     Step 4: 访问 A (如果 LRU: A 在 Step 2 被重新访问后变成 MRU,
//                           E 是 LRU, F 把 E 挤出, A 还在)
//             (如果 FIFO: A 是最老的, Step 2 不改变顺序, F 把 A 挤出)
//
//   但我们无法控制 set mapping! 所以改用一个统计方法:
//   在一个大的 working set 里, 随机 re-access 一部分 "old" elements,
//   看整体吞吐. 如果 LRU: re-access 使 old 变 young → 保留率高 → 快.
//   如果 FIFO: re-access 不改变位置 → 保留率无变化.

// 我们简化设计: 对比两种访问模式在一个超过 L1 的 working set 上的吞吐.
//
// 模式 A (sequential): 顺序访问 → 自然的 LRU/FIFO 行为
// 模式 B (temporal-locality): 访问新数据 + 周期性地回溯访问老数据
//   如果 LRU: 回溯访问让老数据"续命" → 后续访问更容易命中
//   如果 FIFO: 回溯访问不改变位置 → 老数据最终被挤出

__global__ void sequential_access(
    const float * __restrict__ x,
    float * __restrict__ sink,
    int n_per_block,
    int iters
) {
    int tid = threadIdx.x;
    const float *my_x = x + blockIdx.x * n_per_block;
    float sum = 0.0f;

    for (int iter = 0; iter < iters; iter++) {
        for (int i = tid; i < n_per_block; i += BLOCK_SIZE) {
            sum += my_x[i];
        }
    }
    if (tid == 0) sink[blockIdx.x] = sum;
}

__global__ void temporal_locality_access(
    const float * __restrict__ x,
    float * __restrict__ sink,
    int n_per_block,
    int iters
) {
    int tid = threadIdx.x;
    int tt = BLOCK_SIZE;
    const float *my_x = x + blockIdx.x * n_per_block;
    float sum = 0.0f;

    // Pattern: access new chunk, then re-access an old chunk
    // This creates temporal locality that LRU exploits but FIFO doesn't
    //
    // Specifically: each thread walks its stride, but every K iterations
    // it goes back and re-reads the first chunk it accessed.
    // Under LRU: those old chunks stay because we keep touching them.
    // Under FIFO: those old chunks are evicted regardless.

    int chunk_size = tt;  // one element per thread per "chunk"
    int num_chunks = n_per_block / tt;

    for (int iter = 0; iter < iters; iter++) {
        // Forward pass: access all chunks in order
        for (int c = 0; c < num_chunks; c++) {
            int i = c * tt + tid;
            if (i < n_per_block) {
                sum += my_x[i];
            }
        }
        // Backward pass: re-access old chunks (LRU should keep them alive)
        // Only do this every 4th iteration to see the effect
        if (iter % 4 == 0) {
            for (int c = 0; c < num_chunks / 4; c++) {
                int i = c * tt + tid;
                if (i < n_per_block) {
                    sum += my_x[i];  // touch the oldest chunks
                }
            }
        }
    }
    if (tid == 0) sink[blockIdx.x] = sum;
}

// ============================================================================
// 实验 3C: 检测 Eviction 是 "all-or-nothing cache line" 还是 "sector-level"
// ============================================================================
// Cache line = 128 bytes = 4 sectors × 32 bytes
// 如果只访问 32 bytes (1 sector):
//   - 全部 128 bytes 被加载到 cache line
//   - 但只有那 1 个 sector 标记为 valid
//   - 另外 3 个 sector 的 cache 空间被浪费
//
// 测试: 访问同样数量的数据, 但用 dense (128B aligned) vs sparse (32B aligned, stride=128B)
// dense: 每个 cache line 全部 4 sectors 都被用到 → 有效利用 cache
// sparse: 每个 cache line 只有 1 sector 被用到 → 浪费 75% cache 容量
//
// 如果 sparse 版本在更小的 working set 就 thrash → 证明 sector cache 的存在

__global__ void dense_access(
    const float * __restrict__ x,
    float * __restrict__ sink,
    int n,
    int iters
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    float sum = 0.0f;
    for (int iter = 0; iter < iters; iter++) {
        for (int i = tid; i < n; i += stride) {
            sum += x[i];   // 128B aligned, sequential → uses all 4 sectors
        }
    }
    if (tid == 0) sink[blockIdx.x] = sum;
}

__global__ void sparse_access(
    const float * __restrict__ x,
    float * __restrict__ sink,
    int n,
    int iters
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tt = gridDim.x * blockDim.x;
    float sum = 0.0f;

    // Access 1 float out of every 32 (to hit only 1 sector per 128B cache line)
    // stride = 32 floats = 128 bytes = 1 cache line
    int stride_per_thread = tt * 32;
    for (int iter = 0; iter < iters; iter++) {
        for (int i = tid * 32; i < n; i += stride_per_thread) {
            sum += x[i];   // each access = new cache line, only 1 sector used
        }
    }
    if (tid == 0) sink[blockIdx.x] = sum;
}

// ============================================================================
// Bench
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

float bench(void (*kernel)(), dim3 grid, dim3 block,
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
    printf("============================================================\n");
    printf("  实验 3: L1 Eviction 行为探测\n");
    printf("============================================================\n\n");

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");

    // ========================================================================
    // 实验 3A: Working Set Size Scan
    // ========================================================================
    printf("--- 3A: Working Set Size Scan (L1 Capacity Detection) ---\n\n");
    printf("  Hypothesis: throughput drops when working set > L1 per SM.\n");
    printf("  GA107 L1 per SM = 128 KB.\n\n");

    int grid_size = 20;  // 1 block per SM
    dim3 grid_3a(grid_size), block_3a(BLOCK_SIZE);

    // Working set sizes to test (in KB per block)
    int ws_kb[]   = {2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256};
    int n_ws = sizeof(ws_kb) / sizeof(ws_kb[0]);

    printf("  %-12s %-12s %-12s %s\n",
           "WS(KB)", "Time(ms)", "BW(GB/s)", "Note");
    printf("  ------------------------------------------------\n");

    float *huge_x;
    int max_ws = 256 * 1024 / 4;  // 256K floats per block
    int total_floats = grid_size * max_ws;
    check(cudaMallocHost(&huge_x, total_floats * sizeof(float)), "host alloc");
    for (int i = 0; i < total_floats; i++) huge_x[i] = (float)(rand()) / RAND_MAX;

    float *d_x, *d_sink;
    check(cudaMalloc(&d_x, total_floats * sizeof(float)), "d_x");
    check(cudaMalloc(&d_sink, grid_size * sizeof(float)), "d_sink");
    check(cudaMemcpy(d_x, huge_x, total_floats * sizeof(float), cudaMemcpyHostToDevice), "H2D");

    for (int w = 0; w < n_ws; w++) {
        int elems = ws_kb[w] * 1024 / 4;  // working set in floats per block

        float best = 1e9;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            working_set_scan<<<grid_3a, block_3a>>>(d_x, d_sink, elems, 1);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best) best = t;
        }

        // Data accessed in this kernel: elems * grid_size * iters * 4 bytes
        long long total_access = (long long)elems * grid_size * 128 * 4;
        float bw = total_access / (best / 1000.0f) / 1e9f;

        // If everything hits L1 after first pass, effective HBM BW should be low
        // If thrashing, HBM BW should approach DRAM peak (~192 GB/s)
        const char *note = "";
        if (ws_kb[w] <= 128) note = "<= L1";
        else                 note = "> L1, expect thrash";

        printf("  %-12d %-12.4f %-12.1f %s\n", ws_kb[w], best, bw, note);
    }

    printf("\n  Expected: BW stays low (L1 hits) until ~128 KB, then jumps up.\n");
    printf("  The jump point ≈ L1 capacity per SM.\n\n");

    cudaFree(d_x); cudaFree(d_sink);

    // ========================================================================
    // 实验 3B: LRU vs FIFO Pattern Test
    // ========================================================================
    printf("--- 3B: Sequential vs Temporal-Locality Access ---\n\n");

    // Use a working set of 192 KB per block (larger than L1)
    // So eviction happens in both patterns.
    // If LRU: temporal-locality pattern keeps "touched" data alive → fewer misses.
    // If FIFO: no difference.

    int n_per_block_3b = 192 * 1024 / 4;  // 192K floats = 192 KB per block
    int iters_3b = 16;

    total_floats = grid_size * n_per_block_3b;
    check(cudaMallocHost(&huge_x, total_floats * sizeof(float)), "host 3b");
    for (int i = 0; i < total_floats; i++) huge_x[i] = (float)(rand()) / RAND_MAX;

    check(cudaMalloc(&d_x, total_floats * sizeof(float)), "d_x 3b");
    check(cudaMalloc(&d_sink, grid_size * sizeof(float)), "d_sink 3b");
    check(cudaMemcpy(d_x, huge_x, total_floats * sizeof(float), cudaMemcpyHostToDevice), "H2D 3b");

    dim3 grid_3b(grid_size), block_3b(BLOCK_SIZE);

    // Sequential
    float t_seq;
    {
        float best = 1e9;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            sequential_access<<<grid_3b, block_3b>>>(d_x, d_sink,
                n_per_block_3b, iters_3b);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best) best = t;
        }
        t_seq = best;
    }

    // Temporal locality
    float t_temp;
    {
        float best = 1e9;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            temporal_locality_access<<<grid_3b, block_3b>>>(d_x, d_sink,
                n_per_block_3b, iters_3b);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best) best = t;
        }
        t_temp = best;
    }

    printf("  Sequential access:          %.4f ms\n", t_seq);
    printf("  Temporal-locality access:   %.4f ms\n", t_temp);
    printf("  Ratio (temp/seq):           %.2f\n", t_temp / t_seq);
    printf("\n");
    printf("  If LRU: temp < seq (re-access keeps data alive in L1).\n");
    printf("  If FIFO: temp ≈ seq (re-access doesn't change eviction order).\n");
    printf("  If temp >> seq: temporal pattern has extra overhead.\n");
    printf("  NOTE: This test is suggestive, not conclusive without Nsight.\n\n");

    cudaFree(d_x); cudaFree(d_sink);

    // ========================================================================
    // 实验 3C: Sector Cache Detection
    // ========================================================================
    printf("--- 3C: Dense vs Sparse Access (Sector Cache Detection) ---\n\n");

    // Same total number of floats accessed, but:
    // Dense:  every float sequentially → 4 sectors used per cache line
    // Sparse: 1 float every 128 bytes → only 1 sector per cache line
    //
    // If sector cache: sparse should thrash at a smaller effective working set
    // (because 75% of cache capacity is wasted on unused sectors)

    int n_3c = 128 * 1024 / 4;  // 128K floats (spans many cache lines)
    int iters_3c = 64;

    printf("  Array: %d floats = %.1f MB, %d iterations\n",
           n_3c, n_3c * 4.0f / 1e6f, iters_3c);
    printf("  Dense: %d cache lines, all sectors used.\n",
           n_3c / 32);  // 32 floats per 128B cache line
    printf("  Sparse: same floats touched, but 1 sector per cache line.\n\n");

    float *d_x_3c, *d_sink_3c;
    check(cudaMalloc(&d_x_3c, n_3c * sizeof(float)), "d_x 3c");
    check(cudaMalloc(&d_sink_3c, grid_size * sizeof(float)), "d_sink 3c");

    float *h_3c = (float*)malloc(n_3c * sizeof(float));
    for (int i = 0; i < n_3c; i++) h_3c[i] = (float)(rand()) / RAND_MAX;
    check(cudaMemcpy(d_x_3c, h_3c, n_3c * sizeof(float), cudaMemcpyHostToDevice), "H2D 3c");

    // Dense
    float t_dense;
    {
        float best = 1e9;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            dense_access<<<grid_3b, block_3b>>>(d_x_3c, d_sink_3c, n_3c, iters_3c);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best) best = t;
        }
        t_dense = best;
    }

    // Sparse
    float t_sparse;
    {
        float best = 1e9;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(start, 0);
            sparse_access<<<grid_3b, block_3b>>>(d_x_3c, d_sink_3c, n_3c, iters_3c);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            float t = millis(start, stop);
            if (t < best) best = t;
        }
        t_sparse = best;
    }

    // Both kernels access the same number of floats!
    // But sparse uses 4× more cache lines
    printf("  Dense:   %.4f ms\n", t_dense);
    printf("  Sparse:  %.4f ms\n", t_sparse);
    printf("  Ratio (sparse/dense): %.2f\n\n", t_sparse / t_dense);

    if (t_sparse > t_dense * 2.0f) {
        printf("  ✓ Sparse access significantly slower.\n");
        printf("    Consistent with sector cache: sparse wastes 75%% cache capacity\n");
        printf("    (each 128B cache line only uses 1 of 4 sectors),\n");
        printf("    causing earlier eviction and more L1 misses.\n");
    } else if (t_sparse > t_dense * 1.5f) {
        printf("  △ Sparse somewhat slower — sector cache effect present but mild.\n");
        printf("    L2 may be absorbing some of the miss penalty.\n");
    } else {
        printf("  △ Sparse ≈ Dense — sector effect not detected at this data size.\n");
        printf("    Try larger array or check L1 hit rate with Nsight Compute.\n");
    }

    printf("\n");
    printf("  If your card is A100 (40 MB L2): L2 masks L1 effects more.\n");
    printf("  If your card is 3050 Ti (2 MB L2): L1 effects are more visible.\n\n");

    // Cleanup
    cudaFree(d_x_3c); cudaFree(d_sink_3c);
    cudaFreeHost(huge_x);
    free(h_3c);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    printf("============================================================\n");
    printf("  Summary & How to Verify with Nsight Compute\n");
    printf("============================================================\n\n");
    printf("  For definitive answers on L1 LRU, run:\n");
    printf("    ncu --metrics \\\n");
    printf("      l1tex__t_sectors_hit_rate,\\\n");
    printf("      l1tex__t_sectors_lookup_hit,\\\n");
    printf("      l1tex__t_sectors_lookup_miss,\\\n");
    printf("      l1tex__t_sectors_pipe_lsu_mem_global_op_ld_hit_rate \\\n");
    printf("      ./exp3_l1\n\n");
    printf("  Key metrics:\n");
    printf("    l1tex__t_sectors_hit_rate       — fraction of L1 lookups that hit\n");
    printf("    l1tex__t_sectors_lookup_hit     — absolute L1 hit count\n");
    printf("    l1tex__t_sectors_lookup_miss    — absolute L1 miss count\n");
    printf("  If L1 hit rate drops at working set > 128 KB → L1 capacity confirmed.\n");
    printf("  If sequential hit rate = temporal hit rate → LRU not distinguishable.\n");
    printf("  If temporal hit rate > sequential → consistent with LRU.\n");

    return 0;
}
