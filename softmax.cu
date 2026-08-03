/**
 * CUDA Softmax Kernel
 *
 * Softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
 *
 * 和 RMSNorm 的相似点:
 *   - 都是 "reduce + broadcast + element-wise" 的融合 kernel
 *   - 都是 per-row 操作，每个 block 处理一行
 *
 * 和 RMSNorm 的不同点:
 *   - Softmax 需要 TWO reduces (max + sum)，不是 one
 *   - exp() 是超越函数 (~8-16 cycles)，比 mul/add (~1 cycle) 贵很多
 *   - Phase 1 的 reduce 是 MAX (不是 SUM)
 *   - 因此比 RMSNorm 慢一些，但瓶颈仍然是带宽
 *
 * 3-phase 执行流程:
 *   Phase 1: read x → reduce (MAX) → broadcast max
 *   Phase 2: read x → reduce (SUM of exp(x-max)) → broadcast sum
 *   Phase 3: read x → exp(x_i-max) / sum → write out
 *
 * 内存流量: 3 reads + 1 write = 4x data size
 * 相比 RMSNorm 的 1 read + 1 write = 2x，softmax 多读了 2 次
 *
 * 编译: nvcc -o softmax_cuda softmax.cu && ./softmax_cuda
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N_ROWS         1000
#define HIDDEN_SIZE    4096
#define BLOCK_SIZE     256
#define WARMUP         10
#define RUNS           30

// ============================================================================
// Reference Softmax Kernel
// ============================================================================
// Each block handles one row (hidden_size elements).
// 256 threads per block, each thread handles hidden_size/256 = 16 elements.

__global__ void softmax_kernel(
    const float *x,        // [N_ROWS, HIDDEN_SIZE]
    float *out,            // [N_ROWS, HIDDEN_SIZE]
    int n_rows,
    int hidden_size
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    const float *x_row = x + row * hidden_size;
    float *out_row = out + row * hidden_size;

    __shared__ float smem[BLOCK_SIZE];

    // ================================================================
    // Phase 1: find max(x) — reduce with MAX operator
    // ================================================================
    float local_max = -INFINITY;
    for (int i = tid; i < hidden_size; i += total_threads) {
        float v = x_row[i];
        if (v > local_max) local_max = v;
    }

    smem[tid] = local_max;
    __syncthreads();

    // cross-warp reduce (MAX)
    for (int stride = BLOCK_SIZE / 2; stride >= 32; stride >>= 1) {
        if (tid < stride && smem[tid + stride] > smem[tid]) {
            smem[tid] = smem[tid + stride];
        }
        __syncthreads();
    }

    // in-warp reduce (MAX)
    float val = smem[tid];
    float other;

    other = __shfl_down_sync(0xffffffff, val, 16);
    if (other > val) val = other;
    other = __shfl_down_sync(0xffffffff, val, 8);
    if (other > val) val = other;
    other = __shfl_down_sync(0xffffffff, val, 4);
    if (other > val) val = other;
    other = __shfl_down_sync(0xffffffff, val, 2);
    if (other > val) val = other;
    other = __shfl_down_sync(0xffffffff, val, 1);
    if (other > val) val = other;

    // Broadcast max to all 256 threads
    if (tid == 0) smem[0] = val;
    __syncthreads();
    float row_max = smem[0];

    // ================================================================
    // Phase 2: sum(exp(x - max)) — reduce with SUM operator
    // ================================================================
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += total_threads) {
        local_sum += expf(x_row[i] - row_max);
    }

    smem[tid] = local_sum;
    __syncthreads();

    // cross-warp reduce (SUM)
    for (int stride = BLOCK_SIZE / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }

    // in-warp reduce (SUM)
    val = smem[tid];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);

    // Broadcast sum to all 256 threads
    if (tid == 0) smem[0] = val;
    __syncthreads();
    float row_sum = smem[0];

    // ================================================================
    // Phase 3: exp(x_i - max) / sum → write output
    // ================================================================
    for (int i = tid; i < hidden_size; i += total_threads) {
        out_row[i] = expf(x_row[i] - row_max) / row_sum;
    }
}


// ============================================================================
// L2 Cache Verification Kernels
// ============================================================================
// Hypothesis: Phase 2/3 reads of x hit L2 cache (not HBM), so actual HBM
// traffic is less than our 3-read estimate. This explains BW > 100%.
//
// Experiment:
//   K_normal: 3 phases back-to-back (L2 can help)
//   K_evict:  same 3 phases but between phases, each thread reads a large
//             dummy buffer (> L2 size) to force L2 eviction.
//             If L2 was helping, this should be measurably slower.
//
// The dummy buffer is 20 MB (larger than typical L2 of 2-6 MB), and we stride
// through it so each thread touches different cache lines.

#define DUMMY_SIZE     (5 * 1024 * 1024)  // 5M floats = 20 MB (> L2 cache)

__global__ void softmax_kernel_evict(
    const float *x,
    float *out,
    const float *dummy1,    // 20 MB dummy array 1
    const float *dummy2,    // 20 MB dummy array 2
    const float *dummy3,    // 20 MB dummy array 3
    int n_rows,
    int hidden_size
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    const float *x_row = x + row * hidden_size;
    float *out_row = out + row * hidden_size;

    __shared__ float smem[BLOCK_SIZE];

    // ================================================================
    // Phase 1: find max(x)
    // ================================================================
    float local_max = -INFINITY;
    for (int i = tid; i < hidden_size; i += total_threads) {
        float v = x_row[i];
        if (v > local_max) local_max = v;
    }

    smem[tid] = local_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride >= 32; stride >>= 1) {
        if (tid < stride && smem[tid + stride] > smem[tid])
            smem[tid] = smem[tid + stride];
        __syncthreads();
    }
    float val = smem[tid];
    float other;
    other = __shfl_down_sync(0xffffffff, val, 16); if (other > val) val = other;
    other = __shfl_down_sync(0xffffffff, val, 8);  if (other > val) val = other;
    other = __shfl_down_sync(0xffffffff, val, 4);  if (other > val) val = other;
    other = __shfl_down_sync(0xffffffff, val, 2);  if (other > val) val = other;
    other = __shfl_down_sync(0xffffffff, val, 1);  if (other > val) val = other;
    if (tid == 0) smem[0] = val;
    __syncthreads();
    float row_max = smem[0];

    // === EVICT L2: read 20 MB dummy buffer ===
    // Each thread strides through the buffer touching different cache lines.
    // 256 threads x 16 elems/thread x 4096 iterations ≈ 20 MB touched.
    // The dummy buffer has DUMMY_SIZE elements; we stride by blockDim.x
    // so threads' accesses are coalesced.
    {
        float sink = 0.0f;
        for (int i = tid; i < DUMMY_SIZE; i += total_threads) {
            sink += dummy1[i];  // touches every element of 20MB buffer
        }
        if (sink < -1e9f) out_row[0] = sink;  // prevent compiler optimizing away
    }
    __threadfence_block();  // ensure dummy reads complete before reading x again

    // ================================================================
    // Phase 2: sum(exp(x - max))
    // ================================================================
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += total_threads) {
        local_sum += expf(x_row[i] - row_max);
    }

    smem[tid] = local_sum;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    val = smem[tid];
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    if (tid == 0) smem[0] = val;
    __syncthreads();
    float row_sum = smem[0];

    // === EVICT L2 again: read 2nd dummy buffer ===
    {
        float sink = 0.0f;
        for (int i = tid; i < DUMMY_SIZE; i += total_threads) {
            sink += dummy2[i];
        }
        if (sink < -1e9f) out_row[0] = sink;
    }
    __threadfence_block();

    // ================================================================
    // Phase 3: exp(x_i - max) / sum → write output
    // ================================================================
    for (int i = tid; i < hidden_size; i += total_threads) {
        out_row[i] = expf(x_row[i] - row_max) / row_sum;
    }

    // === EVICT again: read 3rd dummy (force write to be measured too) ===
    {
        float sink = 0.0f;
        for (int i = tid; i < DUMMY_SIZE; i += total_threads) {
            sink += dummy3[i];
        }
        if (sink < -1e9f) out_row[0] = sink;
    }
}


// ============================================================================
// Pure read-only kernel: measure true HBM read BW
// Reads x 3 times (same count as our softmax) with zero compute, zero writes.
// This isolates: "what is the absolute fastest possible 3-read for this data?"
// ============================================================================
__global__ void read_three_times(const float *x, float *sink, int n_rows, int hidden_size) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;
    int hidden = hidden_size;
    const float *x_row = x + row * hidden;

    float dummy = 0.0f;

    // Read 1
    for (int i = tid; i < hidden; i += total_threads) {
        dummy += x_row[i];
    }

    // Read 2
    for (int i = tid; i < hidden; i += total_threads) {
        dummy += x_row[i];
    }

    // Read 3
    for (int i = tid; i < hidden; i += total_threads) {
        dummy += x_row[i];
    }

    // Prevent compiler from optimizing away the reads
    if (tid == 0) sink[row] = dummy;
}


// ============================================================================
// Your own softmax kernel — TODO: implement this!
// ============================================================================
#define MY_BLOCK_SIZE 256

__global__ void softmax_my_own(
    const float *x,
    float *out,
    int n_rows,
    int hidden_size
) {
    // TODO: your implementation
    // Hints:
    //   Phase 1: reduce to find max(x) — use MAX operator, not SUM
    //   Phase 2: reduce to sum(exp(x - max))
    //   Phase 3: exp(x_i - max) / sum, write to out
    //   Don't forget to broadcast intermediate results via shared memory!
}


// ============================================================================
// Benchmark tools
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
    int n_rows = N_ROWS;
    int hidden = HIDDEN_SIZE;
    int total_elems = n_rows * hidden;

    printf("===== CUDA Softmax: %d rows x %d hidden =====\n\n", n_rows, hidden);

    // ---- Allocate host memory ----
    float *h_x     = (float *)malloc(total_elems * sizeof(float));
    float *h_out   = (float *)malloc(total_elems * sizeof(float));

    for (int i = 0; i < total_elems; i++) {
        h_x[i] = (float)(rand()) / RAND_MAX - 0.5f;  // [-0.5, 0.5]
    }

    // ---- Allocate device memory ----
    float *d_x, *d_out;
    check(cudaMalloc(&d_x,   total_elems * sizeof(float)), "d_x");
    check(cudaMalloc(&d_out, total_elems * sizeof(float)), "d_out");

    check(cudaMemcpy(d_x, h_x, total_elems * sizeof(float), cudaMemcpyHostToDevice), "H2D x");

    // ---- Kernel config ----
    dim3 grid(n_rows);
    dim3 block(BLOCK_SIZE);

    // ---- Warmup ----
    for (int i = 0; i < WARMUP; i++) {
        softmax_kernel<<<grid, block>>>(d_x, d_out, n_rows, hidden);
    }
    check(cudaDeviceSynchronize(), "warmup");

    // ---- Benchmark reference kernel ----
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event");
    check(cudaEventCreate(&stop), "event");

    float best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start, 0);
        softmax_kernel<<<grid, block>>>(d_x, d_out, n_rows, hidden);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }

    // ---- BW & Analysis ----
    // Memory: 3 reads (x for max, x for sum, x for normalize) + 1 write (out)
    // = 4 * total_elems * 4 bytes
    float read_bytes  = (float)total_elems * sizeof(float) * 3.0f;
    float write_bytes = (float)total_elems * sizeof(float);
    float total_bytes = read_bytes + write_bytes;
    float bw = total_bytes / (best / 1000.0f) / 1e9f;

    printf("--- Reference softmax_kernel ---\n");
    printf("  Time:       %.4f ms\n", best);
    printf("  BW (est):   %.1f GB/s  (%.1f%% of 192 GB/s)\n",
           bw, bw / 192.0f * 100);
    printf("  Mem traffic: 3 reads + 1 write = %.1f MB\n",
           total_bytes / 1e6f);

    // ---- Verify correctness (row 0) ----
    check(cudaMemcpy(h_out, d_out, hidden * sizeof(float), cudaMemcpyDeviceToHost), "D2H verify");

    // CPU reference for row 0: numerically stable softmax
    float cpu_max = -INFINITY;
    for (int i = 0; i < hidden; i++) {
        if (h_x[i] > cpu_max) cpu_max = h_x[i];
    }
    float cpu_sum = 0.0f;
    for (int i = 0; i < hidden; i++) {
        cpu_sum += expf(h_x[i] - cpu_max);
    }

    float max_diff = 0.0f;
    for (int i = 0; i < hidden; i++) {
        float expected = expf(h_x[i] - cpu_max) / cpu_sum;
        float diff = fabsf(h_out[i] - expected);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  Max diff vs CPU: %.6e %s\n", max_diff,
           (max_diff < 1e-4f) ? "PASS" : "FAIL");

    // ====================================================================
    // L2 Cache Verification Experiment
    // ====================================================================
    printf("\n");
    printf("===== L2 Cache Experiment =====\n");
    printf("  Hypothesis: Phase 2/3 reads of x hit L2 cache.\n");
    printf("  Prediction: If we evict L2 between phases, time should increase.\n");
    printf("  If no L2 effect: normal ≈ evicted.\n\n");

    // Allocate dummy buffers and sink
    float *d_dummy1, *d_dummy2, *d_dummy3, *d_sink;
    int dummy_bytes = DUMMY_SIZE * sizeof(float);
    check(cudaMalloc(&d_dummy1, dummy_bytes), "dummy1");
    check(cudaMalloc(&d_dummy2, dummy_bytes), "dummy2");
    check(cudaMalloc(&d_dummy3, dummy_bytes), "dummy3");
    check(cudaMalloc(&d_sink, n_rows * sizeof(float)), "sink");

    // Fill dummy buffers with non-zero values (prevent compiler from optimizing away)
    float *h_dummy = (float *)malloc(dummy_bytes);
    for (int i = 0; i < DUMMY_SIZE; i++) h_dummy[i] = 1.0f;
    check(cudaMemcpy(d_dummy1, h_dummy, dummy_bytes, cudaMemcpyHostToDevice), "H2D dummy");
    check(cudaMemcpy(d_dummy2, h_dummy, dummy_bytes, cudaMemcpyHostToDevice), "H2D dummy");
    check(cudaMemcpy(d_dummy3, h_dummy, dummy_bytes, cudaMemcpyHostToDevice), "H2D dummy");

    // --- Benchmark: normal kernel (re-measure for fair comparison) ---
    printf("--- Kernel: normal (3-phase, back-to-back) ---\n");
    best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start, 0);
        softmax_kernel<<<grid, block>>>(d_x, d_out, n_rows, hidden);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }
    float t_normal = best;
    printf("  Time:       %.4f ms\n", t_normal);

    // --- Benchmark: evicted kernel ---
    printf("--- Kernel: evicted (read 20MB dummy between each phase) ---\n");
    for (int i = 0; i < WARMUP; i++) {
        softmax_kernel_evict<<<grid, block>>>(d_x, d_out,
            d_dummy1, d_dummy2, d_dummy3, n_rows, hidden);
    }
    check(cudaDeviceSynchronize(), "evict warmup");

    best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start, 0);
        softmax_kernel_evict<<<grid, block>>>(d_x, d_out,
            d_dummy1, d_dummy2, d_dummy3, n_rows, hidden);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }
    float t_evict = best;
    // Subtract the pure dummy read time to isolate just the softmax part
    printf("  Time:       %.4f ms  (includes 60MB dummy reads)\n", t_evict);

    // --- Benchmark: pure "read x 3 times" — the HBM lower bound ---
    printf("--- Kernel: read x three times (zero compute, zero writes) ---\n");
    for (int i = 0; i < WARMUP; i++) {
        read_three_times<<<grid, block>>>(d_x, d_sink, n_rows, hidden);
    }
    check(cudaDeviceSynchronize(), "read3 warmup");

    best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start, 0);
        read_three_times<<<grid, block>>>(d_x, d_sink, n_rows, hidden);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }
    float t_read3 = best;
    // What would "3 reads at 192 GB/s" take?
    float theoretical_read3 = (total_elems * 3 * sizeof(float)) / 192e9f * 1000.0f;
    printf("  Time:       %.4f ms\n", t_read3);
    printf("  Theoretical minimum (3 reads @ 192 GB/s): %.4f ms\n", theoretical_read3);

    // --- Analysis ---
    printf("\n===== Analysis =====\n");
    printf("  Normal kernel:      %.4f ms  (3 reads + 1 write + compute)\n", t_normal);
    printf("  Read-3-only:        %.4f ms  (3 reads, NO write, NO compute)\n", t_read3);
    printf("  Theoret.min read3:  %.4f ms  (3 reads @ 192 GB/s)\n", theoretical_read3);
    printf("\n");
    printf("  If L2 caches re-reads:\n");
    printf("    → t_normal ≈ t_read3  (re-reads are 'free' from L2)\n");
    printf("    → t_read3 < theoretical_min  (effective BW > 192 GB/s)\n");
    printf("  If re-reads always go to HBM:\n");
    printf("    → t_normal > theoretical_min  (compute adds cost)\n");
    printf("    → t_read3 ≈ theoretical_min\n");

    // Verify evicted kernel correctness too
    check(cudaMemcpy(h_out, d_out, hidden * sizeof(float), cudaMemcpyDeviceToHost), "D2H verify evict");
    max_diff = 0.0f;
    for (int i = 0; i < hidden; i++) {
        float expected = expf(h_x[i] - cpu_max) / cpu_sum;
        float diff = fabsf(h_out[i] - expected);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  Evict kernel max diff: %.6e %s\n", max_diff,
           (max_diff < 1e-4f) ? "PASS" : "FAIL");

    // Cleanup verification buffers
    cudaFree(d_dummy1); cudaFree(d_dummy2); cudaFree(d_dummy3); cudaFree(d_sink);
    free(h_dummy);

    // ---- Benchmark: softmax_my_own (if implemented) ----
    printf("\n--- softmax_my_own ---\n");

    // warmup
    for (int i = 0; i < WARMUP; i++) {
        softmax_my_own<<<grid, block>>>(d_x, d_out, n_rows, hidden);
    }
    check(cudaDeviceSynchronize(), "my_own warmup");

    best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start, 0);
        softmax_my_own<<<grid, block>>>(d_x, d_out, n_rows, hidden);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }

    bw = total_bytes / (best / 1000.0f) / 1e9f;
    printf("  Time:       %.4f ms\n", best);
    printf("  BW (est):   %.1f GB/s\n", bw);

    // Verify my_own
    check(cudaMemcpy(h_out, d_out, hidden * sizeof(float), cudaMemcpyDeviceToHost), "D2H verify my_own");
    max_diff = 0.0f;
    for (int i = 0; i < hidden; i++) {
        float expected = expf(h_x[i] - cpu_max) / cpu_sum;
        float diff = fabsf(h_out[i] - expected);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  Max diff vs CPU: %.6e %s\n", max_diff,
           (max_diff < 1e-4f) ? "PASS" : "FAIL");

    // ---- Compare with RMSNorm ----
    printf("\n  ===== Softmax vs RMSNorm Comparison =====\n");
    printf("  Both: fuse reduce + broadcast + element-wise\n");
    printf("  RMSNorm:  1 reduce (sum)   + element-wise  → 2x mem traffic\n");
    printf("  Softmax:  2 reduces (max + sum) + element-wise  → 4x mem traffic\n");
    printf("  Softmax also uses exp() which is expensive (~8-16 cycles)\n");
    printf("  Expect: softmax > RMSNorm time (maybe 1.5-2x)\n");

    // ---- Cleanup ----
    cudaFree(d_x);
    cudaFree(d_out);
    free(h_x);
    free(h_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
