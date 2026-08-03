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
// Reference Softmax Kernel (V1: naive 3-pass)
// ============================================================================
// Reads x THREE times (once per phase).  L2 cache usually saves the re-reads,
// but the extra passes cost sync barriers and L2 tag lookups.

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

    // Phase 1: find max(x)
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

    // Phase 2: sum(exp(x - max))
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

    // Phase 3: normalize + write
    for (int i = tid; i < hidden_size; i += total_threads) {
        out_row[i] = expf(x_row[i] - row_max) / row_sum;
    }
}


// ============================================================================
// V2: Online Softmax — max + sum 合并到一次读
// ============================================================================
// Algorithm (per thread):
//   m = -inf, d = 0
//   for each x_i:
//     new_m = max(m, x_i)
//     d = d * exp(m - new_m) + exp(x_i - new_m)   // 调旧 sum 到新 max 尺度
//     m = new_m
//
// After the loop, (m, d) satisfies: d = sum(exp(x_i - m))
//
// Cross-thread merge:  combine (m_a, d_a) and (m_b, d_b)
//   m = max(m_a, m_b)
//   d = d_a * exp(m_a - m) + d_b * exp(m_b - m)
//
// Then one more pass to normalize + write.
// Memory: 2 reads + 1 write (down from V1's 3 reads + 1 write)

__device__ inline void merge_pair(
    float m_a, float d_a, float m_b, float d_b,
    float *m_out, float *d_out
) {
    if (m_a >= m_b) {
        *m_out = m_a;
        *d_out = d_a + d_b * expf(m_b - m_a);  // exp(m_a - m_a) = 1 省略
    } else {
        *m_out = m_b;
        *d_out = d_b + d_a * expf(m_a - m_b);
    }
}

__global__ void softmax_kernel_v2_online(
    const float *x,
    float *out,
    int n_rows,
    int hidden_size
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int total_threads = blockDim.x;

    const float *x_row = x + row * hidden_size;
    float *out_row = out + row * hidden_size;

    __shared__ float smem_m[BLOCK_SIZE];
    __shared__ float smem_d[BLOCK_SIZE];

    // ================================================================
    // Pass 1: online max + sum in ONE read through x
    // ================================================================
    float m = -INFINITY;
    float d = 0.0f;
    for (int i = tid; i < hidden_size; i += total_threads) {
        float v = x_row[i];
        if (v > m) {
            d = d * expf(m - v) + 1.0f;  // exp(x_i - new_max) = exp(v - v) = 1
            m = v;
        } else {
            d += expf(v - m);
        }
    }

    // ---- Reduce (m, d) pairs across threads with merge_pair ----
    smem_m[tid] = m;
    smem_d[tid] = d;
    __syncthreads();

    // Cross-warp merge
    for (int stride = BLOCK_SIZE / 2; stride >= 32; stride >>= 1) {
        if (tid < stride) {
            float m_new, d_new;
            merge_pair(smem_m[tid], smem_d[tid],
                       smem_m[tid + stride], smem_d[tid + stride],
                       &m_new, &d_new);
            smem_m[tid] = m_new;
            smem_d[tid] = d_new;
        }
        __syncthreads();
    }

    // In-warp merge
    m = smem_m[tid];
    d = smem_d[tid];
    for (int delta = 16; delta >= 1; delta >>= 1) {
        float other_m = __shfl_down_sync(0xffffffff, m, delta);
        float other_d = __shfl_down_sync(0xffffffff, d, delta);
        float m_new, d_new;
        merge_pair(m, d, other_m, other_d, &m_new, &d_new);
        m = m_new;
        d = d_new;
    }

    // Broadcast global (m, d) to all threads
    if (tid == 0) { smem_m[0] = m; smem_d[0] = d; }
    __syncthreads();
    float row_max = smem_m[0];
    float row_sum = smem_d[0];

    // ================================================================
    // Pass 2: normalize + write (read x again, but hits L2 cache)
    // ================================================================
    for (int i = tid; i < hidden_size; i += total_threads) {
        out_row[i] = expf(x_row[i] - row_max) / row_sum;
    }
}


// ============================================================================
// V3: Register-Cached Softmax — 所有数据读进寄存器，只读一次 HBM
// ============================================================================
// Each thread handles hidden_size/256 = 16 elements.
// 16 floats = 64 bytes = 16 registers.  No pressure at all.
//
// Memory: 1 read + 1 write.  Zero re-reads, zero L2 dependency.
// This is the theoretical minimum memory traffic for softmax.

#define ITEMS_PER_THREAD (HIDDEN_SIZE / BLOCK_SIZE)  // 16

__global__ void softmax_kernel_v3_regcache(
    const float *x,
    float *out,
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
    // Read all my elements into registers (1 HBM read, coalesced)
    // ================================================================
    float x_reg[ITEMS_PER_THREAD];
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        x_reg[i] = x_row[tid + i * total_threads];
    }

    // ---- Phase 1: find max from registers ----
    float local_max = -INFINITY;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        float v = x_reg[i];
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

    // ---- Phase 2: sum(exp(x - max)) from registers ----
    float local_sum = 0.0f;
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        local_sum += expf(x_reg[i] - row_max);
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

    // ---- Phase 3: normalize + write from registers ----
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        out_row[tid + i * total_threads] = expf(x_reg[i] - row_max) / row_sum;
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

    int tid = threadIdx.x;
    int gap = blockIdx.x * hidden_size;
    const float* x_cur_ptr = x + gap;
    float* out_cur_ptr = out + gap;
    
    __shared__ float smem_max[MY_BLOCK_SIZE];
    __shared__ float smem_exp[MY_BLOCK_SIZE];
    float _max = -INFINITY;
    float exp_sum = 0.0f;
    float temp1 = 0.0f;
    for (int i = tid; i < hidden_size; i += MY_BLOCK_SIZE) {
        temp1 = x_cur_ptr[i];
        if (temp1 > _max) {
            exp_sum = exp_sum * expf(_max - temp1) + 1.0f;
            _max = temp1;
        } else{
            exp_sum += expf(temp1 - _max);
        }
    }
    smem_max[tid] = _max;
    smem_exp[tid] = exp_sum;
    __syncthreads();

    // reduce 
    for (int stride = MY_BLOCK_SIZE / 2; stride >= 32; stride = stride / 2) {
        if (tid < stride) {
            if (smem_max[tid] < smem_max[tid + stride]) {
                // 希望编译器自己做了smem_max[tid]和smem_max[tid + stride]取值的优化 这样我不用专门写一个局部变量定义
                // 以及这个公式的逻辑是什么？我每次都得自己推一遍保证正确 感觉很费时费力
                //记这个公式就行
                //  merge( (m_a, d_a), (m_b, d_b) ):
                //  m = max(m_a, m_b)
                //  d = d_a * exp(m_a - m) + d_b * exp(m_b - m)
                //  永远是这个形式。其中 max 那一方的 exp = exp(0) = 1，所以代码里省略了那一步乘法。
                
                smem_exp[tid] = smem_exp[tid] * expf(smem_max[tid] - smem_max[tid + stride]) + smem_exp[tid + stride];
                smem_max[tid] = smem_max[tid + stride];
            } else {
                smem_exp[tid] = smem_exp[tid] + smem_exp[tid + stride] * expf(smem_max[tid + stride] - smem_max[tid]);
            }
        }
        __syncthreads();
    }

    float val = smem_exp[tid];
    float _max2 = smem_max[tid];

    // 这里0~15的thread会拿到temp的值 那15~31的thread的temp是什么？以及15~31的thread为什么不会往上取数？是哪个参数限制了？
    float temp = __shfl_down_sync(0xffffffff, _max2, 16);
    if (_max2 < temp) { 
        val = val * expf(_max2 - temp) + __shfl_down_sync(0xffffffff, val, 16);
        _max2 = temp;
    } else {
        val = val + __shfl_down_sync(0xffffffff, val, 16) * expf(temp - _max2);
    }
    temp = __shfl_down_sync(0xffffffff, _max2, 8);
    if (_max2 < temp) { 
        val = val * expf(_max2 - temp) + __shfl_down_sync(0xffffffff, val, 8);
        _max2 = temp;
    } else {
        val = val + __shfl_down_sync(0xffffffff, val, 8) * expf(temp - _max2);
    }
    temp = __shfl_down_sync(0xffffffff, _max2, 4);
    if (_max2 < temp) { 
        val = val * expf(_max2 - temp) + __shfl_down_sync(0xffffffff, val, 4);
        _max2 = temp;
    } else {
        val = val + __shfl_down_sync(0xffffffff, val, 4) * expf(temp - _max2);
    }
    temp = __shfl_down_sync(0xffffffff, _max2, 2);
    if (_max2 < temp) { 
        val = val * expf(_max2 - temp) + __shfl_down_sync(0xffffffff, val, 2);
        _max2 = temp;
    } else {
        val = val + __shfl_down_sync(0xffffffff, val, 2) * expf(temp - _max2);
    }
    temp = __shfl_down_sync(0xffffffff, _max2, 1);
    if (_max2 < temp) { 
        val = val * expf(_max2 - temp) + __shfl_down_sync(0xffffffff, val, 1);
        _max2 = temp;
    } else {
        val = val + __shfl_down_sync(0xffffffff, val, 1) * expf(temp - _max2);
    }
   

    // 在什么时候要__syncthreads？ 规则:  任何线程 A 写了 shared memory，线程 B（B ≠ A）要读这个位置，中间必须夹一个 __syncthreads()
    if (tid == 0) {
        smem_max[tid] = _max2;
        smem_exp[tid] = val;
    }

    __syncthreads();
    float exp_final_sum = smem_exp[0];
    float max_final = smem_max[0];
    
    //  你在 reduce 阶段求了 sum(exp(x)) 而非 sum(exp(x - max))。虽然最终公式 exp(x-max)/sum(exp)*exp(-max) 等价，但中间 exp(x) 可能       
    // INF。应该先求 max，再 exp(x - max)。  看一下这样做和正确的做法的区别 以及什么时候会INF
    
    /********
    错误的做法！！
    for(int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        out_cur_ptr[i] = (expf(x_cur_ptr[i]) * expf(-max_final)) / (exp_final_sum * expf(-max_final))
    }
    错误的做法！！
    ********/

    for(int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        out_cur_ptr[i] = expf(x_cur_ptr[i] - max_final) / exp_final_sum;
    }

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
    float t_normal = best;  // save V1 time for comparison

    // ---- BW & Analysis ----
    // Memory: 3 reads (x for max, x for sum, x for normalize) + 1 write (out)
    // = 4 * total_elems * 4 bytes
    float read_bytes  = (float)total_elems * sizeof(float) * 3.0f;
    float write_bytes = (float)total_elems * sizeof(float);
    float total_bytes = read_bytes + write_bytes;
    float bw = total_bytes / (best / 1000.0f) / 1e9f;

    printf("--- V1 (naive 3-pass) ---\n");
    printf("  Time:       %.4f ms  (3R + 1W)\n", best);

    // ---- CPU reference (used by all verification below) ----
    float cpu_max = -INFINITY;
    for (int i = 0; i < hidden; i++) {
        if (h_x[i] > cpu_max) cpu_max = h_x[i];
    }
    float cpu_sum = 0.0f;
    for (int i = 0; i < hidden; i++) {
        cpu_sum += expf(h_x[i] - cpu_max);
    }

    // ---- Benchmark: V2 (online softmax) ----
    printf("--- V2 (online: max+sum merged into 1 pass) ---\n");
    for (int i = 0; i < WARMUP; i++) {
        softmax_kernel_v2_online<<<grid, block>>>(d_x, d_out, n_rows, hidden);
    }
    check(cudaDeviceSynchronize(), "v2 warmup");

    best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start, 0);
        softmax_kernel_v2_online<<<grid, block>>>(d_x, d_out, n_rows, hidden);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }
    float t_v2 = best;
    printf("  Time:       %.4f ms  (2R + 1W)  vs V1: %.2fx\n",
           t_v2, t_v2 > 0 ? t_normal / t_v2 : 0);

    // Verify V2
    check(cudaMemcpy(h_out, d_out, hidden * sizeof(float), cudaMemcpyDeviceToHost), "D2H v2");
    {
        float v2_diff = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float expected = expf(h_x[i] - cpu_max) / cpu_sum;
            float diff = fabsf(h_out[i] - expected);
            if (diff > v2_diff) v2_diff = diff;
        }
        printf("  Max diff:   %.6e %s\n", v2_diff, (v2_diff < 1e-4f) ? "PASS" : "FAIL");
    }

    // ---- Benchmark: V3 (register-cached) ----
    printf("--- V3 (register-cached: 1 read + 1 write) ---\n");
    for (int i = 0; i < WARMUP; i++) {
        softmax_kernel_v3_regcache<<<grid, block>>>(d_x, d_out, n_rows, hidden);
    }
    check(cudaDeviceSynchronize(), "v3 warmup");

    best = 1e9;
    for (int r = 0; r < RUNS; r++) {
        cudaEventRecord(start, 0);
        softmax_kernel_v3_regcache<<<grid, block>>>(d_x, d_out, n_rows, hidden);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t = millis(start, stop);
        if (t < best) best = t;
    }
    float t_v3 = best;
    printf("  Time:       %.4f ms  (1R + 1W)  vs V1: %.2fx  vs V2: %.2fx\n",
           t_v3, t_v3 > 0 ? t_normal / t_v3 : 0, t_v3 > 0 ? t_v2 / t_v3 : 0);

    // Verify V3
    check(cudaMemcpy(h_out, d_out, hidden * sizeof(float), cudaMemcpyDeviceToHost), "D2H v3");
    {
        float v3_diff = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float expected = expf(h_x[i] - cpu_max) / cpu_sum;
            float diff = fabsf(h_out[i] - expected);
            if (diff > v3_diff) v3_diff = diff;
        }
        printf("  Max diff:   %.6e %s\n", v3_diff, (v3_diff < 1e-4f) ? "PASS" : "FAIL");
    }

    // ---- Summary comparison ----
    printf("\n===== Comparison =====\n");
    printf("  V1 (naive 3-pass):    %.4f ms  (3R + 1W = %.1f MB)\n",
           t_normal, total_bytes / 1e6f);
    printf("  V2 (online merge):    %.4f ms  (2R + 1W = %.1f MB)\n",
           t_v2, (total_elems * sizeof(float) * 3.0f) / 1e6f);
    float v3_traffic = total_elems * sizeof(float) * 2.0f;
    printf("  V3 (reg-cached):      %.4f ms  (1R + 1W = %.1f MB)\n",
           t_v3, v3_traffic / 1e6f);
    printf("  PyTorch eager:        0.198 ms\n\n");

    // Check if V3 needs L2 by computing pure-HBM time
    float hbm_theoretical = v3_traffic / 192e9f * 1000.0f;
    printf("  V3 theoretical floor (all HBM):        %.4f ms\n", hbm_theoretical);
    printf("  V3 effective BW (1R+1W model):         %.1f GB/s  (%.1f%%)\n\n",
           v3_traffic / (t_v3 / 1000.0f) / 1e9f,
           (v3_traffic / (t_v3 / 1000.0f) / 1e9f) / 192.0f * 100);

    // ---- Verify V1 correctness (row 0) ----
    check(cudaMemcpy(h_out, d_out, hidden * sizeof(float), cudaMemcpyDeviceToHost), "D2H verify V1");

    float max_diff = 0.0f;
    for (int i = 0; i < hidden; i++) {
        float expected = expf(h_x[i] - cpu_max) / cpu_sum;
        float diff = fabsf(h_out[i] - expected);
        if (diff > max_diff) max_diff = diff;
    }
    printf("  V1 max diff vs CPU: %.6e %s\n", max_diff,
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
    // t_normal already set above in main benchmark section
    printf("  Time:       %.4f ms\n", best);

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
