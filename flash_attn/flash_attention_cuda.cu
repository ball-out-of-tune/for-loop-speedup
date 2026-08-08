/**
 * Flash Attention CUDA Kernel — Forward Pass
 *
 * Uses the same optimization techniques as our GEMM kernels:
 * - WMMA FP16 Tensor Cores for matmuls
 * - cp.async double-buffered shared memory
 * - Online softmax with tiling
 *
 * Algorithm: Flash Attention V2 (Q-outer loop)
 *   For each Q block:
 *     For each K,V block:
 *       S = Q_block @ K_block^T   (WMMA, Br×Bc)
 *       P = softmax(S)            (online, in SRAM)
 *       O_block += P @ V_block   (WMMA, Br×d)
 *
 * Parameters: Br=128, Bc=64, d=64, FP16
 * Each thread block handles one Q_block × all K,V blocks
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define CHECK(c) do{cudaError_t e=c;if(e!=cudaSuccess){fprintf(stderr,"CUDA %d@%d\n",e,__LINE__);exit(1);}}while(0)

// ============================================================================
// Parameters
// ============================================================================
#define Br 128   // Q block rows
#define Bc 64    // K,V block rows (also S cols)
#define d  64    // head dimension
#define WARP_M 4  // WMMA tiles in M dim (Br/16=8, but 4 per warp with 2 warps)
#define WARP_N 4  // WMMA tiles in N dim
#define WARP_K 4  // WMMA tiles in K dim (d/16=4)

// ============================================================================
// FP32→FP16 conversion
// ============================================================================
__global__ void convert_f32_to_f16(int n, const float* src, half* dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// ============================================================================
// Flash Attention Forward — Single Head
// Q,K,V: [seq_len, d] in row-major, FP16
// O: [seq_len, d] in row-major, FP32 (accumulated)
// ============================================================================
__global__ void flash_attn_forward(
    int seq_len,
    const half* __restrict__ Q,    // [seq_len, d]
    const half* __restrict__ K,    // [seq_len, d]
    const half* __restrict__ V,    // [seq_len, d]
    float* __restrict__ O,         // [seq_len, d]
    float* __restrict__ L,         // [seq_len] log-sum-exp (optional, for backward)
    float softmax_scale
) {
    // Each block handles one Q tile (Br rows)
    int q_start = blockIdx.x * Br;
    int tid = threadIdx.x;
    int wid = tid / 32;
    int lane = tid % 32;

    // For 128 threads (4 warps):
    // Warps 0-1: handle upper 64 rows (M=64 each)
    // Warps 2-3: handle lower 64 rows
    // Within each M=64: 4×4=16 WMMA operations
    int wy = wid / 2;  // 0 or 1 (which half of Br)
    int wx = wid % 2;  // 0 or 1 (which column group in Bc)

    // This warp's output region in O
    int r0 = q_start + wy * 64;  // this warp's row start in O
    int r_end = min(r0 + 64, seq_len);

    // === Shared memory ===
    // Double buffered K,V tiles + Q tile
    __shared__ half Q_smem[Br][d];       // Q tile for this block: 128×64 = 16KB
    __shared__ half Kj_smem[2][Bc][d];   // K tile buffers: 2×64×64 = 16KB
    __shared__ half Vj_smem[2][Bc][d];   // V tile buffers: 2×64×64 = 16KB
    // Total: 48KB — exactly fills smem!

    // === Registers ===
    // WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[WARP_M];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[WARP_M][WARP_N]; // S = Q@K^T

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frag[WARP_M][WARP_K]; // O

    // Online softmax state (per row this warp handles)
    float m_i[64];  // running max per row
    float l_i[64];  // running sum per row

    // Initialize O accumulators and softmax state
    #pragma unroll
    for (int mi = 0; mi < WARP_M; mi++) {
        #pragma unroll
        for (int ki = 0; ki < WARP_K; ki++) {
            wmma::fill_fragment(o_frag[mi][ki], 0.0f);
        }
    }
    for (int i = 0; i < 64; i++) {
        m_i[i] = -1e9f;
        l_i[i] = 0.0f;
    }

    // Prefetch Q tile to smem
    for (int i = tid; i < Br * d; i += blockDim.x) {
        int r = i / d, c = i % d;
        int gr = q_start + r;
        if (gr < seq_len)
            Q_smem[r][c] = Q[gr * d + c];
        else
            Q_smem[r][c] = __float2half(0.0f);
    }

    // Prefetch first K,V tile to buf[0]
    for (int i = tid; i < Bc * d; i += blockDim.x) {
        int r = i / d, c = i % d;
        if (r < seq_len)
            Kj_smem[0][r][c] = K[r * d + c];
        else
            Kj_smem[0][r][c] = __float2half(0.0f);
        Vj_smem[0][r][c] = (r < seq_len) ? V[r * d + c] : __float2half(0.0f);
    }
    __syncthreads();

    int read_buf = 0;
    int num_kv_blocks = (seq_len + Bc - 1) / Bc;

    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int k_start = kv_block * Bc;

        // cp.async prefetch next K,V tile (if not last)
        if (kv_block + 1 < num_kv_blocks) {
            int write_buf = 1 - read_buf;
            int next_k_start = (kv_block + 1) * Bc;

            // cp.async load K
            for (int chunk = tid; chunk < Bc * d / 8; chunk += blockDim.x) {
                int pos = chunk * 8;
                int r = pos / d, c = pos % d;
                int gr = next_k_start + r;
                if (gr < seq_len && c + 7 < d) {
                    unsigned sa = __cvta_generic_to_shared(&Kj_smem[write_buf][r][c]);
                    asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"
                                 :: "r"(sa), "l"(&K[gr * d + c]));
                }
            }
            // cp.async load V
            for (int chunk = tid; chunk < Bc * d / 8; chunk += blockDim.x) {
                int pos = chunk * 8;
                int r = pos / d, c = pos % d;
                int gr = next_k_start + r;
                if (gr < seq_len && c + 7 < d) {
                    unsigned sa = __cvta_generic_to_shared(&Vj_smem[write_buf][r][c]);
                    asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"
                                 :: "r"(sa), "l"(&V[gr * d + c]));
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        }

        // === STEP 1: S = Q_block @ K_block^T ===
        // Q rows [r0:r0+64] @ K^T columns [0:Bc-1]
        // S is [64][Bc] per warp
        #pragma unroll
        for (int mi = 0; mi < WARP_M; mi++) {
            #pragma unroll
            for (int nj = 0; nj < WARP_N; nj++) {
                wmma::fill_fragment(s_frag[mi][nj], 0.0f);
            }
        }

        for (int kk = 0; kk < d; kk += 16) {
            // Load Q fragment
            #pragma unroll
            for (int mi = 0; mi < WARP_M; mi++) {
                wmma::load_matrix_sync(a_frag[mi],
                    &Q_smem[wy*64 + mi*16][kk], d);
            }
            // Load K^T fragment (K is [Bc][d], we need K^T[d][Bc])
            // WMMA matrix_b is col_major for K^T
            wmma::load_matrix_sync(b_frag,
                &Kj_smem[read_buf][0][kk], d);  // This loads 16 columns of K^T

            // For each Q row tile
            #pragma unroll
            for (int mi = 0; mi < WARP_M; mi++) {
                #pragma unroll
                for (int nj = 0; nj < WARP_N; nj++) {
                    // Load K fragment for this column
                    wmma::load_matrix_sync(b_frag,
                        &Kj_smem[read_buf][0][kk], d);
                    wmma::mma_sync(s_frag[mi][nj], a_frag[mi], b_frag, s_frag[mi][nj]);
                }
            }
        }

        // === STEP 2: Extract S values, online softmax rescale ===
        // For each row: find max, compute exp, update running sum
        float S_local[16][16];  // one 16×16 WMMA tile (worst case)
        for (int mi = 0; mi < WARP_M; mi++) {
            for (int nj = 0; nj < WARP_N; nj++) {
                wmma::store_matrix_sync(S_local, s_frag[mi][nj], 16, wmma::mem_row_major);

                // Online softmax rescaling for these 16 rows
                for (int r = 0; r < 16; r++) {
                    int row = mi * 16 + r;
                    if (r0 + row >= seq_len) continue;

                    // Find max in this row
                    float row_max = m_i[row];
                    for (int c = 0; c < 16; c++) {
                        row_max = fmaxf(row_max, S_local[r][c] * softmax_scale);
                    }
                    float m_new = row_max;

                    // Rescale
                    float exp_diff = expf(m_i[row] - m_new);
                    l_i[row] *= exp_diff;

                    // Accumulate new exp sum
                    float row_sum = 0.0f;
                    for (int c = 0; c < 16; c++) {
                        row_sum += expf(S_local[r][c] * softmax_scale - m_new);
                    }
                    l_i[row] += row_sum;
                    m_i[row] = m_new;

                    // Store P = softmax(S) for next step
                    for (int c = 0; c < 16; c++) {
                        S_local[r][c] = expf(S_local[r][c] * softmax_scale - m_new);
                    }
                }
                // Store P back to WMMA fragment for O += P@V
                wmma::load_matrix_sync(p_frag, S_local, 16, wmma::mem_row_major);
            }
        }

        // === STEP 3: O += P @ V ===
        // P is [Br][Bc], V is [Bc][d]
        // Each warp updates its O rows
        // ... This is complex! For now, let me simplify

        // Actually, the full Flash Attention with WMMA + online softmax
        // requires careful handling of per-element softmax between matmuls.
        // This is significantly more complex than pure GEMM.
        // Let me focus on getting a working simplified version first.

        // Wait for prefetched tile
        if (kv_block + 1 < num_kv_blocks) {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        read_buf = 1 - read_buf;
    }

    // Store O results
    // (Placeholder)
}

// ============================================================================
// PyTorch reference implementation for comparison
// ============================================================================
void pytorch_flash_attn_ref(int seq_len, int head_dim,
    const float* Q, const float* K, const float* V,
    float* O, float softmax_scale)
{
    // Naive: S = Q@K^T, P = softmax(S), O = P@V
    // O(N^2) memory, just for correctness checking
    float* S = (float*)malloc(seq_len * seq_len * sizeof(float));

    // S = Q @ K^T * scale
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < seq_len; j++) {
            float sum = 0.0f;
            for (int k = 0; k < head_dim; k++) {
                sum += Q[i * head_dim + k] * K[j * head_dim + k];
            }
            S[i * seq_len + j] = sum * softmax_scale;
        }
    }

    // P = softmax(S, dim=-1)
    for (int i = 0; i < seq_len; i++) {
        float row_max = -1e9f;
        for (int j = 0; j < seq_len; j++) {
            row_max = fmaxf(row_max, S[i * seq_len + j]);
        }
        float row_sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            S[i * seq_len + j] = expf(S[i * seq_len + j] - row_max);
            row_sum += S[i * seq_len + j];
        }
        for (int j = 0; j < seq_len; j++) {
            S[i * seq_len + j] /= row_sum;
        }
    }

    // O = P @ V
    for (int i = 0; i < seq_len; i++) {
        for (int k = 0; k < head_dim; k++) {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                sum += S[i * seq_len + j] * V[j * head_dim + k];
            }
            O[i * head_dim + k] = sum;
        }
    }

    free(S);
}

// ============================================================================
// Benchmark harness
// ============================================================================
int main() {
    printf("Flash Attention CUDA — Forward Pass\n");
    printf("GPU: RTX 3050 Ti Laptop\n\n");

    int seq_lens[] = {512, 1024, 2048, 4096};
    int head_dim = 64;
    float scale = 1.0f / sqrtf((float)head_dim);  // 1/sqrt(d)

    for (int si = 0; si < 4; si++) {
        int N = seq_lens[si];
        printf("=== seqlen=%d, d=%d ===\n", N, head_dim);

        // Allocate host
        float *hQ = (float*)malloc(N * head_dim * sizeof(float));
        float *hK = (float*)malloc(N * head_dim * sizeof(float));
        float *hV = (float*)malloc(N * head_dim * sizeof(float));
        float *hO_ref = (float*)malloc(N * head_dim * sizeof(float));

        // Random data
        for (int i = 0; i < N * head_dim; i++) {
            hQ[i] = (float)rand() / RAND_MAX - 0.5f;
            hK[i] = (float)rand() / RAND_MAX - 0.5f;
            hV[i] = (float)rand() / RAND_MAX - 0.5f;
        }

        // CPU reference
        pytorch_flash_attn_ref(N, head_dim, hQ, hK, hV, hO_ref, scale);

        // Device allocations
        float *dQ32, *dK32, *dV32, *dO;
        half *dQ16, *dK16, *dV16;
        CHECK(cudaMalloc(&dQ32, N * head_dim * sizeof(float)));
        CHECK(cudaMalloc(&dK32, N * head_dim * sizeof(float)));
        CHECK(cudaMalloc(&dV32, N * head_dim * sizeof(float)));
        CHECK(cudaMalloc(&dQ16, N * head_dim * sizeof(half)));
        CHECK(cudaMalloc(&dK16, N * head_dim * sizeof(half)));
        CHECK(cudaMalloc(&dV16, N * head_dim * sizeof(half)));
        CHECK(cudaMalloc(&dO, N * head_dim * sizeof(float)));

        CHECK(cudaMemcpy(dQ32, hQ, N * head_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dK32, hK, N * head_dim * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dV32, hV, N * head_dim * sizeof(float), cudaMemcpyHostToDevice));
        convert_f32_to_f16<<<(N*head_dim+255)/256, 256>>>(N*head_dim, dQ32, dQ16);
        convert_f32_to_f16<<<(N*head_dim+255)/256, 256>>>(N*head_dim, dK32, dK16);
        convert_f32_to_f16<<<(N*head_dim+255)/256, 256>>>(N*head_dim, dV32, dV16);
        CHECK(cudaDeviceSynchronize());

        // Warmup
        dim3 grid((N + Br - 1) / Br);
        dim3 block(128);
        for (int i = 0; i < 5; i++)
            flash_attn_forward<<<grid, block>>>(N, dQ16, dK16, dV16, dO, nullptr, scale);
        CHECK(cudaDeviceSynchronize());

        // Benchmark
        int reps = 20;
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < reps; i++)
            flash_attn_forward<<<grid, block>>>(N, dQ16, dK16, dV16, dO, nullptr, scale);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float t;
        cudaEventElapsedTime(&t, start, stop);

        // Verify correctness
        float *hO = (float*)malloc(N * head_dim * sizeof(float));
        CHECK(cudaMemcpy(hO, dO, N * head_dim * sizeof(float), cudaMemcpyDeviceToHost));

        float max_err = 0.0f;
        for (int i = 0; i < N * head_dim; i++) {
            max_err = fmaxf(max_err, fabsf(hO[i] - hO_ref[i]));
        }

        // Compute TFLOPS
        double flops = 4.0 * (double)N * N * head_dim;  // 2×(S=P@V) + 2×(O=P@V)
        double tflops = flops / (t / (reps * 1000.0)) / 1e12;

        printf("  Time: %.3f ms  TFLOPS: %.3f  MaxErr: %.6f  Blocks: %d\n",
               t / reps, tflops, max_err, (N + Br - 1) / Br);

        cudaEventDestroy(start); cudaEventDestroy(stop);

        free(hQ); free(hK); free(hV); free(hO_ref); free(hO);
        CHECK(cudaFree(dQ32)); CHECK(cudaFree(dK32)); CHECK(cudaFree(dV32));
        CHECK(cudaFree(dQ16)); CHECK(cudaFree(dK16)); CHECK(cudaFree(dV16));
        CHECK(cudaFree(dO));
    }

    printf("\nDone.\n");
    return 0;
}
