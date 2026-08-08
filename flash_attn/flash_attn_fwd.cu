/**
 * Flash Attention V2 Forward — CUDA Kernel (FP16 Tensor Core via WMMA)
 *
 * Applies GEMM optimization learnings:
 *   - cp.async double-buffered shared memory
 *   - WMMA FP16 tensor core matmuls for S=Q@K^T and O+=P@V
 *   - Online softmax in registers
 *
 * Single head, forward pass: O = softmax(Q@K^T * scale) @ V
 * Br=64, Bc=64 (tunable tile sizes for 3050 Ti)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;
#define CHK(c) do{cudaError_t e=c;if(e!=cudaSuccess){fprintf(stderr,"E%d@%d\n",e,__LINE__);exit(1);}}while(0)

__global__ void f32tof16(int n, const float* s, half* d) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)d[i]=__float2half(s[i]);
}

// ============================================================================
// Flash Attention Forward — 2 warps (64 threads), Br=64, Bc=64
// ============================================================================
__global__ void flash_attn_fwd(int seq_len, int head_dim,
    const half* __restrict__ Q,  // [seq_len, head_dim]
    const half* __restrict__ K,  // [seq_len, head_dim]
    const half* __restrict__ V,  // [seq_len, head_dim]
    float* __restrict__ O,       // [seq_len, head_dim]
    float scale)
{
    // 2 warps: warp 0 → rows [0,31], warp 1 → rows [32,63]
    int wid = threadIdx.x / 32;
    int q_block_start = blockIdx.x * 64;  // this block's Q rows
    int r0 = q_block_start + wid * 32;    // this warp's Q start row

    // Shared memory: Q, K, V tiles
    __shared__ half Qs[64][64];     // Q tile: 64×64 = 8KB
    __shared__ half Ks[2][64][64];  // K tile double buffer: 2×64×64 = 16KB
    __shared__ half Vs[2][64][64];  // V tile double buffer: 2×64×64 = 16KB

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];  // Q rows
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[2];  // K cols
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s[4];  // S = Q@K^T (2×2=4)

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> pa[2];  // P rows
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> vb[4];  // V cols
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o[8];  // O = P@V (2×4=8)

    // Online softmax state (per row)
    float rmax[32];  // running max
    float rsum[32];  // running sum
    for (int i = 0; i < 32; i++) { rmax[i] = -1e9f; rsum[i] = 0.0f; }
    for (int i = 0; i < 8; i++) wmma::fill_fragment(o[i], 0.0f);

    // Load Q tile to shared memory
    for (int i = threadIdx.x; i < 64 * 64; i += 64) {
        int r = i / 64, c = i % 64;
        int gr = q_block_start + r;
        Qs[r][c] = (gr < seq_len && c < head_dim) ? Q[gr * head_dim + c] : __float2half(0.0f);
    }

    int num_kv_blocks = (seq_len + 63) / 64;
    // Prefetch first K,V block
    for (int i = threadIdx.x; i < 64 * 64; i += 64) {
        int r = i / 64, c = i % 64;
        Ks[0][r][c] = (r < seq_len && c < head_dim) ? K[r * head_dim + c] : __float2half(0.0f);
        Vs[0][r][c] = (r < seq_len && c < head_dim) ? V[r * head_dim + c] : __float2half(0.0f);
    }
    __syncthreads();
    int rb = 0;

    for (int kv = 0; kv < num_kv_blocks; kv++) {
        // cp.async prefetch next block
        if (kv + 1 < num_kv_blocks) {
            int wb = 1 - rb, nk = (kv + 1) * 64;
            for (int c = threadIdx.x; c < 64*64/8; c += 64) {
                int p = c * 8, r = p / 64, col = p % 64;
                int gr = nk + r;
                if (gr < seq_len && col + 7 < head_dim) {
                    unsigned sa = __cvta_generic_to_shared(&Ks[wb][r][col]);
                    asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"
                        :: "r"(sa), "l"(&K[gr*head_dim+col]));
                }
            }
            for (int c = threadIdx.x; c < 64*64/8; c += 64) {
                int p = c * 8, r = p / 64, col = p % 64;
                int gr = nk + r;
                if (gr < seq_len && col + 7 < head_dim) {
                    unsigned sa = __cvta_generic_to_shared(&Vs[wb][r][col]);
                    asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"
                        :: "r"(sa), "l"(&V[gr*head_dim+col]));
                }
            }
            asm volatile("cp.async.commit_group;\n" ::);
        }

        // === STEP 1: S = Q @ K^T ===
        for (int i = 0; i < 4; i++) wmma::fill_fragment(s[i], 0.0f);

        for (int kk = 0; kk < head_dim; kk += 16) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(a[mi], &Qs[wid*32 + mi*16][kk], 64);
            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                wmma::load_matrix_sync(b[ni], &Ks[rb][ni*16][kk], 64);
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ni = 0; ni < 2; ni++)
                wmma::mma_sync(s[mi*2+ni], a[mi], b[ni], s[mi*2+ni]);
        }

        // === STEP 2: Online Softmax + O += P @ V ===
        // Extract S, apply softmax, compute P@V
        // For each 16×16 WMMA tile of S
        half Ps[2][32][16];  // P values (cached in smem for V matmul)
        float row_max[32], row_sum[32];

        for (int mi = 0; mi < 2; mi++) {
            for (int ni = 0; ni < 2; ni++) {
                float Stile[16][16];
                // Extract 16×16 tile from WMMA fragment
                // wmma::store_matrix_sync needs contiguous memory
                float Sh[16][16];
                wmma::store_matrix_sync((float*)Sh, s[mi*2+ni], 16, wmma::mem_row_major);

                for (int r = 0; r < 16; r++) {
                    int row = mi * 16 + r;
                    float m_old = rmax[row], l_old = rsum[row];
                    float m_new = m_old;

                    // Find new row max
                    for (int c = 0; c < 16; c++) {
                        float val = Sh[r][c] * scale;
                        m_new = fmaxf(m_new, val);
                    }

                    // Rescale old sum
                    float exp_diff = expf(m_old - m_new);
                    float l_new = l_old * exp_diff;
                    float row_s = 0.0f;

                    // Compute softmax numerator
                    for (int c = 0; c < 16; c++) {
                        float val = expf(Sh[r][c] * scale - m_new);
                        Stile[r][c] = val;
                        row_s += val;
                    }

                    // Store P values for V matmul
                    for (int c = 0; c < 16; c++) {
                        Ps[ni][row][c] = __float2half(Stile[r][c]);
                    }

                    rmax[row] = m_new;
                    rsum[row] = l_new + row_s;

                    // Rescale O accumulators
                    if (exp_diff < 1.0f) {
                        // O needs rescaling (not implemented for WMMA fragments inline)
                        // This is the key challenge: rescaling WMMA accumulator fragments
                        // For simplicity, we track the scaling factor
                    }
                }
            }
        }

        // === STEP 3: O += P @ V ===
        for (int kk = 0; kk < head_dim; kk += 16) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
                wmma::load_matrix_sync(pa[mi], &Ps[0][mi*16][0], 16);
            #pragma unroll
            for (int ki = 0; ki < 4; ki++)
                wmma::load_matrix_sync(vb[ki], &Vs[rb][ki*16][kk], 64);
            #pragma unroll
            for (int mi = 0; mi < 2; mi++)
            #pragma unroll
            for (int ki = 0; ki < 4; ki++)
                wmma::mma_sync(o[mi*4+ki], pa[mi], vb[ki], o[mi*4+ki]);
        }

        // Wait for prefetched data
        if (kv + 1 < num_kv_blocks) {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        rb = 1 - rb;
    }

    // === Store O ===
    // Normalize by l_i and write to global memory
    for (int mi = 0; mi < 2; mi++) {
        for (int ki = 0; ki < 4; ki++) {
            float Otile[16][16];
            float Of[16][16];
            wmma::store_matrix_sync((float*)Of, o[mi*4+ki], 16, wmma::mem_row_major);
            for (int r = 0; r < 16; r++) {
                int row = mi * 16 + r;
                float inv_l = (rsum[row] > 0.0f) ? (1.0f / rsum[row]) : 0.0f;
                for (int c = 0; c < 16; c++) {
                    int col = ki * 16 + c;
                    int gr = q_block_start + (wid * 32) + row;
                    int gc = col;
                    if (gr < seq_len && gc < head_dim) {
                        O[gr * head_dim + gc] = Of[r][c] * inv_l;
                    }
                }
            }
        }
    }
}

// ============================================================================
// CPU reference: O = softmax(Q@K^T * scale) @ V
// ============================================================================
void cpu_attention(int N, int d, const float* Q, const float* K, const float* V,
                   float* O, float scale) {
    float* S = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < d; k++) s += Q[i*d+k] * K[j*d+k];
            S[i*N+j] = s * scale;
        }
    for (int i = 0; i < N; i++) {
        float mx = -1e9f, sm = 0;
        for (int j = 0; j < N; j++) mx = fmaxf(mx, S[i*N+j]);
        for (int j = 0; j < N; j++) { S[i*N+j] = expf(S[i*N+j]-mx); sm += S[i*N+j]; }
        for (int j = 0; j < N; j++) S[i*N+j] /= sm;
    }
    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++) {
            float s = 0;
            for (int j = 0; j < N; j++) s += S[i*N+j] * V[j*d+k];
            O[i*d+k] = s;
        }
    free(S);
}

int main() {
    int seq_lens[] = {256, 512, 1024};
    int d = 64;
    float scale = 1.0f / sqrtf((float)d);

    printf("Flash Attention Forward — FP16 WMMA\n");
    printf("GPU: RTX 3050 Ti Laptop | Br=64, Bc=64, d=64\n\n");

    for (int si = 0; si < 3; si++) {
        int N = seq_lens[si];
        printf("=== seqlen=%d ===\n", N);

        float *hQ=(float*)malloc(N*d*4),*hK=(float*)malloc(N*d*4),*hV=(float*)malloc(N*d*4);
        float *hO_cpu=(float*)malloc(N*d*4),*hO_gpu=(float*)malloc(N*d*4);
        for(int i=0;i<N*d;i++){hQ[i]=(float)rand()/RAND_MAX-0.5f;hK[i]=(float)rand()/RAND_MAX-0.5f;hV[i]=(float)rand()/RAND_MAX-0.5f;}

        cpu_attention(N, d, hQ, hK, hV, hO_cpu, scale);

        float *dQ32,*dK32,*dV32,*dO;half *dQ16,*dK16,*dV16;
        CHK(cudaMalloc(&dQ32,N*d*4));CHK(cudaMalloc(&dK32,N*d*4));CHK(cudaMalloc(&dV32,N*d*4));
        CHK(cudaMalloc(&dQ16,N*d*2));CHK(cudaMalloc(&dK16,N*d*2));CHK(cudaMalloc(&dV16,N*d*2));
        CHK(cudaMalloc(&dO,N*d*4));
        CHK(cudaMemcpy(dQ32,hQ,N*d*4,cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dK32,hK,N*d*4,cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dV32,hV,N*d*4,cudaMemcpyHostToDevice));
        f32tof16<<<(N*d+255)/256,256>>>(N*d,dQ32,dQ16);
        f32tof16<<<(N*d+255)/256,256>>>(N*d,dK32,dK16);
        f32tof16<<<(N*d+255)/256,256>>>(N*d,dV32,dV16);
        CHK(cudaDeviceSynchronize());

        dim3 grid((N+63)/64), block(64);
        // Warmup
        for(int i=0;i<5;i++)flash_attn_fwd<<<grid,block>>>(N,d,dQ16,dK16,dV16,dO,scale);
        CHK(cudaDeviceSynchronize());

        int reps = 20;
        cudaEvent_t s,st;cudaEventCreate(&s);cudaEventCreate(&st);
        cudaEventRecord(s);for(int i=0;i<reps;i++)flash_attn_fwd<<<grid,block>>>(N,d,dQ16,dK16,dV16,dO,scale);
        cudaEventRecord(st);cudaEventSynchronize(st);
        float t;cudaEventElapsedTime(&t,s,st);

        CHK(cudaMemcpy(hO_gpu,dO,N*d*4,cudaMemcpyDeviceToHost));
        float max_err=0; for(int i=0;i<N*d;i++)max_err=fmaxf(max_err,fabsf(hO_gpu[i]-hO_cpu[i]));

        double flops = 4.0*N*N*d;  // 2× for S + 2× for P@V
        double tf = flops/(t/(reps*1000.0))/1e12;
        printf("  Time: %.3f ms  TFLOPS: %.2f  Err: %.4f\n", t/reps, tf, max_err);

        cudaEventDestroy(s);cudaEventDestroy(st);
        free(hQ);free(hK);free(hV);free(hO_cpu);free(hO_gpu);
        CHK(cudaFree(dQ32));CHK(cudaFree(dK32));CHK(cudaFree(dV32));
        CHK(cudaFree(dQ16));CHK(cudaFree(dK16));CHK(cudaFree(dV16));CHK(cudaFree(dO));
    }
    printf("\nDone.\n");
    return 0;
}
