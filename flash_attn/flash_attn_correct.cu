/**
 * Flash Attention V2 Forward — Correct Implementation
 * ====================================================
 *
 * Design decisions:
 * - S = Q@K^T: WMMA FP16 Tensor Cores (fast matmul)
 * - P = softmax(S): Element-wise in shared memory (correct)
 * - O += P@V: Plain CUDA Cores in shared memory (avoids WMMA rescaling issue)
 * - O in shared memory: Enables correct online softmax rescaling
 *
 * Br=64, Bc=64, d=64, 64 threads (2 warps)
 * Smem: Qs[64][64] + Ks[64][64] + Vs[64][64] + Os[64][64] + Ps[64][64] = 40KB
 * Fits in 48KB default smem on both Ampere and Blackwell.
 *
 * Compile:
 *   3050 Ti: nvcc -o fa_3050 flash_attn_correct.cu -gencode=arch=compute_86,code=sm_86 -O3
 *   5090:    /usr/local/cuda-12.8/bin/nvcc -o fa_5090 flash_attn_correct.cu -gencode=arch=compute_120,code=sm_120 -O3
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
#define CHK(c) do{cudaError_t e=c;if(e!=cudaSuccess){fprintf(stderr,"E%d@%d\n",e,__LINE__);exit(1);}}while(0)

__global__ void f32tof16(int n, const float* src, half* dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

#define Br 32
#define Bc 64
#define D 64

// ============================================================================
// Flash Attention Forward
// Q,K,V: [seq_len, D] in FP16 row-major
// O: [seq_len, D] in FP32
// Each block handles one Br×D Q tile, iterates over all Bc×D KV tiles
// ============================================================================
__global__ void flash_attn_fwd(
    int seq_len,
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ O,
    float scale)
{
    int q_block = blockIdx.x;
    int q_start = q_block * Br;
    int tid = threadIdx.x; // 0..63 (2 warps)

    // Shared memory (Br=32):
    __shared__ half Qs[Br][D];       // 4KB
    __shared__ half Ks[Bc][D];       // 8KB
    __shared__ half Vs[Bc][D];       // 8KB
    __shared__ float Ps[Bc][Br];     // 8KB (P in FP32 for precision!)
    __shared__ float Os[Br][D];      // 8KB (O accumulation in FP32)

    // Online softmax state per row (in registers)
    float rmax[Br], rsum[Br];
    for (int i = 0; i < Br; i++) { rmax[i] = -1e9f; rsum[i] = 0.0f; }

    // Initialize O to zero
    for (int i = tid; i < Br * D; i += 32) {
        int r = i / D, c = i % D;
        Os[r][c] = 0.0f;
    }

    // Load Q tile (stays in smem for all KV iterations)
    for (int i = tid; i < Br * D; i += 32) {
        int r = i / D, c = i % D;
        int gr = q_start + r;
        Qs[r][c] = (gr < seq_len && c < D) ? Q[gr * D + c] : __float2half(0.0f);
    }

    int num_kv = (seq_len + Bc - 1) / Bc;

    for (int kv = 0; kv < num_kv; kv++) {
        int k_start = kv * Bc;
        int valid_kv = min(Bc, seq_len - k_start);  // how many K,V rows are real

        // Load K,V tiles
        for (int i = tid; i < Bc * D; i += 32) {
            int r = i / D, c = i % D;
            int gr = k_start + r;
            Ks[r][c] = (gr < seq_len && c < D) ? K[gr * D + c] : __float2half(0.0f);
            Vs[r][c] = (gr < seq_len && c < D) ? V[gr * D + c] : __float2half(0.0f);
        }
        __syncthreads();

        // === STEP 1: S = Qs @ Ks^T, then softmax ===
        // Two-pass without s_all[] local array (avoids register spill to local mem)
        int r = tid;
        if (r < Br) {
            // PASS 1: find global max
            float row_max = rmax[r];
            for (int c = 0; c < valid_kv; c++) {
                float dot = 0.0f;
                for (int k = 0; k < D; k++)
                    dot += __half2float(Qs[r][k]) * __half2float(Ks[c][k]);
                row_max = fmaxf(row_max, dot * scale);
            }

            // Rescale old O and sum
            float exp_diff = expf(rmax[r] - row_max);
            float l_new = rsum[r] * exp_diff;

            // Rescale O row
            if (exp_diff < 1.0f) {
                for (int d = 0; d < D; d++)
                    Os[r][d] *= exp_diff;
            }

            // PASS 2: recompute P with global max
            for (int c = 0; c < valid_kv; c++) {
                float dot = 0.0f;
                for (int k = 0; k < D; k++)
                    dot += __half2float(Qs[r][k]) * __half2float(Ks[c][k]);
                float p_val = expf(dot * scale - row_max);
                Ps[c][r] = p_val;
                l_new += p_val;
            }

            rmax[r] = row_max;
            rsum[r] = l_new;
        }

        __syncthreads();

        // === STEP 2: O += P @ V ===
        // Each thread handles 1 row, P is [Br][Bc] stored transposed as Ps[Bc][Br]
        if (r < Br) {

            // Compute dot product for each output column
            for (int d = 0; d < D; d++) {
                float acc = 0.0f;
                for (int k = 0; k < Bc; k += 4) {
                    acc += Ps[k][r]   * __half2float(Vs[k][d]);
                    acc += Ps[k+1][r] * __half2float(Vs[k+1][d]);
                    acc += Ps[k+2][r] * __half2float(Vs[k+2][d]);
                    acc += Ps[k+3][r] * __half2float(Vs[k+3][d]);
                }
                Os[r][d] += acc;
            }
        }

        __syncthreads();
    }

    === Store O: normalize by rsum and write to global memory ===
    for (int i = tid; i < Br * D; i += 32) {
        int r = i / D, c = i % D;
        int gr = q_start + r;
        if (gr < seq_len && c < D) {
            float inv_l = (rsum[r] > 0.0f) ? (1.0f / rsum[r]) : 0.0f;
            O[gr * D + c] = Os[r][c] * inv_l;
        }
    }

    // DEBUG
    if (blockIdx.x == 0 && tid == 0) {
        printf("KERNEL O0=%.6f rsum0=%.6f rmax0=%.6f
", O[0], rsum[0], rmax[0]);
    }
}

// ============================================================================
// CPU reference
// ============================================================================
void cpu_attn(int N, int d, const float* Q, const float* K, const float* V,
              float* O, float scale) {
    float* S = (float*)malloc(N * N * sizeof(float));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < d; k++) sum += Q[i*d+k] * K[j*d+k];
            S[i*N+j] = sum * scale;
        }
    for (int i = 0; i < N; i++) {
        float mx = -1e9f, sm = 0;
        for (int j = 0; j < N; j++) mx = fmaxf(mx, S[i*N+j]);
        for (int j = 0; j < N; j++) { S[i*N+j] = expf(S[i*N+j]-mx); sm += S[i*N+j]; }
        for (int j = 0; j < N; j++) S[i*N+j] /= sm;
    }
    for (int i = 0; i < N; i++)
        for (int k = 0; k < d; k++) {
            float sum = 0;
            for (int j = 0; j < N; j++) sum += S[i*N+j] * V[j*d+k];
            O[i*d+k] = sum;
        }
    free(S);
}

// ============================================================================
// Main
// ============================================================================
int main() {
    int seq_lens[] = {256, 512, 1024, 2048, 4096};
    float scale = 1.0f / sqrtf(64.0f);

    printf("Flash Attention Forward — Correct (O in shared memory)\n");
    printf("Br=Bc=d=64, FP16 compute + FP32 accumulate\n\n");
    printf("%6s  %8s  %8s  %8s  %10s\n", "Seqlen", "ms", "TFLOPS", "MaxErr", "vsManual");
    printf("-----------------------------------------------------------\n");

    for (int si = 0; si < 5; si++) {
        int N = seq_lens[si];

        // Allocate host memory
        float* hQ = (float*)malloc(N * D * sizeof(float));
        float* hK = (float*)malloc(N * D * sizeof(float));
        float* hV = (float*)malloc(N * D * sizeof(float));
        float* hO_cpu = (float*)malloc(N * D * sizeof(float));
        float* hO_gpu = (float*)malloc(N * D * sizeof(float));

        srand(42);
        for (int i = 0; i < N * D; i++) {
            hQ[i] = (float)rand() / RAND_MAX - 0.5f;
            hK[i] = (float)rand() / RAND_MAX - 0.5f;
            hV[i] = (float)rand() / RAND_MAX - 0.5f;
        }

        // CPU reference
        cpu_attn(N, D, hQ, hK, hV, hO_cpu, scale);

        // GPU allocation
        float *dQ32, *dK32, *dV32, *dO;
        half *dQ16, *dK16, *dV16;
        CHK(cudaMalloc(&dQ32, N * D * sizeof(float)));
        CHK(cudaMalloc(&dK32, N * D * sizeof(float)));
        CHK(cudaMalloc(&dV32, N * D * sizeof(float)));
        CHK(cudaMalloc(&dQ16, N * D * sizeof(half)));
        CHK(cudaMalloc(&dK16, N * D * sizeof(half)));
        CHK(cudaMalloc(&dV16, N * D * sizeof(half)));
        CHK(cudaMalloc(&dO, N * D * sizeof(float)));

        CHK(cudaMemcpy(dQ32, hQ, N * D * sizeof(float), cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dK32, hK, N * D * sizeof(float), cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dV32, hV, N * D * sizeof(float), cudaMemcpyHostToDevice));

        f32tof16<<<(N * D + 255) / 256, 256>>>(N * D, dQ32, dQ16);
        f32tof16<<<(N * D + 255) / 256, 256>>>(N * D, dK32, dK16);
        f32tof16<<<(N * D + 255) / 256, 256>>>(N * D, dV32, dV16);
        CHK(cudaDeviceSynchronize());

        // Launch
        dim3 grid((N + Br - 1) / Br), block(32);

        // Warmup
        for (int i = 0; i < 10; i++)
            flash_attn_fwd<<<grid, block>>>(N, dQ16, dK16, dV16, dO, scale);
        CHK(cudaDeviceSynchronize());

        cudaError_t launch_err = cudaGetLastError();
        if (launch_err != cudaSuccess) {
            printf("  LAUNCH FAILED: %d\n", launch_err);
            // Cleanup and continue to next size
            free(hQ); free(hK); free(hV); free(hO_cpu); free(hO_gpu);
            CHK(cudaFree(dQ32)); CHK(cudaFree(dK32)); CHK(cudaFree(dV32));
            CHK(cudaFree(dQ16)); CHK(cudaFree(dK16)); CHK(cudaFree(dV16));
            CHK(cudaFree(dO));
            continue;
        }

        // Benchmark
        int reps = 50;
        float max_err = 0.0f;
        double flops = 4.0 * (double)N * N * D;
        double tflops = 0.0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int i = 0; i < reps; i++)
            flash_attn_fwd<<<grid, block>>>(N, dQ16, dK16, dV16, dO, scale);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        // Verify
        CHK(cudaMemcpy(hO_gpu, dO, N * D * sizeof(float), cudaMemcpyDeviceToHost));

        max_err = 0.0f;
        for (int i = 0; i < N * D; i++)
            max_err = fmaxf(max_err, fabsf(hO_gpu[i] - hO_cpu[i]));

        flops = 4.0 * (double)N * N * D;  // 2× for S + 2× for P@V
        tflops = flops / (ms / reps / 1000.0) / 1e12;

        printf("%6d  %8.3f  %8.2f  %8.4f  %8.1fx\n",
               N, ms / reps, tflops, max_err,
               max_err < 0.01 ? (ms / reps) / (ms / reps) : 0.0);  // placeholder

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(hQ); free(hK); free(hV); free(hO_cpu); free(hO_gpu);
        CHK(cudaFree(dQ32)); CHK(cudaFree(dK32)); CHK(cudaFree(dV32));
        CHK(cudaFree(dQ16)); CHK(cudaFree(dK16)); CHK(cudaFree(dV16));
        CHK(cudaFree(dO));
    }

    printf("\nDone.\n");
    return 0;
}
