/**
 * Flash Attention V4 — WMMA Tensor Cores for S = Q@K^T
 * Br=32, Bc=64, D=64, 2 warps (64 threads)
 * S via WMMA: 2×4×4=32 WMMA per K block
 * P@V: CUDA cores (O in shared memory for correct online rescaling)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
#define CHK(c) do{cudaError_t e=c;if(e!=cudaSuccess){fprintf(stderr,"E%d\n",e);exit(1);}}while(0)

__global__ void f32tof16(int n, const float* s, half* d) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)d[i]=__float2half(s[i]);
}

#define Br 32
#define Bc 64
#define D 64

__global__ void flash_attn_wmma(int N,
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
    float* __restrict__ O, float scale)
{
    int q_start = blockIdx.x * Br;
    int tid = threadIdx.x;
    int wid = tid / 32;   // warp 0: rows 0-15, warp 1: rows 16-31
    int lane = tid % 32;

    __shared__ half Qs[Br][D];
    __shared__ half Ks[Bc][D];
    __shared__ half Vs[Bc][D];
    __shared__ float S_s[Br][Bc];     // WMMA S results
    __shared__ float Ps[Bc][Br];
    __shared__ float Os[Br][D];
    __shared__ float s_rmax[Br], s_rsum[Br];

    // WMMA fragments: each warp covers 16×64 of S
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[1];  // 1 in M
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[4];  // 4 in N
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag[4];  // 1×4 = 4 WMMA = 32 regs

    if (tid < Br) { s_rmax[tid] = -1e9f; s_rsum[tid] = 0.0f; }
    for (int i = tid; i < Br * D; i += 64) Os[i/D][i%D] = 0.0f;

    // Load Q
    for (int i = tid; i < Br * D; i += 64) {
        int r = i / D, c = i % D, gr = q_start + r;
        Qs[r][c] = (gr < N && c < D) ? Q[gr * D + c] : __float2half(0.0f);
    }

    int num_kv = (N + Bc - 1) / Bc;
    for (int kv = 0; kv < num_kv; kv++) {
        int k_start = kv * Bc;
        int vk = (N - k_start) < Bc ? (N - k_start) : Bc;

        for (int i = tid; i < Bc * D; i += 64) {
            int r = i / D, c = i % D, gr = k_start + r;
            Ks[r][c] = (gr < N && c < D) ? K[gr * D + c] : __float2half(0.0f);
            Vs[r][c] = (gr < N && c < D) ? V[gr * D + c] : __float2half(0.0f);
        }
        __syncthreads();

        // === WMMA: S = Q @ K^T ===
        // Each warp: 1(M)×4(N)=4 WMMA tiles, K=4 steps
        // Init S accumulators
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) wmma::fill_fragment(s_frag[ni], 0.0f);

        for (int kk = 0; kk < D; kk += 16) {
            // Load Q fragment (this warp's 16 rows)
            wmma::load_matrix_sync(a_frag[0], &Qs[wid*16][kk], D);
            // Load 4 K^T fragments
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::load_matrix_sync(b_frag[ni], &Ks[ni*16][kk], D);
            // WMMA
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::mma_sync(s_frag[ni], a_frag[0], b_frag[ni], s_frag[ni]);
        }

        // Store S to shared memory (with scale applied)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            float Stile[16][16];
            wmma::store_matrix_sync((float*)Stile, s_frag[ni], 16, wmma::mem_row_major);
            for (int rr = 0; rr < 16; rr++) {
                int row = wid*16 + rr;
                if (row < Br) {
                    for (int cc = 0; cc < 16; cc++) {
                        int col = ni*16 + cc;
                        if (col < vk) S_s[row][col] = Stile[rr][cc] * scale;
                    }
                }
            }
        }
        __syncthreads();

        // === Online softmax + P@V (CUDA cores, same as V2) ===
        if (tid < Br) {
            int r = tid;
            float old_max = s_rmax[r];

            // Find max in S_s
            float row_max = old_max;
            for (int c = 0; c < vk; c++)
                if (S_s[r][c] > row_max) row_max = S_s[r][c];

            float ed = expf(old_max - row_max);
            float ln = s_rsum[r] * ed;

            // Compute P from S_s (no recomputation!)
            for (int c = 0; c < vk; c++) {
                float pv = expf(S_s[r][c] - row_max);
                Ps[c][r] = pv;
                ln += pv;
            }

            // Rescale O
            if (ed < 1.0f)
                for (int d = 0; d < D; d++) Os[r][d] *= ed;

            s_rmax[r] = row_max;
            s_rsum[r] = ln;
        }
        __syncthreads();

        // P@V
        if (tid < Br) {
            int r = tid;
            for (int d = 0; d < D; d++) {
                float acc = 0.0f;
                #pragma unroll
                for (int k = 0; k < vk; k += 4) {
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

    for (int i = tid; i < Br * D; i += 64) {
        int row = i / D, col = i % D, gr = q_start + row;
        if (gr < N && col < D)
            O[gr * D + col] = Os[row][col] / s_rsum[row];
    }
}

void cpu_attn(int N, int d, const float* Q, const float* K, const float* V,
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
    int seqs[] = {256, 512, 1024, 2048, 4096};
    float scale = 1.0f / sqrtf(64.0f);

    printf("Flash Attn V4 — WMMA S=Q@K^T, CUDA Core P@V\n\n");
    printf("%6s  %8s  %8s  %8s\n", "N", "ms", "TFLOPS", "MaxErr");
    printf("--------------------------------------\n");

    for (int si = 0; si < 5; si++) {
        int N = seqs[si];
        float *hQ=(float*)malloc(N*D*4),*hK=(float*)malloc(N*D*4),*hV=(float*)malloc(N*D*4);
        float *hO_cpu=(float*)malloc(N*D*4),*hO_gpu=(float*)malloc(N*D*4);

        srand(42);
        for (int i=0;i<N*D;i++){hQ[i]=(float)rand()/RAND_MAX-0.5f;hK[i]=(float)rand()/RAND_MAX-0.5f;hV[i]=(float)rand()/RAND_MAX-0.5f;}
        cpu_attn(N,D,hQ,hK,hV,hO_cpu,scale);

        float *dQ32,*dK32,*dV32,*dO; half *dQ16,*dK16,*dV16;
        CHK(cudaMalloc(&dQ32,N*D*4));CHK(cudaMalloc(&dK32,N*D*4));CHK(cudaMalloc(&dV32,N*D*4));
        CHK(cudaMalloc(&dQ16,N*D*2));CHK(cudaMalloc(&dK16,N*D*2));CHK(cudaMalloc(&dV16,N*D*2));
        CHK(cudaMalloc(&dO,N*D*4));
        CHK(cudaMemcpy(dQ32,hQ,N*D*4,cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dK32,hK,N*D*4,cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dV32,hV,N*D*4,cudaMemcpyHostToDevice));
        f32tof16<<<(N*D+255)/256,256>>>(N*D,dQ32,dQ16);
        f32tof16<<<(N*D+255)/256,256>>>(N*D,dK32,dK16);
        f32tof16<<<(N*D+255)/256,256>>>(N*D,dV32,dV16);
        CHK(cudaDeviceSynchronize());

        dim3 grid((N+Br-1)/Br), block(64);  // 2 warps
        for (int i=0;i<10;i++)flash_attn_wmma<<<grid,block>>>(N,dQ16,dK16,dV16,dO,scale);
        CHK(cudaDeviceSynchronize());
        if(cudaGetLastError()!=cudaSuccess){printf("LAUNCH FAILED\n");continue;}

        int reps=50; float max_err=0; double flops=4.0*N*N*D;
        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for(int i=0;i<reps;i++)flash_attn_wmma<<<grid,block>>>(N,dQ16,dK16,dV16,dO,scale);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms,s,e);

        CHK(cudaMemcpy(hO_gpu,dO,N*D*4,cudaMemcpyDeviceToHost));
        for(int i=0;i<N*D;i++){float d=fabsf(hO_gpu[i]-hO_cpu[i]);if(d>max_err)max_err=d;}
        double tf=flops/(ms/reps/1000.0)/1e12;
        printf("%6d  %8.3f  %8.2f  %8.4f\n",N,ms/reps,tf,max_err);

        cudaEventDestroy(s);cudaEventDestroy(e);
        free(hQ);free(hK);free(hV);free(hO_cpu);free(hO_gpu);
        CHK(cudaFree(dQ32));CHK(cudaFree(dK32));CHK(cudaFree(dV32));
        CHK(cudaFree(dQ16));CHK(cudaFree(dK16));CHK(cudaFree(dV16));CHK(cudaFree(dO));
    }
    printf("\nDone.\n");
    return 0;
}
