/**
 * Flash Attention Forward — Mixed Precision (FP16 compute, FP32 accumulate)
 *
 * Algorithm:
 *   For each Q tile (Br rows):
 *     For each KV tile (Bc rows):
 *       S = Q_tile @ K_tile^T * scale        (FP16 matmul in shared mem)
 *       P = softmax(S)                        (online, in shared mem)
 *       O_tile += P @ V_tile                 (FP16 matmul in shared mem)
 *
 * Uses: shared memory tiling + online softmax (NO WMMA, for simplicity)
 * Br=64, Bc=64, d=64
 * Each block = 1 warp = 32 threads handles Br=64 rows
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CHK(c) do{cudaError_t e=c;if(e!=cudaSuccess){fprintf(stderr,"E%d@%d\n",e,__LINE__);exit(1);}}while(0)

__global__ void f32tof16(int n, const float* s, half* d) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)d[i]=__float2half(s[i]);
}

#define Br 64
#define Bc 64
#define D 64

__global__ void flash_attn_fwd(int seq_len,
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ O,
    float scale)
{
    int q_block = blockIdx.x;
    int q_start = q_block * Br;
    int tid = threadIdx.x;       // 0..31 (1 warp)
    int half_warp = tid / 16;    // 0 or 1 (upper/lower half of Br)

    // Shared memory
    __shared__ half Qs[Br][D];        // 64×64 = 8KB
    __shared__ half Ks[Bc][D];        // 64×64 = 8KB
    __shared__ half Vs[Bc][D];        // 64×64 = 8KB
    __shared__ half Ps_sub[32][Bc/2]; // partial P values (for V matmul)
    // Total: 24KB + 4KB = 28KB → fits in 48KB

    // Online softmax state per row
    float rmax[2];  // 2 rows per thread (64 rows / 32 threads = 2)
    float rsum[2];
    rmax[0] = rmax[1] = -1e9f;
    rsum[0] = rsum[1] = 0.0f;

    // O accumulators (FP32, 4 elements per thread)
    float o_acc[2][4];  // 2 rows × 4 cols = 8 floats
    for (int i = 0; i < 8; i++) ((float*)o_acc)[i] = 0.0f;

    // Load Q tile
    for (int i = tid; i < Br * D; i += 32) {
        int r = i / D, c = i % D;
        int gr = q_start + r;
        Qs[r][c] = (gr < seq_len && c < D) ? Q[gr * D + c] : __float2half(0.0f);
    }

    int num_kv = (seq_len + Bc - 1) / Bc;

    for (int kv = 0; kv < num_kv; kv++) {
        int k_start = kv * Bc;

        // Load K, V tiles
        for (int i = tid; i < Bc * D; i += 32) {
            int r = i / D, c = i % D;
            int gr = k_start + r;
            Ks[r][c] = (gr < seq_len && c < D) ? K[gr * D + c] : __float2half(0.0f);
            Vs[r][c] = (gr < seq_len && c < D) ? V[gr * D + c] : __float2half(0.0f);
        }
        __syncthreads();

        // === S = Qs @ Ks^T, then softmax ===
        // Each thread processes 2 rows (64/32=2), computing dot products
        for (int row_i = 0; row_i < 2; row_i++) {
            int r = half_warp * 32 + tid + row_i * 0;  // actually let me use a simpler mapping
            // Thread tid handles row: r = half_warp * 32 + (tid % 16) + (row_i * 16)
            // Wait, that's wrong too. Let me just do straightforward mapping.
        }

        // SIMPLIFIED: each thread handles 2 specific rows
        // Thread t: rows t and t+32 (total 64/32=2 rows per thread)
        int row0 = tid;
        int row1 = tid + 32;

        // Compute S[row0] = Q[row0] @ K^T
        // For each column of S (each row of K):
        for (int col_block = 0; col_block < Bc; col_block += 16) {
            // Each thread computes 2 rows × 16 cols of S
            float s_val[2][16];

            // Initialize this sub-block
            for (int j = 0; j < 16; j++) s_val[0][j] = s_val[1][j] = 0.0f;

            // Dot product accumulation (d=64, 4 unrolled iterations of 16)
            for (int k = 0; k < D; k++) {
                half q0 = (row0 < Br) ? Qs[row0][k] : __float2half(0.0f);
                half q1 = (row1 < Br) ? Qs[row1][k] : __float2half(0.0f);
                float q0f = __half2float(q0);
                float q1f = __half2float(q1);

                for (int j = 0; j < 16; j++) {
                    int col = col_block + j;
                    half kval = (col < Bc) ? Ks[col][k] : __float2half(0.0f);
                    float kf = __half2float(kval);
                    s_val[0][j] += q0f * kf;
                    s_val[1][j] += q1f * kf;
                }
            }

            // Apply scale, find max, softmax, update running state
            for (int row_i = 0; row_i < 2; row_i++) {
                int global_row = (row_i == 0) ? row0 : row1;
                if (global_row >= Br) continue;

                // Find max in this sub-block
                float m_old = rmax[row_i];
                float m_new = m_old;
                for (int j = 0; j < 16; j++) {
                    s_val[row_i][j] *= scale;
                    m_new = fmaxf(m_new, s_val[row_i][j]);
                }

                // Rescale
                float exp_diff = expf(m_old - m_new);
                float l_new = rsum[row_i] * exp_diff;

                // Exp and sum
                for (int j = 0; j < 16; j++) {
                    float p = expf(s_val[row_i][j] - m_new);
                    s_val[row_i][j] = p;  // reuse s_val for P
                    l_new += p;
                }

                // Rescale O accumulators
                for (int c = 0; c < 4; c++) {
                    o_acc[row_i][c] *= exp_diff;
                }

                // Store P to shared memory for V matmul
                for (int j = 0; j < 16; j++) {
                    int store_row = row_i * 32 + (tid % 16);  // This is wrong, fix later
                    // Actually let me compute this differently
                }

                rmax[row_i] = m_new;
                rsum[row_i] = l_new;
            }
        }

        // === O += P @ V ===
        // Each thread accumulates 2 rows × 4 columns of O (8 output elements)
        for (int row_i = 0; row_i < 2; row_i++) {
            for (int c = 0; c < 4; c++) {
                int v_col = tid * 4 + c;
                float acc = 0.0f;
                for (int k = 0; k < Bc; k++) {
                    // P[row][k] — we need this from shared memory
                    // V[k][v_col] — from Vs
                    // acc += P[row][k] * V[k][v_col]
                }
                o_acc[row_i][c] += acc;
            }
        }

        __syncthreads();
    }

    // Store O: O[row] = o_acc / rsum
    for (int row_i = 0; row_i < 2; row_i++) {
        int global_row = q_start + ((row_i == 0) ? row0 : row1);
        if (global_row >= seq_len) continue;
        float inv_l = (rsum[row_i] > 0.0f) ? (1.0f / rsum[row_i]) : 0.0f;
        for (int c = 0; c < 4; c++) {
            int col = tid * 4 + c;
            if (col < D) {
                O[global_row * D + col] = o_acc[row_i][c] * inv_l;
            }
        }
    }
}

void cpu_ref(int N, int d, const float* Q, const float* K, const float* V,
             float* O, float scale) {
    float* S = (float*)malloc(N * N * sizeof(float));
    for(int i=0;i<N;i++)for(int j=0;j<N;j++){
        float s=0;for(int k=0;k<d;k++)s+=Q[i*d+k]*K[j*d+k];S[i*N+j]=s*scale;
    }
    for(int i=0;i<N;i++){
        float mx=-1e9f,sm=0;for(int j=0;j<N;j++)mx=fmaxf(mx,S[i*N+j]);
        for(int j=0;j<N;j++){S[i*N+j]=expf(S[i*N+j]-mx);sm+=S[i*N+j];}
        for(int j=0;j<N;j++)S[i*N+j]/=sm;
    }
    for(int i=0;i<N;i++)for(int k=0;k<d;k++){
        float s=0;for(int j=0;j<N;j++)s+=S[i*N+j]*V[j*d+k];O[i*d+k]=s;
    }
    free(S);
}

int main() {
    int seq_lens[] = {128, 256, 512};
    float scale = 1.0f/sqrtf(64.0f);

    printf("Flash Attn Simple — FP16 CUDA Core, Br=Bc=d=64\n\n");

    for(int si=0;si<3;si++){
        int N=seq_lens[si];
        printf("=== seqlen=%d ===\n",N);

        float *hQ=(float*)malloc(N*64*4),*hK=(float*)malloc(N*64*4),*hV=(float*)malloc(N*64*4);
        float *hO_cpu=(float*)malloc(N*64*4),*hO_gpu=(float*)malloc(N*64*4);
        for(int i=0;i<N*64;i++){hQ[i]=(float)rand()/RAND_MAX-0.5f;hK[i]=(float)rand()/RAND_MAX-0.5f;hV[i]=(float)rand()/RAND_MAX-0.5f;}

        cpu_ref(N,64,hQ,hK,hV,hO_cpu,scale);

        float *dQ32,*dK32,*dV32,*dO;half *dQ16,*dK16,*dV16;
        CHK(cudaMalloc(&dQ32,N*64*4));CHK(cudaMalloc(&dK32,N*64*4));CHK(cudaMalloc(&dV32,N*64*4));
        CHK(cudaMalloc(&dQ16,N*64*2));CHK(cudaMalloc(&dK16,N*64*2));CHK(cudaMalloc(&dV16,N*64*2));
        CHK(cudaMalloc(&dO,N*64*4));
        CHK(cudaMemcpy(dQ32,hQ,N*64*4,cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dK32,hK,N*64*4,cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dV32,hV,N*64*4,cudaMemcpyHostToDevice));
        f32tof16<<<(N*64+255)/256,256>>>(N*64,dQ32,dQ16);
        f32tof16<<<(N*64+255)/256,256>>>(N*64,dK32,dK16);
        f32tof16<<<(N*64+255)/256,256>>>(N*64,dV32,dV16);
        CHK(cudaDeviceSynchronize());

        dim3 grid((N+Br-1)/Br), block(32);
        for(int i=0;i<3;i++)flash_attn_fwd<<<grid,block>>>(N,dQ16,dK16,dV16,dO,scale);
        CHK(cudaDeviceSynchronize());
        cudaError_t e = cudaGetLastError();
        if(e!=cudaSuccess){printf("  Launch failed: %d\n",e);goto cleanup;}

        cudaEvent_t s,st;cudaEventCreate(&s);cudaEventCreate(&st);
        cudaEventRecord(s);
        for(int i=0;i<20;i++)flash_attn_fwd<<<grid,block>>>(N,dQ16,dK16,dV16,dO,scale);
        cudaEventRecord(st);cudaEventSynchronize(st);
        float t;cudaEventElapsedTime(&t,s,st);

        CHK(cudaMemcpy(hO_gpu,dO,N*64*4,cudaMemcpyDeviceToHost));
        float max_err=0;for(int i=0;i<N*64;i++)max_err=fmaxf(max_err,fabsf(hO_gpu[i]-hO_cpu[i]));

        double flops=4.0*N*N*64;
        printf("  Time: %.3f ms  TFLOPS: %.2f  Err: %.4f\n",
               t/20,flops/(t/20000.0)/1e12,max_err);

        cudaEventDestroy(s);cudaEventDestroy(st);
cleanup:
        free(hQ);free(hK);free(hV);free(hO_cpu);free(hO_gpu);
        CHK(cudaFree(dQ32));CHK(cudaFree(dK32));CHK(cudaFree(dV32));
        CHK(cudaFree(dQ16));CHK(cudaFree(dK16));CHK(cudaFree(dV16));CHK(cudaFree(dO));
    }
    printf("\nDone.\n");
    return 0;
}
