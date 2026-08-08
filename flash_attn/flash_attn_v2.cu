/**
 * Flash Attention V2 Forward — Clean Implementation
 * Br=64, Bc=64, D=64, 32 threads, O in shared memory (correct online softmax rescaling)
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define CHK(c) do{cudaError_t e=c;if(e!=cudaSuccess){fprintf(stderr,"E%d@%d\n",e,__LINE__);exit(1);}}while(0)

__global__ void f32tof16(int n, const float* s, half* d) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)d[i]=__float2half(s[i]);
}

#define Br 32
#define Bc 64
#define D 64

__global__ void flash_attn_fwd(int seq_len,
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
    float* __restrict__ O, float scale)
{
    int q_start = blockIdx.x * Br;
    int tid = threadIdx.x;

    __shared__ half Qs[Br][D];
    __shared__ half Ks[Bc][D];
    __shared__ half Vs[Bc][D];
    __shared__ float Ps[Bc][Br];  // FP32 P values (transposed)
    __shared__ float Os[Br][D];   // FP32 O accumulator

    __shared__ float s_s_rmax[Br];
    __shared__ float s_s_s_rsum[Br];
    if (tid < Br) { s_s_rmax[tid] = -1e9f; s_s_s_rsum[tid] = 0.0f; }
    __syncthreads();
    for (int i = tid; i < Br * D; i += 32) { int r = i / D, c = i % D; Os[r][c] = 0.0f; }

    // Load Q
    for (int i = tid; i < Br * D; i += 32) {
        int r = i / D, c = i % D;
        int gr = q_start + r;
        Qs[r][c] = (gr < seq_len && c < D) ? Q[gr * D + c] : __float2half(0.0f);
    }

    int num_kv = (seq_len + Bc - 1) / Bc;
    for (int kv = 0; kv < num_kv; kv++) {
        int k_start = kv * Bc;
        int valid_kv = (seq_len - k_start) < Bc ? (seq_len - k_start) : Bc;

        for (int i = tid; i < Bc * D; i += 32) {
            int r = i / D, c = i % D;
            int gr = k_start + r;
            Ks[r][c] = (gr < seq_len && c < D) ? K[gr * D + c] : __float2half(0.0f);
            Vs[r][c] = (gr < seq_len && c < D) ? V[gr * D + c] : __float2half(0.0f);
        }
        __syncthreads();

        // S = Q @ K^T + online softmax (two-pass: find global max, then compute P)
        if (tid < Br) {
            int r = tid;

            // Pass 1: find global max for this KV block
            float row_max = s_s_rmax[r];
            for (int c = 0; c < valid_kv; c++) {
                float dot = 0.0f;
                for (int k = 0; k < D; k++)
                    dot += __half2float(Qs[r][k]) * __half2float(Ks[c][k]);
                float s_val = dot * scale;
                if (s_val > row_max) row_max = s_val;
            }

            // Rescale O and running sum
            float exp_diff = expf(s_s_rmax[r] - row_max);
            float l_new = s_s_s_rsum[r] * exp_diff;
            if (exp_diff < 1.0f)
                for (int d = 0; d < D; d++) Os[r][d] *= exp_diff;

            // Pass 2: compute P with correct global max
            for (int c = 0; c < valid_kv; c++) {
                float dot = 0.0f;
                for (int k = 0; k < D; k++)
                    dot += __half2float(Qs[r][k]) * __half2float(Ks[c][k]);
                float p_val = expf(dot * scale - row_max);
                Ps[c][r] = p_val;
                l_new += p_val;
            }

            s_s_rmax[r] = row_max;
            s_s_s_rsum[r] = l_new;
        }
        __syncthreads();

        // O += P @ V
        if (tid < Br) {
            int r = tid;
            for (int d = 0; d < D; d++) {
                float acc = 0.0f;
                for (int k = 0; k < Bc; k++)
                    acc += Ps[k][r] * __half2float(Vs[k][d]);
                Os[r][d] += acc;
            }
        }
        __syncthreads();
    }

    // Store O (normalized)
    for (int i = tid; i < Br * D; i += 32) {
        int r = i / D, c = i % D;
        int gr = q_start + r;
        if (gr < seq_len && c < D) {
            float inv_l = (s_s_s_rsum[r] > 0.0f) ? (1.0f / s_s_s_rsum[r]) : 0.0f;
            O[gr * D + c] = Os[r][c] * inv_l;
        }
    }
}

// CPU reference
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
    int seqs[] = {256, 512, 1024, 2048, 4096};  // minimal test: 1 Q block, 1 K block, no online rescaling
    float scale = 1.0f / sqrtf(64.0f);

    printf("Flash Attn V2 — Br=32, Bc=D=64, FP16 compute + FP32 O\n\n");
    printf("%6s  %8s  %8s  %8s  %8s\n", "N", "ms", "TFLOPS", "ErrFP32", "ErrFP16");
    printf("--------------------------------------\n");

    for (int si = 0; si < 5; si++) {
        int N = seqs[si];
        float *hQ=(float*)malloc(N*D*4),*hK=(float*)malloc(N*D*4),*hV=(float*)malloc(N*D*4);
        float *hO_cpu=(float*)malloc(N*D*4),*hO_gpu=(float*)malloc(N*D*4);

        srand(12345);  // different seed — if error changes, it's FP16 quantization
        for (int i=0;i<N*D;i++){hQ[i]=(float)rand()/RAND_MAX-0.5f;hK[i]=(float)rand()/RAND_MAX-0.5f;hV[i]=(float)rand()/RAND_MAX-0.5f;}
        cpu_attn(N, D, hQ, hK, hV, hO_cpu, scale);

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

        dim3 grid((N+Br-1)/Br), block(32);
        for (int i=0;i<10;i++)flash_attn_fwd<<<grid,block>>>(N,dQ16,dK16,dV16,dO,scale);
        CHK(cudaDeviceSynchronize());
        if (cudaGetLastError()!=cudaSuccess){printf("LAUNCH FAILED\n");continue;}

        int reps=50; float max_err=0; double flops=4.0*N*N*D, tflops=0;
        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for (int i=0;i<reps;i++) flash_attn_fwd<<<grid,block>>>(N,dQ16,dK16,dV16,dO,scale);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms,s,e);

        CHK(cudaMemcpy(hO_gpu,dO,N*D*4,cudaMemcpyDeviceToHost));
                for (int i=0;i<N*D;i++){float d=fabsf(hO_gpu[i]-hO_cpu[i]);if(d>max_err)max_err=d;}
        tflops=flops/(ms/reps/1000.0)/1e12;
        // Also compute using FP16 inputs (same as kernel sees)
        float *hQ16=(float*)malloc(N*D*4),*hK16=(float*)malloc(N*D*4),*hV16=(float*)malloc(N*D*4);
        float *hO_fp16ref=(float*)malloc(N*D*4);
        for(int i=0;i<N*D;i++){
            hQ16[i]=__half2float(__float2half(hQ[i]));
            hK16[i]=__half2float(__float2half(hK[i]));
            hV16[i]=__half2float(__float2half(hV[i]));
        }
        cpu_attn(N,D,hQ16,hK16,hV16,hO_fp16ref,scale);
        float max_err_fp16=0;
        for(int i=0;i<N*D;i++){float d=fabsf(hO_gpu[i]-hO_fp16ref[i]);if(d>max_err_fp16)max_err_fp16=d;}
        printf("%6d  %8.3f  %8.2f  %8.4f  %8.4f\n",N,ms/reps,tflops,max_err,max_err_fp16);
        free(hQ16);free(hK16);free(hV16);free(hO_fp16ref);

        cudaEventDestroy(s); cudaEventDestroy(e);
        free(hQ);free(hK);free(hV);free(hO_cpu);free(hO_gpu);
        CHK(cudaFree(dQ32));CHK(cudaFree(dK32));CHK(cudaFree(dV32));
        CHK(cudaFree(dQ16));CHK(cudaFree(dK16));CHK(cudaFree(dV16));CHK(cudaFree(dO));
    }
    printf("\nDone.\n");
    return 0;
}
