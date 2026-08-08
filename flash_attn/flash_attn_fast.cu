/**
 * Flash Attention V3 — Optimized Forward
 * - S stored in shared memory (no recomputation)
 * - float4 vectorized loads for Q,K,V
 * - Merged online softmax rescaling + P@V
 * Br=32, Bc=64, D=64, 32 threads
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define CHK(c) do{cudaError_t e=c;if(e!=cudaSuccess){fprintf(stderr,"E%d\n",e);exit(1);}}while(0)

__global__ void f32tof16(int n, const float* s, half* d) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)d[i]=__float2half(s[i]);
}

#define Br 32
#define Bc 64
#define D 64

__global__ void flash_attn_fwd(int N,
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
    float* __restrict__ O, float scale)
{
    int q_start = blockIdx.x * Br;
    int tid = threadIdx.x;
    int r = tid;  // each thread handles one Q row (0..Br-1)

    __shared__ half Qs[Br][D];       // 4KB
    __shared__ half Ks[Bc][D];       // 8KB
    __shared__ half Vs[Bc][D];       // 8KB
    __shared__ float S_s[Br][Bc];    // 8KB — cached S values
    __shared__ float Ps[Bc][Br];     // 8KB
    __shared__ float Os[Br][D];      // 8KB
    __shared__ float s_rmax[Br], s_rsum[Br];  // 0.25KB

    if (tid < Br) { s_rmax[tid] = -1e9f; s_rsum[tid] = 0.0f; }
    for (int i = tid; i < Br * D; i += 32) Os[i/D][i%D] = 0.0f;

    // Load Q (float4 = 8 fp16 per load)
    for (int i = tid; i < Br * D / 8; i += 32) {
        int idx = i * 8, row = idx / D, col = idx % D;
        int gr = q_start + row;
        if (gr < N && col + 7 < D)
            *(float4*)&Qs[row][col] = *(const float4*)&Q[gr * D + col];
    }

    int num_kv = (N + Bc - 1) / Bc;
    for (int kv = 0; kv < num_kv; kv++) {
        int k_start = kv * Bc;
        int vk = (N - k_start) < Bc ? (N - k_start) : Bc;

        // Load K,V (float4)
        for (int i = tid; i < Bc * D / 8; i += 32) {
            int idx = i * 8, row = idx / D, col = idx % D;
            int gr = k_start + row;
            if (gr < N && col + 7 < D) {
                *(float4*)&Ks[row][col] = *(const float4*)&K[gr * D + col];
                *(float4*)&Vs[row][col] = *(const float4*)&V[gr * D + col];
            }
        }
        __syncthreads();

        // ==========================================
        // STEP 1: Compute S = Q @ K^T, store in S_s, save old rmax
        // ==========================================
        float old_rmax[Br], old_rsum[Br];
        if (tid < Br) {
            old_rmax[tid] = s_rmax[tid];
            old_rsum[tid] = s_rsum[tid];

            float row_max = -1e9f;
            for (int c = 0; c < vk; c++) {
                float dot = 0.0f;
                #pragma unroll
                for (int k = 0; k < D; k += 4) {
                    dot += __half2float(Qs[r][k])   * __half2float(Ks[c][k]);
                    dot += __half2float(Qs[r][k+1]) * __half2float(Ks[c][k+1]);
                    dot += __half2float(Qs[r][k+2]) * __half2float(Ks[c][k+2]);
                    dot += __half2float(Qs[r][k+3]) * __half2float(Ks[c][k+3]);
                }
                float sv = dot * scale;
                S_s[r][c] = sv;
                if (sv > row_max) row_max = sv;
            }
            // Update global max (for THIS K block's new max)
            // The "new max" is max(old max, max of this K block)
            float new_max = fmaxf(old_rmax[r], row_max);
            float ed = expf(old_rmax[r] - new_max);
            float ln = old_rsum[r] * ed;

            // Compute P and update running state
            for (int c = 0; c < vk; c++) {
                float pv = expf(S_s[r][c] - new_max);
                Ps[c][r] = pv;
                ln += pv;
            }

            // Rescale old O
            if (ed < 1.0f)
                for (int d = 0; d < D; d++) Os[r][d] *= ed;

            s_rmax[r] = new_max;
            s_rsum[r] = ln;
        }
        __syncthreads();

        // ==========================================
        // STEP 2: O += P @ V
        // ==========================================
        if (tid < Br) {
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

    // ==========================================
    // Store O
    // ==========================================
    for (int i = tid; i < Br * D; i += 32) {
        int row = i / D, col = i % D;
        int gr = q_start + row;
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
    // PyTorch SDPA ref on 3050 Ti (FP16, measured previously)
    float pt_ms[] = {0.04f, 0.07f, 0.16f, 0.48f, 1.61f};  // rough estimates

    printf("Flash Attn V3 — Optimized (S in smem, float4 loads)\n\n");
    printf("%6s  %8s  %8s  %8s  %8s\n", "N", "ms", "TFLOPS", "MaxErr", "vsPT");
    printf("--------------------------------------------\n");

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

        dim3 grid((N+Br-1)/Br), block(32);
        for (int i=0;i<10;i++)flash_attn_fwd<<<grid,block>>>(N,dQ16,dK16,dV16,dO,scale);
        CHK(cudaDeviceSynchronize());
        if(cudaGetLastError()!=cudaSuccess){printf("LAUNCH FAILED\n");continue;}

        int reps=50; float max_err=0; double flops=4.0*N*N*D;
        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        for(int i=0;i<reps;i++)flash_attn_fwd<<<grid,block>>>(N,dQ16,dK16,dV16,dO,scale);
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms,s,e);

        CHK(cudaMemcpy(hO_gpu,dO,N*D*4,cudaMemcpyDeviceToHost));
        for(int i=0;i<N*D;i++){float d=fabsf(hO_gpu[i]-hO_cpu[i]);if(d>max_err)max_err=d;}
        double tf=flops/(ms/reps/1000.0)/1e12;
        printf("%6d  %8.3f  %8.2f  %8.4f  %5.1fx\n",N,ms/reps,tf,max_err,pt_ms[si]/(ms/reps));

        cudaEventDestroy(s);cudaEventDestroy(e);
        free(hQ);free(hK);free(hV);free(hO_cpu);free(hO_gpu);
        CHK(cudaFree(dQ32));CHK(cudaFree(dK32));CHK(cudaFree(dV32));
        CHK(cudaFree(dQ16));CHK(cudaFree(dK16));CHK(cudaFree(dV16));CHK(cudaFree(dO));
    }
    printf("\nDone.\n");
    return 0;
}
