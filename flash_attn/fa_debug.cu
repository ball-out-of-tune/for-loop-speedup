/**
 * Minimal Flash Attention — reads binary data, writes binary output
 * N=64, Br=32, Bc=64, D=64
 * Python generates data, C computes, Python verifies
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define CHK(c) do{cudaError_t e=c;if(e!=cudaSuccess){fprintf(stderr,"E%d\n",e);exit(1);}}while(0)

#define Br 32
#define Bc 64
#define D 64

__global__ void fa_debug(int N,
    const half* __restrict__ Q, const half* __restrict__ K, const half* __restrict__ V,
    float* __restrict__ O, float* __restrict__ debug_S, float* __restrict__ debug_P,
    float scale)
{
    int q_start = blockIdx.x * Br;
    int tid = threadIdx.x;

    __shared__ half Qs[Br][D];
    __shared__ half Ks[Bc][D];
    __shared__ half Vs[Bc][D];
    __shared__ float Ps[Bc][Br];
    __shared__ float Os[Br][D];
    __shared__ float s_rmax[Br];  // MUST be shared (all threads read O store)
    __shared__ float s_rsum[Br];  // MUST be shared

    if (tid < Br) { s_rmax[tid] = -1e9f; s_rsum[tid] = 0.0f; }
    for (int i = tid; i < Br * D; i += 32) Os[i/D][i%D] = 0.0f;
    __syncthreads();
    for (int i = tid; i < Br * D; i += 32) {
        int r = i / D, c = i % D;
        int gr = q_start + r;
        Qs[r][c] = (gr < N && c < D) ? Q[gr * D + c] : __float2half(0.0f);
    }

    int num_kv = (N + Bc - 1) / Bc;
    for (int kv = 0; kv < num_kv; kv++) {
        int k_start = kv * Bc;
        int vk = (N - k_start) < Bc ? (N - k_start) : Bc;

        for (int i = tid; i < Bc * D; i += 32) {
            int r = i / D, c = i % D;
            int gr = k_start + r;
            Ks[r][c] = (gr < N && c < D) ? K[gr * D + c] : __float2half(0.0f);
            Vs[r][c] = (gr < N && c < D) ? V[gr * D + c] : __float2half(0.0f);
        }
        __syncthreads();

        if (tid < Br) {
            int r = tid;
            float row_max = s_rmax[r];
            // Single pass: compute S, find max, P, rescale
            // First find max
            for (int c = 0; c < vk; c++) {
                float dot = 0.0f;
                for (int k = 0; k < D; k++)
                    dot += __half2float(Qs[r][k]) * __half2float(Ks[c][k]);
                float sv = dot * scale;
                if (sv > row_max) row_max = sv;
            }
            float ed = expf(s_rmax[r] - row_max);
            float ln = s_rsum[r] * ed;
            if (ed < 1.0f)
                for (int d = 0; d < D; d++) Os[r][d] *= ed;

            for (int c = 0; c < vk; c++) {
                float dot = 0.0f;
                for (int k = 0; k < D; k++)
                    dot += __half2float(Qs[r][k]) * __half2float(Ks[c][k]);
                float pv = expf(dot * scale - row_max);
                Ps[c][r] = pv;
                ln += pv;

                // Debug: for first Q block, first row, first K block, store S and P
                if (blockIdx.x == 0 && r == 0 && kv == 0 && c < 8) {
                    debug_S[c] = dot * scale;
                    debug_P[c] = pv;
                }
            }
            s_rmax[r] = row_max;
            s_rsum[r] = ln;
        }
        __syncthreads();

        if (tid < Br) {
            int r = tid;
            for (int d = 0; d < D; d++) {
                float acc = 0.0f;
                for (int k = 0; k < vk; k++)
                    acc += Ps[k][r] * __half2float(Vs[k][d]);
                Os[r][d] += acc;
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < Br * D; i += 32) {
        int r = i / D, c = i % D;
        int gr = q_start + r;
        if (gr < N && c < D)
            O[gr * D + c] = Os[r][c] / s_rsum[r];
    }
}

int main() {
    int N = 64;
    float scale = 1.0f / sqrtf(64.0f);

    // Read binary data from files
    float* hQ = (float*)malloc(N * D * sizeof(float));
    float* hK = (float*)malloc(N * D * sizeof(float));
    float* hV = (float*)malloc(N * D * sizeof(float));

    FILE* f = fopen("q_data.bin", "rb");
    fread(hQ, sizeof(float), N * D, f); fclose(f);
    f = fopen("k_data.bin", "rb");
    fread(hK, sizeof(float), N * D, f); fclose(f);
    f = fopen("v_data.bin", "rb");
    fread(hV, sizeof(float), N * D, f); fclose(f);

    printf("Loaded data: Q[0]=%f, K[0]=%f, V[0]=%f\n", hQ[0], hK[0], hV[0]);

    float *dQ32, *dK32, *dV32, *dO, *dS, *dP;
    half *dQ16, *dK16, *dV16;
    CHK(cudaMalloc(&dQ32, N*D*4)); CHK(cudaMalloc(&dK32, N*D*4)); CHK(cudaMalloc(&dV32, N*D*4));
    CHK(cudaMalloc(&dQ16, N*D*2)); CHK(cudaMalloc(&dK16, N*D*2)); CHK(cudaMalloc(&dV16, N*D*2));
    CHK(cudaMalloc(&dO, N*D*4));
    CHK(cudaMalloc(&dS, 8*4));  // debug: first 8 S values
    CHK(cudaMalloc(&dP, 8*4));  // debug: first 8 P values
    CHK(cudaMemset(dS, 0, 8*4));
    CHK(cudaMemset(dP, 0, 8*4));

    CHK(cudaMemcpy(dQ32, hQ, N*D*4, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dK32, hK, N*D*4, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dV32, hV, N*D*4, cudaMemcpyHostToDevice));

    // Use cudaMemcpy with half conversion instead of kernel
    half* tmp = (half*)malloc(N*D*2);
    for (int i = 0; i < N*D; i++) tmp[i] = __float2half(hQ[i]);
    CHK(cudaMemcpy(dQ16, tmp, N*D*2, cudaMemcpyHostToDevice));
    for (int i = 0; i < N*D; i++) tmp[i] = __float2half(hK[i]);
    CHK(cudaMemcpy(dK16, tmp, N*D*2, cudaMemcpyHostToDevice));
    for (int i = 0; i < N*D; i++) tmp[i] = __float2half(hV[i]);
    CHK(cudaMemcpy(dV16, tmp, N*D*2, cudaMemcpyHostToDevice));
    free(tmp);

    dim3 grid((N+Br-1)/Br), block(32);
    fa_debug<<<grid, block>>>(N, dQ16, dK16, dV16, dO, dS, dP, scale);
    CHK(cudaDeviceSynchronize());
    printf("Kernel launched OK\n");

    // Read back
    float* hO = (float*)malloc(N*D*4);
    float hS[8], hP[8];
    CHK(cudaMemcpy(hO, dO, N*D*4, cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(hS, dS, 8*4, cudaMemcpyDeviceToHost));
    CHK(cudaMemcpy(hP, dP, 8*4, cudaMemcpyDeviceToHost));

    printf("O[0,0]=%f  O[0,1]=%f\n", hO[0], hO[1]);
    printf("S[0,0..7]: ");
    for (int i = 0; i < 8; i++) printf("%f ", hS[i]);
    printf("\nP[0,0..7]: ");
    for (int i = 0; i < 8; i++) printf("%f ", hP[i]);
    printf("\n");

    // Write O to binary for Python comparison
    f = fopen("o_data.bin", "wb");
    fwrite(hO, sizeof(float), N*D, f); fclose(f);
    printf("Wrote o_data.bin\n");

    free(hQ); free(hK); free(hV); free(hO);
    CHK(cudaFree(dQ32)); CHK(cudaFree(dK32)); CHK(cudaFree(dV32));
    CHK(cudaFree(dQ16)); CHK(cudaFree(dK16)); CHK(cudaFree(dV16));
    CHK(cudaFree(dO)); CHK(cudaFree(dS)); CHK(cudaFree(dP));
    return 0;
}
