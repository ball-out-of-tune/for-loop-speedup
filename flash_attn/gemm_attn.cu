/**
 * GEMM-based Attention: Q@K^T (GEMM) + scale+softmax + P@V (GEMM)
 * Uses our V9 WMMA GEMM kernel + fused scale/softmax
 * For single-head: Q,K,V = [N, D]
 * M=K=N for square GEMM, M=N, K=D for Q@K^T (or K=N for P@V)
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
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)d[i]=__float2half(s[i]);
}

// ============================================================
// V9 GEMM kernel (verified: beats PyTorch at 1024, 4096 on 5090)
// ============================================================
__global__ void gemm_v9(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64;
    int c0 = blockIdx.x * 128 + wx * 64;
    __shared__ half Ab[2][128][16], Bb[2][16][128];
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator,16,16,16,float> c[16];
    for(int i=0;i<16;i++)wmma::fill_fragment(c[i],0.0f);
    for(int idx=threadIdx.x;idx<128*16/2;idx+=128){int pos=idx*2,r=pos/16,col=pos%16,gr=blockIdx.y*128+r;if(gr<M&&col+1<K)*(short2*)&Ab[0][r][col]=*(const short2*)&A[gr*K+col];}
    for(int idx=threadIdx.x;idx<16*128/2;idx+=128){int pos=idx*2,r=pos/128,col=pos%128,gc=blockIdx.x*128+col;if(r<K&&gc+1<N)*(short2*)&Bb[0][r][col]=*(const short2*)&B[r*N+gc];}
    __syncthreads();int rb=0;
    for(int kb=16;kb<K;kb+=16){int wb=1-rb;
        for(int c=threadIdx.x;c<128*16/8;c+=128){int p=c*8,r=p/16,col=p%16,gr=blockIdx.y*128+r,gc=kb+col;if(gr<M&&gc+7<K){unsigned sa=__cvta_generic_to_shared(&Ab[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&A[gr*K+gc]));}}
        for(int c=threadIdx.x;c<16*128/8;c+=128){int p=c*8,r=p/128,col=p%128,gr=kb+r,gc=blockIdx.x*128+col;if(gr<K&&gc+7<N){unsigned sa=__cvta_generic_to_shared(&Bb[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&B[gr*N+gc]));}}
        asm volatile("cp.async.commit_group;\n"::);
        for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*64+mi*16][0],16);
        for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][0][wx*64+ni*16],128);
        for(int mi=0;mi<4;mi++)for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
        asm volatile("cp.async.wait_group 0;\n"::);__syncthreads();rb=wb;}
    for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*64+mi*16][0],16);
    for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][0][wx*64+ni*16],128);
    for(int mi=0;mi<4;mi++)for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
    for(int mi=0;mi<4;mi++)for(int ni=0;ni<4;ni++)wmma::store_matrix_sync(C+(r0+mi*16)*N+c0+ni*16,c[mi*4+ni],N,wmma::mem_row_major);
}

// ============================================================
// Fused scale + softmax kernel
// Input: S [M, N] in FP32 (from GEMM output)
// Output: P [M, N] in FP16 (softmax over dim=-1, ready for P@V)
// ============================================================
__global__ void scale_softmax(int M, int N, const float* __restrict__ S, half* __restrict__ P, float scale) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // Find max in this row
    float mx = -1e9f;
    for (int j = 0; j < N; j++) {
        float v = S[row * N + j] * scale;
        if (v > mx) mx = v;
    }
    // Compute exp sum
    float sm = 0.0f;
    for (int j = 0; j < N; j++) {
        float v = expf(S[row * N + j] * scale - mx);
        sm += v;
    }
    // Normalize and store as FP16
    float inv_sm = 1.0f / sm;
    for (int j = 0; j < N; j++) {
        float v = expf(S[row * N + j] * scale - mx) * inv_sm;
        P[row * N + j] = __float2half(v);
    }
}

// ============================================================
// CPU reference
// ============================================================
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
    int seqs[] = {512, 1024, 2048, 4096};
    int D = 64;
    float scale = 1.0f / sqrtf(64.0f);

    printf("GEMM-based Attention (V9 WMMA x2 + softmax)

");
    printf("%6s  %8s  %8s  %8s
", "N", "ms", "TFLOPS", "MaxErr");
    printf("--------------------------------------
");

    for (int si = 0; si < 4; si++) {
        int N = seqs[si];
        double flops = 4.0 * N * N * D;

        // Pad dimensions to 128 for V9 tile alignment
        int N_pad = ((N + 127) / 128) * 128;

        float *hQ=(float*)calloc(N_pad*D, sizeof(float));
        float *hK=(float*)calloc(N_pad*D, sizeof(float));
        float *hV=(float*)calloc(N_pad*D, sizeof(float));
        float *hO_cpu=(float*)calloc(N*D, sizeof(float));
        float *hO_gpu=(float*)calloc(N_pad*D, sizeof(float));

        srand(42);
        for(int i=0;i<N*D;i++){hQ[i]=(float)rand()/RAND_MAX-0.5f;hK[i]=(float)rand()/RAND_MAX-0.5f;hV[i]=(float)rand()/RAND_MAX-0.5f;}
        cpu_attn(N,D,hQ,hK,hV,hO_cpu,scale);

        // GPU allocations (padded)
        float *dQ32,*dK32,*dV32,*dS,*dO; half *dQ16,*dK16,*dV16,*dP;
        CHK(cudaMalloc(&dQ32,N_pad*D*4)); CHK(cudaMalloc(&dK32,N_pad*D*4)); CHK(cudaMalloc(&dV32,N_pad*D*4));
        CHK(cudaMalloc(&dQ16,N_pad*D*2)); CHK(cudaMalloc(&dK16,N_pad*D*2)); CHK(cudaMalloc(&dV16,N_pad*D*2));
        CHK(cudaMalloc(&dS,N_pad*N_pad*4));
        CHK(cudaMalloc(&dP,N_pad*N_pad*2));
        CHK(cudaMalloc(&dO,N_pad*D*4));

        // Copy and pad with zeros
        CHK(cudaMemset(dQ32,0,N_pad*D*4)); CHK(cudaMemset(dK32,0,N_pad*D*4)); CHK(cudaMemset(dV32,0,N_pad*D*4));
        CHK(cudaMemcpy(dQ32,hQ,N*D*4,cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dK32,hK,N*D*4,cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dV32,hV,N*D*4,cudaMemcpyHostToDevice));
        f32tof16<<<(N_pad*D+255)/256,256>>>(N_pad*D,dQ32,dQ16);
        f32tof16<<<(N_pad*D+255)/256,256>>>(N_pad*D,dK32,dK16);
        f32tof16<<<(N_pad*D+255)/256,256>>>(N_pad*D,dV32,dV16);
        CHK(cudaMemset(dO,0,N_pad*D*4));
        CHK(cudaDeviceSynchronize());

        // Grid for GEMM
        dim3 gemmBlock(128);

        // Step 1: S = Q @ K^T  (M=N_pad, K=D, N=N_pad)
        dim3 grid1((N_pad+127)/128, (N_pad+127)/128);
        gemm_v9<<<grid1,gemmBlock>>>(N_pad, N_pad, D, dQ16, dK16, dS);
        CHK(cudaDeviceSynchronize());

        // Step 2: scale + softmax
        scale_softmax<<<(N+255)/256,256>>>(N, N_pad, dS, dP, scale);
        CHK(cudaDeviceSynchronize());

        // Step 3: O = P @ V  (M=N, K=N_pad, N=D)
        dim3 grid2(((D+127)/128), (N+127)/128);
        gemm_v9<<<grid2,gemmBlock>>>(N, D, N_pad, dP, dV16, dO);
        CHK(cudaDeviceSynchronize());
        CHK(cudaGetLastError());

        // Benchmark
        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        for(int w=0;w<10;w++){
            gemm_v9<<<grid1,gemmBlock>>>(N_pad,N_pad,D,dQ16,dK16,dS);
            scale_softmax<<<(N+255)/256,256>>>(N,N_pad,dS,dP,scale);
            gemm_v9<<<grid2,gemmBlock>>>(N,D,N_pad,dP,dV16,dO);
        }
        CHK(cudaDeviceSynchronize());

        cudaEventRecord(s);
        for(int w=0;w<50;w++){
            gemm_v9<<<grid1,gemmBlock>>>(N_pad,N_pad,D,dQ16,dK16,dS);
            scale_softmax<<<(N+255)/256,256>>>(N,N_pad,dS,dP,scale);
            gemm_v9<<<grid2,gemmBlock>>>(N,D,N_pad,dP,dV16,dO);
        }
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms; cudaEventElapsedTime(&ms,s,e);
        double tf=flops/(ms/50/1000.0)/1e12;

        CHK(cudaMemcpy(hO_gpu,dO,N*D*4,cudaMemcpyDeviceToHost));
        float max_err=0;
        for(int i=0;i<N*D;i++){float d=fabsf(hO_gpu[i]-hO_cpu[i]);if(d>max_err)max_err=d;}
        printf("%6d  %8.3f  %8.2f  %8.4f
",N,ms/50,tf,max_err);

        cudaEventDestroy(s);cudaEventDestroy(e);
        free(hQ);free(hK);free(hV);free(hO_cpu);free(hO_gpu);
        CHK(cudaFree(dQ32));CHK(cudaFree(dK32));CHK(cudaFree(dV32));
        CHK(cudaFree(dQ16));CHK(cudaFree(dK16));CHK(cudaFree(dV16));
        CHK(cudaFree(dS));CHK(cudaFree(dP));CHK(cudaFree(dO));
    }
    printf("
Done.
");
    return 0;
}