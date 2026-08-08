/**
 * Debug kernel: Isolate alignment issues on sm_120
 */
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
using namespace nvcuda;

#define CE(call) do { cudaError_t e = call; if(e) printf("ERR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); } while(0)

/* Test 1: Pure WMMA without cp.async, loads from global directly */
__global__ void test_wmma_only(const half* A, const half* B, float* C, int M, int N, int K) {
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64, c0 = blockIdx.x * 128 + wx * 64;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];
    for(int i=0;i<16;i++) wmma::fill_fragment(c[i], 0.0f);

    // Load directly from global (SLOW but should work on any arch)
    for(int kb=0; kb<K; kb+=16) {
        for(int mi=0;mi<4;mi++) {
            int row = wy*64 + mi*16;
            if(row < M) wmma::load_matrix_sync(a[mi], &A[(blockIdx.y*128+row)*K + kb], K);
        }
        for(int ni=0;ni<4;ni++) {
            int col = wx*64 + ni*16;
            if(col < N) wmma::load_matrix_sync(b[ni], &B[kb*N + blockIdx.x*128 + col], N);
        }
        for(int mi=0;mi<4;mi++)
        for(int ni=0;ni<4;ni++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);
    }

    for(int mi=0;mi<4;mi++)
    for(int ni=0;ni<4;ni++) {
        int fr=r0+mi*16, fc=c0+ni*16;
        if(fr+16<=M && fc+16<=N)
            wmma::store_matrix_sync(&C[fr*N+fc], c[mi*4+ni], N, wmma::mem_row_major);
    }
}

/* Test 2: WMMA with shared memory (sync loads, no cp.async) */
__global__ void test_wmma_smem(const half* A, const half* B, float* C, int M, int N, int K) {
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64, c0 = blockIdx.x * 128 + wx * 64;

    __shared__ half As[128][16];
    __shared__ half Bs[16][128];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];
    for(int i=0;i<16;i++) wmma::fill_fragment(c[i], 0.0f);

    for(int kb=0; kb<K; kb+=16) {
        // Sync load into shared memory (no cp.async)
        for(int idx=threadIdx.x; idx<128*16; idx+=128) {
            int r=idx/16, c2=idx%16;
            int gr=blockIdx.y*128+r, gc=kb+c2;
            if(gr<M && gc<K) As[r][c2] = A[gr*K+gc];
        }
        for(int idx=threadIdx.x; idx<16*128; idx+=128) {
            int r=idx/128, c2=idx%128;
            int gr=kb+r, gc=blockIdx.x*128+c2;
            if(gr<K && gc<N) Bs[r][c2] = B[gr*N+gc];
        }
        __syncthreads();

        for(int mi=0;mi<4;mi++) wmma::load_matrix_sync(a[mi], &As[wy*64+mi*16][0], 16);
        for(int ni=0;ni<4;ni++) wmma::load_matrix_sync(b[ni], &Bs[0][wx*64+ni*16], 128);
        for(int mi=0;mi<4;mi++)
        for(int ni=0;ni<4;ni++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);
        __syncthreads();
    }

    for(int mi=0;mi<4;mi++)
    for(int ni=0;ni<4;ni++) {
        int fr=r0+mi*16, fc=c0+ni*16;
        if(fr+16<=M && fc+16<=N)
            wmma::store_matrix_sync(&C[fr*N+fc], c[mi*4+ni], N, wmma::mem_row_major);
    }
}

int main() {
    cudaDeviceProp p; CE(cudaGetDeviceProperties(&p,0));
    printf("GPU: %s CC %d.%d\n", p.name, p.major, p.minor);

    int M=512, N=512, K=512;
    half *dA, *dB, *hA=(half*)malloc(M*K*2), *hB=(half*)malloc(K*N*2);
    float *dC;
    for(int i=0;i<M*K;i++) hA[i]=__float2half(0.1f);
    for(int i=0;i<K*N;i++) hB[i]=__float2half(0.1f);
    CE(cudaMalloc(&dA, M*K*2)); CE(cudaMemcpy(dA,hA,M*K*2,cudaMemcpyHostToDevice));
    CE(cudaMalloc(&dB, K*N*2)); CE(cudaMemcpy(dB,hB,K*N*2,cudaMemcpyHostToDevice));
    CE(cudaMalloc(&dC, M*N*4));
    free(hA); free(hB);

    dim3 blk(128), grd(N/128, M/128);

    printf("\n=== Test 1: WMMA from global (no shared mem, no cp.async) ===\n");
    CE(cudaGetLastError());
    test_wmma_only<<<grd,blk>>>(dA,dB,dC,M,N,K);
    cudaError_t e = cudaDeviceSynchronize();
    printf("Result: %s\n", cudaGetErrorString(e));

    printf("\n=== Test 2: WMMA with shared mem (sync loads, no cp.async) ===\n");
    CE(cudaGetLastError());
    test_wmma_smem<<<grd,blk>>>(dA,dB,dC,M,N,K);
    e = cudaDeviceSynchronize();
    printf("Result: %s\n", cudaGetErrorString(e));

    CE(cudaFree(dA)); CE(cudaFree(dB)); CE(cudaFree(dC));
    return 0;
}
