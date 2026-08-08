/**
 * FP16 Tensor Core GEMM: separate conversion + WMMA
 * vs PyTorch FP16 (which also uses tensor cores)
 *
 * nvcc -o tcore_gemm tensorcore_gemm.cu -arch=sm_86
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define CUDA_CHECK(c) do{cudaError_t e=c;if(e!=cudaSuccess){fprintf(stderr,"CUDA %d\n",e);exit(1);}}while(0)
float ms(cudaEvent_t s,cudaEvent_t e){float t;cudaEventElapsedTime(&t,s,e);return t;}

// Step 1: FP32 -> FP16
__global__ void convert_f32tof16(int n, const float* src, half* dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// Step 2 (V1): FP16 WMMA GEMM, 256 threads, 64x32 tile, 1 WMMA/warp/K
__global__ void gemm_wmma_fp16_v1(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int warp_id = threadIdx.x / 32;
    int wy = warp_id / 2;  // 0..3
    int wx = warp_id % 2;  // 0..1
    int warpM = blockIdx.y * 64 + wy * 16;
    int warpN = blockIdx.x * 32 + wx * 16;

    __shared__ half As[64][16];
    __shared__ half Bs[16][32];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int kb = 0; kb < K; kb += 16) {
        for (int i = threadIdx.x; i < 64*16; i += 256) {
            int r = i / 16, c = i % 16;
            int gr = blockIdx.y * 64 + r, gc = kb + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
        }
        for (int i = threadIdx.x; i < 16*32; i += 256) {
            int r = i / 32, c = i % 32;
            int gr = kb + r, gc = blockIdx.x * 32 + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, &As[wy * 16][0], 16);
        wmma::load_matrix_sync(b_frag, &Bs[0][wx * 16], 32);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }
    wmma::store_matrix_sync(C + warpM * N + warpN, c_frag, N, wmma::mem_row_major);
}

// Step 2 (V2): 64x64 tile, 2 WMMA/warp/K, 4KB smem → high occupancy
__global__ void gemm_wmma_fp16_v2(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int warp_id = threadIdx.x / 32;
    int wy = warp_id / 2;  // 0..3
    int wx = warp_id % 2;  // 0..1
    int warpM = blockIdx.y * 64 + wy * 16;
    int warpN = blockIdx.x * 64 + wx * 32;

    __shared__ half As[64][16];
    __shared__ half Bs[16][64];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b0, b1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0, c1;
    wmma::fill_fragment(c0, 0.0f);
    wmma::fill_fragment(c1, 0.0f);

    for (int kb = 0; kb < K; kb += 16) {
        for (int i = threadIdx.x; i < 64*16; i += 256) {
            int r = i / 16, c = i % 16;
            int gr = blockIdx.y * 64 + r, gc = kb + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
        }
        for (int i = threadIdx.x; i < 16*64; i += 256) {
            int r = i / 64, c = i % 64;
            int gr = kb + r, gc = blockIdx.x * 64 + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
        }
        __syncthreads();

        wmma::load_matrix_sync(a, &As[wy * 16][0], 16);
        wmma::load_matrix_sync(b0, &Bs[0][wx * 32 + 0],  64);
        wmma::load_matrix_sync(b1, &Bs[0][wx * 32 + 16], 64);
        wmma::mma_sync(c0, a, b0, c0);
        wmma::mma_sync(c1, a, b1, c1);
        __syncthreads();
    }
    wmma::store_matrix_sync(C + warpM * N + warpN + 0,  c0, N, wmma::mem_row_major);
    wmma::store_matrix_sync(C + warpM * N + warpN + 16, c1, N, wmma::mem_row_major);
}

// V3: 128x64 tile, 4 WMMA/warp/K, 6KB smem
__global__ void gemm_wmma_fp16_v3(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int warp_id = threadIdx.x / 32;  // 0..7
    int warp_row = blockIdx.y * 128 + warp_id * 16;
    int warp_col = blockIdx.x * 64;

    __shared__ half As[128][16];
    __shared__ half Bs[16][64];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b0,b1,b2,b3;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c0,c1,c2,c3;
    wmma::fill_fragment(c0,0);wmma::fill_fragment(c1,0);
    wmma::fill_fragment(c2,0);wmma::fill_fragment(c3,0);

    for (int kb = 0; kb < K; kb += 16) {
        for (int i = threadIdx.x; i < 128*16; i += 256) {
            int r = i / 16, c = i % 16;
            int gr = blockIdx.y * 128 + r, gc = kb + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
        }
        for (int i = threadIdx.x; i < 16*64; i += 256) {
            int r = i / 64, c = i % 64;
            int gr = kb + r, gc = blockIdx.x * 64 + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
        }
        __syncthreads();
        wmma::load_matrix_sync(a,  &As[warp_id * 16][0], 16);
        wmma::load_matrix_sync(b0, &Bs[0][0],  64);
        wmma::load_matrix_sync(b1, &Bs[0][16], 64);
        wmma::load_matrix_sync(b2, &Bs[0][32], 64);
        wmma::load_matrix_sync(b3, &Bs[0][48], 64);
        wmma::mma_sync(c0,a,b0,c0); wmma::mma_sync(c1,a,b1,c1);
        wmma::mma_sync(c2,a,b2,c2); wmma::mma_sync(c3,a,b3,c3);
        __syncthreads();
    }
    wmma::store_matrix_sync(C + warp_row * N + warp_col + 0,  c0, N, wmma::mem_row_major);
    wmma::store_matrix_sync(C + warp_row * N + warp_col + 16, c1, N, wmma::mem_row_major);
    wmma::store_matrix_sync(C + warp_row * N + warp_col + 32, c2, N, wmma::mem_row_major);
    wmma::store_matrix_sync(C + warp_row * N + warp_col + 48, c3, N, wmma::mem_row_major);
}

// V4: 128x128 tile, 4 warps (128 threads), 16 WMMA/warp/K, 8KB smem
__global__ void gemm_wmma_fp16_v4(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int warp_id = threadIdx.x / 32;  // 0..3
    int wy = warp_id / 2;  // 0..1
    int wx = warp_id % 2;  // 0..1
    int warp_row = blockIdx.y * 128 + wy * 64;
    int warp_col = blockIdx.x * 128 + wx * 64;

    __shared__ half As[128][16];
    __shared__ half Bs[16][128];

    // 4×4 = 16 WMMA per warp: 4 M-blocks × 4 N-blocks = 16 accumulators
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    for (int kb = 0; kb < K; kb += 16) {
        // Cooperative load: 128 threads load As[128][16] + Bs[16][128]
        for (int i = threadIdx.x; i < 128*16; i += 128) {
            int r = i / 16, c = i % 16;
            int gr = blockIdx.y * 128 + r, gc = kb + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
        }
        for (int i = threadIdx.x; i < 16*128; i += 128) {
            int r = i / 128, c = i % 128;
            int gr = kb + r, gc = blockIdx.x * 128 + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
        }
        __syncthreads();

        // Load 4 A-tiles and 4 B-tiles per warp
        int base_row = wy * 64;
        int base_col = wx * 64;
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &As[base_row + mi*16][0], 16);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &Bs[0][base_col + ni*16], 128);

        // 16 WMMA ops
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);

        __syncthreads();
    }

    // Store 16 output tiles per warp
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::store_matrix_sync(C + (warp_row + mi*16) * N + warp_col + ni*16,
                                c[mi*4+ni], N, wmma::mem_row_major);
}

// V5: 128x128 tile, 8 warps (256 threads), 8 WMMA/warp, 8KB smem
__global__ void gemm_wmma_fp16_v5(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int warp_id = threadIdx.x / 32;  // 0..7, 4x2 layout
    int wy = warp_id / 2;  // 0..3 (M dir)
    int wx = warp_id % 2;  // 0..1 (N dir)
    int warp_row = blockIdx.y * 128 + wy * 32;  // 32 M-rows per warp
    int warp_col = blockIdx.x * 128 + wx * 64;  // 64 N-cols per warp

    __shared__ half As[128][16];
    __shared__ half Bs[16][128];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[8];
    for (int i = 0; i < 8; i++) wmma::fill_fragment(c[i], 0.0f);

    for (int kb = 0; kb < K; kb += 16) {
        for (int i = threadIdx.x; i < 128*16; i += 256) {
            int r = i / 16, c = i % 16;
            int gr = blockIdx.y * 128 + r, gc = kb + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
        }
        for (int i = threadIdx.x; i < 16*128; i += 256) {
            int r = i / 128, c = i % 128;
            int gr = kb + r, gc = blockIdx.x * 128 + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
        }
        __syncthreads();

        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            wmma::load_matrix_sync(a[mi], &As[wy*32 + mi*16][0], 16);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &Bs[0][wx*64 + ni*16], 128);

        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);

        __syncthreads();
    }
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::store_matrix_sync(C + (warp_row+mi*16)*N + warp_col+ni*16,
                                c[mi*4+ni], N, wmma::mem_row_major);
}

// V6: 128x128 tile, 4 warps, K-step=32 (double), 32 WMMA/warp/sync, 16KB smem
__global__ void gemm_wmma_fp16_v6(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int warp_id = threadIdx.x / 32;
    int wy = warp_id / 2, wx = warp_id % 2;
    int warp_row = blockIdx.y * 128 + wy * 64;
    int warp_col = blockIdx.x * 128 + wx * 64;

    __shared__ half As[128][32];  // doubled K-dim
    __shared__ half Bs[32][128];  // doubled K-dim

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    for (int kb = 0; kb < K; kb += 32) {  // 32 K at a time = 2 WMMA K-steps
        for (int i = threadIdx.x; i < 128*32; i += 128) {
            int r = i / 32, c = i % 32;
            int gr = blockIdx.y * 128 + r, gc = kb + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
        }
        for (int i = threadIdx.x; i < 32*128; i += 128) {
            int r = i / 128, c = i % 128;
            int gr = kb + r, gc = blockIdx.x * 128 + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
        }
        __syncthreads();

        for (int ki = 0; ki < 2; ki++) {  // 2 WMMA sub-steps (16 K each)
            int k_off = ki * 16;
            #pragma unroll
            for (int mi = 0; mi < 4; mi++)
                wmma::load_matrix_sync(a[mi], &As[wy*64 + mi*16][k_off], 32);
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::load_matrix_sync(b[ni], &Bs[k_off][wx*64 + ni*16], 128);
            #pragma unroll
            for (int mi = 0; mi < 4; mi++)
            #pragma unroll
            for (int ni = 0; ni < 4; ni++)
                wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::store_matrix_sync(C + (warp_row+mi*16)*N + warp_col+ni*16,
                                c[mi*4+ni], N, wmma::mem_row_major);
}

// V7: Warp-parallel, no smem, no sync. Each warp loads directly from global.
__global__ void gemm_wmma_fp16_v7(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int warp_id = threadIdx.x / 32;  // 0..7, 4x2 layout
    int wy = warp_id / 2, wx = warp_id % 2;
    int warpM = blockIdx.y * 128 + wy * 32;
    int warpN = blockIdx.x * 128 + wx * 64;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[8];
    for (int i = 0; i < 8; i++) wmma::fill_fragment(c[i], 0.0f);

    for (int kb = 0; kb < K; kb += 16) {
        // Load directly from global memory — no smem, no sync!
        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
            wmma::load_matrix_sync(a[mi], &A[(warpM + mi*16) * K + kb], K);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &B[kb * N + (warpN + ni*16)], N);

        #pragma unroll
        for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);
    }
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::store_matrix_sync(C + (warpM + mi*16)*N + warpN + ni*16,
                                c[mi*4+ni], N, wmma::mem_row_major);
}

// V8: 4 warps (128 threads), 16 WMMA/warp, NO smem, NO sync
__global__ void gemm_wmma_fp16_v8(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int warp_id = threadIdx.x / 32;
    int wy = warp_id / 2, wx = warp_id % 2;
    int warpM = blockIdx.y * 128 + wy * 64;
    int warpN = blockIdx.x * 128 + wx * 64;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    for (int kb = 0; kb < K; kb += 16) {
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &A[(warpM + mi*16) * K + kb], K);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &B[kb * N + (warpN + ni*16)], N);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);
    }
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::store_matrix_sync(C + (warpM + mi*16)*N + warpN + ni*16,
                                c[mi*4+ni], N, wmma::mem_row_major);
}

int main() {
    int sizes[] = {1024, 2048, 4096};
    int n_sizes = 3;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));

    // Check TF32 WMMA availability
    printf("Tensor Core status:\n");
    printf("  FP16 WMMA:     COMPILES (using it now)\n");
    printf("  TF32 WMMA:     NOT available (fragment incomplete type)\n");
    printf("  cuBLAS TF32:   via PyTorch (5.6-8.4 TFLOPS)\n\n");

    printf("Our FP16 WMMA kernel vs PyTorch FP16\n\n");
    printf("%6s | %15s %8s %8s | %15s %8s %8s\n",
           "size", "kernel", "ms", "TFLOPS", "PyTorch FP16", "ms", "TFLOPS");
    printf("-------+---------------------------------------------------------\n");

    float pytorch_t[] = {0.191f, 1.278f, 9.319f};
    double pytorch_tf[] = {11.232, 13.440, 14.749};

    for (int si = 0; si < n_sizes; si++) {
        int M = sizes[si], N = M, K = M;
        size_t bA32 = (size_t)M * K * 4;
        size_t bA16 = (size_t)M * K * 2;
        size_t bB32 = (size_t)K * N * 4;
        size_t bB16 = (size_t)K * N * 2;
        size_t bC   = (size_t)M * N * 4;
        double flops = 2.0 * (double)M * N * K;

        float *hA = (float*)malloc(bA32);
        float *hB = (float*)malloc(bB32);
        float *hC = (float*)malloc(bC);
        for (size_t i = 0; i < (size_t)M*K; i++) {
            hA[i] = (float)rand() / RAND_MAX - 0.5f;
            hB[i] = (float)rand() / RAND_MAX - 0.5f;
        }

        float *dA32, *dB32, *dC;
        half *dA16, *dB16;
        CUDA_CHECK(cudaMalloc(&dA32, bA32));
        CUDA_CHECK(cudaMalloc(&dB32, bB32));
        CUDA_CHECK(cudaMalloc(&dA16, bA16));
        CUDA_CHECK(cudaMalloc(&dB16, bB16));
        CUDA_CHECK(cudaMalloc(&dC, bC));
        CUDA_CHECK(cudaMemcpy(dA32, hA, bA32, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB32, hB, bB32, cudaMemcpyHostToDevice));

        // Convert FP32 -> FP16 (once, not in inner loop!)
        int blocks32 = (M * K + 255) / 256;
        convert_f32tof16<<<blocks32, 256>>>(M * K, dA32, dA16);
        convert_f32tof16<<<(K * N + 255) / 256, 256>>>(K * N, dB32, dB16);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Benchmark V1 and V2
        void (*all_kernels[])(int,int,int,const half*,const half*,float*) = {
            gemm_wmma_fp16_v4, gemm_wmma_fp16_v8, gemm_wmma_fp16_v9};
        const char* all_names[] = {"V4_smem_sync","V8_nosmem","V9_cp_async"};
        int all_tile_n[] = {128, 128, 128};
        int all_tile_m[] = {128, 128, 128};
        int all_block[] = {128, 128, 128};
        int n_kernels = 3;

        for (int v = 0; v < n_kernels; v++) {
            void (*kernel)(int,int,int,const half*,const half*,float*) = all_kernels[v];
            const char* name = all_names[v];
            int tile_n = all_tile_n[v], tile_m = all_tile_m[v];
            dim3 grid((N + tile_n - 1) / tile_n, (M + tile_m - 1) / tile_m);
            dim3 block(all_block[v]);

            for (int i = 0; i < 5; i++)
                kernel<<<grid, block>>>(M, N, K, dA16, dB16, dC);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(start, 0));
            for (int i = 0; i < 20; i++)
                kernel<<<grid, block>>>(M, N, K, dA16, dB16, dC);
            CUDA_CHECK(cudaEventRecord(stop, 0));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float t_ours = ms(start, stop) / 20;
            double tf_ours = flops / (t_ours / 1000) / 1e12;

            double pct = pytorch_tf[si] > 0 ? tf_ours / pytorch_tf[si] * 100 : 0;
            printf("%6d | %15s %8.3f %8.3f %5.0f%% | %15s %8.3f %8.3f\n",
                   M, name, t_ours, tf_ours, pct,
                   (v==0)?"PyTorch":"", (v==0)?pytorch_t[si]:-1.0f, (v==0)?pytorch_tf[si]:-1.0);
        }

        // PyTorch FP16 reference (from Python benchmark)
        float pytorch_t[] = {0.191f, 1.278f, 9.319f};
        double pytorch_tf[] = {11.232, 13.440, 14.749};

        CUDA_CHECK(cudaFree(dA32)); CUDA_CHECK(cudaFree(dB32));
        CUDA_CHECK(cudaFree(dA16)); CUDA_CHECK(cudaFree(dB16));
        CUDA_CHECK(cudaFree(dC));
        free(hA); free(hB); free(hC);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}
