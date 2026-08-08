/**
 * SGEMM with WMMA (Tensor Core) — TF32 precision
 * One warp per 16x16 output tile, no shared memory
 *
 * nvcc -o sgemm_wmma sgemm_wmma.cu -lcublas -arch=sm_86
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32

#define CUDA_CHECK(c)  do { cudaError_t e = c; if (e != cudaSuccess) { \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); }} while(0)
#define CUBLAS_CHECK(c) do { cublasStatus_t s = c; if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr,"cuBLAS %s:%d: %d\n",__FILE__,__LINE__,s); exit(1); }} while(0)

float ms(cudaEvent_t s, cudaEvent_t e) { float t; cudaEventElapsedTime(&t,s,e); return t; }

// ============================================================
// V1: WMMA FP16 — 1 warp per block, shared memory for FP32→FP16
// ============================================================
__global__ void sgemm_wmma_fp16(int M, int N, int K,
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
    int warpM = blockIdx.y * WMMA_M;
    int warpN = blockIdx.x * WMMA_N;
    int lane = threadIdx.x;

    // Shared memory for FP16 conversion
    __shared__ half As[WMMA_M][WMMA_K];
    __shared__ half Bs[WMMA_K][WMMA_N];

    // No need for FP16 — use float directly (Ampere+ uses TF32 on tensor cores)
    // Shared memory for cooperative loading (float, no conversion needed)
    __shared__ float As[WMMA_M][WMMA_K];
    __shared__ float Bs[WMMA_K][WMMA_N];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, float, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, float, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k_block = 0; k_block < K; k_block += WMMA_K) {
        // Load A tile: cooperative load from FP32 global memory
        #pragma unroll
        for (int i = lane; i < WMMA_M * WMMA_K; i += WARP_SIZE) {
            int r = i / WMMA_K, c = i % WMMA_K;
            int g_row = warpM + r, g_col = k_block + c;
            As[r][c] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }
        // Load B tile
        #pragma unroll
        for (int i = lane; i < WMMA_K * WMMA_N; i += WARP_SIZE) {
            int r = i / WMMA_N, c = i % WMMA_N;
            int g_row = k_block + r, g_col = warpN + c;
            Bs[r][c] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag, &As[0][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &Bs[0][0], WMMA_N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    float* c_ptr = C + warpM * N + warpN;
    wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
}

// ============================================================
// V2: WMMA TF32 — 4 warps per block (128 threads), 32x32 tile
//     Cooperative loading via shared memory
// ============================================================
__global__ void sgemm_wmma_4warp(int M, int N, int K,
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
    // Block = 4 warps, 2×2 layout covering 32×32 output
    // Warp (wy, wx) where wy = threadIdx.x / 16, wx = threadIdx.x % 16...
    // Actually: 128 threads = 4 warps. Layout: warps 0,1 on first row, 2,3 on second
    int warp_id = threadIdx.x / WARP_SIZE;     // 0..3
    int wy = warp_id / 2;                       // 0 or 1
    int wx = warp_id % 2;                       // 0 or 1

    int warpM = blockIdx.y * (WMMA_M * 2) + wy * WMMA_M;
    int warpN = blockIdx.x * (WMMA_N * 2) + wx * WMMA_N;

    __shared__ float As[WMMA_M * 2][WMMA_K];
    __shared__ float Bs[WMMA_K][WMMA_N * 2];

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, float, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, float, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int k_block = 0; k_block < K; k_block += WMMA_K) {
        // Cooperative load of 32×16 As tile and 16×32 Bs tile
        int lane = threadIdx.x % WARP_SIZE;

        // Load As: 128 threads load 32×16 = 512 floats → each loads 4
        for (int i = threadIdx.x; i < WMMA_M * 2 * WMMA_K; i += blockDim.x) {
            int row = i / WMMA_K;
            int col = i % WMMA_K;
            int a_row = blockIdx.y * WMMA_M * 2 + row;
            int a_col = k_block + col;
            As[row][col] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }
        // Load Bs: 128 threads load 16×32 = 512 floats → each loads 4
        for (int i = threadIdx.x; i < WMMA_K * WMMA_N * 2; i += blockDim.x) {
            int row = i / (WMMA_N * 2);
            int col = i % (WMMA_N * 2);
            int b_row = k_block + row;
            int b_col = blockIdx.x * WMMA_N * 2 + col;
            Bs[row][col] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }
        __syncthreads();

        // Each warp loads its 16×16 sub-tile from shared memory
        wmma::load_matrix_sync(a_frag, &As[wy * WMMA_M][0], WMMA_K);
        wmma::load_matrix_sync(b_frag, &Bs[0][wx * WMMA_N], WMMA_N * 2);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }

    float* c_ptr = C + warpM * N + warpN;
    wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
}

// ============================================================
// Benchmark
// ============================================================
int main() {
    int sizes[] = {512, 1024, 2048, 4096};
    int n_sizes = 4;
    const int WARM = 5, RUN = 20;

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    // Enable TF32 for cuBLAS too (fair comparison)
    cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH);

    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    typedef void (*kfn_t)(int,int,int,const float*,const float*,float*);
    struct { kfn_t fn; const char* name; dim3 block; int tile; } kernels[] = {
        {sgemm_wmma_fp16,    "V1_wmma_float",  dim3(32,1,1), 16},
        {sgemm_wmma_4warp,   "V2_wmma_4warp",  dim3(128,1,1), 32},
    };

    printf("SGEMM WMMA (Tensor Core TF32) vs cuBLAS TF32\n");
    printf("==============================================\n\n");
    printf("%6s | %20s %8s %8s | %20s %8s %8s | %8s\n",
           "size", "kernel", "ms", "TFLOPS", "cuBLAS_TF32", "ms", "TFLOPS", "max_err");
    printf("-------+----------------------------------------------+----------------------------------------+----------\n");

    for (int si = 0; si < n_sizes; si++) {
        int M = sizes[si], N = M, K = M;
        size_t bA = (size_t)M * K * 4, bB = (size_t)K * N * 4, bC = (size_t)M * N * 4;
        double flops = 2.0 * (double)M * N * K;

        float *hA = (float*)malloc(bA), *hB = (float*)malloc(bB);
        float *hC = (float*)malloc(bC), *hRef = (float*)malloc(bC);
        for (size_t i=0;i<(size_t)M*K;i++){hA[i]=(float)rand()/RAND_MAX-0.5f;hB[i]=(float)rand()/RAND_MAX-0.5f;}

        float *dA,*dB,*dC;
        CUDA_CHECK(cudaMalloc(&dA,bA));CUDA_CHECK(cudaMalloc(&dB,bB));CUDA_CHECK(cudaMalloc(&dC,bC));
        CUDA_CHECK(cudaMemcpy(dA,hA,bA,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB,hB,bB,cudaMemcpyHostToDevice));

        // cuBLAS TF32 reference
        float t_cublas, tf_cublas;
        {
            for (int i=0;i<WARM;i++)
                CUBLAS_CHECK(cublasGemmEx(cublas,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,
                    &alpha,dB,CUDA_R_32F,N,dA,CUDA_R_32F,K,&beta,dC,CUDA_R_32F,N,
                    CUBLAS_COMPUTE_32F_FAST_TF32,CUBLAS_GEMM_DEFAULT));
            CUDA_CHECK(cudaDeviceSynchronize());
            float best=1e9f;
            for(int r=0;r<RUN;r++){CUDA_CHECK(cudaEventRecord(start,0));
                CUBLAS_CHECK(cublasGemmEx(cublas,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,
                    &alpha,dB,CUDA_R_32F,N,dA,CUDA_R_32F,K,&beta,dC,CUDA_R_32F,N,
                    CUBLAS_COMPUTE_32F_FAST_TF32,CUBLAS_GEMM_DEFAULT));
                CUDA_CHECK(cudaEventRecord(stop,0));CUDA_CHECK(cudaEventSynchronize(stop));
                float t=ms(start,stop);if(t<best)best=t;}
            t_cublas=best;tf_cublas=flops/(best/1000.0)/1e12;
            CUDA_CHECK(cudaMemcpy(hRef,dC,bC,cudaMemcpyDeviceToHost));
        }

        for (int k=0;k<2;k++) {
            int tsz = kernels[k].tile;
            dim3 grid((N+tsz-1)/tsz, (M+tsz-1)/tsz);

            for(int i=0;i<WARM;i++)kernels[k].fn<<<grid,kernels[k].block>>>(M,N,K,dA,dB,dC);
            CUDA_CHECK(cudaDeviceSynchronize());
            float best=1e9f;
            for(int r=0;r<RUN;r++){CUDA_CHECK(cudaEventRecord(start,0));
                kernels[k].fn<<<grid,kernels[k].block>>>(M,N,K,dA,dB,dC);
                CUDA_CHECK(cudaEventRecord(stop,0));CUDA_CHECK(cudaEventSynchronize(stop));
                float t=ms(start,stop);if(t<best)best=t;}

            float tflops=flops/(best/1000.0)/1e12;
            CUDA_CHECK(cudaMemcpy(hC,dC,bC,cudaMemcpyDeviceToHost));

            double max_err=0.0;
            for(size_t i=0;i<(size_t)M*N;i++){double e=fabs((double)hC[i]-(double)hRef[i]);if(e>max_err)max_err=e;}

            printf("%6d | %20s %8.3f %8.3f | %20s %8.3f %8.3f | %8.1e\n",
                   M,kernels[k].name,best,tflops,"cuBLAS_TF32",t_cublas,tf_cublas,max_err);
        }
        printf("-------+----------------------------------------------+----------------------------------------+----------\n");

        CUDA_CHECK(cudaFree(dA));CUDA_CHECK(cudaFree(dB));CUDA_CHECK(cudaFree(dC));
        free(hA);free(hB);free(hC);free(hRef);
    }

    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDA_CHECK(cudaEventDestroy(start));CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}
