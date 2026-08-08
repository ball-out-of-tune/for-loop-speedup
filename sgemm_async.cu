/**
 * V6: V4 + cp.async double buffering (64x64 tile, 32KB×2 smem)
 * Uses PTX cp.async for async global→shared copy
 *
 * nvcc -o sgemm_async sgemm_async.cu -lcublas -arch=sm_86
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(c)  do { cudaError_t e = c; if (e != cudaSuccess) { \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); }} while(0)
#define CUBLAS_CHECK(c) do { cublasStatus_t s = c; if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr,"cuBLAS %s:%d: %d\n",__FILE__,__LINE__,s); exit(1); }} while(0)

float ms(cudaEvent_t s, cudaEvent_t e) { float t; cudaEventElapsedTime(&t,s,e); return t; }

#define T64 64

// ============================================================
// V6: 64x64 tile, double-buffered with cp.async
// ============================================================
__global__ void __launch_bounds__(256, 2)
sgemm_tiled_64_async(int M, int N, int K,
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
    const int T = T64;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row0 = blockIdx.y * T + ty * 4;
    int col0 = blockIdx.x * T + tx * 4;

    // Double buffer: 2 sets, each 64×64 floats for A and B
    // 4 × 64 × 64 × 4 = 64KB total (needs carveout)
    __shared__ float As[2][T][T];
    __shared__ float Bs[2][T][T];

    float c[4][4] = {{0}};

    // Prefetch tile 0 (buf 0)
    {
        const float* a_src = A + row0 * K;
        const float* b_src = B;
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int g_row = row0 + r;
            if (g_row < M) {
                // cp.async: copy 16 bytes (4 floats) from A to As[0]
                unsigned smem_addr = __cvta_generic_to_shared(&As[0][ty * 4 + r][tx * 4]);
                asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(&A[g_row * K + tx * 4]));
            }
        }
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int k_row = ty * 4 + r;
            if (k_row < K) {
                unsigned smem_addr = __cvta_generic_to_shared(&Bs[0][ty * 4 + r][tx * 4]);
                asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(&B[k_row * N + col0]));
            }
        }
    }
    asm("cp.async.commit_group;\n" ::);
    asm("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    int read_buf = 0;

    for (int k_block = T; k_block < K; k_block += T) {
        int write_buf = 1 - read_buf;

        // Prefetch next tile (buf write_buf) while computing current (buf read_buf)
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int g_row = row0 + r;
            if (g_row < M) {
                unsigned smem_addr = __cvta_generic_to_shared(&As[write_buf][ty * 4 + r][tx * 4]);
                asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(&A[g_row * K + (k_block + tx * 4)]));
            }
        }
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int k_row = k_block + ty * 4 + r;
            if (k_row < K) {
                unsigned smem_addr = __cvta_generic_to_shared(&Bs[write_buf][ty * 4 + r][tx * 4]);
                asm("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(&B[k_row * N + col0]));
            }
        }
        asm("cp.async.commit_group;\n" ::);

        // Compute from read_buf
        #pragma unroll
        for (int k = 0; k < T; k++) {
            float a0 = As[read_buf][ty * 4 + 0][k];
            float a1 = As[read_buf][ty * 4 + 1][k];
            float a2 = As[read_buf][ty * 4 + 2][k];
            float a3 = As[read_buf][ty * 4 + 3][k];
            float b0 = Bs[read_buf][k][tx * 4 + 0];
            float b1 = Bs[read_buf][k][tx * 4 + 1];
            float b2 = Bs[read_buf][k][tx * 4 + 2];
            float b3 = Bs[read_buf][k][tx * 4 + 3];

            c[0][0] += a0*b0; c[0][1] += a0*b1; c[0][2] += a0*b2; c[0][3] += a0*b3;
            c[1][0] += a1*b0; c[1][1] += a1*b1; c[1][2] += a1*b2; c[1][3] += a1*b3;
            c[2][0] += a2*b0; c[2][1] += a2*b1; c[2][2] += a2*b2; c[2][3] += a2*b3;
            c[3][0] += a3*b0; c[3][1] += a3*b1; c[3][2] += a3*b2; c[3][3] += a3*b3;
        }

        // Wait for prefetch to complete
        asm("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        read_buf = write_buf;
    }

    // Compute last tile (read_buf points to the last loaded tile)
    #pragma unroll
    for (int k = 0; k < T; k++) {
        float a0 = As[read_buf][ty * 4 + 0][k];
        float a1 = As[read_buf][ty * 4 + 1][k];
        float a2 = As[read_buf][ty * 4 + 2][k];
        float a3 = As[read_buf][ty * 4 + 3][k];
        float b0 = Bs[read_buf][k][tx * 4 + 0];
        float b1 = Bs[read_buf][k][tx * 4 + 1];
        float b2 = Bs[read_buf][k][tx * 4 + 2];
        float b3 = Bs[read_buf][k][tx * 4 + 3];

        c[0][0] += a0*b0; c[0][1] += a0*b1; c[0][2] += a0*b2; c[0][3] += a0*b3;
        c[1][0] += a1*b0; c[1][1] += a1*b1; c[1][2] += a1*b2; c[1][3] += a1*b3;
        c[2][0] += a2*b0; c[2][1] += a2*b1; c[2][2] += a2*b2; c[2][3] += a2*b3;
        c[3][0] += a3*b0; c[3][1] += a3*b1; c[3][2] += a3*b2; c[3][3] += a3*b3;
    }

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int row = row0 + r;
        if (row >= M) continue;
        #pragma unroll
        for (int cc = 0; cc < 4; cc++) {
            int col = col0 + cc;
            if (col < N) C[row * N + col] = c[r][cc];
        }
    }
}

// ============================================================
// Benchmark
// ============================================================
int main() {
    int sizes[] = {512, 1024, 2048, 4096};
    const int WARM = 5, RUN = 20;

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH);
    float alpha=1.0f,beta=0.0f;

    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start));CUDA_CHECK(cudaEventCreate(&stop));

    // Need 64KB shared memory per block for double buffering
    cudaFuncSetAttribute(sgemm_tiled_64_async, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);

    printf("SGEMM cp.async vs cuBLAS TF32\n");
    printf("%6s | %20s %8s %8s | %8s\n", "size", "kernel", "ms", "TFLOPS", "max_err");
    printf("-------+----------------------------------------------+----------\n");

    for (int si=0;si<4;si++) {
        int M=sizes[si],N=M,K=M;
        size_t bA=(size_t)M*K*4,bB=(size_t)K*N*4,bC=(size_t)M*N*4;
        double flops=2.0*(double)M*N*K;

        float *hA=(float*)malloc(bA),*hB=(float*)malloc(bB),*hC=(float*)malloc(bC),*hRef=(float*)malloc(bC);
        for(size_t i=0;i<(size_t)M*K;i++){hA[i]=(float)rand()/RAND_MAX-0.5f;hB[i]=(float)rand()/RAND_MAX-0.5f;}

        float *dA,*dB,*dC;
        CUDA_CHECK(cudaMalloc(&dA,bA));CUDA_CHECK(cudaMalloc(&dB,bB));CUDA_CHECK(cudaMalloc(&dC,bC));
        CUDA_CHECK(cudaMemcpy(dA,hA,bA,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB,hB,bB,cudaMemcpyHostToDevice));

        // cuBLAS TF32 reference
        for(int i=0;i<WARM;i++)
            CUBLAS_CHECK(cublasGemmEx(cublas,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,dB,CUDA_R_32F,N,dA,CUDA_R_32F,K,&beta,dC,CUDA_R_32F,N,CUBLAS_COMPUTE_32F_FAST_TF32,CUBLAS_GEMM_DEFAULT));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hRef,dC,bC,cudaMemcpyDeviceToHost));

        // V6 async
        dim3 grid((N+T64-1)/T64,(M+T64-1)/T64), block(16,16);
        for(int i=0;i<WARM;i++)sgemm_tiled_64_async<<<grid,block>>>(M,N,K,dA,dB,dC);
        CUDA_CHECK(cudaDeviceSynchronize());
        float best=1e9f;
        for(int r=0;r<RUN;r++){CUDA_CHECK(cudaEventRecord(start,0));
            sgemm_tiled_64_async<<<grid,block>>>(M,N,K,dA,dB,dC);
            CUDA_CHECK(cudaEventRecord(stop,0));CUDA_CHECK(cudaEventSynchronize(stop));
            float t=ms(start,stop);if(t<best)best=t;}
        float tflops=flops/(best/1000.0)/1e12;
        CUDA_CHECK(cudaMemcpy(hC,dC,bC,cudaMemcpyDeviceToHost));
        double max_err=0.0;
        for(size_t i=0;i<(size_t)M*N;i++){double e=fabs((double)hC[i]-(double)hRef[i]);if(e>max_err)max_err=e;}
        printf("%6d | %20s %8.3f %8.3f | %8.1e\n",M,"V6_cp_async",best,tflops,max_err);

        printf("-------+----------------------------------------------+----------\n");
        CUDA_CHECK(cudaFree(dA));CUDA_CHECK(cudaFree(dB));CUDA_CHECK(cudaFree(dC));
        free(hA);free(hB);free(hC);free(hRef);
    }
    CUBLAS_CHECK(cublasDestroy(cublas));
    return 0;
}
