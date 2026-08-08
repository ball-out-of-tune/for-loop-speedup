/**
 * PTX mma.sync TF32 — direct Tensor Core programming
 * Last resort to match PyTorch TF32 (8+ TFLOPS)
 *
 * nvcc -o sgemm_ptx sgemm_ptx.cu -lcublas -arch=sm_86
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(c)  do { cudaError_t e=c; if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define CUBLAS_CHECK(c) do { cublasStatus_t s=c; if(s!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cuBLAS %s:%d: %d\n",__FILE__,__LINE__,s);exit(1);}}while(0)
float ms(cudaEvent_t s,cudaEvent_t e){float t;cudaEventElapsedTime(&t,s,e);return t;}

// PTX mma.sync.m16n8k16 for TF32: each warp does 16x8 output
// Cooperative: 4 warps per block → 32x16 output per block
__global__ void sgemm_ptx_mma(int M, int N, int K,
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
    // 128 threads = 4 warps, laid out 2×2 covering 32×16 output
    int warp_id = threadIdx.x / 32;
    int wy = warp_id / 2;  // 0 or 1
    int wx = warp_id % 2;  // 0 or 1

    int blk_m = blockIdx.y * 32;
    int blk_n = blockIdx.x * 16;
    int warp_m = blk_m + wy * 16;
    int warp_n = blk_n + wx * 8;

    // Shared memory: 32(M)×16(K) for A + 16(K)×16(N) for B
    __shared__ float As[32][16];
    __shared__ float Bs[16][16];

    // Accumulators: each thread holds 4 f32 registers for the 16×8 C fragment
    float c[4] = {0,0,0,0};
    int lane = threadIdx.x % 32;

    for (int k_block = 0; k_block < K; k_block += 16) {
        // Cooperative load: 128 threads load 32×16 As + 16×16 Bs
        for (int i = threadIdx.x; i < 32*16; i += 128) {
            int r = i / 16, c = i % 16;
            int g_row = blk_m + r, g_col = k_block + c;
            As[r][c] = (g_row<M && g_col<K) ? A[g_row*K + g_col] : 0.0f;
        }
        for (int i = threadIdx.x; i < 16*16; i += 128) {
            int r = i / 16, c = i % 16;
            int g_row = k_block + r, g_col = blk_n + c;
            Bs[r][c] = (g_row<K && g_col<N) ? B[g_row*N + g_col] : 0.0f;
        }
        __syncthreads();

        // Load warp's A(16x16) from shared memory using ldmatrix
        // ldmatrix loads 4 8x8 matrices per warp
        unsigned a0,a1,a2,a3,b0,b1,b2,b3;
        // A fragment: rows wy*16..wy*16+15 of As, cols 0..15
        // 4 ldmatrix.sync.aligned.m8n8x4 instructions
        asm volatile(
            "ldmatrix.sync.aligned.m8n8x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
            : "r"(__cvta_generic_to_shared(&As[wy*16][0]))
        );
        asm volatile(
            "ldmatrix.sync.aligned.m8n8x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
            : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3)
            : "r"(__cvta_generic_to_shared(&As[wy*16+8][0]))
        );

        // B fragment: for col_major B (transposed from Bs)
        // ldmatrix loads B^T from Bs as col_major
        unsigned ba0,ba1,bb0,bb1;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8x4.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(ba0), "=r"(ba1)
            : "r"(__cvta_generic_to_shared(&Bs[0][wx*8]))
        );
        asm volatile(
            "ldmatrix.sync.aligned.m8n8x4.shared.b16 {%0,%1}, [%2];\n"
            : "=r"(bb0), "=r"(bb1)
            : "r"(__cvta_generic_to_shared(&Bs[8][wx*8]))
        );

        // mma.sync: C += A × B
        // m16n8k16, row.col, f32.tf32.tf32.f32
        // Inputs: 8 regs for A, 4 regs for B, 4 regs for C
        // The register assignments follow a specific pattern defined by PTX ISA
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7,%8,%9,%10,%11}, {%12,%13,%14,%15}, {%16,%17,%18,%19};\n"
            : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
              "r"(b0), "r"(b1), "r"(b2), "r"(b3),
              "r"(ba0), "r"(ba1), "r"(bb0), "r"(bb1),
              "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3])
        );

        __syncthreads();
    }

    // Store: each warp writes its 16×8 result
    // Thread mapping uses the stmatrix or manual scatter
    // For simplicity: each thread writes its 4 elements
    // (exact mapping depends on the fragment layout)
    int row_map[4] = {0,0,8,8};  // approximate
    int col_map[4] = {0,1,0,1};
    for (int i=0;i<4;i++) {
        int r = warp_m + row_map[i] + (lane/4)*2;
        int c = warp_n + col_map[i] + (lane%4);
        if (r<M && c<N) C[r*N+c] = c[i];
    }
}

int main() {
    printf("PTX mma.sync SGEMM (TF32 Tensor Core)\n");
    printf("Attempting to match cuBLAS TF32 performance...\n");

    int M=1024,N=1024,K=1024;
    size_t bA=(size_t)M*K*4,bB=(size_t)K*N*4,bC=(size_t)M*N*4;
    double flops=2.0*(double)M*N*K;

    float *hA,*hB,*hC,*hRef;
    hA=(float*)malloc(bA);hB=(float*)malloc(bB);hC=(float*)malloc(bC);hRef=(float*)malloc(bC);
    for(size_t i=0;i<(size_t)M*K;i++){hA[i]=(float)rand()/RAND_MAX-0.5f;hB[i]=(float)rand()/RAND_MAX-0.5f;}

    float *dA,*dB,*dC;
    CUDA_CHECK(cudaMalloc(&dA,bA));CUDA_CHECK(cudaMalloc(&dB,bB));CUDA_CHECK(cudaMalloc(&dC,bC));
    CUDA_CHECK(cudaMemcpy(dA,hA,bA,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB,bB,cudaMemcpyHostToDevice));

    // cuBLAS TF32 reference
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    cublasSetMathMode(cublas,CUBLAS_TF32_TENSOR_OP_MATH);
    float alpha=1.0f,beta=0.0f;
    CUBLAS_CHECK(cublasGemmEx(cublas,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,dB,CUDA_R_32F,N,dA,CUDA_R_32F,K,&beta,dC,CUDA_R_32F,N,CUBLAS_COMPUTE_32F_FAST_TF32,CUBLAS_GEMM_DEFAULT));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hRef,dC,bC,cudaMemcpyDeviceToHost));

    // Our PTX kernel
    dim3 grid((N+15)/16,(M+15)/32),block(128);
    for(int i=0;i<5;i++)sgemm_ptx_mma<<<grid,block>>>(M,N,K,dA,dB,dC);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC,dC,bC,cudaMemcpyDeviceToHost));

    double max_err=0.0;
    for(size_t i=0;i<(size_t)M*N;i++){double e=fabs((double)hC[i]-(double)hRef[i]);if(e>max_err)max_err=e;}
    printf("Max error: %e (should be ~1e-5 for correct result)\n",max_err);

    if(max_err>0.01) printf("NOTE: PTX kernel has correctness bugs. Fragment layout needs fixing.\n");
    else printf("Kernel produces correct results!\n");

    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDA_CHECK(cudaFree(dA));CUDA_CHECK(cudaFree(dB));CUDA_CHECK(cudaFree(dC));
    free(hA);free(hB);free(hC);free(hRef);
    return 0;
}
