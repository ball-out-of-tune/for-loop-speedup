/**
 * Two-kernel approach: FP32→FP16 conversion + pure FP16 WMMA GEMM
 * Eliminates per-tile conversion overhead
 *
 * nvcc -o sgemm_fp16 sgemm_fp16.cu -lcublas -arch=sm_86
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define CUDA_CHECK(c)  do { cudaError_t e=c; if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define CUBLAS_CHECK(c) do { cublasStatus_t s=c; if(s!=CUBLAS_STATUS_SUCCESS){ \
    fprintf(stderr,"cuBLAS %s:%d: %d\n",__FILE__,__LINE__,s);exit(1);}}while(0)
float ms(cudaEvent_t s,cudaEvent_t e){float t;cudaEventElapsedTime(&t,s,e);return t;}

// Step 1: Convert FP32 A(M×K) + B(K×N) → FP16
__global__ void f32tof16(int n, const float* __restrict__ src, half* __restrict__ dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// Step 2: FP16 WMMA GEMM — 4 warps per block, 32×32 tile
__global__ void sgemm_wmma_fp16only(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int warp_id = threadIdx.x / 32;
    int wy = warp_id / 2, wx = warp_id % 2;
    int warpM = blockIdx.y * 32 + wy * 16;
    int warpN = blockIdx.x * 32 + wx * 16;

    __shared__ half As[32][16];
    __shared__ half Bs[16][32];

    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator,16,16,16,float> c_frag;
    wmma::fill_fragment(c_frag,0.0f);

    for (int kb=0; kb<K; kb+=16) {
        // Cooperative load FP16 tiles (no conversion needed!)
        for (int i=threadIdx.x; i<32*16; i+=128) {
            int r=i/16, c=i%16;
            int gr=blockIdx.y*32+r, gc=kb+c;
            As[r][c]=(gr<M&&gc<K)?A[gr*K+gc]:__float2half(0.0f);
        }
        for (int i=threadIdx.x; i<16*32; i+=128) {
            int r=i/32, c=i%32;
            int gr=kb+r, gc=blockIdx.x*32+c;
            Bs[r][c]=(gr<K&&gc<N)?B[gr*N+gc]:__float2half(0.0f);
        }
        __syncthreads();

        wmma::load_matrix_sync(a_frag,&As[wy*16][0],16);
        wmma::load_matrix_sync(b_frag,&Bs[0][wx*16],32);
        wmma::mma_sync(c_frag,a_frag,b_frag,c_frag);
        __syncthreads();
    }
    wmma::store_matrix_sync(C+warpM*N+warpN,c_frag,N,wmma::mem_row_major);
}

int main() {
    int M=1024,N=1024,K=1024;
    printf("FP16 WMMA GEMM (separate conversion) | M=N=K=%d\n",M);
    printf("----------------------------------------\n");

    // Allocate FP32 host
    float *hA=(float*)malloc(M*K*4),*hB=(float*)malloc(K*N*4),*hC=(float*)malloc(M*N*4);
    for(int i=0;i<M*K;i++)hA[i]=(float)rand()/RAND_MAX-0.5f;
    for(int i=0;i<K*N;i++)hB[i]=(float)rand()/RAND_MAX-0.5f;

    // GPU: FP32 inputs, FP16 intermediates, FP32 output
    float *dA_f32,*dB_f32,*dC;
    half *dA_f16,*dB_f16;
    CUDA_CHECK(cudaMalloc(&dA_f32,M*K*4)); CUDA_CHECK(cudaMalloc(&dB_f32,K*N*4));
    CUDA_CHECK(cudaMalloc(&dA_f16,M*K*2)); CUDA_CHECK(cudaMalloc(&dB_f16,K*N*2));
    CUDA_CHECK(cudaMalloc(&dC,M*N*4));
    CUDA_CHECK(cudaMemcpy(dA_f32,hA,M*K*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB_f32,hB,K*N*4,cudaMemcpyHostToDevice));

    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start));CUDA_CHECK(cudaEventCreate(&stop));

    // Step 1: Convert (timed)
    int blocks=(M*K+255)/256;
    CUDA_CHECK(cudaEventRecord(start,0));
    f32tof16<<<blocks,256>>>(M*K,dA_f32,dA_f16);
    f32tof16<<<(K*N+255)/256,256>>>(K*N,dB_f32,dB_f16);
    CUDA_CHECK(cudaEventRecord(stop,0));CUDA_CHECK(cudaEventSynchronize(stop));
    float t_conv=ms(start,stop);
    printf("FP32->FP16 conversion: %.4f ms\n",t_conv);

    // Step 2: WMMA GEMM (timed)
    dim3 grid_ge((N+31)/32,(M+31)/32), block_ge(128);
    for(int i=0;i<5;i++)sgemm_wmma_fp16only<<<grid_ge,block_ge>>>(M,N,K,dA_f16,dB_f16,dC);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start,0));
    for(int i=0;i<20;i++)sgemm_wmma_fp16only<<<grid_ge,block_ge>>>(M,N,K,dA_f16,dB_f16,dC);
    CUDA_CHECK(cudaEventRecord(stop,0));CUDA_CHECK(cudaEventSynchronize(stop));
    float t_ge=ms(start,stop)/20;
    double flops=2.0*M*N*K;
    double tflops=flops/(t_ge/1000)/1e12;
    printf("FP16 WMMA GEMM:       %.4f ms → %.3f TFLOPS\n",t_ge,tflops);
    printf("Total (conv+GEMM):    %.4f ms → %.3f TFLOPS\n",t_conv+t_ge,flops/((t_conv+t_ge)/1000)/1e12);

    // cuBLAS TF32 reference
    cublasHandle_t cb;cublasCreate(&cb);cublasSetMathMode(cb,CUBLAS_TF32_TENSOR_OP_MATH);
    float a=1,b=0;
    for(int i=0;i<5;i++)cublasGemmEx(cb,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,dB_f32,CUDA_R_32F,N,dA_f32,CUDA_R_32F,K,&b,dC,CUDA_R_32F,N,CUBLAS_COMPUTE_32F_FAST_TF32,CUBLAS_GEMM_DEFAULT);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start,0));
    for(int i=0;i<20;i++)cublasGemmEx(cb,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&a,dB_f32,CUDA_R_32F,N,dA_f32,CUDA_R_32F,K,&b,dC,CUDA_R_32F,N,CUBLAS_COMPUTE_32F_FAST_TF32,CUBLAS_GEMM_DEFAULT);
    CUDA_CHECK(cudaEventRecord(stop,0));CUDA_CHECK(cudaEventSynchronize(stop));
    float t_cublas=ms(start,stop)/20;
    printf("\ncuBLAS TF32:          %.4f ms → %.3f TFLOPS\n",t_cublas,flops/(t_cublas/1000)/1e12);
    printf("Our FP16 / cuBLAS TF32: %.1f%%\n",tflops/(flops/(t_cublas/1000)/1e12)*100);

    CUDA_CHECK(cudaMemcpy(hC,dC,M*N*4,cudaMemcpyDeviceToHost));
    printf("Result sample: C[0,0]=%f\n",hC[0]);

    cublasDestroy(cb);cudaFree(dA_f32);cudaFree(dB_f32);cudaFree(dA_f16);cudaFree(dB_f16);cudaFree(dC);
    free(hA);free(hB);free(hC);
    return 0;
}
