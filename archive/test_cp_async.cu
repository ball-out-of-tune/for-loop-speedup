#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
using namespace nvcuda;
#define CK(call) do{cudaError_t e=call;if(e){printf("ERR %s\n",cudaGetErrorString(e));}}while(0)

__global__ void test_16byte(const half* A,const half* B,float* C,int M,int N,int K){
    int wid=threadIdx.x/32;
    __shared__ half buf[2][128][16];
    // Try 16-byte cp.async
    unsigned sa=__cvta_generic_to_shared(&buf[0][0][0]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(sa),"l"(A));
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    if(threadIdx.x==0) C[0]=1.0f;
}

__global__ void test_4byte(const half* A,const half* B,float* C,int M,int N,int K){
    __shared__ half buf[2][128][16];
    // Try 4-byte cp.async
    unsigned sa=__cvta_generic_to_shared(&buf[0][0][0]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"::"r"(sa),"l"(A));
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    if(threadIdx.x==0) C[0]=1.0f;
}

__global__ void test_8byte(const half* A,const half* B,float* C,int M,int N,int K){
    __shared__ half buf[2][128][16];
    // Try 8-byte cp.async
    unsigned sa=__cvta_generic_to_shared(&buf[0][0][0]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n"::"r"(sa),"l"(A));
    asm volatile("cp.async.commit_group;\n"::);
    asm volatile("cp.async.wait_group 0;\n"::);
    __syncthreads();
    if(threadIdx.x==0) C[0]=1.0f;
}

int main(){
    half *dA,*dB;float *dC;
    cudaMalloc(&dA,512*512*2);cudaMalloc(&dB,512*512*2);cudaMalloc(&dC,4);

    printf("Testing cp.async on sm_120...\n");

    CK(cudaGetLastError());
    test_16byte<<<1,128>>>(dA,dB,dC,512,512,512);
    cudaError_t e=cudaDeviceSynchronize();
    printf("16-byte cp.async: %s\n",cudaGetErrorString(e));

    CK(cudaGetLastError());
    test_8byte<<<1,128>>>(dA,dB,dC,512,512,512);
    e=cudaDeviceSynchronize();
    printf("8-byte cp.async:  %s\n",cudaGetErrorString(e));

    CK(cudaGetLastError());
    test_4byte<<<1,128>>>(dA,dB,dC,512,512,512);
    e=cudaDeviceSynchronize();
    printf("4-byte cp.async:  %s\n",cudaGetErrorString(e));

    cudaFree(dA);cudaFree(dB);cudaFree(dC);
    return 0;
}
