/**
 * Baseline GEMM: WMMA + shared mem (sync loads, NO cp.async)
 * Clean, working baseline for sm_120 comparison
 */
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
using namespace nvcuda;

double now_ms(){struct timeval tv;gettimeofday(&tv,NULL);return tv.tv_sec*1000.0+tv.tv_usec/1000.0;}
#define CK(call) do{cudaError_t e=call;if(e!=cudaSuccess){fprintf(stderr,"ERR %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define CBLAS(call) do{cublasStatus_t e=call;if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"CUBLAS err %d\n",e);exit(1);}}while(0)

__global__ void gemm_baseline(const half* A,const half* B,float* C,int M,int N,int K){
    int wid=threadIdx.x/32,wy=wid/2,wx=wid%2;
    int r0=blockIdx.y*128+wy*64,c0=blockIdx.x*128+wx*64;
    __shared__ half As[128][16],Bs[16][128];

    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator,16,16,16,float> c[16];
    #pragma unroll
    for(int i=0;i<16;i++) wmma::fill_fragment(c[i],0.0f);

    for(int kb=0;kb<K;kb+=16){
        // Sync load A: 128x16 tile
        for(int idx=threadIdx.x;idx<128*16;idx+=128){
            int r=idx/16,c2=idx%16;
            int gr=blockIdx.y*128+r,gc=kb+c2;
            As[r][c2]=(gr<M&&gc<K)?A[gr*K+gc]:__float2half(0.0f);
        }
        // Sync load B: 16x128 tile
        for(int idx=threadIdx.x;idx<16*128;idx+=128){
            int r=idx/128,c2=idx%128;
            int gr=kb+r,gc=blockIdx.x*128+c2;
            Bs[r][c2]=(gr<K&&gc<N)?B[gr*N+gc]:__float2half(0.0f);
        }
        __syncthreads();

        // WMMA compute
        #pragma unroll
        for(int mi=0;mi<4;mi++) wmma::load_matrix_sync(a[mi],&As[wy*64+mi*16][0],16);
        #pragma unroll
        for(int ni=0;ni<4;ni++) wmma::load_matrix_sync(b[ni],&Bs[0][wx*64+ni*16],128);
        #pragma unroll
        for(int mi=0;mi<4;mi++)
        #pragma unroll
        for(int ni=0;ni<4;ni++) wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
        __syncthreads();
    }

    // Store
    #pragma unroll
    for(int mi=0;mi<4;mi++)
    #pragma unroll
    for(int ni=0;ni<4;ni++){
        int fr=r0+mi*16,fc=c0+ni*16;
        if(fr+16<=M&&fc+16<=N)
            wmma::store_matrix_sync(&C[fr*N+fc],c[mi*4+ni],N,wmma::mem_row_major);
    }
}

int main(){
    cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));
    printf("GPU: %s | CC %d.%d | SMs %d | MaxThr %d | ShMem %zuKB\n\n",
           p.name,p.major,p.minor,p.multiProcessorCount,
           p.maxThreadsPerBlock,p.sharedMemPerBlock/1024);
    cublasHandle_t h;CBLAS(cublasCreate(&h));

    struct{int M,N,K;const char* t;}T[]={
        {512,512,512,"sq-512"},{1024,1024,1024,"sq-1K"},
        {2048,2048,2048,"sq-2K"},{4096,4096,4096,"sq-4K"},
        {256,256,8192,"lK-8K"},{512,512,16384,"lK-16K"},{1024,1024,32768,"lK-32K"},
        {1,4096,4096,"d-1"},{1,8192,8192,"d-L"},{4,4096,4096,"d-4"},
        {8,4096,4096,"d-8"},{16,4096,4096,"d-16"},{32,4096,4096,"d-32"},
        {64,4096,4096,"d-64"},{128,4096,4096,"d-128"},
        {512,4096,4096,"p-512"},{1024,4096,4096,"p-1K"},{2048,4096,4096,"p-2K"},
        {4096,4096,64,"f-64"},{8192,8192,128,"f-128"},
        {1000,2000,500,"unaligned"},{100,100,100,"small"},
        {1,12288,12288,"gpt3-d"},{128,12288,12288,"gpt3-p"},
        {1,8192,28672,"llama-d"},{128,8192,28672,"llama-p"},
        {1,1024,3072,"qwen3-d"},{128,1024,3072,"qwen3-p"},
        {0,0,0,NULL}};
    int n=0;while(T[n].M)n++;

    printf("BASELINE: WMMA + shared mem, NO cp.async (sync loads only)\n");
    printf("%-20s %-10s %10s %10s %8s %10s\n","Shape","Category","Custom ms","cuBLAS ms","Speedup","CusTF");
    printf("%.20s %-10s %10s %10s %8s %10s\n","--------------------","----------","----------","----------","--------","----------");

    double totC=0,totB=0;
    for(int i=0;i<n;i++){
        int M=T[i].M,Nn=T[i].N,K=T[i].K;
        half *dA,*dB,*hA=(half*)malloc(M*K*2),*hB=(half*)malloc(K*Nn*2);
        float *dC,*dCb;
        for(int j=0;j<M*K;j++)hA[j]=__float2half(((float)rand()/RAND_MAX-0.5f)*0.1f);
        for(int j=0;j<K*Nn;j++)hB[j]=__float2half(((float)rand()/RAND_MAX-0.5f)*0.1f);
        CK(cudaMalloc(&dA,M*K*2));CK(cudaMemcpy(dA,hA,M*K*2,cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dB,K*Nn*2));CK(cudaMemcpy(dB,hB,K*Nn*2,cudaMemcpyHostToDevice));
        CK(cudaMalloc(&dC,M*Nn*4));CK(cudaMalloc(&dCb,M*Nn*4));
        free(hA);free(hB);

        dim3 blk(128),grd((Nn+127)/128,(M+127)/128);
        int wup=3,itr=20;
        long long fl=2LL*M*Nn*K;
        if(fl>1e8){wup=2;itr=10;}if(fl>5e8){wup=1;itr=5;}

        // Custom
        cudaGetLastError();
        for(int j=0;j<wup;j++)gemm_baseline<<<grd,blk>>>(dA,dB,dC,M,Nn,K);
        CK(cudaDeviceSynchronize());
        double t0=now_ms();
        for(int j=0;j<itr;j++)gemm_baseline<<<grd,blk>>>(dA,dB,dC,M,Nn,K);
        CK(cudaDeviceSynchronize());
        double ct=(now_ms()-t0)/itr;

        // cuBLAS (float output for fair comparison)
        float al=1,be=0;
        for(int j=0;j<wup;j++)
            CBLAS(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,Nn,M,K,&al,dB,CUDA_R_16F,Nn,dA,CUDA_R_16F,K,&be,dCb,CUDA_R_32F,Nn,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CK(cudaDeviceSynchronize());
        t0=now_ms();
        for(int j=0;j<itr;j++)
            CBLAS(cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,Nn,M,K,&al,dB,CUDA_R_16F,Nn,dA,CUDA_R_16F,K,&be,dCb,CUDA_R_32F,Nn,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CK(cudaDeviceSynchronize());
        double bt=(now_ms()-t0)/itr;

        double sp=bt/ct,tf=(2.0*M*Nn*K)/(ct/1000.0)/1e12;
        totC+=ct;totB+=bt;
        char sh[32];snprintf(sh,sizeof(sh),"%dx%dx%d",M,Nn,K);
        printf("%-20s %-10s %10.4f %10.4f %7.2fx %10.1f\n",sh,T[i].t,ct,bt,sp,tf);
        CK(cudaFree(dA));CK(cudaFree(dB));CK(cudaFree(dC));CK(cudaFree(dCb));
    }
    printf("%.20s %-10s %10s %10s %8s %10s\n","--------------------","----------","----------","----------","--------","----------");
    printf("%-20s %-10s %10.2f %10.2f %7.2fx\n","TOTAL","",totC,totB,totB/totC);
    cublasDestroy(h);
    printf("\nDone!\n");
    return 0;
}
