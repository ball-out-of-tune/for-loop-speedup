#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
#define CHK(c) do{cudaError_t e=c;if(e!=cudaSuccess){fprintf(stderr,"E%d\n",e);exit(1);}}while(0)

__global__ void convert(int n, const float* s, half* d) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)d[i]=__float2half(s[i]);
}

// V10: 160x160 tile, 25 WMMA/warp, cp.async 16-byte
__global__ void gemm_v10(int M,int N,int K,const half* __restrict__ A,const half* __restrict__ B,float* __restrict__ C)
{
    int wid=threadIdx.x/32,wy=wid/2,wx=wid%2;
    int r0=blockIdx.y*160+wy*80,c0=blockIdx.x*160+wx*80;
    __shared__ half Ab[2][160][16],Bb[2][16][160];

    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a[5];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b[5];
    wmma::fragment<wmma::accumulator,16,16,16,float> c[25];
    for(int i=0;i<25;i++)wmma::fill_fragment(c[i],0.0f);

    // Prefetch buf 0
    for(int i=threadIdx.x;i<160*16/2;i+=128){
        int p=i*2,r=p/16,col=p%16,gr=blockIdx.y*160+r;
        if(gr<M&&col+1<K)*(short2*)&Ab[0][r][col]=*(const short2*)&A[gr*K+col];
    }
    for(int i=threadIdx.x;i<16*160/2;i+=128){
        int p=i*2,r=p/160,col=p%160,gc=blockIdx.x*160+col;
        if(r<K&&gc+1<N)*(short2*)&Bb[0][r][col]=*(const short2*)&B[r*N+gc];
    }
    __syncthreads();
    int rb=0;

    for(int kb=16;kb<K;kb+=16){
        int wb=1-rb;
        for(int i=threadIdx.x;i<160*16/8;i+=128){
            int p=i*8,r=p/16,col=p%16,gr=blockIdx.y*160+r,gc=kb+col;
            if(gr<M&&gc+7<K){unsigned sa=__cvta_generic_to_shared(&Ab[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&A[gr*K+gc]));}
        }
        for(int i=threadIdx.x;i<16*160/8;i+=128){
            int p=i*8,r=p/160,col=p%160,gr=kb+r,gc=blockIdx.x*160+col;
            if(gr<K&&gc+7<N){unsigned sb=__cvta_generic_to_shared(&Bb[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sb),"l"(&B[gr*N+gc]));}
        }
        asm volatile("cp.async.commit_group;\n"::);
        for(int mi=0;mi<5;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*80+mi*16][0],16);
        for(int ni=0;ni<5;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][0][wx*80+ni*16],160);
        for(int mi=0;mi<5;mi++)for(int ni=0;ni<5;ni++)wmma::mma_sync(c[mi*5+ni],a[mi],b[ni],c[mi*5+ni]);
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();rb=wb;
    }
    for(int mi=0;mi<5;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*80+mi*16][0],16);
    for(int ni=0;ni<5;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][0][wx*80+ni*16],160);
    for(int mi=0;mi<5;mi++)for(int ni=0;ni<5;ni++)wmma::mma_sync(c[mi*5+ni],a[mi],b[ni],c[mi*5+ni]);
    for(int mi=0;mi<5;mi++)for(int ni=0;ni<5;ni++)wmma::store_matrix_sync(C+(r0+mi*16)*N+c0+ni*16,c[mi*5+ni],N,wmma::mem_row_major);
}

int main(){
    int sz=4096;
    printf("V10 160x160 25MMA: M=N=K=%d\n",sz);
    float *hA=(float*)malloc(sz*sz*4),*hB=(float*)malloc(sz*sz*4);
    for(int i=0;i<sz*sz;i++){hA[i]=(float)rand()/RAND_MAX-0.5f;hB[i]=(float)rand()/RAND_MAX-0.5f;}
    float *dA32,*dB32,*dC;half *dA16,*dB16;
    CHK(cudaMalloc(&dA32,sz*sz*4));CHK(cudaMalloc(&dB32,sz*sz*4));
    CHK(cudaMalloc(&dA16,sz*sz*2));CHK(cudaMalloc(&dB16,sz*sz*2));
    CHK(cudaMalloc(&dC,sz*sz*4));
    CHK(cudaMemcpy(dA32,hA,sz*sz*4,cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dB32,hB,sz*sz*4,cudaMemcpyHostToDevice));
    convert<<<(sz*sz+255)/256,256>>>(sz*sz,dA32,dA16);
    convert<<<(sz*sz+255)/256,256>>>(sz*sz,dB32,dB16);
    CHK(cudaDeviceSynchronize());
    dim3 grid((sz+159)/160,(sz+159)/160),block(128);
    for(int i=0;i<5;i++)gemm_v10<<<grid,block>>>(sz,sz,sz,dA16,dB16,dC);
    CHK(cudaDeviceSynchronize());
    cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);
    cudaEventRecord(s);for(int i=0;i<20;i++)gemm_v10<<<grid,block>>>(sz,sz,sz,dA16,dB16,dC);
    cudaEventRecord(e);cudaEventSynchronize(e);
    float t;cudaEventElapsedTime(&t,s,e);
    printf("V10: %.3f ms -> %.3f TFLOPS (V9: 13.0T, PyTorch: 14.75T)\n",t/20,2.0*sz*sz*sz/(t/20000)/1e12);
    return 0;
}
