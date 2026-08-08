/**
 * GEMM shared library for Python ctypes loading
 * Compile: nvcc -O3 -shared -Xcompiler -fPIC -o libgemm.so gemm_lib.cu -arch=sm_120
 */
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

extern "C" {

__global__ void gemm_kernel(const half* A,const half* B,float* C,int M,int N,int K){
    int wid=threadIdx.x/32,wy=wid/2,wx=wid%2;
    int r0=blockIdx.y*128+wy*64,c0=blockIdx.x*128+wx*64;
    __shared__ half As[2][128][16],Bs[2][16][128];
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator,16,16,16,float> c[16];
    #pragma unroll
    for(int i=0;i<16;i++)wmma::fill_fragment(c[i],0.0f);

    for(int idx=threadIdx.x;idx<128*16;idx+=128){
        int r=idx/16,c2=idx%16;
        int gr=blockIdx.y*128+r,gc=c2;
        As[0][r][c2]=(gr<M&&gc<K)?A[gr*K+gc]:__float2half(0.0f);
    }
    for(int idx=threadIdx.x;idx<16*128;idx+=128){
        int r=idx/128,c2=idx%128;
        int gr=r,gc=blockIdx.x*128+c2;
        Bs[0][r][c2]=(gr<K&&gc<N)?B[gr*N+gc]:__float2half(0.0f);
    }
    __syncthreads();
    int rb=0;

    bool use_cp=((unsigned long long)K&7)==0;
    for(int kb=16;kb<K;kb+=16){
        int wb=1-rb;
        for(int ch=threadIdx.x;ch<128*16/8;ch+=128){
            int pos=ch*8,r=pos/16,c8=pos%16;
            int gr=blockIdx.y*128+r,gc=kb+c8;
            if(gr<M&&gc+7<K){
                unsigned sa=__cvta_generic_to_shared(&As[wb][r][c8]);
                if(use_cp)asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(sa),"l"(&A[gr*K+gc]));
                else for(int j=0;j<8&&gc+j<K;j++)As[wb][r][c8+j]=A[gr*K+gc+j];
            }
        }
        for(int ch=threadIdx.x;ch<16*128/8;ch+=128){
            int pos=ch*8,r=pos/128,c8=pos%128;
            int gr=kb+r,gc=blockIdx.x*128+c8;
            if(gr<K&&gc+7<N){
                unsigned sa=__cvta_generic_to_shared(&Bs[wb][r][c8]);
                if(use_cp)asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(sa),"l"(&B[gr*N+gc]));
                else for(int j=0;j<8&&gc+j<N;j++)Bs[wb][r][c8+j]=B[gr*N+gc+j];
            }
        }
        if(use_cp)asm volatile("cp.async.commit_group;\n"::);
        else __syncthreads();

        #pragma unroll
        for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&As[rb][wy*64+mi*16][0],16);
        #pragma unroll
        for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bs[rb][0][wx*64+ni*16],128);
        #pragma unroll
        for(int mi=0;mi<4;mi++)
        #pragma unroll
        for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);

        if(use_cp)asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
        rb=wb;
    }

    #pragma unroll
    for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&As[rb][wy*64+mi*16][0],16);
    #pragma unroll
    for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bs[rb][0][wx*64+ni*16],128);
    #pragma unroll
    for(int mi=0;mi<4;mi++)
    #pragma unroll
    for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);

    #pragma unroll
    for(int mi=0;mi<4;mi++)
    #pragma unroll
    for(int ni=0;ni<4;ni++){
        int fr=r0+mi*16,fc=c0+ni*16;
        if(fr+16<=M&&fc+16<=N)wmma::store_matrix_sync(&C[fr*N+fc],c[mi*4+ni],N,wmma::mem_row_major);
    }
}

int gemm_launch(const half* A,const half* B,float* C,int M,int N,int K){
    dim3 blk(128),grd((N+127)/128,(M+127)/128);
    gemm_kernel<<<grd,blk>>>(A,B,C,M,N,K);
    return (int)cudaDeviceSynchronize();
}

} // extern "C"
