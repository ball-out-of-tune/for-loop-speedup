/**
 * cp.async debug on Blackwell sm_120
 * Test various cp.async patterns to find what works
 */
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
using namespace nvcuda;
#define CK(call) do{cudaError_t e=call;if(e){printf("ERR %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));}}while(0)

// Test 1: cp.async in a K-loop WITHOUT WMMA (just async copy)
__global__ void t1_cp_loop(const half* A, const half* B, float* C, int M, int N, int K){
    __shared__ half As[2][128][16];
    __shared__ half Bs[2][16][128];
    int rb=0;
    // Prefetch
    for(int idx=threadIdx.x;idx<128*16;idx+=128){
        int r=idx/16,c2=idx%16;
        As[0][r][c2]=A[(blockIdx.y*128+r)*K+c2];
    }
    __syncthreads();
    for(int kb=16;kb<K;kb+=16){
        int wb=1-rb;
        for(int ch=threadIdx.x;ch<128*16/8;ch+=128){
            int pos=ch*8,r=pos/16,c8=pos%16;
            unsigned sa=__cvta_generic_to_shared(&As[wb][r][c8]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(sa),"l"(&A[(blockIdx.y*128+r)*K+kb+c8]));
        }
        for(int ch=threadIdx.x;ch<16*128/8;ch+=128){
            int pos=ch*8,r=pos/128,c8=pos%128;
            unsigned sa=__cvta_generic_to_shared(&Bs[wb][r][c8]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(sa),"l"(&B[(kb+r)*N+blockIdx.x*128+c8]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
        rb=wb;
    }
    if(threadIdx.x==0)C[0]=1.0f;
}

// Test 2: cp.async + WMMA but SEPARATED (no overlap)
__global__ void t2_cp_wmma_separate(const half* A,const half* B,float* C,int M,int N,int K){
    int wid=threadIdx.x/32,wy=wid/2,wx=wid%2;
    int r0=blockIdx.y*128+wy*64,c0=blockIdx.x*128+wx*64;
    __shared__ half As[2][128][16],Bs[2][16][128];
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator,16,16,16,float> c[16];
    for(int i=0;i<16;i++)wmma::fill_fragment(c[i],0.0f);

    int rb=0;
    // Prefetch with sync
    for(int idx=threadIdx.x;idx<128*16;idx+=128){
        int r=idx/16,c2=idx%16;
        As[0][r][c2]=A[(blockIdx.y*128+r)*K+c2];
    }
    for(int idx=threadIdx.x;idx<16*128;idx+=128){
        int r=idx/128,c2=idx%128;
        Bs[0][r][c2]=B[r*N+blockIdx.x*128+c2];
    }
    __syncthreads();

    for(int kb=16;kb<K;kb+=16){
        int wb=1-rb;
        // Step 1: cp.async load (no WMMA during this)
        for(int ch=threadIdx.x;ch<128*16/8;ch+=128){
            int pos=ch*8,r=pos/16,c8=pos%16;
            unsigned sa=__cvta_generic_to_shared(&As[wb][r][c8]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(sa),"l"(&A[(blockIdx.y*128+r)*K+kb+c8]));
        }
        for(int ch=threadIdx.x;ch<16*128/8;ch+=128){
            int pos=ch*8,r=pos/128,c8=pos%128;
            unsigned sa=__cvta_generic_to_shared(&Bs[wb][r][c8]);
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(sa),"l"(&B[(kb+r)*N+blockIdx.x*128+c8]));
        }
        asm volatile("cp.async.commit_group;\n"::);
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
        rb=wb;

        // Step 2: WMMA compute (after cp.async done, no overlap)
        for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&As[rb][wy*64+mi*16][0],16);
        for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bs[rb][0][wx*64+ni*16],128);
        for(int mi=0;mi<4;mi++)
        for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
        __syncthreads();
    }
    // Last tile
    for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&As[rb][wy*64+mi*16][0],16);
    for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bs[rb][0][wx*64+ni*16],128);
    for(int mi=0;mi<4;mi++)
    for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
    for(int mi=0;mi<4;mi++)
    for(int ni=0;ni<4;ni++){
        int fr=r0+mi*16,fc=c0+ni*16;
        if(fr+16<=M&&fc+16<=N)wmma::store_matrix_sync(&C[fr*N+fc],c[mi*4+ni],N,wmma::mem_row_major);
    }
}

// Test 3: cp.async + WMMA OVERLAPPED (the original design but with borders always in-bounds)
__global__ void t3_cp_wmma_overlap(const half* A,const half* B,float* C,int M,int N,int K){
    int wid=threadIdx.x/32,wy=wid/2,wx=wid%2;
    int r0=blockIdx.y*128+wy*64,c0=blockIdx.x*128+wx*64;
    __shared__ half As[2][128][16],Bs[2][16][128];
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator,16,16,16,float> c[16];
    for(int i=0;i<16;i++)wmma::fill_fragment(c[i],0.0f);

    // Sync prefetch (avoid cp.async alignment issues for first tile)
    for(int idx=threadIdx.x;idx<128*16;idx+=128){
        int r=idx/16,c2=idx%16;
        if(blockIdx.y*128+r<M && c2<K) As[0][r][c2]=A[(blockIdx.y*128+r)*K+c2];
    }
    for(int idx=threadIdx.x;idx<16*128;idx+=128){
        int r=idx/128,c2=idx%128;
        if(r<K && blockIdx.x*128+c2<N) Bs[0][r][c2]=B[r*N+blockIdx.x*128+c2];
    }
    __syncthreads();
    int rb=0;

    for(int kb=16;kb<K;kb+=16){
        int wb=1-rb;
        // cp.async: ONLY inside valid range
        for(int ch=threadIdx.x;ch<128*16/8;ch+=128){
            int pos=ch*8,r=pos/16,c8=pos%16;
            if(blockIdx.y*128+r<M && kb+c8+7<K){
                unsigned sa=__cvta_generic_to_shared(&As[wb][r][c8]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(sa),"l"(&A[(blockIdx.y*128+r)*K+kb+c8]));
            }
        }
        for(int ch=threadIdx.x;ch<16*128/8;ch+=128){
            int pos=ch*8,r=pos/128,c8=pos%128;
            if(kb+r<K && blockIdx.x*128+c8+7<N){
                unsigned sa=__cvta_generic_to_shared(&Bs[wb][r][c8]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"::"r"(sa),"l"(&B[(kb+r)*N+blockIdx.x*128+c8]));
            }
        }
        asm volatile("cp.async.commit_group;\n"::);

        // WMMA
        for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&As[rb][wy*64+mi*16][0],16);
        for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bs[rb][0][wx*64+ni*16],128);
        for(int mi=0;mi<4;mi++)
        for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);

        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
        rb=wb;
    }

    for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&As[rb][wy*64+mi*16][0],16);
    for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bs[rb][0][wx*64+ni*16],128);
    for(int mi=0;mi<4;mi++)
    for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
    for(int mi=0;mi<4;mi++)
    for(int ni=0;ni<4;ni++){
        int fr=r0+mi*16,fc=c0+ni*16;
        if(fr+16<=M&&fc+16<=N)wmma::store_matrix_sync(&C[fr*N+fc],c[mi*4+ni],N,wmma::mem_row_major);
    }
}

int main(){
    cudaDeviceProp p;CK(cudaGetDeviceProperties(&p,0));
    printf("GPU: %s (CC %d.%d)\n\n",p.name,p.major,p.minor);

    int M=512,N=512,K=512;
    half *dA,*dB;float *dC;
    CK(cudaMalloc(&dA,M*K*2));CK(cudaMalloc(&dB,K*N*2));CK(cudaMalloc(&dC,M*N*4));

    dim3 blk(128),grd(N/128,M/128);

    printf("Test 1: cp.async loop (no WMMA)... ");
    CK(cudaGetLastError());
    t1_cp_loop<<<grd,blk>>>(dA,dB,dC,M,N,K);
    printf("%s\n",cudaGetErrorString(cudaDeviceSynchronize()));

    printf("Test 2: cp.async + WMMA separated... ");
    CK(cudaGetLastError());
    t2_cp_wmma_separate<<<grd,blk>>>(dA,dB,dC,M,N,K);
    printf("%s\n",cudaGetErrorString(cudaDeviceSynchronize()));

    printf("Test 3: cp.async + WMMA overlapped... ");
    CK(cudaGetLastError());
    t3_cp_wmma_overlap<<<grd,blk>>>(dA,dB,dC,M,N,K);
    printf("%s\n",cudaGetErrorString(cudaDeviceSynchronize()));

    CK(cudaFree(dA));CK(cudaFree(dB));CK(cudaFree(dC));
    printf("\nDone!\n");
    return 0;
}
