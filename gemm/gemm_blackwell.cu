/**
 * GEMM for Blackwell SM 120 (RTX 5090)
 *
 * Key constraints:
 *   - 170 SMs, 48 KB default smem, 99 KB opt-in smem
 *   - 65536 regs/SM, 1536 threads/SM
 *   - K=64 double-buffered fits in 99 KB (2*(128*64+64*128)*2 = 64 KB)
 *   - 5th gen Tensor Cores via WMMA API
 *
 * Compile: /usr/local/cuda-12.8/bin/nvcc -o gemm_bw gemm_blackwell.cu \
 *          -gencode=arch=compute_120,code=sm_120 -O3 -Xptxas -O3 --fmad=true -use_fast_math --restrict
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
using namespace nvcuda;
#define CHK(c) do{cudaError_t e=c;if(e!=cudaSuccess){fprintf(stderr,"E%d@%d\n",e,__LINE__);exit(1);}}while(0)

__global__ void convert(int n, const float* s, half* d) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)d[i]=__float2half(s[i]);
}

// ============================================================================
// V9 port: 128x128 tile, K=16 double-buffered, 4 warps
// Default smem (48 KB) — baseline
// ============================================================================
__global__ void gemm_bw_v9(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64;
    int c0 = blockIdx.x * 128 + wx * 64;
    __shared__ half Ab[2][128][16], Bb[2][16][128];
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator,16,16,16,float> c[16];
    for(int i=0;i<16;i++)wmma::fill_fragment(c[i],0.0f);
    for(int idx=threadIdx.x;idx<128*16/2;idx+=128){int pos=idx*2,r=pos/16,col=pos%16,gr=blockIdx.y*128+r;if(gr<M&&col+1<K)*(short2*)&Ab[0][r][col]=*(const short2*)&A[gr*K+col];}
    for(int idx=threadIdx.x;idx<16*128/2;idx+=128){int pos=idx*2,r=pos/128,col=pos%128,gc=blockIdx.x*128+col;if(r<K&&gc+1<N)*(short2*)&Bb[0][r][col]=*(const short2*)&B[r*N+gc];}
    __syncthreads();int rb=0;
    for(int kb=16;kb<K;kb+=16){int wb=1-rb;
        for(int c=threadIdx.x;c<128*16/8;c+=128){int p=c*8,r=p/16,col=p%16,gr=blockIdx.y*128+r,gc=kb+col;if(gr<M&&gc+7<K){unsigned sa=__cvta_generic_to_shared(&Ab[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&A[gr*K+gc]));}}
        for(int c=threadIdx.x;c<16*128/8;c+=128){int p=c*8,r=p/128,col=p%128,gr=kb+r,gc=blockIdx.x*128+col;if(gr<K&&gc+7<N){unsigned sa=__cvta_generic_to_shared(&Bb[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&B[gr*N+gc]));}}
        asm volatile("cp.async.commit_group;\n"::);
        for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*64+mi*16][0],16);
        for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][0][wx*64+ni*16],128);
        for(int mi=0;mi<4;mi++)for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
        asm volatile("cp.async.wait_group 0;\n"::);__syncthreads();rb=wb;}
    for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*64+mi*16][0],16);
    for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][0][wx*64+ni*16],128);
    for(int mi=0;mi<4;mi++)for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
    for(int mi=0;mi<4;mi++)for(int ni=0;ni<4;ni++)wmma::store_matrix_sync(C+(r0+mi*16)*N+c0+ni*16,c[mi*4+ni],N,wmma::mem_row_major);
}

// ============================================================================
// BW_V1: K=32 double-buffered (64 KB smem) — fewer syncs, needs opt-in smem
// 128×128 tile, 4 warps, 2× WMMA per sync (32 WMMA/sync)
// With 99 KB opt-in smem: 99/64 = 1.5 → 1 blk/SM (but 170 SMs gives plenty of parallelism)
// ============================================================================
__global__ void gemm_bw_v1_k32(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64;
    int c0 = blockIdx.x * 128 + wx * 64;
    __shared__ half Ab[2][128][32], Bb[2][32][128];
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator,16,16,16,float> c[16];
    for(int i=0;i<16;i++)wmma::fill_fragment(c[i],0.0f);
    // Prefetch buf 0
    for(int idx=threadIdx.x;idx<128*32/2;idx+=128){int pos=idx*2,r=pos/32,col=pos%32,gr=blockIdx.y*128+r;if(gr<M&&col+1<K)*(short2*)&Ab[0][r][col]=*(const short2*)&A[gr*K+col];}
    for(int idx=threadIdx.x;idx<32*128/2;idx+=128){int pos=idx*2,r=pos/128,col=pos%128,gc=blockIdx.x*128+col;if(r<K&&gc+1<N)*(short2*)&Bb[0][r][col]=*(const short2*)&B[r*N+gc];}
    __syncthreads();int rb=0;
    for(int kb=32;kb<K;kb+=32){int wb=1-rb;
        for(int c=threadIdx.x;c<128*32/8;c+=128){int p=c*8,r=p/32,col=p%32,gr=blockIdx.y*128+r,gc=kb+col;if(gr<M&&gc+7<K){unsigned sa=__cvta_generic_to_shared(&Ab[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&A[gr*K+gc]));}}
        for(int c=threadIdx.x;c<32*128/8;c+=128){int p=c*8,r=p/128,col=p%128,gr=kb+r,gc=blockIdx.x*128+col;if(gr<K&&gc+7<N){unsigned sa=__cvta_generic_to_shared(&Bb[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&B[gr*N+gc]));}}
        asm volatile("cp.async.commit_group;\n"::);
        // Step 0: K[0..15]
        for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*64+mi*16][0],32);
        for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][0][wx*64+ni*16],128);
        for(int mi=0;mi<4;mi++)for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
        // Step 1: K[16..31]
        for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*64+mi*16][16],32);
        for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][16][wx*64+ni*16],128);
        for(int mi=0;mi<4;mi++)for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
        asm volatile("cp.async.wait_group 0;\n"::);__syncthreads();rb=wb;}
    for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*64+mi*16][0],32);
    for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][0][wx*64+ni*16],128);
    for(int mi=0;mi<4;mi++)for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
    for(int mi=0;mi<4;mi++)for(int ni=0;ni<4;ni++)wmma::store_matrix_sync(C+(r0+mi*16)*N+c0+ni*16,c[mi*4+ni],N,wmma::mem_row_major);
}

// ============================================================================
// BW_V2: 256x128 tile, K=32 double-buffered, 8 warps
// Smem: 2*(256*32+32*128)*2 = 2*(8192+4096)*2 = 48 KB → fits default!
// 8 warps × 4×4 WMMA/warp = 128 WMMA per block per K-step (vs 64 in V9)
// More work per block → fewer blocks → less grid launch overhead
// ============================================================================
__global__ void gemm_bw_v2_256x128(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid = threadIdx.x / 32, wy = wid / 4, wx = wid % 4;
    int r0 = blockIdx.y * 256 + wy * 64;
    int c0 = blockIdx.x * 128 + wx * 32;

    __shared__ half Ab[2][256][32], Bb[2][32][128];
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b[2];
    wmma::fragment<wmma::accumulator,16,16,16,float> c[8];  // 4×2=8 WMMA/warp
    for(int i=0;i<8;i++)wmma::fill_fragment(c[i],0.0f);

    for(int idx=threadIdx.x;idx<256*32/2;idx+=256){int pos=idx*2,r=pos/32,col=pos%32,gr=blockIdx.y*256+r;if(gr<M&&col+1<K)*(short2*)&Ab[0][r][col]=*(const short2*)&A[gr*K+col];}
    for(int idx=threadIdx.x;idx<32*128/2;idx+=256){int pos=idx*2,r=pos/128,col=pos%128,gc=blockIdx.x*128+col;if(r<K&&gc+1<N)*(short2*)&Bb[0][r][col]=*(const short2*)&B[r*N+gc];}
    __syncthreads();int rb=0;
    for(int kb=32;kb<K;kb+=32){int wb=1-rb;
        for(int c=threadIdx.x;c<256*32/8;c+=256){int p=c*8,r=p/32,col=p%32,gr=blockIdx.y*256+r,gc=kb+col;if(gr<M&&gc+7<K){unsigned sa=__cvta_generic_to_shared(&Ab[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&A[gr*K+gc]));}}
        for(int c=threadIdx.x;c<32*128/8;c+=256){int p=c*8,r=p/128,col=p%128,gr=kb+r,gc=blockIdx.x*128+col;if(gr<K&&gc+7<N){unsigned sa=__cvta_generic_to_shared(&Bb[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&B[gr*N+gc]));}}
        asm volatile("cp.async.commit_group;\n"::);
        for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*64+mi*16][0],32);
        for(int ni=0;ni<2;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][0][wx*32+ni*16],128);
        for(int mi=0;mi<4;mi++)for(int ni=0;ni<2;ni++)wmma::mma_sync(c[mi*2+ni],a[mi],b[ni],c[mi*2+ni]);
        for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*64+mi*16][16],32);
        for(int ni=0;ni<2;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][16][wx*32+ni*16],128);
        for(int mi=0;mi<4;mi++)for(int ni=0;ni<2;ni++)wmma::mma_sync(c[mi*2+ni],a[mi],b[ni],c[mi*2+ni]);
        asm volatile("cp.async.wait_group 0;\n"::);__syncthreads();rb=wb;}
    for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*64+mi*16][0],32);
    for(int ni=0;ni<2;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][0][wx*32+ni*16],128);
    for(int mi=0;mi<4;mi++)for(int ni=0;ni<2;ni++)wmma::mma_sync(c[mi*2+ni],a[mi],b[ni],c[mi*2+ni]);
    for(int mi=0;mi<4;mi++)for(int ni=0;ni<2;ni++)wmma::store_matrix_sync(C+(r0+mi*16)*N+c0+ni*16,c[mi*2+ni],N,wmma::mem_row_major);
}

// ============================================================================
// BW_V3: 128x128 tile, K=64 double-buffered (max smem)
// Smem: 2*(128*64+64*128)*2 = 64 KB → needs opt-in 99 KB smem
// K=64 → 3 pipeline stages (triple-buffered K=16 equivalent)
// WMMA/sync ratio = 64 (4× more than V9!)
// Only 64 syncs for K=4096 (vs 256 for V9)
// ============================================================================
__global__ void gemm_bw_v3_k64(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64;
    int c0 = blockIdx.x * 128 + wx * 64;
    __shared__ half Ab[2][128][64], Bb[2][64][128];
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator,16,16,16,float> c[16];
    for(int i=0;i<16;i++)wmma::fill_fragment(c[i],0.0f);
    for(int idx=threadIdx.x;idx<128*64/2;idx+=128){int pos=idx*2,r=pos/64,col=pos%64,gr=blockIdx.y*128+r;if(gr<M&&col+1<K)*(short2*)&Ab[0][r][col]=*(const short2*)&A[gr*K+col];}
    for(int idx=threadIdx.x;idx<64*128/2;idx+=128){int pos=idx*2,r=pos/128,col=pos%128,gc=blockIdx.x*128+col;if(r<K&&gc+1<N)*(short2*)&Bb[0][r][col]=*(const short2*)&B[r*N+gc];}
    __syncthreads();int rb=0;
    for(int kb=64;kb<K;kb+=64){int wb=1-rb;
        for(int c=threadIdx.x;c<128*64/8;c+=128){int p=c*8,r=p/64,col=p%64,gr=blockIdx.y*128+r,gc=kb+col;if(gr<M&&gc+7<K){unsigned sa=__cvta_generic_to_shared(&Ab[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&A[gr*K+gc]));}}
        for(int c=threadIdx.x;c<64*128/8;c+=128){int p=c*8,r=p/128,col=p%128,gr=kb+r,gc=blockIdx.x*128+col;if(gr<K&&gc+7<N){unsigned sa=__cvta_generic_to_shared(&Bb[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&B[gr*N+gc]));}}
        asm volatile("cp.async.commit_group;\n"::);
        // 4 WMMA steps from 64 K-columns: K[0..15], K[16..31], K[32..47], K[48..63]
        for(int ks=0;ks<64;ks+=16){
            for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*64+mi*16][ks],64);
            for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][ks][wx*64+ni*16],128);
            for(int mi=0;mi<4;mi++)for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
        }
        asm volatile("cp.async.wait_group 0;\n"::);__syncthreads();rb=wb;}
    for(int ks=0;ks<64;ks+=16){
        for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*64+mi*16][ks],64);
        for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][ks][wx*64+ni*16],128);
        for(int mi=0;mi<4;mi++)for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
    }
    for(int mi=0;mi<4;mi++)for(int ni=0;ni<4;ni++)wmma::store_matrix_sync(C+(r0+mi*16)*N+c0+ni*16,c[mi*4+ni],N,wmma::mem_row_major);
}

// ============================================================================
// Benchmark
// ============================================================================
int main() {
    // PyTorch baselines for RTX 5090 (measured above)
    double pytorch_tf_5090[] = {153.2, 177.4, 216.3, 216.7}; // 1024, 2048, 4096, 8192
    int sizes[] = {1024, 2048, 4096, 8192};

    printf("GEMM — RTX 5090 (Blackwell SM 120)\n");
    printf("PyTorch ref: 1024=%.0fT, 2048=%.0fT, 4096=%.0fT, 8192=%.0fT\n\n",
           pytorch_tf_5090[0], pytorch_tf_5090[1], pytorch_tf_5090[2], pytorch_tf_5090[3]);

    for (int si = 0; si < 4; si++) {
        int M = sizes[si], N=M, K=M;
        double flops = 2.0 * (double)M * N * K;
        printf("=== %d^2 | PT=%.1fT ===\n", M, pytorch_tf_5090[si]);

        float *hA=(float*)malloc(M*K*4),*hB=(float*)malloc(K*N*4);
        for(size_t i=0;i<(size_t)M*K;i++)hA[i]=(float)rand()/RAND_MAX-0.5f;
        for(size_t i=0;i<(size_t)K*N;i++)hB[i]=(float)rand()/RAND_MAX-0.5f;
        float *dA32,*dB32,*dC;half *dA16,*dB16;
        CHK(cudaMalloc(&dA32,M*K*4));CHK(cudaMalloc(&dB32,K*N*4));
        CHK(cudaMalloc(&dA16,M*K*2));CHK(cudaMalloc(&dB16,K*N*2));
        CHK(cudaMalloc(&dC,M*N*4));
        CHK(cudaMemcpy(dA32,hA,M*K*4,cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dB32,hB,K*N*4,cudaMemcpyHostToDevice));
        convert<<<(M*K+255)/256,256>>>(M*K,dA32,dA16);
        convert<<<(K*N+255)/256,256>>>(K*N,dB32,dB16);
        CHK(cudaDeviceSynchronize());

        // Test V9 (default smem)
        {
            dim3 grid((N+127)/128,(M+127)/128),block(128);
            for(int i=0;i<30;i++)gemm_bw_v9<<<grid,block>>>(M,N,K,dA16,dB16,dC);
            CHK(cudaDeviceSynchronize());

            cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);
            cudaEventRecord(s);for(int i=0;i<50;i++)gemm_bw_v9<<<grid,block>>>(M,N,K,dA16,dB16,dC);
            cudaEventRecord(e);cudaEventSynchronize(e);
            float t;cudaEventElapsedTime(&t,s,e);
            double tf=flops/(t/50000.0)/1e12;
            printf("V9_K16:   %8.3fms %8.1fT %5.0f%% PT\n",t/50,tf,tf/pytorch_tf_5090[si]*100);
            cudaEventDestroy(s);cudaEventDestroy(e);
        }

        // Test V1 (K=32, needs opt-in smem)
        {
            dim3 grid((N+127)/128,(M+127)/128),block(128);
            cudaFuncSetAttribute(gemm_bw_v1_k32, cudaFuncAttributePreferredSharedMemoryCarveout,
                                 cudaSharedmemCarveoutMaxShared);
            for(int i=0;i<10;i++)gemm_bw_v1_k32<<<grid,block>>>(M,N,K,dA16,dB16,dC);
            CHK(cudaDeviceSynchronize());
            cudaError_t e=cudaGetLastError();
            if(e!=cudaSuccess){printf("V1_K32:   LAUNCH FAILED (E%d)\n",e);}
            else{
                cudaEvent_t s,st;cudaEventCreate(&s);cudaEventCreate(&st);
                cudaEventRecord(s);for(int i=0;i<50;i++)gemm_bw_v1_k32<<<grid,block>>>(M,N,K,dA16,dB16,dC);
                cudaEventRecord(st);cudaEventSynchronize(st);
                float t;cudaEventElapsedTime(&t,s,st);
                double tf=flops/(t/50000.0)/1e12;
                printf("V1_K32:   %8.3fms %8.1fT %5.0f%% PT\n",t/50,tf,tf/pytorch_tf_5090[si]*100);
                cudaEventDestroy(s);cudaEventDestroy(st);
            }
        }

        // Test V2 (256x128, K=32, 8 warps)
        {
            dim3 grid((N+127)/128,(M+255)/256),block(256);
            for(int i=0;i<10;i++)gemm_bw_v2_256x128<<<grid,block>>>(M,N,K,dA16,dB16,dC);
            CHK(cudaDeviceSynchronize());
            cudaError_t e=cudaGetLastError();
            if(e!=cudaSuccess){printf("V2_256:   LAUNCH FAILED (E%d)\n",e);}
            else{
                cudaEvent_t s,st;cudaEventCreate(&s);cudaEventCreate(&st);
                cudaEventRecord(s);for(int i=0;i<50;i++)gemm_bw_v2_256x128<<<grid,block>>>(M,N,K,dA16,dB16,dC);
                cudaEventRecord(st);cudaEventSynchronize(st);
                float t;cudaEventElapsedTime(&t,s,st);
                double tf=flops/(t/50000.0)/1e12;
                printf("V2_256:   %8.3fms %8.1fT %5.0f%% PT\n",t/50,tf,tf/pytorch_tf_5090[si]*100);
                cudaEventDestroy(s);cudaEventDestroy(st);
            }
        }

        // Test V3 (K=64, opt-in smem)
        {
            dim3 grid((N+127)/128,(M+127)/128),block(128);
            cudaFuncSetAttribute(gemm_bw_v3_k64, cudaFuncAttributePreferredSharedMemoryCarveout,
                                 cudaSharedmemCarveoutMaxShared);
            for(int i=0;i<10;i++)gemm_bw_v3_k64<<<grid,block>>>(M,N,K,dA16,dB16,dC);
            CHK(cudaDeviceSynchronize());
            cudaError_t e=cudaGetLastError();
            if(e!=cudaSuccess){printf("V3_K64:   LAUNCH FAILED (E%d)\n",e);}
            else{
                cudaEvent_t s,st;cudaEventCreate(&s);cudaEventCreate(&st);
                cudaEventRecord(s);for(int i=0;i<50;i++)gemm_bw_v3_k64<<<grid,block>>>(M,N,K,dA16,dB16,dC);
                cudaEventRecord(st);cudaEventSynchronize(st);
                float t;cudaEventElapsedTime(&t,s,st);
                double tf=flops/(t/50000.0)/1e12;
                printf("V3_K64:   %8.3fms %8.1fT %5.0f%% PT\n",t/50,tf,tf/pytorch_tf_5090[si]*100);
                cudaEventDestroy(s);cudaEventDestroy(st);
            }
        }

        printf("\n");
        CHK(cudaFree(dA32));CHK(cudaFree(dB32));CHK(cudaFree(dA16));CHK(cudaFree(dB16));CHK(cudaFree(dC));
        free(hA);free(hB);
    }
    printf("Done.\n");
    return 0;
}
