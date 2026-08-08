/**
 * GEMM with PTX mma.sync for Ampere (3050 Ti / SM 86)
 *
 * Key advantage over WMMA:
 *   - mma.sync.aligned.m16n8k16 uses 4 accumulator registers (vs 8 for WMMA m16n16k16)
 *   - More granular tiles: can cover 64×32 per warp with 16 MMA = 64 regs (vs 128 for WMMA)
 *   - This leaves room for software pipelining: preload A/B for next K-step while computing
 *
 * Approach: K=32 pipelined — load K[0..31], then alternate mma steps without syncs
 *   - Load A0,B0 (K[0..15]), compute 8 mma, load A1,B1 (K[16..31]), compute 8 mma
 *   - 16 MMA per block-sync instead of 16 (same) BUT we can pipeline A1 loading during computation
 *   - The key benefit: PTX gives explicit control over register allocation
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

/**
 * PTX MMA GEMM — 128×128 tile, 4 warps, K=16 step, double-buffered cp.async
 * Each warp: 64×64 output, using PTX mma.sync.aligned.m16n8k16
 * Warp layout: 4×8 = 32 MMA operations (same compute as WMMA m16n16k16 4×4=16)
 *
 * Register count per warp:
 *   - 32 accumulator fragments × 4 registers = 128 registers (SAME as WMMA!)
 *
 * So for the same output area, PTX mma uses the same # of registers.
 * The true advantage of PTX mma.sync is:
 *   1. Can pipeline K-loads manually: preload K-step N+1 while computing step N
 *   2. More precise instruction scheduling
 *   3. Can use m16n8k8 (smaller fragments) for register-limited scenarios
 *
 * For 4096² bottleneck (256 syncs):
 *   - Pipeline: load K[0..31] at once, then do 2 mma-sets without sync
 *   - With cp.async double-buffered: load K[32..63] while computing K[0..31]
 *   - Within K[0..31]: 2 sub-steps (0..15, 16..31), each sub-step is 32 mma
 *   - So 128 syncs for K=4096 (vs 256 for WMMA K=16)
 *
 * This is what V10 (K=32 WMMA) tried to do but couldn't because:
 *   - WMMA K=32 needed 32KB smem → 1 block/SM (occupancy loss)
 *   - PTX can do K=32 with 16KB smem? No — still need 32-K columns in smem
 *   - The real advantage: PTX can pipeline within a single K-step without syncs!
 */

// For now: PTX-based kernel with WMMA-like structure but finer control
__global__ void gemm_ptx_v1(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    // Same structure as V9 but with PTX-level instruction scheduling hints
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64;
    int c0 = blockIdx.x * 128 + wx * 64;

    __shared__ half A_buf[2][128][16];
    __shared__ half B_buf[2][16][128];

    // PTX mma fragments for m16n8k16:
    // A: 4 fp16 registers per thread per fragment
    // B: 2 fp16 registers per thread per fragment
    // C/D: 4 fp32 registers per thread per fragment
    // To cover 64×64: need 4 (M) × 8 (N) = 32 mma operations

    // For simplicity, declare WMMA-compatible fragments (compiler maps to PTX)
    // and use __syncwarp() for warp-level sync between pipeline stages
    wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator,16,16,16,float> c[16];
    for(int i=0;i<16;i++)wmma::fill_fragment(c[i],0.0f);

    // Prefetch
    for(int idx=threadIdx.x;idx<128*16/2;idx+=128){int pos=idx*2,r=pos/16,col=pos%16,gr=blockIdx.y*128+r;if(gr<M&&col+1<K)*(short2*)&A_buf[0][r][col]=*(const short2*)&A[gr*K+col];}
    for(int idx=threadIdx.x;idx<16*128/2;idx+=128){int pos=idx*2,r=pos/128,col=pos%128,gc=blockIdx.x*128+col;if(r<K&&gc+1<N)*(short2*)&B_buf[0][r][col]=*(const short2*)&B[r*N+gc];}
    __syncthreads();int rb=0;

    // K-loop: use __syncwarp() between pipeline stages instead of __syncthreads()
    // This only works if we don't share data between warps within a K-step
    // But ALL warps share the A_buf/B_buf — so we still need __syncthreads() for buffer swap
    // The optimization: use fewer, wider K-steps

    for(int kb=16;kb<K;kb+=16){int wb=1-rb;
        for(int c=threadIdx.x;c<128*16/8;c+=128){int p=c*8,r=p/16,col=p%16,gr=blockIdx.y*128+r,gc=kb+col;if(gr<M&&gc+7<K){unsigned sa=__cvta_generic_to_shared(&A_buf[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&A[gr*K+gc]));}}
        for(int c=threadIdx.x;c<16*128/8;c+=128){int p=c*8,r=p/128,col=p%128,gr=kb+r,gc=blockIdx.x*128+col;if(gr<K&&gc+7<N){unsigned sa=__cvta_generic_to_shared(&B_buf[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&B[gr*N+gc]));}}
        asm volatile("cp.async.commit_group;\n"::);
        for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&A_buf[rb][wy*64+mi*16][0],16);
        for(int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&B_buf[rb][0][wx*64+ni*16],128);
        // Interleave: for each B, do all A-MMAs
        for(int ni=0;ni<4;ni++){
            for(int mi=0;mi<4;mi++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
        }
        asm volatile("cp.async.wait_group 0;\n"::);__syncthreads();rb=wb;}
    for(int mi=0;mi<4;mi++)wmma::load_matrix_sync(a[mi],&A_buf[rb][wy*64+mi*16][0],16);
    for(int ni=0;ni<4;ni++){
        wmma::load_matrix_sync(b[ni],&B_buf[rb][0][wx*64+ni*16],128);
        for(int mi=0;mi<4;mi++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
    }
    for(int mi=0;mi<4;mi++)for(int ni=0;ni<4;ni++)wmma::store_matrix_sync(C+(r0+mi*16)*N+c0+ni*16,c[mi*4+ni],N,wmma::mem_row_major);
}

int main() {
    int sizes[]={1024,2048,4096};
    double pt_tf[]={11.257,13.437,14.751};
    printf("PTX MMA GEMM (Interleaved WMMA) — 3050 Ti\n\n");

    for(int si=0;si<3;si++){
        int M=sizes[si],N=M,K=M;
        double flops=2.0*(double)M*N*K;
        printf("=== %d^2 | PT=%.1fT ===\n",M,pt_tf[si]);

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

        dim3 grid((N+127)/128,(M+127)/128),block(128);
        for(int i=0;i<30;i++)gemm_ptx_v1<<<grid,block>>>(M,N,K,dA16,dB16,dC);
        CHK(cudaDeviceSynchronize());

        cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);
        cudaEventRecord(s);for(int i=0;i<20;i++)gemm_ptx_v1<<<grid,block>>>(M,N,K,dA16,dB16,dC);
        cudaEventRecord(e);cudaEventSynchronize(e);
        float t;cudaEventElapsedTime(&t,s,e);
        double tf=flops/(t/20000.0)/1e12;
        printf("  PTX_v1: %.3fms %.2fT %.0f%% PT\n",t/20,tf,tf/pt_tf[si]*100);

        cudaEventDestroy(s);cudaEventDestroy(e);
        CHK(cudaFree(dA32));CHK(cudaFree(dB32));CHK(cudaFree(dA16));CHK(cudaFree(dB16));CHK(cudaFree(dC));
        free(hA);free(hB);
    }
    return 0;
}
