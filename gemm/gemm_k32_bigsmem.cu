/**
 * V16: K=32 double-buffered WITH increased shared memory carout
 * GA107: 128KB L1/smem unified, default=48KB smem
 * Set to 100KB smem → K=32 dbl buf (32KB) can have 3 blk/SM (100/32=3)!
 * This gives: 2× sync reduction + 3× occupancy = both worlds
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

// V16: K=32 double-buffered, 128x128 tile, 4 warps
// smem = 2*(128*32 + 32*128)*2 = 32KB
// With 100KB smem carout: 3 blocks/SM (vs 1 with 48KB)
__global__ void gemm_v16_k32_bigsmem(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64;
    int c0 = blockIdx.x * 128 + wx * 64;

    __shared__ half A_buf[2][128][32];
    __shared__ half B_buf[2][32][128];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    // Prefetch buf 0
    for (int idx = threadIdx.x; idx < 128*32/2; idx += 128) {
        int pos = idx * 2, r = pos / 32, col = pos % 32;
        int gr = blockIdx.y*128 + r;
        if (gr < M && col + 1 < K)
            *(short2*)&A_buf[0][r][col] = *(const short2*)&A[gr*K + col];
    }
    for (int idx = threadIdx.x; idx < 32*128/2; idx += 128) {
        int pos = idx * 2, r = pos / 128, col = pos % 128;
        int gc = blockIdx.x*128 + col;
        if (r < K && gc + 1 < N)
            *(short2*)&B_buf[0][r][col] = *(const short2*)&B[r*N + gc];
    }
    __syncthreads();
    int read_buf = 0;

    for (int kb = 32; kb < K; kb += 32) {
        int write_buf = 1 - read_buf;
        for (int chunk = threadIdx.x; chunk < 128*32/8; chunk += 128) {
            int pos = chunk * 8, r = pos / 32, c = pos % 32;
            int gr = blockIdx.y*128 + r, gc = kb + c;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c]);
                asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&A[gr*K+gc]));
            }
        }
        for (int chunk = threadIdx.x; chunk < 32*128/8; chunk += 128) {
            int pos = chunk * 8, r = pos / 128, c = pos % 128;
            int gr = kb + r, gc = blockIdx.x*128 + c;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c]);
                asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&B[gr*N+gc]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        // Step 0: K[0..15]
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy*64+mi*16][0], 32);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx*64+ni*16], 128);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);
        // Step 1: K[16..31]
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy*64+mi*16][16], 32);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &B_buf[read_buf][16][wx*64+ni*16], 128);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        read_buf = write_buf;
    }
    // Last tile (already loaded by prefetch)
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy*64+mi*16][0], 32);
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx*64+ni*16], 128);
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::store_matrix_sync(C+(r0+mi*16)*N+c0+ni*16, c[mi*4+ni], N, wmma::mem_row_major);
}

// V9 reference
__global__ void gemm_v9(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64, c0 = blockIdx.x * 128 + wx * 64;
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

int main() {
    // Test at 4096² only (other sizes already beaten or close)
    int sizes[] = {2048, 4096};
    double pytorch_tf[] = {13.437, 14.751};

    for (int si = 0; si < 2; si++) {
        int M = sizes[si], N_sz = M, K_sz = M;
        double flops = 2.0 * (double)M * N_sz * K_sz;
        printf("=== %d^2 | PT=%.1fT ===\n", M, pytorch_tf[si]);

        float *hA=(float*)malloc(M*K_sz*4),*hB=(float*)malloc(K_sz*N_sz*4);
        for(size_t i=0;i<(size_t)M*K_sz;i++)hA[i]=(float)rand()/RAND_MAX-0.5f;
        for(size_t i=0;i<(size_t)K_sz*N_sz;i++)hB[i]=(float)rand()/RAND_MAX-0.5f;
        float *dA32,*dB32,*dC;half *dA16,*dB16;
        CHK(cudaMalloc(&dA32,M*K_sz*4));CHK(cudaMalloc(&dB32,K_sz*N_sz*4));
        CHK(cudaMalloc(&dA16,M*K_sz*2));CHK(cudaMalloc(&dB16,K_sz*N_sz*2));
        CHK(cudaMalloc(&dC,M*N_sz*4));
        CHK(cudaMemcpy(dA32,hA,M*K_sz*4,cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dB32,hB,K_sz*N_sz*4,cudaMemcpyHostToDevice));
        convert<<<(M*K_sz+255)/256,256>>>(M*K_sz,dA32,dA16);
        convert<<<(K_sz*N_sz+255)/256,256>>>(K_sz*N_sz,dB32,dB16);
        CHK(cudaDeviceSynchronize());

        dim3 grid((N_sz+127)/128,(M+127)/128),block(128);

        // Warmup
        for(int i=0;i<30;i++)gemm_v9<<<grid,block>>>(M,N_sz,K_sz,dA16,dB16,dC);
        CHK(cudaDeviceSynchronize());

        printf("%-18s %8s %8s %6s\n","Kernel","ms","TFLOPs","%PT");
        printf("-----------------------------------------\n");

        // Test V9 baseline (default 48KB smem)
        cudaFuncSetAttribute(gemm_v9, cudaFuncAttributePreferredSharedMemoryCarveout, 0); // default
        cudaEvent_t s,e;cudaEventCreate(&s);cudaEventCreate(&e);
        cudaEventRecord(s);for(int i=0;i<20;i++)gemm_v9<<<grid,block>>>(M,N_sz,K_sz,dA16,dB16,dC);
        cudaEventRecord(e);cudaEventSynchronize(e);
        float t;cudaEventElapsedTime(&t,s,e);
        double tf=flops/(t/(20*1000.0))/1e12;
        printf("%-18s %8.3f %8.2f %5.0f%%\n","V9_default",t/20,tf,tf/pytorch_tf[si]*100);

        // Test V16 with increased smem (100KB)
        cudaFuncSetAttribute(gemm_v16_k32_bigsmem, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        cudaEventRecord(s);for(int i=0;i<20;i++)gemm_v16_k32_bigsmem<<<grid,block>>>(M,N_sz,K_sz,dA16,dB16,dC);
        cudaEventRecord(e);cudaEventSynchronize(e);
        cudaEventElapsedTime(&t,s,e);
        tf=flops/(t/(20*1000.0))/1e12;
        printf("%-18s %8.3f %8.2f %5.0f%%\n","V16_K32+smem100",t/20,tf,tf/pytorch_tf[si]*100);

        // Test V16 with max smem
        cudaFuncSetAttribute(gemm_v16_k32_bigsmem, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaEventRecord(s);for(int i=0;i<20;i++)gemm_v16_k32_bigsmem<<<grid,block>>>(M,N_sz,K_sz,dA16,dB16,dC);
        cudaEventRecord(e);cudaEventSynchronize(e);
        cudaEventElapsedTime(&t,s,e);
        tf=flops/(t/(20*1000.0))/1e12;
        printf("%-18s %8.3f %8.2f %5.0f%%\n","V16_K32+maxSmem",t/20,tf,tf/pytorch_tf[si]*100);

        cudaEventDestroy(s);cudaEventDestroy(e);
        printf("\n");
        CHK(cudaFree(dA32));CHK(cudaFree(dB32));CHK(cudaFree(dA16));CHK(cudaFree(dB16));CHK(cudaFree(dC));
        free(hA);free(hB);
    }
    printf("Done.\n");
    return 0;
}
