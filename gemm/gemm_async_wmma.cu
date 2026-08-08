/**
 * Async WMMA V9: cp.async double buffer + FP16 WMMA
 * Overlap global->smem load with Tensor Core compute
 *
 * nvcc -o gemm_async gemm_async_wmma.cu -arch=sm_86
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define CHECK(c) do{cudaError_t e=c;if(e!=cudaSuccess){fprintf(stderr,"CUDA %d\n",e);exit(1);}}while(0)

__global__ void convert(int n, const float* src, half* dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// V9: async double-buffered WMMA, 4 warps (128 threads), 128x128 tile
__global__ void gemm_async_wmma(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64;
    int c0 = blockIdx.x * 128 + wx * 64;

    __shared__ half A_buf[2][128][16];
    __shared__ half B_buf[2][16][128];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    // Prefetch buf 0: load 2xFP16 at once using short2*
    for (int idx = threadIdx.x; idx < 128*16/2; idx += 128) {
        int pos = idx * 2;
        int r = pos / 16, c = pos % 16;
        int gr = blockIdx.y*128 + r, gc = c;
        if (gr < M && gc + 1 < K)
            *(short2*)&A_buf[0][r][c] = *(const short2*)&A[gr*K + gc];
    }
    for (int idx = threadIdx.x; idx < 16*128/2; idx += 128) {
        int pos = idx * 2;
        int r = pos / 128, c = pos % 128;
        int gr = r, gc = blockIdx.x*128 + c;
        if (gr < K && gc + 1 < N)
            *(short2*)&B_buf[0][r][c] = *(const short2*)&B[gr*N + gc];
    }
    __syncthreads();

    int read_buf = 0;

    for (int kb = 16; kb < K; kb += 16) {
        int write_buf = 1 - read_buf;

        // cp.async next tile: 16 bytes = 8xFP16 per copy
        for (int chunk = threadIdx.x; chunk < 128*16/8; chunk += 128) {
            int pos = chunk * 8;  // element offset
            int r = pos / 16, c = pos % 16;  // c is 0 or 8
            int gr = blockIdx.y*128 + r, gc = kb + c;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c]);
                const half* ga = &A[gr * K + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(ga));
            }
        }
        for (int chunk = threadIdx.x; chunk < 16*128/8; chunk += 128) {
            int pos = chunk * 8;
            int r = pos / 128, c = pos % 128;  // c in {0,8,...,120}
            int gr = kb + r, gc = blockIdx.x*128 + c;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c]);
                const half* gb = &B[gr * N + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(gb));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);

        // WMMA compute from read_buf (async copy runs in background!)
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy*64 + mi*16][0], 16);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx*64 + ni*16], 128);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);

        // Wait for cp.async to finish, then swap buffers
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        read_buf = write_buf;
    }

    // Last tile (already in read_buf)
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy*64 + mi*16][0], 16);
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx*64 + ni*16], 128);
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);

    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::store_matrix_sync(C + (r0+mi*16)*N + c0+ni*16, c[mi*4+ni], N, wmma::mem_row_major);
}

__global__ void async_gemm_myown() {
    
}

// V12: 64x128 tile, 8 WMMA/warp, 6KB smem (for small sizes)
__global__ void gemm_async_wmma_64x128(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 64 + wy * 32;
    int c0 = blockIdx.x * 128 + wx * 64;

    __shared__ half Ab[2][64][16];
    __shared__ half Bb[2][16][128];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[8];
    for (int i = 0; i < 8; i++) wmma::fill_fragment(c[i], 0.0f);

    for (int i = threadIdx.x; i < 64*16/2; i += 128) {
        int p = i * 2, r = p / 16, col = p % 16;
        int gr = blockIdx.y * 64 + r;
        if (gr < M && col + 1 < K) *(short2*)&Ab[0][r][col] = *(const short2*)&A[gr*K+col];
    }
    for (int i = threadIdx.x; i < 16*128/2; i += 128) {
        int p = i * 2, r = p / 128, col = p % 128;
        int gc = blockIdx.x*128+col;
        if (r<K&&gc+1<N)*(short2*)&Bb[0][r][col]=*(const short2*)&B[r*N+gc];
    }
    __syncthreads();
    int rb = 0;

    for (int kb = 16; kb < K; kb += 16) {
        int wb = 1 - rb;
        for (int i = threadIdx.x; i < 64*16/8; i += 128) {
            int p = i * 8, r = p / 16, col = p % 16;
            int gr = blockIdx.y*64+r, gc = kb+col;
            if (gr<M&&gc+7<K){unsigned sa=__cvta_generic_to_shared(&Ab[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&A[gr*K+gc]));}
        }
        for (int i = threadIdx.x; i < 16*128/8; i += 128) {
            int p = i * 8, r = p / 128, col = p % 128;
            int gr = kb+r, gc = blockIdx.x*128+col;
            if (gr<K&&gc+7<N){unsigned sb=__cvta_generic_to_shared(&Bb[wb][r][col]);asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sb),"l"(&B[gr*N+gc]));}
        }
        asm volatile("cp.async.commit_group;\n"::);
        for (int mi=0;mi<2;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*32+mi*16][0],16);
        for (int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][0][wx*64+ni*16],128);
        for (int mi=0;mi<2;mi++)for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();rb=wb;
    }
    for (int mi=0;mi<2;mi++)wmma::load_matrix_sync(a[mi],&Ab[rb][wy*32+mi*16][0],16);
    for (int ni=0;ni<4;ni++)wmma::load_matrix_sync(b[ni],&Bb[rb][0][wx*64+ni*16],128);
    for (int mi=0;mi<2;mi++)for(int ni=0;ni<4;ni++)wmma::mma_sync(c[mi*4+ni],a[mi],b[ni],c[mi*4+ni]);
    for (int mi=0;mi<2;mi++)for(int ni=0;ni<4;ni++)wmma::store_matrix_sync(C+(r0+mi*16)*N+c0+ni*16,c[mi*4+ni],N,wmma::mem_row_major);
}

// V11: 128x160 tile, 20 WMMA/warp (4x5), 9KB smem
__global__ void gemm_async_wmma_v11(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64;
    int c0 = blockIdx.x * 160 + wx * 80;

    __shared__ half Ab[2][128][16];
    __shared__ half Bb[2][16][160];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[5];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[20];
    for (int i = 0; i < 20; i++) wmma::fill_fragment(c[i], 0.0f);

    for (int i = threadIdx.x; i < 128*16/2; i += 128) {
        int p = i * 2, r = p / 16, col = p % 16;
        int gr = blockIdx.y * 128 + r;
        if (gr < M && col + 1 < K)
            *(short2*)&Ab[0][r][col] = *(const short2*)&A[gr * K + col];
    }
    for (int i = threadIdx.x; i < 16*160/2; i += 128) {
        int p = i * 2, r = p / 160, col = p % 160;
        int gc = blockIdx.x * 160 + col;
        if (r < K && gc + 1 < N)
            *(short2*)&Bb[0][r][col] = *(const short2*)&B[r * N + gc];
    }
    __syncthreads();
    int rb = 0;

    for (int kb = 16; kb < K; kb += 16) {
        int wb = 1 - rb;
        for (int i = threadIdx.x; i < 128*16/8; i += 128) {
            int p = i * 8, r = p / 16, col = p % 16;
            int gr = blockIdx.y*128 + r, gc = kb + col;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&Ab[wb][r][col]);
                asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n" :: "r"(sa), "l"(&A[gr*K+gc]));
            }
        }
        for (int i = threadIdx.x; i < 16*160/8; i += 128) {
            int p = i * 8, r = p / 160, col = p % 160;
            int gr = kb + r, gc = blockIdx.x*160 + col;
            if (gr < K && gc + 7 < N) {
                unsigned sb = __cvta_generic_to_shared(&Bb[wb][r][col]);
                asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n" :: "r"(sb), "l"(&B[gr*N+gc]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        for (int mi = 0; mi < 4; mi++) wmma::load_matrix_sync(a[mi], &Ab[rb][wy*64 + mi*16][0], 16);
        for (int ni = 0; ni < 5; ni++) wmma::load_matrix_sync(b[ni], &Bb[rb][0][wx*80 + ni*16], 160);
        for (int mi = 0; mi < 4; mi++) for (int ni = 0; ni < 5; ni++)
            wmma::mma_sync(c[mi*5+ni], a[mi], b[ni], c[mi*5+ni]);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads(); rb = wb;
    }
    for (int mi = 0; mi < 4; mi++) wmma::load_matrix_sync(a[mi], &Ab[rb][wy*64 + mi*16][0], 16);
    for (int ni = 0; ni < 5; ni++) wmma::load_matrix_sync(b[ni], &Bb[rb][0][wx*80 + ni*16], 160);
    for (int mi = 0; mi < 4; mi++) for (int ni = 0; ni < 5; ni++)
        wmma::mma_sync(c[mi*5+ni], a[mi], b[ni], c[mi*5+ni]);
    for (int mi = 0; mi < 4; mi++) for (int ni = 0; ni < 5; ni++)
        wmma::store_matrix_sync(C + (r0+mi*16)*N + c0+ni*16, c[mi*5+ni], N, wmma::mem_row_major);
}

// V12: 64x64 tile variant for small sizes (more blocks = better occupancy)
__global__ void gemm_async_wmma_64(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 64 + wy * 32;
    int c0 = blockIdx.x * 64 + wx * 32;

    __shared__ half Ab[2][64][16];
    __shared__ half Bb[2][16][64];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    for (int i = threadIdx.x; i < 64*16/2; i += 128) {
        int p = i * 2, r = p / 16, col = p % 16;
        int gr = blockIdx.y * 64 + r;
        if (gr < M && col + 1 < K) *(short2*)&Ab[0][r][col] = *(const short2*)&A[gr*K+col];
    }
    for (int i = threadIdx.x; i < 16*64/2; i += 128) {
        int p = i * 2, r = p / 64, col = p % 64;
        int gc = blockIdx.x * 64 + col;
        if (r < K && gc + 1 < N) *(short2*)&Bb[0][r][col] = *(const short2*)&B[r*N+gc];
    }
    __syncthreads();
    int rb = 0;

    for (int kb = 16; kb < K; kb += 16) {
        int wb = 1 - rb;
        for (int i = threadIdx.x; i < 64*16/8; i += 128) {
            int p = i * 8, r = p / 16, col = p % 16;
            int gr = blockIdx.y*64 + r, gc = kb + col;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&Ab[wb][r][col]);
                asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n" :: "r"(sa), "l"(&A[gr*K+gc]));
            }
        }
        for (int i = threadIdx.x; i < 16*64/8; i += 128) {
            int p = i * 8, r = p / 64, col = p % 64;
            int gr = kb + r, gc = blockIdx.x*64 + col;
            if (gr < K && gc + 7 < N) {
                unsigned sb = __cvta_generic_to_shared(&Bb[wb][r][col]);
                asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n" :: "r"(sb), "l"(&B[gr*N+gc]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        for (int mi = 0; mi < 2; mi++) wmma::load_matrix_sync(a[mi], &Ab[rb][wy*32+mi*16][0], 16);
        for (int ni = 0; ni < 2; ni++) wmma::load_matrix_sync(b[ni], &Bb[rb][0][wx*32+ni*16], 64);
        for (int mi = 0; mi < 2; mi++) for (int ni = 0; ni < 2; ni++)
            wmma::mma_sync(c[mi*2+ni], a[mi], b[ni], c[mi*2+ni]);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads(); rb = wb;
    }
    for (int mi = 0; mi < 2; mi++) wmma::load_matrix_sync(a[mi], &Ab[rb][wy*32+mi*16][0], 16);
    for (int ni = 0; ni < 2; ni++) wmma::load_matrix_sync(b[ni], &Bb[rb][0][wx*32+ni*16], 64);
    for (int mi = 0; mi < 2; mi++) for (int ni = 0; ni < 2; ni++)
        wmma::mma_sync(c[mi*2+ni], a[mi], b[ni], c[mi*2+ni]);
    for (int mi = 0; mi < 2; mi++) for (int ni = 0; ni < 2; ni++)
        wmma::store_matrix_sync(C+(r0+mi*16)*N+c0+ni*16, c[mi*2+ni], N, wmma::mem_row_major);
}

int main() {
    int sizes[] = {1024, 2048, 4096};
    printf("V9 cp.async final\n\n");

    // PyTorch FP16 ref (measured)
    float pytorch_t[] = {0.191f, 1.279f, 9.317f};
    double pytorch_tf[] = {11.257, 13.437, 14.751};

    printf("%6s | %15s %8s %8s %6s | %15s %8s %8s\n",
           "size", "V9_cp_async", "ms", "TFLOPS", "%PT", "PyTorch FP16", "ms", "TFLOPS");
    printf("-------+---------------------------------------------------------\n");

    for (int si = 0; si < 3; si++) {
        int M = sizes[si], N = M, K = M;
        double flops = 2.0 * (double)M * N * K;

        float *hA = (float*)malloc(M*K*4), *hB = (float*)malloc(K*N*4);
        for (size_t i = 0; i < (size_t)M*K; i++) hA[i] = (float)rand()/RAND_MAX-0.5f;
        for (size_t i = 0; i < (size_t)K*N; i++) hB[i] = (float)rand()/RAND_MAX-0.5f;

        float *dA32, *dB32, *dC; half *dA16, *dB16;
        CHECK(cudaMalloc(&dA32, M*K*4)); CHECK(cudaMalloc(&dB32, K*N*4));
        CHECK(cudaMalloc(&dA16, M*K*2)); CHECK(cudaMalloc(&dB16, K*N*2));
        CHECK(cudaMalloc(&dC, M*N*4));
        CHECK(cudaMemcpy(dA32, hA, M*K*4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB32, hB, K*N*4, cudaMemcpyHostToDevice));
        convert<<<(M*K+255)/256,256>>>(M*K, dA32, dA16);
        convert<<<(K*N+255)/256,256>>>(K*N, dB32, dB16);
        CHECK(cudaDeviceSynchronize());

        dim3 grid((N+127)/128, (M+127)/128), block(128);
        for (int i = 0; i < 50; i++) gemm_async_wmma<<<grid,block>>>(M,N,K,dA16,dB16,dC);
        CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        for (int i = 0; i < 20; i++) gemm_async_wmma<<<grid,block>>>(M,N,K,dA16,dB16,dC);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float t; cudaEventElapsedTime(&t, start, stop);
        double tflops = flops / (t/20000) / 1e12;
        double pct = tflops / pytorch_tf[si] * 100;
        printf("%6d | %15s %8.3f %8.3f %5.0f%% | %15s %8.3f %8.3f\n",
               M, "V9_cp_async", t/20, tflops, pct,
               "PyTorch FP16", pytorch_t[si], pytorch_tf[si]);
        cudaEventDestroy(start); cudaEventDestroy(stop);

        CHECK(cudaFree(dA32)); CHECK(cudaFree(dB32));
        CHECK(cudaFree(dA16)); CHECK(cudaFree(dB16)); CHECK(cudaFree(dC));
        free(hA); free(hB);
    }
    return 0;
}
