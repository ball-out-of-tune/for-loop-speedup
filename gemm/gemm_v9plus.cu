/**
 * GEMM V9+ — Targeted optimizations to beat PyTorch at all sizes
 * V9 baseline: 1024=104%, 2048=92%, 4096=89%
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
    if(i<n) d[i]=__float2half(s[i]);
}

// ============================================================================
// V9 BASELINE (from gemm_async_wmma.cu — exact copy)
// ============================================================================
__global__ void gemm_v9(int M, int N, int K,
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

    for (int idx = threadIdx.x; idx < 128*16/2; idx += 128) {
        int pos = idx * 2, r = pos / 16, col = pos % 16;
        int gr = blockIdx.y*128 + r;
        if (gr < M && col + 1 < K)
            *(short2*)&A_buf[0][r][col] = *(const short2*)&A[gr*K + col];
    }
    for (int idx = threadIdx.x; idx < 16*128/2; idx += 128) {
        int pos = idx * 2, r = pos / 128, col = pos % 128;
        int gc = blockIdx.x*128 + col;
        if (r < K && gc + 1 < N)
            *(short2*)&B_buf[0][r][col] = *(const short2*)&B[r*N + gc];
    }
    __syncthreads();
    int read_buf = 0;

    for (int kb = 16; kb < K; kb += 16) {
        int write_buf = 1 - read_buf;
        for (int chunk = threadIdx.x; chunk < 128*16/8; chunk += 128) {
            int pos = chunk * 8, r = pos / 16, c = pos % 16;
            int gr = blockIdx.y*128 + r, gc = kb + c;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(&A[gr*K+gc]));
            }
        }
        for (int chunk = threadIdx.x; chunk < 16*128/8; chunk += 128) {
            int pos = chunk * 8, r = pos / 128, c = pos % 128;
            int gr = kb + r, gc = blockIdx.x*128 + c;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(&B[gr*N+gc]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
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
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        read_buf = write_buf;
    }
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

// ============================================================================
// V10: K=32 double-buffered — halve sync overhead (256→128 syncs at 4096²)
// A_buf[2][128][32] + B_buf[2][32][128] = 16KB+16KB = 32KB smem → 1 blk/SM
// But 2× WMMA sets between syncs → 32 WMMA/sync ratio (up from 16)
// ============================================================================
__global__ void gemm_v10_k32(int M, int N, int K,
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

    // Prefetch buf 0 (2x data of V9)
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
        // cp.async load 32 K-columns (2x V9 data)
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

        // Step 0: WMMA for K columns [0..15]
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy*64 + mi*16][0], 32);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx*64 + ni*16], 128);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);

        // Step 1: WMMA for K columns [16..31]
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy*64 + mi*16][16], 32);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &B_buf[read_buf][16][wx*64 + ni*16], 128);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);

        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        read_buf = write_buf;
    }
    // Last tile: 2 WMMA steps from remaining data (prefetch loaded 32 cols)
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy*64 + mi*16][0], 32);
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

// ============================================================================
// V11: K=32 SINGLE-buffered — 16KB smem → 3 blk/SM, but no cp.async overlap
// Theory: 256 clocks of WMMA can hide some memory latency via wave-level parallelism
// 128 K-steps (vs 256 for V9), but loads are serial
// ============================================================================
__global__ void gemm_v11_single_k32(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64;
    int c0 = blockIdx.x * 128 + wx * 64;

    __shared__ half As[128][32];
    __shared__ half Bs[32][128];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    for (int kb = 0; kb < K; kb += 32) {
        // Load 32 K-columns (synchronous)
        for (int idx = threadIdx.x; idx < 128*32/2; idx += 128) {
            int pos = idx * 2, r = pos / 32, col = pos % 32;
            int gr = blockIdx.y*128 + r, gc = kb + col;
            if (gr < M && gc + 1 < K)
                *(short2*)&As[r][col] = *(const short2*)&A[gr*K + gc];
        }
        for (int idx = threadIdx.x; idx < 32*128/2; idx += 128) {
            int pos = idx * 2, r = pos / 128, col = pos % 128;
            int gr = kb + r, gc = blockIdx.x*128 + col;
            if (r < K && gr < K && gc + 1 < N)
                *(short2*)&Bs[r][col] = *(const short2*)&B[gr*N + gc];
        }
        __syncthreads();

        // WMMA step 0: K[0..15]
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &As[wy*64 + mi*16][0], 32);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &Bs[0][wx*64 + ni*16], 128);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);

        // WMMA step 1: K[16..31]
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &As[wy*64 + mi*16][16], 32);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &Bs[16][wx*64 + ni*16], 128);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);

        __syncthreads();
    }

    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::store_matrix_sync(C + (r0+mi*16)*N + c0+ni*16, c[mi*4+ni], N, wmma::mem_row_major);
}

// ============================================================================
// V13: Triple-buffered cp.async — 2 async groups in flight, 2x compute window
// Smem: 3×(128×16 + 16×128)×2 = 24KB → 2 blk/SM
// Pipeline: load buf[N+2] | WMMA buf[N+1] | wait for buf[N]
// ============================================================================
__global__ void gemm_v13_triple(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64;
    int c0 = blockIdx.x * 128 + wx * 64;

    __shared__ half A_buf[3][128][16];
    __shared__ half B_buf[3][16][128];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    // Prefetch buf[0] and buf[1]
    for (int idx = threadIdx.x; idx < 128*16/2; idx += 128) {
        int pos = idx * 2, r = pos / 16, col = pos % 16;
        int gr = blockIdx.y*128 + r;
        if (gr < M && col + 1 < K)
            *(short2*)&A_buf[0][r][col] = *(const short2*)&A[gr*K + col];
    }
    for (int idx = threadIdx.x; idx < 16*128/2; idx += 128) {
        int pos = idx * 2, r = pos / 128, col = pos % 128;
        int gc = blockIdx.x*128 + col;
        if (r < K && gc + 1 < N)
            *(short2*)&B_buf[0][r][col] = *(const short2*)&B[r*N + gc];
    }
    // Prefetch buf[1]: cp.async tile at K=16
    for (int chunk = threadIdx.x; chunk < 128*16/8; chunk += 128) {
        int pos = chunk * 8, r = pos / 16, c = pos % 16;
        int gr = blockIdx.y*128 + r, gc = 16 + c;
        if (gr < M && gc + 7 < K) {
            unsigned sa = __cvta_generic_to_shared(&A_buf[1][r][c]);
            asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&A[gr*K+gc]));
        }
    }
    for (int chunk = threadIdx.x; chunk < 16*128/8; chunk += 128) {
        int pos = chunk * 8, r = pos / 128, c = pos % 128;
        int gr = 16 + r, gc = blockIdx.x*128 + c;
        if (gr < K && gc + 7 < N) {
            unsigned sa = __cvta_generic_to_shared(&B_buf[1][r][c]);
            asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&B[gr*N+gc]));
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);  // group 0 (buf[1])

    // First WMMA: buf[0], while buf[1] is loading
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        wmma::load_matrix_sync(a[mi], &A_buf[0][wy*64 + mi*16][0], 16);
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::load_matrix_sync(b[ni], &B_buf[0][0][wx*64 + ni*16], 128);
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);
    asm volatile("cp.async.wait_group 0;\n" ::);  // wait for buf[1]
    __syncthreads();

    // State: buf[0] done, buf[1] ready, buf[2] empty
    // Pipeline: compute_buf → buf[1] first, load buf[2] with K=[32..47]
    // cur_buf = (kb/16 - 1) % 3, write_buf = (kb/16) % 3

    for (int kb = 32; kb < K; kb += 16) {
        int write_buf = (kb/16) % 3;
        int cur_buf = (kb/16 - 1) % 3;
        // cp.async load buf[write_buf] with K=[kb..kb+15]
        int k_src = kb;
        for (int chunk = threadIdx.x; chunk < 128*16/8; chunk += 128) {
            int pos = chunk * 8, r = pos / 16, c = pos % 16;
            int gr = blockIdx.y*128 + r, gc = k_src + c;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c]);
                asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&A[gr*K+gc]));
            }
        }
        for (int chunk = threadIdx.x; chunk < 16*128/8; chunk += 128) {
            int pos = chunk * 8, r = pos / 128, c = pos % 128;
            int gr = k_src + r, gc = blockIdx.x*128 + c;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c]);
                asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&B[gr*N+gc]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);

        // WMMA from cur_buf
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &A_buf[cur_buf][wy*64 + mi*16][0], 16);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &B_buf[cur_buf][0][wx*64 + ni*16], 128);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);

        // wait_group 1: wait for ALL committed groups except the latest
        // → cur_buf's load (committed 2 groups ago) is done
        // → write_buf's load (committed 1 group ago) is done
        // → current commit (write_buf) stays in flight
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
    }

    // Last two tiles (already loaded in bufs, no new loads)
    for (int last = 0; last < 2; last++) {
        int buf_idx = (K/16 - 2 + last) % 3;
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &A_buf[buf_idx][wy*64 + mi*16][0], 16);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &B_buf[buf_idx][0][wx*64 + ni*16], 128);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);
    }

    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::store_matrix_sync(C + (r0+mi*16)*N + c0+ni*16, c[mi*4+ni], N, wmma::mem_row_major);
}
// 4 warps, 32 WMMA/warp (4x8 grid)
// Each warp covers 64x128 output = 4×8 = 32 WMMA (vs 16 in V9)
// Registers: 32*8=256 → will spill → need more warps!
// Let's try 8 warps instead: each covers 32×64 = 2×4 = 8 WMMA, 8*8=64 regs
// ============================================================================
__global__ void gemm_v12_256x128_8w(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    // 8 warps (256 threads): 4 rows x 2 cols of 64x64 blocks
    int wid = threadIdx.x / 32;
    int wy = wid / 2;  // 0..3 (M direction: 4 groups of 64 rows = 256 total M)
    int wx = wid % 2;  // 0..1 (N direction: 2 groups of 64 cols = 128 total N)
    int r0 = blockIdx.y * 256 + wy * 64;
    int c0 = blockIdx.x * 128 + wx * 64;

    __shared__ half A_buf[2][256][16];
    __shared__ half B_buf[2][16][128];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    // Prefetch
    for (int idx = threadIdx.x; idx < 256*16/2; idx += 256) {
        int pos = idx * 2, r = pos / 16, col = pos % 16;
        int gr = blockIdx.y*256 + r;
        if (gr < M && col + 1 < K)
            *(short2*)&A_buf[0][r][col] = *(const short2*)&A[gr*K + col];
    }
    for (int idx = threadIdx.x; idx < 16*128/2; idx += 256) {
        int pos = idx * 2, r = pos / 128, col = pos % 128;
        int gc = blockIdx.x*128 + col;
        if (r < K && gc + 1 < N)
            *(short2*)&B_buf[0][r][col] = *(const short2*)&B[r*N + gc];
    }
    __syncthreads();
    int read_buf = 0;

    for (int kb = 16; kb < K; kb += 16) {
        int write_buf = 1 - read_buf;
        for (int chunk = threadIdx.x; chunk < 256*16/8; chunk += 256) {
            int pos = chunk * 8, r = pos / 16, c = pos % 16;
            int gr = blockIdx.y*256 + r, gc = kb + c;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c]);
                asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&A[gr*K+gc]));
            }
        }
        for (int chunk = threadIdx.x; chunk < 16*128/8; chunk += 256) {
            int pos = chunk * 8, r = pos / 128, c = pos % 128;
            int gr = kb + r, gc = blockIdx.x*128 + c;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c]);
                asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&B[gr*N+gc]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy*64+mi*16][0], 16);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx*64+ni*16], 128);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        read_buf = write_buf;
    }
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy*64+mi*16][0], 16);
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
        wmma::store_matrix_sync(C + (r0+mi*16)*N + c0+ni*16, c[mi*4+ni], N, wmma::mem_row_major);
}

// ============================================================================
// V14: Interleaved WMMA loads + MMAs — better ILP within each K-step
// Instead of: load all A, load all B, then all MMAs
// Do: load A, then for each B: load B, mma A×B
// This lets B loads and A-reuse happen concurrently
// ============================================================================
__global__ void gemm_v14_interleave(int M, int N, int K,
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
    #pragma unroll
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    for (int idx = threadIdx.x; idx < 128*16/2; idx += 128) {
        int pos = idx * 2, r = pos / 16, col = pos % 16;
        int gr = blockIdx.y*128 + r;
        if (gr < M && col + 1 < K)
            *(short2*)&A_buf[0][r][col] = *(const short2*)&A[gr*K + col];
    }
    for (int idx = threadIdx.x; idx < 16*128/2; idx += 128) {
        int pos = idx * 2, r = pos / 128, col = pos % 128;
        int gc = blockIdx.x*128 + col;
        if (r < K && gc + 1 < N)
            *(short2*)&B_buf[0][r][col] = *(const short2*)&B[r*N + gc];
    }
    __syncthreads();
    int read_buf = 0;

    for (int kb = 16; kb < K; kb += 16) {
        int write_buf = 1 - read_buf;
        for (int chunk = threadIdx.x; chunk < 128*16/8; chunk += 128) {
            int pos = chunk * 8, r = pos / 16, c = pos % 16;
            int gr = blockIdx.y*128 + r, gc = kb + c;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(&A[gr*K+gc]));
            }
        }
        for (int chunk = threadIdx.x; chunk < 16*128/8; chunk += 128) {
            int pos = chunk * 8, r = pos / 128, c = pos % 128;
            int gr = kb + r, gc = blockIdx.x*128 + c;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(&B[gr*N+gc]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);

        // INTERLEAVED: load all A fragments first, then B+MMA interleaved
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy*64 + mi*16][0], 16);

        // For each B fragment: load, then immediately do all A-MMAs
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx*64 + ni*16], 128);
            #pragma unroll
            for (int mi = 0; mi < 4; mi++)
                wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);
        }

        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        read_buf = write_buf;
    }
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy*64 + mi*16][0], 16);
    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx*64 + ni*16], 128);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);
    }
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::store_matrix_sync(C + (r0+mi*16)*N + c0+ni*16, c[mi*4+ni], N, wmma::mem_row_major);
}

// ============================================================================
// V15: V14 interleaved + V13 triple buffering = deeper pipeline + better ILP
// Triple buffered smem (24KB) + interleaved WMMA/B loads
// ============================================================================
__global__ void gemm_v15_hybrid(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64;
    int c0 = blockIdx.x * 128 + wx * 64;

    __shared__ half A_buf[3][128][16];
    __shared__ half B_buf[3][16][128];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    // Prefetch buf[0] sync
    for (int idx = threadIdx.x; idx < 128*16/2; idx += 128) {
        int pos = idx * 2, r = pos / 16, col = pos % 16;
        int gr = blockIdx.y*128 + r;
        if (gr < M && col + 1 < K)
            *(short2*)&A_buf[0][r][col] = *(const short2*)&A[gr*K + col];
    }
    for (int idx = threadIdx.x; idx < 16*128/2; idx += 128) {
        int pos = idx * 2, r = pos / 128, col = pos % 128;
        int gc = blockIdx.x*128 + col;
        if (r < K && gc + 1 < N)
            *(short2*)&B_buf[0][r][col] = *(const short2*)&B[r*N + gc];
    }
    // Prefetch buf[1] via cp.async
    for (int chunk = threadIdx.x; chunk < 128*16/8; chunk += 128) {
        int pos = chunk * 8, r = pos / 16, c = pos % 16;
        int gr = blockIdx.y*128 + r, gc = 16 + c;
        if (gr < M && gc + 7 < K) {
            unsigned sa = __cvta_generic_to_shared(&A_buf[1][r][c]);
            asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&A[gr*K+gc]));
        }
    }
    for (int chunk = threadIdx.x; chunk < 16*128/8; chunk += 128) {
        int pos = chunk * 8, r = pos / 128, c = pos % 128;
        int gr = 16 + r, gc = blockIdx.x*128 + c;
        if (gr < K && gc + 7 < N) {
            unsigned sa = __cvta_generic_to_shared(&B_buf[1][r][c]);
            asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&B[gr*N+gc]));
        }
    }
    asm volatile("cp.async.commit_group;\n" ::);
    // Compute buf[0]
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        wmma::load_matrix_sync(a[mi], &A_buf[0][wy*64 + mi*16][0], 16);
    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        wmma::load_matrix_sync(b[ni], &B_buf[0][0][wx*64 + ni*16], 128);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    for (int kb = 32; kb < K; kb += 16) {
        int write_buf = (kb/16) % 3;
        int cur_buf = (kb/16 - 1) % 3;
        int k_src = kb;

        for (int chunk = threadIdx.x; chunk < 128*16/8; chunk += 128) {
            int pos = chunk * 8, r = pos / 16, c = pos % 16;
            int gr = blockIdx.y*128 + r, gc = k_src + c;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c]);
                asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&A[gr*K+gc]));
            }
        }
        for (int chunk = threadIdx.x; chunk < 16*128/8; chunk += 128) {
            int pos = chunk * 8, r = pos / 128, c = pos % 128;
            int gr = k_src + r, gc = blockIdx.x*128 + c;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c]);
                asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&B[gr*N+gc]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);

        // Interleaved loads from cur_buf
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &A_buf[cur_buf][wy*64 + mi*16][0], 16);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::load_matrix_sync(b[ni], &B_buf[cur_buf][0][wx*64 + ni*16], 128);
            #pragma unroll
            for (int mi = 0; mi < 4; mi++)
                wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);
        }

        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();
    }
    // Last 2 tiles
    for (int last = 0; last < 2; last++) {
        int buf_idx = (K/16 - 2 + last) % 3;
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &A_buf[buf_idx][wy*64 + mi*16][0], 16);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            wmma::load_matrix_sync(b[ni], &B_buf[buf_idx][0][wx*64 + ni*16], 128);
            #pragma unroll
            for (int mi = 0; mi < 4; mi++)
                wmma::mma_sync(c[mi*4+ni], a[mi], b[ni], c[mi*4+ni]);
        }
    }
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::store_matrix_sync(C + (r0+mi*16)*N + c0+ni*16, c[mi*4+ni], N, wmma::mem_row_major);
}
// ============================================================================
__global__ void gemm_v14_v9_unroll(int M, int N, int K,
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
    #pragma unroll
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    for (int idx = threadIdx.x; idx < 128*16/2; idx += 128) {
        int pos = idx * 2, r = pos / 16, col = pos % 16;
        int gr = blockIdx.y*128 + r;
        if (gr < M && col + 1 < K)
            *(short2*)&A_buf[0][r][col] = *(const short2*)&A[gr*K + col];
    }
    for (int idx = threadIdx.x; idx < 16*128/2; idx += 128) {
        int pos = idx * 2, r = pos / 128, col = pos % 128;
        int gc = blockIdx.x*128 + col;
        if (r < K && gc + 1 < N)
            *(short2*)&B_buf[0][r][col] = *(const short2*)&B[r*N + gc];
    }
    __syncthreads();
    int read_buf = 0;

    for (int kb = 16; kb < K; kb += 16) {
        int write_buf = 1 - read_buf;
        for (int chunk = threadIdx.x; chunk < 128*16/8; chunk += 128) {
            int pos = chunk * 8, r = pos / 16, c = pos % 16;
            int gr = blockIdx.y*128 + r, gc = kb + c;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(&A[gr*K+gc]));
            }
        }
        for (int chunk = threadIdx.x; chunk < 16*128/8; chunk += 128) {
            int pos = chunk * 8, r = pos / 128, c = pos % 128;
            int gr = kb + r, gc = blockIdx.x*128 + c;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(&B[gr*N+gc]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);
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
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        read_buf = write_buf;
    }
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

// ============================================================================
// Benchmark
// ============================================================================
int main() {
    int sizes[] = {1024, 2048, 4096};
    double pytorch_tf[] = {11.257, 13.437, 14.751};

    printf("V9+ GEMM Optimization\n");
    printf("GPU: RTX 3050 Ti Laptop (FW via WSL)\n\n");

    typedef void (*gemm_fn)(int,int,int,const half*,const half*,float*);
    typedef struct { const char* n; gemm_fn k; int tm, tn, th; } KInfo;
    KInfo ki[] = {
        {"V9_baseline",   gemm_v9,          128, 128, 128},
        {"V13_triple",    gemm_v13_triple,  128, 128, 128},
        {"V14_interleave",gemm_v14_interleave,128,128,128},
        {"V15_hybrid",    gemm_v15_hybrid,  128, 128, 128},
    };
    int nk = 4;

    for (int si = 0; si < 3; si++) {
        int M = sizes[si], N_sz = M, K_sz = M;
        double flops = 2.0 * (double)M * N_sz * K_sz;
        printf("=== %d^2 | PT=%.1fT ===\n", M, pytorch_tf[si]);
        printf("%-18s %8s %8s %6s\n", "Kernel", "ms", "TFLOPs", "%PT");
        printf("-------------------------------------------\n");

        float *hA=(float*)malloc(M*K_sz*4),*hB=(float*)malloc(K_sz*N_sz*4);
        for(size_t i=0;i<(size_t)M*K_sz;i++)hA[i]=(float)rand()/RAND_MAX-0.5f;
        for(size_t i=0;i<(size_t)K_sz*N_sz;i++)hB[i]=(float)rand()/RAND_MAX-0.5f;

        float *dA32,*dB32,*dC; half *dA16,*dB16;
        CHK(cudaMalloc(&dA32,M*K_sz*4)); CHK(cudaMalloc(&dB32,K_sz*N_sz*4));
        CHK(cudaMalloc(&dA16,M*K_sz*2)); CHK(cudaMalloc(&dB16,K_sz*N_sz*2));
        CHK(cudaMalloc(&dC,M*N_sz*4));
        CHK(cudaMemcpy(dA32,hA,M*K_sz*4,cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dB32,hB,K_sz*N_sz*4,cudaMemcpyHostToDevice));
        convert<<<(M*K_sz+255)/256,256>>>(M*K_sz,dA32,dA16);
        convert<<<(K_sz*N_sz+255)/256,256>>>(K_sz*N_sz,dB32,dB16);
        CHK(cudaDeviceSynchronize());

        // Warmup: run V9 enough to stabilize clocks
        { dim3 g((N_sz+127)/128,(M+127)/128);
          for(int i=0;i<50;i++)gemm_v9<<<g,128>>>(M,N_sz,K_sz,dA16,dB16,dC);
          CHK(cudaDeviceSynchronize()); }

        for (int j = 0; j < nk; j++) {
            dim3 grid((N_sz+ki[j].tn-1)/ki[j].tn, (M+ki[j].tm-1)/ki[j].tm);
            dim3 block(ki[j].th);

            // verify
            ki[j].k<<<grid,block>>>(M,N_sz,K_sz,dA16,dB16,dC);
            cudaError_t e = cudaDeviceSynchronize();
            if (e != cudaSuccess) {
                printf("%-18s LAUNCH FAILED (err %d)\n", ki[j].n, e);
                continue;
            }

            int reps = 20;
            cudaEvent_t s,stop; cudaEventCreate(&s); cudaEventCreate(&stop);
            cudaEventRecord(s);
            for(int i=0;i<reps;i++)
                ki[j].k<<<grid,block>>>(M,N_sz,K_sz,dA16,dB16,dC);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float t; cudaEventElapsedTime(&t,s,stop);
            double tf = flops/(t/(reps*1000.0))/1e12;
            double pct = tf/pytorch_tf[si]*100;
            printf("%-18s %8.3f %8.2f %5.0f%%", ki[j].n, t/reps, tf, pct);
            if(pct>=100)printf(" <<<");
            printf("\n");
            cudaEventDestroy(s); cudaEventDestroy(stop);
        }
        printf("\n");
        CHK(cudaFree(dA32));CHK(cudaFree(dB32));CHK(cudaFree(dA16));CHK(cudaFree(dB16));CHK(cudaFree(dC));
        free(hA);free(hB);
    }
    printf("Done.\n");
    return 0;
}
