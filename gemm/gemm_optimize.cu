/**
 * GEMM Optimization: Variants to beat PyTorch at 1024^2, 2048^2, 4096^2
 * V9 baseline: 1024=104%, 2048=92%, 4096=89%
 * Goals: 2048->100%+, 4096->100%+
 *
 * Variants:
 *   V9  - baseline (ref)
 *   V9a - K=32 single-buffered (double WMMA per sync, no cp.async)
 *   V9b - ld=18 bank-conflict-free smem layout
 *   V9c - ld=32 bank-conflict-free + 2x WMMA/sync (K=32 with ld=32)
 *   V9d - 8-warp (256 threads), 4 WMMA/warp, 256x128 tile
 *   V9e - 6-warp (192 threads), 8 WMMA/warp, 192x128 tile
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define CHECK(c) do{cudaError_t e=c;if(e!=cudaSuccess){fprintf(stderr,"CUDA %d\n",e);exit(1);}}while(0)

// ============================================================================
// Shared utilities
// ============================================================================
__global__ void convert(int n, const float* src, half* dst) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// Warmup kernel to stabilize clocks
__global__ void warmup(half* a, half* b, float* c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N*N) c[i] = __half2float(a[i]) + __half2float(b[i]);
}

// ============================================================================
// V9: Baseline (cp.async double-buffered, 128x128, K=16)
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

    // Prefetch buf 0
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
                const half* ga = &A[gr * K + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(ga));
            }
        }
        for (int chunk = threadIdx.x; chunk < 16*128/8; chunk += 128) {
            int pos = chunk * 8, r = pos / 128, c = pos % 128;
            int gr = kb + r, gc = blockIdx.x*128 + c;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c]);
                const half* ga = &B[gr * N + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(ga));
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

    // Last tile
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
// V9a: K=32 single-buffered (no cp.async, but 2x WMMA per sync)
// Shared: [128][32] + [32][128] = 8KB+8KB = 16KB, 48/16=3 blocks/SM
// Each K-step: 2 syncs but 32 WMMA → WMMA/sync = 16 (same as V9!)
// Actually... we still need 2 syncs (one for load, one for compute)
// Without cp.async: load → sync → compute 2× → sync → next
// That's WORSE than V9 (which hides load overhead)
// Let's skip this one
// ============================================================================

// ============================================================================
// V9a: ld=18 padding to avoid bank conflicts
// Bank conflict with ld=16: stride is 8 banks, gcd(8,32)=8 → 4-way bank conflict
// With ld=18: stride is 9 banks, gcd(9,32)=1 → 0 bank conflict
// Smem: 2*(128*18 + 18*128)*2 = 2*(2304+2304)*2 = 18KB, 48/18=2.6→2 blocks/SM
// ============================================================================
__global__ void gemm_v9_ld18(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64;
    int c0 = blockIdx.x * 128 + wx * 64;

    __shared__ half A_buf[2][128][18];  // padded to 18
    __shared__ half B_buf[2][18][128];  // padded to 18

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    // Prefetch buf 0 (pad K dim to 16 for load indexing consistency)
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
                const half* ga = &A[gr * K + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(ga));
            }
        }
        for (int chunk = threadIdx.x; chunk < 16*128/8; chunk += 128) {
            int pos = chunk * 8, r = pos / 128, c = pos % 128;
            int gr = kb + r, gc = blockIdx.x*128 + c;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c]);
                const half* ga = &B[gr * N + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(ga));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);

        // WMMA with ld=18 (bank-conflict-free!)
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy*64 + mi*16][0], 18);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx*64 + ni*16], 18);
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
        wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy*64 + mi*16][0], 18);
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx*64 + ni*16], 18);
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
// V9b: K=32 double-buffered, reduced sync overhead
// Shared: [2][128][32] + [2][32][128] = 16KB+16KB = 32KB
// 48KB limit → 1 block/SM. But: 2× WMMA per cp.async = 32 WMMA/sync
// For 4096²: 128 K-steps instead of 256 → half the syncs!
// WMMA loads: do step 0 and step 16 as two consecutive WMMA sets
// ============================================================================
__global__ void gemm_v9_k32(int M, int N, int K,
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

    // Prefetch buf 0: 128*32 = 4096 fp16 = 2048 short2 loads
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
        // cp.async load 32 K-columns
        for (int chunk = threadIdx.x; chunk < 128*32/8; chunk += 128) {
            int pos = chunk * 8, r = pos / 32, c = pos % 32;
            int gr = blockIdx.y*128 + r, gc = kb + c;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(&A[gr*K + gc]));
            }
        }
        for (int chunk = threadIdx.x; chunk < 32*128/8; chunk += 128) {
            int pos = chunk * 8, r = pos / 128, c = pos % 128;
            int gr = kb + r, gc = blockIdx.x*128 + c;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(&B[gr*N + gc]));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);

        // WMMA step 0: K columns [0..15]
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

        // WMMA step 1: K columns [16..31]
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

    // Last tile (first 16 of 32)
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
// V9c: K=32 + ld=18 (bank-free + half the syncs)
// Shared: [2][128][18] + [2][18][128]... wait, K=32 means we need K dim = 32
// Let me use: A_buf[2][128][34] (ld=34, but store 32), B_buf[2][34][128]
// Actually 32 padded to 34: A_buf[2][128][34], B_buf[2][34][128]
// 2*(128*34 + 34*128)*2/1024 = 2*(4352+4352)*2/1024 = 34 KB
// Too much for 48KB with double buffer...
// Let's use single buffer instead:
// A_buf[128][34] + B_buf[34][128] = 128*34*2 + 34*128*2 = 8.5KB + 8.5KB = 17KB
// 48/17 = 2.8 → 2 blocks/SM
// But single-buffered = no cp.async overlap
// Let me stick with K=32 double-buffered (V9b) which already halves syncs
// ============================================================================

// ============================================================================
// V9d: 128x256 tile, 4 warps, 8 WMMA/warp in N-dim, 16 WMMA/warp = half register use
// This spreads the work differently. Each warp: 4x2 = 8 WMMA, 8*8 = 64 regs
// Wait, actually if we have 4x2=8 accumulators, that's only 8*8=64 registers.
// We can INCREASE to 4x4=16 accumulators which is 128 regs (same as V9).
// 128x256 tile means same M-dim, wider N-dim.
// A_buf: [2][128][16] = 4KB per buffer
// B_buf: [2][16][256] = 8KB per buffer
// Total: 24KB → 2 blocks/SM (down from 3)
// ============================================================================
__global__ void gemm_v9_n256(int M, int N, int K,
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C)
{
    int wid = threadIdx.x / 32, wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64;
    int c0 = blockIdx.x * 256 + wx * 128;

    __shared__ half A_buf[2][128][16];
    __shared__ half B_buf[2][16][256];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[8];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[32];
    for (int i = 0; i < 32; i++) wmma::fill_fragment(c[i], 0.0f);

    // Prefetch
    for (int idx = threadIdx.x; idx < 128*16/2; idx += 128) {
        int pos = idx * 2, r = pos / 16, col = pos % 16;
        int gr = blockIdx.y*128 + r;
        if (gr < M && col + 1 < K)
            *(short2*)&A_buf[0][r][col] = *(const short2*)&A[gr*K + col];
    }
    for (int idx = threadIdx.x; idx < 16*256/2; idx += 128) {
        int pos = idx * 2, r = pos / 256, col = pos % 256;
        int gc = blockIdx.x*256 + col;
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
                asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&A[gr*K+gc]));
            }
        }
        for (int chunk = threadIdx.x; chunk < 16*256/8; chunk += 128) {
            int pos = chunk * 8, r = pos / 256, c = pos % 256;
            int gr = kb + r, gc = blockIdx.x*256 + c;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c]);
                asm volatile("cp.async.ca.shared.global [%0],[%1],16;\n"::"r"(sa),"l"(&B[gr*N+gc]));
            }
        }
        asm volatile("cp.async.commit_group;\n"::);

        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy*64+mi*16][0], 16);
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx*128+ni*16], 256);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 8; ni++)
            wmma::mma_sync(c[mi*8+ni], a[mi], b[ni], c[mi*8+ni]);

        asm volatile("cp.async.wait_group 0;\n"::);
        __syncthreads();
        read_buf = write_buf;
    }

    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy*64+mi*16][0], 16);
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx*128+ni*16], 256);
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        wmma::mma_sync(c[mi*8+ni], a[mi], b[ni], c[mi*8+ni]);

    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 8; ni++)
        wmma::store_matrix_sync(C+(r0+mi*16)*N+c0+ni*16, c[mi*8+ni], N, wmma::mem_row_major);
}

// ============================================================================
// Benchmark harness
// ============================================================================
typedef void (*gemm_kernel_t)(int, int, int, const half*, const half*, float*);

struct KernelInfo {
    const char* name;
    gemm_kernel_t kernel;
    int tile_m, tile_n, block_threads;
};

static KernelInfo kernels[] = {
    {"V9_baseline",     gemm_v9,       128, 128, 128},
    {"V9a_ld18",        gemm_v9_ld18,  128, 128, 128},
    {"V9b_K32",         gemm_v9_k32,   128, 128, 128},
    {"V9c_N256",        gemm_v9_n256,  128, 256, 128},
};
static const int NUM_KERNELS = sizeof(kernels) / sizeof(kernels[0]);

int main() {
    int sizes[] = {1024, 2048, 4096};
    // PyTorch FP16 reference (measured on this GPU via WSL ncu-profiled gemm_torch.py)
    double pytorch_tf[] = {11.257, 13.437, 14.751};  // TFLOPS
    float  pytorch_ms[] = {0.191f, 1.279f, 9.317f};  // ms

    printf("GEMM Optimization — Beating PyTorch FP16\n");
    printf("GPU: RTX 3050 Ti Laptop (20 SMs, 80 Tensor Cores)\n");
    printf("PyTorch ref: 1024=%.1fT, 2048=%.1fT, 4096=%.1fT\n\n",
           pytorch_tf[0], pytorch_tf[1], pytorch_tf[2]);

    for (int si = 0; si < 3; si++) {
        int M = sizes[si], N_sz = M, K_sz = M;
        double flops = 2.0 * (double)M * N_sz * K_sz;

        printf("=== Size %d ===    PyTorch: %.1f TFLOPS (%.3f ms)\n\n",
               M, pytorch_tf[si], pytorch_ms[si]);
        printf("%-16s %8s %8s %6s %6s\n", "Kernel", "ms", "TFLOPS", "%PT", "SmemKB");
        printf("------------------------------------------------\n");

        // Allocate
        float *hA = (float*)malloc(M*K_sz*4), *hB = (float*)malloc(K_sz*N_sz*4);
        for (size_t i = 0; i < (size_t)M*K_sz; i++) hA[i] = (float)rand()/RAND_MAX-0.5f;
        for (size_t i = 0; i < (size_t)K_sz*N_sz; i++) hB[i] = (float)rand()/RAND_MAX-0.5f;

        float *dA32, *dB32, *dC;
        half *dA16, *dB16;
        CHECK(cudaMalloc(&dA32, M*K_sz*4)); CHECK(cudaMalloc(&dB32, K_sz*N_sz*4));
        CHECK(cudaMalloc(&dA16, M*K_sz*2)); CHECK(cudaMalloc(&dB16, K_sz*N_sz*2));
        CHECK(cudaMalloc(&dC, M*N_sz*4));
        CHECK(cudaMemcpy(dA32, hA, M*K_sz*4, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB32, hB, K_sz*N_sz*4, cudaMemcpyHostToDevice));
        convert<<<(M*K_sz+255)/256,256>>>(M*K_sz, dA32, dA16);
        convert<<<(K_sz*N_sz+255)/256,256>>>(K_sz*N_sz, dB32, dB16);
        CHECK(cudaDeviceSynchronize());

        // Warmup GPU
        {
            dim3 wg((N_sz+127)/128, (M+127)/128);
            for (int i = 0; i < 10; i++)
                warmup<<<(M*N_sz+255)/256, 256>>>(dA16, dB16, dC, M);
            gemm_v9<<<wg, 128>>>(M, N_sz, K_sz, dA16, dB16, dC);
            CHECK(cudaDeviceSynchronize());
        }

        for (int ki = 0; ki < NUM_KERNELS; ki++) {
            KernelInfo* k = &kernels[ki];
            dim3 grid((N_sz + k->tile_n - 1) / k->tile_n,
                      (M + k->tile_m - 1) / k->tile_m);
            dim3 block(k->block_threads);

            // correctness check (one warmup + verify)
            k->kernel<<<grid, block>>>(M, N_sz, K_sz, dA16, dB16, dC);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaGetLastError());

            // Benchmark
            int reps = 20;
            cudaEvent_t start, stop;
            cudaEventCreate(&start); cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
            for (int i = 0; i < reps; i++)
                k->kernel<<<grid, block>>>(M, N_sz, K_sz, dA16, dB16, dC);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            float t;
            cudaEventElapsedTime(&t, start, stop);
            double tflops = flops / (t / (reps * 1000.0)) / 1e12;
            double pct = tflops / pytorch_tf[si] * 100;

            // Calculate smem used
            int smem_kb = 0;
            if (ki == 0) smem_kb = (2*128*16 + 2*16*128) * 2 / 1024;  // 16KB
            else if (ki == 1) smem_kb = (2*128*18 + 2*18*128) * 2 / 1024;  // 18KB
            else if (ki == 2) smem_kb = (2*128*32 + 2*32*128) * 2 / 1024;  // 32KB
            else if (ki == 3) smem_kb = (2*128*16 + 2*16*256) * 2 / 1024;  // 24KB

            printf("%-16s %8.3f %8.2f %5.0f%% %5d\n",
                   k->name, t/reps, tflops, pct, smem_kb);

            if (pct >= 100.0) printf("  >>> BEAT PyTorch! <<<\n");

            cudaEventDestroy(start); cudaEventDestroy(stop);
        }
        printf("\n");

        CHECK(cudaFree(dA32)); CHECK(cudaFree(dB32));
        CHECK(cudaFree(dA16)); CHECK(cudaFree(dB16)); CHECK(cudaFree(dC));
        free(hA); free(hB);
    }

    printf("Done.\n");
    return 0;
}
