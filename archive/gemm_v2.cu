/**
 * V2: Improved GEMM kernel for RTX 5090 (Blackwell sm_120)
 * Key improvements:
 * - 512 threads per block (16 warps) for better occupancy
 * - Split into more flexible tile sizes
 * - Better warp scheduling
 * - Proper aligned cp.async
 */
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// Large tile kernel: better for square/large matrices
// 256x128 tile with 256 threads (8 warps, 4x2 layout)
__global__ void gemm_v2_256x128(
    const half* __restrict__ A, const half* __restrict__ B,
    float* __restrict__ C, int M, int N, int K
) {
    // 8 warps in 4(row) x 2(col) layout
    int wid = threadIdx.x / 32;
    int wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 256 + wy * 64;
    int c0 = blockIdx.x * 128 + wx * 64;

    __shared__ half A_buf[2][256][16];
    __shared__ half B_buf[2][16][128];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    // Prefetch K-tile 0 using vectorized loads (short2 = 4 bytes)
    for (int idx = threadIdx.x; idx < 256 * 16 / 2; idx += 256) {
        int pos = idx * 2, r = pos / 16, c2 = pos % 16;
        int gr = blockIdx.y * 256 + r, gc = c2;
        if (gr < M && gc + 1 < K)
            *(short2*)&A_buf[0][r][c2] = *(const short2*)&A[gr * K + gc];
    }
    for (int idx = threadIdx.x; idx < 16 * 128 / 2; idx += 256) {
        int pos = idx * 2, r = pos / 128, c2 = pos % 128;
        int gr = r, gc = blockIdx.x * 128 + c2;
        if (gr < K && gc + 1 < N)
            *(short2*)&B_buf[0][r][c2] = *(const short2*)&B[gr * N + gc];
    }
    __syncthreads();

    int read_buf = 0;
    for (int kb = 16; kb < K; kb += 16) {
        int write_buf = 1 - read_buf;

        // cp.async: 16-byte aligned copies (8x half)
        for (int chunk = threadIdx.x; chunk < 256 * 16 / 8; chunk += 256) {
            int pos = chunk * 8, r = pos / 16, c8 = pos % 16;
            int gr = blockIdx.y * 256 + r, gc = kb + c8;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c8]);
                const half* ga = &A[gr * K + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(ga));
            }
        }
        for (int chunk = threadIdx.x; chunk < 16 * 128 / 8; chunk += 256) {
            int pos = chunk * 8, r = pos / 128, c8 = pos % 128;
            int gr = kb + r, gc = blockIdx.x * 128 + c8;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c8]);
                const half* gb = &B[gr * N + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(gb));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);

        // WMMA compute
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy * 64 + mi * 16][0], 16);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx * 64 + ni * 16], 128);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(c[mi * 4 + ni], a[mi], b[ni], c[mi * 4 + ni]);

        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        read_buf = write_buf;
    }

    // Last tile
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy * 64 + mi * 16][0], 16);
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx * 64 + ni * 16], 128);
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::mma_sync(c[mi * 4 + ni], a[mi], b[ni], c[mi * 4 + ni]);

    // Store
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        int fr = r0 + mi * 16, fc = c0 + ni * 16;
        if (fr + 16 <= M && fc + 16 <= N)
            wmma::store_matrix_sync(&C[fr * N + fc], c[mi * 4 + ni], N, wmma::mem_row_major);
    }}
}


// Split-K kernel for small M (LLM decode)
// Multiple blocks work on different K ranges, accumulate via atomicAdd
__global__ void gemm_splitk_smallM(
    const half* __restrict__ A, const half* __restrict__ B,
    float* __restrict__ C, int M, int N, int K,
    int k_start, int k_end  // this block's K range
) {
    int wid = threadIdx.x / 32;
    int wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64;
    int c0 = blockIdx.x * 128 + wx * 64;

    __shared__ half A_buf[2][128][16];
    __shared__ half B_buf[2][16][128];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    // Local K range for this split
    int local_K = k_end - k_start;
    int kb = k_start;

    // Prefetch first tile
    for (int idx = threadIdx.x; idx < 128 * 16 / 2; idx += 128) {
        int pos = idx * 2, r = pos / 16, c2 = pos % 16;
        int gr = blockIdx.y * 128 + r, gc = kb + c2;
        if (gr < M && gc + 1 < k_end)
            *(short2*)&A_buf[0][r][c2] = *(const short2*)&A[gr * K + gc];
    }
    for (int idx = threadIdx.x; idx < 16 * 128 / 2; idx += 128) {
        int pos = idx * 2, r = pos / 128, c2 = pos % 128;
        int gr = kb + r, gc = blockIdx.x * 128 + c2;
        if (gr < k_end && gc + 1 < N)
            *(short2*)&B_buf[0][r][c2] = *(const short2*)&B[gr * N + gc];
    }
    __syncthreads();

    int read_buf = 0;
    kb += 16;

    for (; kb < k_end; kb += 16) {
        int write_buf = 1 - read_buf;

        for (int chunk = threadIdx.x; chunk < 128 * 16 / 8; chunk += 128) {
            int pos = chunk * 8, r = pos / 16, c8 = pos % 16;
            int gr = blockIdx.y * 128 + r, gc = kb + c8;
            if (gr < M && gc + 7 < k_end) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c8]);
                const half* ga = &A[gr * K + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(ga));
            }
        }
        for (int chunk = threadIdx.x; chunk < 16 * 128 / 8; chunk += 128) {
            int pos = chunk * 8, r = pos / 128, c8 = pos % 128;
            int gr = kb + r, gc = blockIdx.x * 128 + c8;
            if (gr < k_end && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c8]);
                const half* gb = &B[gr * N + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(gb));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);

        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
            wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy * 64 + mi * 16][0], 16);
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx * 64 + ni * 16], 128);
        #pragma unroll
        for (int mi = 0; mi < 4; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            wmma::mma_sync(c[mi * 4 + ni], a[mi], b[ni], c[mi * 4 + ni]);

        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        read_buf = write_buf;
    }

    // Last tile
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
        wmma::load_matrix_sync(a[mi], &A_buf[read_buf][wy * 64 + mi * 16][0], 16);
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx * 64 + ni * 16], 128);
    #pragma unroll
    for (int mi = 0; mi < 4; mi++)
    #pragma unroll
    for (int ni = 0; ni < 4; ni++)
        wmma::mma_sync(c[mi * 4 + ni], a[mi], b[ni], c[mi * 4 + ni]);

    // Atomic add to output (split-K accumulation)
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        int fr = r0 + mi * 16, fc = c0 + ni * 16;
        if (fr + 16 <= M && fc + 16 <= N) {
            // Store to temp shared memory first, then atomicAdd
            __shared__ float temp_buf[16][16];
            wmma::store_matrix_sync(&temp_buf[0][0], c[mi * 4 + ni], 16, wmma::mem_row_major);
            __syncthreads();
            for (int i = threadIdx.x; i < 256; i += 128) {
                int tr = i / 16, tc = i % 16;
                if (fr + tr < M && fc + tc < N)
                    atomicAdd(&C[(fr + tr) * N + (fc + tc)], temp_buf[tr][tc]);
            }
            __syncthreads();
        }
    }}
}
