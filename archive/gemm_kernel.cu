/**
 * Standalone CUDA GEMM with WMMA + cp.async for RTX 5090 (Blackwell)
 * Compiled as shared library for Python ctypes loading.
 *
 * Compile on RTX 5090:
 *   nvcc -O3 -shared -Xcompiler -fPIC -o libgemm.so gemm_kernel.cu \
 *        -arch=sm_120 --use_fast_math -maxrregcount=128
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

// ============================================================
// Kernel 1: 128x128 tile (4 warps: 2x2 grid of 64x64 each)
// Best for: medium to large square matrices
// ============================================================
extern "C" __global__ void gemm_128x128(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
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

    // Prefetch A: 128x16 @ K-tile 0
    for (int idx = threadIdx.x; idx < 128 * 16 / 2; idx += 128) {
        int pos = idx * 2, r = pos / 16, c = pos % 16;
        int gr = blockIdx.y * 128 + r, gc = c;
        if (gr < M && gc + 1 < K)
            *(short2*)&A_buf[0][r][c] = *(const short2*)&A[gr * K + gc];
    }
    // Prefetch B: 16x128 @ K-tile 0
    for (int idx = threadIdx.x; idx < 16 * 128 / 2; idx += 128) {
        int pos = idx * 2, r = pos / 128, c = pos % 128;
        int gr = r, gc = blockIdx.x * 128 + c;
        if (gr < K && gc + 1 < N)
            *(short2*)&B_buf[0][r][c] = *(const short2*)&B[gr * N + gc];
    }
    __syncthreads();

    int read_buf = 0;

    // Main K-loop with double buffering + cp.async
    for (int kb = 16; kb < K; kb += 16) {
        int write_buf = 1 - read_buf;

        // cp.async: Load A tile (128x16) from column kb
        for (int chunk = threadIdx.x; chunk < 128 * 16 / 8; chunk += 128) {
            int pos = chunk * 8, r = pos / 16, c = pos % 16;
            int gr = blockIdx.y * 128 + r, gc = kb + c;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c]);
                const half* ga = &A[gr * K + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(ga));
            }
        }
        // cp.async: Load B tile (16x128) from row kb
        for (int chunk = threadIdx.x; chunk < 16 * 128 / 8; chunk += 128) {
            int pos = chunk * 8, r = pos / 128, c = pos % 128;
            int gr = kb + r, gc = blockIdx.x * 128 + c;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c]);
                const half* gb = &B[gr * N + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(gb));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);

        // WMMA compute from read_buf (concurrent with cp.async)
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

    // Last tile (already loaded in read_buf)
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

    // Store with bounds checking
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        int frag_r = r0 + mi * 16, frag_c = c0 + ni * 16;
        if (frag_r + 16 <= M && frag_c + 16 <= N) {
            wmma::store_matrix_sync(
                &C[frag_r * N + frag_c], c[mi * 4 + ni], N, wmma::mem_row_major);
        }
    }}
}


// ============================================================
// Kernel 2: Optimized for LLM decode (small M, large N & K)
// Wider N-tile (256 cols) for better utilization with small batch
// ============================================================
extern "C" __global__ void gemm_smallM(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    // 8 warps: 4x2 grid for 256 rows x 128 cols
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

    // Prefetch
    for (int idx = threadIdx.x; idx < 256 * 16 / 2; idx += 256) {
        int pos = idx * 2, r = pos / 16, c = pos % 16;
        int gr = blockIdx.y * 256 + r, gc = c;
        if (gr < M && gc + 1 < K)
            *(short2*)&A_buf[0][r][c] = *(const short2*)&A[gr * K + gc];
    }
    for (int idx = threadIdx.x; idx < 16 * 128 / 2; idx += 256) {
        int pos = idx * 2, r = pos / 128, c = pos % 128;
        int gr = r, gc = blockIdx.x * 128 + c;
        if (gr < K && gc + 1 < N)
            *(short2*)&B_buf[0][r][c] = *(const short2*)&B[gr * N + gc];
    }
    __syncthreads();

    int read_buf = 0;

    for (int kb = 16; kb < K; kb += 16) {
        int write_buf = 1 - read_buf;

        for (int chunk = threadIdx.x; chunk < 256 * 16 / 8; chunk += 256) {
            int pos = chunk * 8, r = pos / 16, c = pos % 16;
            int gr = blockIdx.y * 256 + r, gc = kb + c;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c]);
                const half* ga = &A[gr * K + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(ga));
            }
        }
        for (int chunk = threadIdx.x; chunk < 16 * 128 / 8; chunk += 256) {
            int pos = chunk * 8, r = pos / 128, c = pos % 128;
            int gr = kb + r, gc = blockIdx.x * 128 + c;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c]);
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

    // Store with bounds checking
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        int frag_r = r0 + mi * 16, frag_c = c0 + ni * 16;
        if (frag_r + 16 <= M && frag_c + 16 <= N) {
            wmma::store_matrix_sync(
                &C[frag_r * N + frag_c], c[mi * 4 + ni], N, wmma::mem_row_major);
        }
    }}
}


// ============================================================
// C-compatible wrapper for Python ctypes
// ============================================================
extern "C" {

int gemm_launch(
    const half* A, const half* B, half* C,
    int M, int N, int K,
    int kernel_type  // 0 = default, 1 = smallM
) {
    cudaError_t err;

    if (kernel_type == 1 || (M <= 128 && N >= 256 && K >= 2048)) {
        // Use 256x128 kernel for better utilization with small M
        dim3 block(256);
        dim3 grid((N + 127) / 128, (M + 255) / 256);
        gemm_smallM<<<grid, block>>>(A, B, C, M, N, K);
    } else {
        // Default 128x128 kernel
        dim3 block(128);
        dim3 grid((N + 127) / 128, (M + 127) / 128);
        gemm_128x128<<<grid, block>>>(A, B, C, M, N, K);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel sync error: %s\n", cudaGetErrorString(err));
        return -2;
    }

    return 0;
}

} // extern "C"
