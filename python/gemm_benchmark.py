"""
RTX 5090 (Blackwell SM120) Optimized GEMM Benchmark
Compare hand-optimized CUDA kernels vs PyTorch/cuBLAS across diverse shapes.

Key features:
- WMMA (Tensor Cores) with 16x16x16 tiles
- cp.async for async global→shared memory transfers
- Multi-stage software pipelining (double buffering)
- Multi-warp cooperation within threadblock
- Specialized kernels for different shape categories
"""

import torch
import torch.utils.cpp_extension
import time
import sys
from typing import List, Tuple, Dict
import json

# ============================================================
# Optimized GEMM Kernel (CUDA)
# ============================================================

gemm_cuda_source = r"""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

// ============================================================
// GEMM Kernel: 128x128 tile, WMMA 16x16x16, double buffering
// Optimized for Blackwell SM120
// ============================================================
// Grid: (N/128, M/128)
// Block: 128 threads (4 warps arranged as 2x2)
//
// Each block computes C[128x128] = A[128xK] x B[Kx128]
// Each warp computes C[64x64] sub-tile
// Uses double-buffered shared memory with cp.async
// ============================================================
__global__ void gemm_wmma_128x128x16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    // Warp mapping: 4 warps in 2x2 grid
    int wid = threadIdx.x / 32;
    int wy = wid / 2, wx = wid % 2;
    int r0 = blockIdx.y * 128 + wy * 64;
    int c0 = blockIdx.x * 128 + wx * 64;

    // Double-buffered shared memory
    __shared__ half A_buf[2][128][16];
    __shared__ half B_buf[2][16][128];

    // WMMA fragments
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    // ============ Prefetch buf[0] for kb=0 ============
    // Load A tile: 128x16 elements
    for (int idx = threadIdx.x; idx < 128 * 16 / 2; idx += 128) {
        int pos = idx * 2;
        int r = pos / 16;
        int c = pos % 16;
        int gr = blockIdx.y * 128 + r;
        int gc = c;
        if (gr < M && gc + 1 < K) {
            *(short2*)&A_buf[0][r][c] = *(const short2*)&A[gr * K + gc];
        }
    }
    // Load B tile: 16x128 elements
    for (int idx = threadIdx.x; idx < 16 * 128 / 2; idx += 128) {
        int pos = idx * 2;
        int r = pos / 128;
        int c = pos % 128;
        int gr = r;
        int gc = blockIdx.x * 128 + c;
        if (gr < K && gc + 1 < N) {
            *(short2*)&B_buf[0][r][c] = *(const short2*)&B[gr * N + gc];
        }
    }
    __syncthreads();

    int read_buf = 0;

    // ============ Main K-loop with double buffering ============
    for (int kb = 16; kb < K; kb += 16) {
        int write_buf = 1 - read_buf;

        // cp.async: load next K-tile into write_buf (async, runs in background!)
        // Load A: 128x16 @ column kb
        for (int chunk = threadIdx.x; chunk < 128 * 16 / 8; chunk += 128) {
            int pos = chunk * 8;
            int r = pos / 16, c = pos % 16;
            int gr = blockIdx.y * 128 + r, gc = kb + c;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c]);
                const half* ga = &A[gr * K + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(ga));
            }
        }
        // Load B: 16x128 @ row kb
        for (int chunk = threadIdx.x; chunk < 16 * 128 / 8; chunk += 128) {
            int pos = chunk * 8;
            int r = pos / 128, c = pos % 128;
            int gr = kb + r, gc = blockIdx.x * 128 + c;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c]);
                const half* gb = &B[gr * N + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(gb));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);

        // WMMA compute from read_buf (runs concurrently with cp.async!)
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

        // Wait for cp.async to finish, sync all threads, swap buffers
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        read_buf = write_buf;
    }

    // ============ Last tile (already in read_buf, no more loading) ============
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

    // ============ Store results with bounds checking ============
    #pragma unroll
    for (int mi = 0; mi < 4; mi++) {
    #pragma unroll
    for (int ni = 0; ni < 4; ni++) {
        int frag_r = r0 + mi * 16;
        int frag_c = c0 + ni * 16;
        // Only store if 16x16 fragment is fully within output bounds
        if (frag_r + 16 <= M && frag_c + 16 <= N) {
            wmma::store_matrix_sync(
                &C[frag_r * N + frag_c],
                c[mi * 4 + ni], N, wmma::mem_row_major
            );
        }
    }}
}


// ============================================================
// Kernel 2: 256x128 tile for larger M (more rows = more reuse)
// ============================================================
__global__ void gemm_wmma_256x128x16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    // 8 warps in 4x2 grid: 256 rows x 128 cols
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

    // Prefetch buf[0]
    for (int idx = threadIdx.x; idx < 256 * 16 / 2; idx += 256) {
        int pos = idx * 2;
        int r = pos / 16, c = pos % 16;
        int gr = blockIdx.y * 256 + r, gc = c;
        if (gr < M && gc + 1 < K)
            *(short2*)&A_buf[0][r][c] = *(const short2*)&A[gr * K + gc];
    }
    for (int idx = threadIdx.x; idx < 16 * 128 / 2; idx += 256) {
        int pos = idx * 2;
        int r = pos / 128, c = pos % 128;
        int gr = r, gc = blockIdx.x * 128 + c;
        if (gr < K && gc + 1 < N)
            *(short2*)&B_buf[0][r][c] = *(const short2*)&B[gr * N + gc];
    }
    __syncthreads();

    int read_buf = 0;

    for (int kb = 16; kb < K; kb += 16) {
        int write_buf = 1 - read_buf;

        // cp.async load next tile
        for (int chunk = threadIdx.x; chunk < 256 * 16 / 8; chunk += 256) {
            int pos = chunk * 8;
            int r = pos / 16, c = pos % 16;
            int gr = blockIdx.y * 256 + r, gc = kb + c;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c]);
                const half* ga = &A[gr * K + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(ga));
            }
        }
        for (int chunk = threadIdx.x; chunk < 16 * 128 / 8; chunk += 256) {
            int pos = chunk * 8;
            int r = pos / 128, c = pos % 128;
            int gr = kb + r, gc = blockIdx.x * 128 + c;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c]);
                const half* gb = &B[gr * N + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(gb));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);

        // Compute from read_buf
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
        int frag_r = r0 + mi * 16;
        int frag_c = c0 + ni * 16;
        if (frag_r + 16 <= M && frag_c + 16 <= N) {
            wmma::store_matrix_sync(
                &C[frag_r * N + frag_c],
                c[mi * 4 + ni], N, wmma::mem_row_major
            );
        }
    }}
}


// ============================================================
// Kernel 3: Optimized for small M (LLM decode, M <= 64)
// Uses multiple blocks along N dimension, fewer warps per block
// ============================================================
// For small M, the key bottleneck is memory bandwidth and launch overhead.
// Strategy: process multiple N-tiles in a single block to increase work per block.
__global__ void gemm_wmma_smallM_64x256x16(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    int M, int N, int K
) {
    // Single warp per row segment, process wider N (256)
    int wid = threadIdx.x / 32;
    int wy = wid / 4, wx = wid % 4;  // 2x4 warp grid: 128 rows x 256 cols
    int r0 = blockIdx.y * 128 + wy * 64;
    int c0 = blockIdx.x * 256 + wx * 64;

    __shared__ half A_buf[2][128][16];
    __shared__ half B_buf[2][16][256];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b[4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) wmma::fill_fragment(c[i], 0.0f);

    // Prefetch
    for (int idx = threadIdx.x; idx < 128 * 16 / 2; idx += 256) {
        int pos = idx * 2;
        int r = pos / 16, c = pos % 16;
        int gr = blockIdx.y * 128 + r, gc = c;
        if (gr < M && gc + 1 < K)
            *(short2*)&A_buf[0][r][c] = *(const short2*)&A[gr * K + gc];
    }
    for (int idx = threadIdx.x; idx < 16 * 256 / 2; idx += 256) {
        int pos = idx * 2;
        int r = pos / 256, c = pos % 256;
        int gr = r, gc = blockIdx.x * 256 + c;
        if (gr < K && gc + 1 < N)
            *(short2*)&B_buf[0][r][c] = *(const short2*)&B[gr * N + gc];
    }
    __syncthreads();

    int read_buf = 0;

    for (int kb = 16; kb < K; kb += 16) {
        int write_buf = 1 - read_buf;

        for (int chunk = threadIdx.x; chunk < 128 * 16 / 8; chunk += 256) {
            int pos = chunk * 8;
            int r = pos / 16, c = pos % 16;
            int gr = blockIdx.y * 128 + r, gc = kb + c;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c]);
                const half* ga = &A[gr * K + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(ga));
            }
        }
        for (int chunk = threadIdx.x; chunk < 16 * 256 / 8; chunk += 256) {
            int pos = chunk * 8;
            int r = pos / 256, c = pos % 256;
            int gr = kb + r, gc = blockIdx.x * 256 + c;
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
            wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx * 64 + ni * 16], 256);
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
        wmma::load_matrix_sync(b[ni], &B_buf[read_buf][0][wx * 64 + ni * 16], 256);
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
        int frag_r = r0 + mi * 16;
        int frag_c = c0 + ni * 16;
        if (frag_r + 16 <= M && frag_c + 16 <= N) {
            wmma::store_matrix_sync(
                &C[frag_r * N + frag_c],
                c[mi * 4 + ni], N, wmma::mem_row_major
            );
        }
    }}
}


// ============================================================
// Launcher functions (C++ wrappers called from Python)
// ============================================================

torch::Tensor gemm_128x128(torch::Tensor A, torch::Tensor B) {
    // A: [M, K] fp16, B: [K, N] fp16 -> C: [M, N] fp16
    TORCH_CHECK(A.dtype() == torch::kHalf, "A must be fp16");
    TORCH_CHECK(B.dtype() == torch::kHalf, "B must be fp16");
    TORCH_CHECK(A.device().is_cuda(), "A must be CUDA");
    TORCH_CHECK(B.device().is_cuda(), "B must be CUDA");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dim mismatch");

    auto C = torch::empty({M, N}, A.options());

    dim3 block(128);
    dim3 grid((N + 127) / 128, (M + 127) / 128);

    if (M >= 128) {
        // Use 256x128 kernel for larger M
        block = dim3(256);
        grid = dim3((N + 127) / 128, (M + 255) / 256);
        gemm_wmma_256x128x16<<<grid, block>>>(
            (half*)A.data_ptr(), (half*)B.data_ptr(),
            (half*)C.data_ptr(), M, N, K
        );
    } else if (M <= 64 && N >= 256) {
        // Use small-M kernel with wider N tiles
        block = dim3(256);
        grid = dim3((N + 255) / 256, (M + 127) / 128);
        gemm_wmma_smallM_64x256x16<<<grid, block>>>(
            (half*)A.data_ptr(), (half*)B.data_ptr(),
            (half*)C.data_ptr(), M, N, K
        );
    } else {
        gemm_wmma_128x128x16<<<grid, block>>>(
            (half*)A.data_ptr(), (half*)B.data_ptr(),
            (half*)C.data_ptr(), M, N, K
        );
    }

    return C;
}

// Auto-select best kernel based on shape
torch::Tensor gemm_optimized(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);

    if (M <= 64 && N >= 256 && K >= 2048) {
        // LLM decode pattern: small M, large N and K
        // Use wider N-tile kernel for better utilization
        const int num_threads = 256;
        dim3 block(num_threads);
        dim3 grid((N + 255) / 256, (M + 127) / 128);

        auto C = torch::empty({M, N}, A.options());
        gemm_wmma_smallM_64x256x16<<<grid, block>>>(
            (half*)A.data_ptr(), (half*)B.data_ptr(),
            (half*)C.data_ptr(), M, N, K
        );
        return C;
    } else if (M >= 256) {
        // Large M: use 256-row tile for more A reuse
        const int num_threads = 256;
        dim3 block(num_threads);
        dim3 grid((N + 127) / 128, (M + 255) / 256);

        auto C = torch::empty({M, N}, A.options());
        gemm_wmma_256x128x16<<<grid, block>>>(
            (half*)A.data_ptr(), (half*)B.data_ptr(),
            (half*)C.data_ptr(), M, N, K
        );
        return C;
    } else {
        // Default: 128x128 tile
        const int num_threads = 128;
        dim3 block(num_threads);
        dim3 grid((N + 127) / 128, (M + 127) / 128);

        auto C = torch::empty({M, N}, A.options());
        gemm_wmma_128x128x16<<<grid, block>>>(
            (half*)A.data_ptr(), (half*)B.data_ptr(),
            (half*)C.data_ptr(), M, N, K
        );
        return C;
    }
}
"""

# ============================================================
# Compile CUDA kernels (with caching)
# ============================================================

print("Compiling CUDA GEMM kernels...")
try:
    gemm_module = torch.utils.cpp_extension.load_inline(
        name="gemm_optimized",
        cpp_sources="",
        cuda_sources=gemm_cuda_source,
        functions=["gemm_128x128", "gemm_optimized"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-maxrregcount=128",
        ],
        with_cuda=True,
        verbose=False,
    )
    print("Kernels compiled successfully!")
except Exception as e:
    print(f"Compilation failed: {e}")
    sys.exit(1)


# ============================================================
# Benchmark
# ============================================================

def benchmark_gemm(name: str, fn, A, B, warmup=5, iters=50):
    """Benchmark a GEMM function, return (avg_time_ms, tflops)"""
    M, K = A.shape
    N = B.shape[1]
    flops = 2.0 * M * N * K  # multiply-add = 2 ops

    # Warmup
    for _ in range(warmup):
        fn(A, B)
    torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(iters):
        fn(A, B)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / iters) * 1000
    tflops = (flops * iters) / elapsed / 1e12
    return avg_time_ms, tflops


def run_benchmarks():
    """Run comprehensive GEMM benchmarks"""

    # Test cases covering all shape categories
    test_cases = [
        # (M, N, K, category)
        # ====== Square matrices ======
        (512, 512, 512, "square-small"),
        (1024, 1024, 1024, "square-medium"),
        (2048, 2048, 2048, "square-large"),
        (4096, 4096, 4096, "square-xl"),

        # ====== Large K (inner product bottleneck) ======
        (256, 256, 8192, "largeK-smallMN"),
        (512, 512, 16384, "largeK-medMN"),
        (1024, 1024, 32768, "largeK-xl"),

        # ====== LLM Decode (small M, large N&K) ======
        (1, 4096, 4096, "llm-decode-1"),
        (1, 8192, 8192, "llm-decode-large"),
        (4, 4096, 4096, "llm-decode-batch4"),
        (8, 4096, 4096, "llm-decode-batch8"),
        (16, 4096, 4096, "llm-decode-batch16"),
        (32, 4096, 4096, "llm-decode-batch32"),
        (64, 4096, 4096, "llm-decode-batch64"),
        (128, 4096, 4096, "llm-decode-batch128"),

        # ====== LLM Prefill (medium M) ======
        (512, 4096, 4096, "llm-prefill-512"),
        (1024, 4096, 4096, "llm-prefill-1024"),
        (2048, 4096, 4096, "llm-prefill-2048"),

        # ====== Flat (large M&N, small K) ======
        (4096, 4096, 64, "flat-smallK"),
        (8192, 8192, 128, "flat-medK"),
        (4096, 1024, 256, "flat-wide"),

        # ====== Unaligned sizes ======
        (1000, 2000, 500, "unaligned-1"),
        (1023, 2047, 511, "unaligned-2"),
        (17, 31, 63, "unaligned-small"),
        (127, 129, 255, "unaligned-med"),
        (100, 100, 100, "unaligned-square"),

        # ====== Real model shapes ======
        # GPT-3 style (hidden=12288)
        (1, 12288, 12288, "gpt3-decode"),
        (128, 12288, 12288, "gpt3-prefill"),
        # LLaMA 70B (hidden=8192, intermediate=28672)
        (1, 8192, 28672, "llama70b-MLP-decode"),
        (128, 8192, 28672, "llama70b-MLP-prefill"),
        # Qwen3 0.6B (hidden=1024, intermediate=3072)
        (1, 1024, 3072, "qwen3-MLP-decode"),
        (128, 1024, 3072, "qwen3-MLP-prefill"),
        (1, 1024, 1024, "qwen3-attn-decode"),
        (128, 1024, 1024, "qwen3-attn-prefill"),
    ]

    device = torch.device("cuda")
    results = []

    print(f"\n{'='*90}")
    print(f"GEMM Benchmark: Custom CUDA Kernel vs PyTorch (cuBLAS)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    print(f"{'='*90}")
    print(f"{'Shape (M,N,K)':<25} {'Category':<22} {'Custom(ms)':<12} {'PyTorch(ms)':<12} {'Speedup':<10} {'Custom TFLOPS':<14}")
    print(f"{'-'*90}")

    total_custom_time = 0
    total_torch_time = 0

    for M, N, K, category in test_cases:
        # Create fp16 inputs
        A = torch.randn(M, K, dtype=torch.float16, device=device)
        B = torch.randn(K, N, dtype=torch.float16, device=device)

        # Our kernel (using auto-dispatched optimized kernel)
        # Need to handle M < 128 and N < 256 for small matrices
        def custom_gemm(a, b):
            return gemm_module.gemm_optimized(a, b)

        # PyTorch reference
        def torch_gemm(a, b):
            return torch.matmul(a, b)

        try:
            # Verify correctness first
            custom_result = custom_gemm(A, B)
            torch_result = torch_gemm(A, B)

            # Check correctness (allow small fp16 errors)
            max_diff = (custom_result.float() - torch_result.float()).abs().max().item()
            rel_diff = max_diff / torch_result.float().abs().mean().item() if torch_result.float().abs().mean().item() > 0 else 0

            if rel_diff > 0.01:  # 1% relative error threshold
                print(f"  WARNING: Large numerical difference for {M}x{N}x{K}: max_diff={max_diff:.6f}, rel_diff={rel_diff:.4f}")

            # Benchmark
            warmup_iters = 2 if M * N * K > 100000000 else 5  # fewer warmups for very large
            timed_iters = 10 if M * N * K > 100000000 else 50

            custom_time, custom_tflops = benchmark_gemm("custom", custom_gemm, A, B, warmup=warmup_iters, iters=timed_iters)
            torch_time, torch_tflops = benchmark_gemm("torch", torch_gemm, A, B, warmup=warmup_iters, iters=timed_iters)

            speedup = torch_time / custom_time
            total_custom_time += custom_time
            total_torch_time += torch_time

            shape_str = f"({M},{N},{K})"
            print(f"{shape_str:<25} {category:<22} {custom_time:<12.4f} {torch_time:<12.4f} {speedup:<10.2f}x {custom_tflops:<14.2f}")

            results.append({
                "shape": (M, N, K),
                "category": category,
                "custom_ms": custom_time,
                "torch_ms": torch_time,
                "speedup": speedup,
                "custom_tflops": custom_tflops,
                "torch_tflops": torch_tflops,
            })

        except Exception as e:
            print(f"{f'({M},{N},{K})':<25} {category:<22} ERROR: {str(e)[:50]}")

    print(f"{'-'*90}")
    overall_speedup = total_torch_time / total_custom_time if total_custom_time > 0 else 0
    print(f"OVERALL: Custom={total_custom_time:.2f}ms, PyTorch={total_torch_time:.2f}ms, Speedup={overall_speedup:.2f}x")

    # Category summary
    print(f"\n{'='*90}")
    print(f"Summary by Category")
    print(f"{'='*90}")

    from collections import defaultdict
    cat_results = defaultdict(list)
    for r in results:
        # Group by category prefix (before the dash)
        cat_prefix = r['category'].rsplit('-', 1)[0] if '-' in r['category'] else r['category']
        cat_results[cat_prefix].append(r['speedup'])

    for cat, speeds in sorted(cat_results.items()):
        avg_speedup = sum(speeds) / len(speeds)
        min_speedup = min(speeds)
        max_speedup = max(speeds)
        print(f"  {cat:<25}: avg={avg_speedup:.2f}x, min={min_speedup:.2f}x, max={max_speedup:.2f}x, n={len(speeds)}")

    return results


if __name__ == "__main__":
    results = run_benchmarks()

    # Save results
    with open("gemm_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to gemm_benchmark_results.json")
