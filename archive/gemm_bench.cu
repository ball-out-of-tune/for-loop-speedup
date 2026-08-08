/**
 * Standalone GEMM benchmark: Custom WMMA kernel vs cuBLAS
 * RTX 5090 (Blackwell sm_120) optimized
 *
 * Compile: /usr/local/cuda-12.8/bin/nvcc -O3 -o gemm_test gemm_bench.cu
 *          -arch=sm_120 -lcublas --use_fast_math -maxrregcount=128 -std=c++17
 */

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

using namespace nvcuda;

double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

#define CUDA_CHECK(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { fprintf(stderr, "CUDA err %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t e = call; \
    if (e != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS err %s:%d: %d\n", __FILE__, __LINE__, e); exit(1); } \
} while(0)

// ============================================================
// Kernel: 128x128 tile, WMMA 16x16x16, float output
// ============================================================
__global__ void gemm_wmma_128x128(
    const half* __restrict__ A, const half* __restrict__ B,
    float* __restrict__ C, int M, int N, int K
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

    // Prefetch A & B @ K-tile 0
    for (int idx = threadIdx.x; idx < 128 * 16 / 2; idx += 128) {
        int pos = idx * 2, r = pos / 16, c2 = pos % 16;
        int gr = blockIdx.y * 128 + r, gc = c2;
        if (gr < M && gc + 1 < K)
            *(short2*)&A_buf[0][r][c2] = *(const short2*)&A[gr * K + gc];
    }
    for (int idx = threadIdx.x; idx < 16 * 128 / 2; idx += 128) {
        int pos = idx * 2, r = pos / 128, c2 = pos % 128;
        int gr = r, gc = blockIdx.x * 128 + c2;
        if (gr < K && gc + 1 < N)
            *(short2*)&B_buf[0][r][c2] = *(const short2*)&B[gr * N + gc];
    }
    __syncthreads();

    int read_buf = 0;
    for (int kb = 16; kb < K; kb += 16) {
        int write_buf = 1 - read_buf;

        // cp.async load A
        for (int chunk = threadIdx.x; chunk < 128 * 16 / 8; chunk += 128) {
            int pos = chunk * 8, r = pos / 16, c8 = pos % 16;
            int gr = blockIdx.y * 128 + r, gc = kb + c8;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c8]);
                const half* ga = &A[gr * K + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(ga));
            }
        }
        // cp.async load B
        for (int chunk = threadIdx.x; chunk < 16 * 128 / 8; chunk += 128) {
            int pos = chunk * 8, r = pos / 128, c8 = pos % 128;
            int gr = kb + r, gc = blockIdx.x * 128 + c8;
            if (gr < K && gc + 7 < N) {
                unsigned sa = __cvta_generic_to_shared(&B_buf[write_buf][r][c8]);
                const half* gb = &B[gr * N + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(gb));
            }
        }
        asm volatile("cp.async.commit_group;\n" ::);

        // WMMA from read_buf
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

    // Store to float* output
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
// Kernel 2: 256x128 tile for large M
// ============================================================
__global__ void gemm_wmma_256x128(
    const half* __restrict__ A, const half* __restrict__ B,
    float* __restrict__ C, int M, int N, int K
) {
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
// Benchmark helpers
// ============================================================

double bench_custom(int M, int N, int K, int warmup, int iters) {
    // Generate half random data
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(((float)rand()/RAND_MAX - 0.5f) * 0.1f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half(((float)rand()/RAND_MAX - 0.5f) * 0.1f);

    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K*N*sizeof(half), cudaMemcpyHostToDevice));
    free(h_A); free(h_B);

    dim3 block, grid;
    if (M >= 256) {
        block = dim3(256);
        grid = dim3((N + 127) / 128, (M + 255) / 256);
    } else {
        block = dim3(128);
        grid = dim3((N + 127) / 128, (M + 127) / 128);
    }

    // Warmup
    for (int i = 0; i < warmup; i++) {
        if (M >= 256)
            gemm_wmma_256x128<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        else
            gemm_wmma_128x128<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed
    double start = get_time_ms();
    for (int i = 0; i < iters; i++) {
        if (M >= 256)
            gemm_wmma_256x128<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        else
            gemm_wmma_128x128<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = get_time_ms() - start;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return elapsed / iters;
}


double bench_cublas(cublasHandle_t handle, int M, int N, int K, int warmup, int iters) {
    half *h_A = (half*)malloc(M * K * sizeof(half));
    half *h_B = (half*)malloc(K * N * sizeof(half));
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(((float)rand()/RAND_MAX - 0.5f) * 0.1f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half(((float)rand()/RAND_MAX - 0.5f) * 0.1f);

    half *d_A, *d_B;
    float *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K*N*sizeof(half), cudaMemcpyHostToDevice));
    free(h_A); free(h_B);

    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            d_B, CUDA_R_16F, N,
            d_A, CUDA_R_16F, K,
            &beta,
            d_C, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed
    double start = get_time_ms();
    for (int i = 0; i < iters; i++) {
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            d_B, CUDA_R_16F, N,
            d_A, CUDA_R_16F, K,
            &beta,
            d_C, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = get_time_ms() - start;

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return elapsed / iters;
}


// ============================================================
// Main
// ============================================================
int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("============================================================\n");
    printf("  GEMM Benchmark: Custom WMMA Kernel vs cuBLAS\n");
    printf("  GPU: %s\n", prop.name);
    printf("  CC: %d.%d | SMs: %d | MaxThreads: %d | SharedMem: %zu KB\n",
           prop.major, prop.minor, prop.multiProcessorCount,
           prop.maxThreadsPerBlock, prop.sharedMemPerBlock / 1024);
    printf("============================================================\n\n");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Test cases
    struct { int M, N, K; const char* cat; const char* desc; } tests[] = {
        // Square
        {512, 512, 512, "square-small", "512^3"},
        {1024, 1024, 1024, "square-med", "1024^3"},
        {2048, 2048, 2048, "square-large", "2048^3"},
        {4096, 4096, 4096, "square-xl", "4096^3"},

        // Large K
        {256, 256, 8192, "largeK", "256x256x8192"},
        {512, 512, 16384, "largeK", "512x512x16384"},
        {1024, 1024, 32768, "largeK", "1024x1024x32768"},

        // LLM Decode
        {1, 4096, 4096, "decode", "1x4096x4096"},
        {1, 8192, 8192, "decode", "1x8192x8192"},
        {4, 4096, 4096, "decode", "4x4096x4096"},
        {8, 4096, 4096, "decode", "8x4096x4096"},
        {16, 4096, 4096, "decode", "16x4096x4096"},
        {32, 4096, 4096, "decode", "32x4096x4096"},
        {64, 4096, 4096, "decode", "64x4096x4096"},
        {128, 4096, 4096, "decode", "128x4096x4096"},

        // LLM Prefill
        {512, 4096, 4096, "prefill", "512x4096x4096"},
        {1024, 4096, 4096, "prefill", "1024x4096x4096"},
        {2048, 4096, 4096, "prefill", "2048x4096x4096"},

        // Flat
        {4096, 4096, 64, "flat", "4096x4096x64"},
        {8192, 8192, 128, "flat", "8192x8192x128"},

        // Unaligned
        {1000, 2000, 500, "unaligned", "1000x2000x500"},
        {1023, 2047, 511, "unaligned", "1023x2047x511"},
        {100, 100, 100, "unaligned", "100x100x100"},

        // Real models
        {1, 12288, 12288, "gpt3-decode", "1x12288x12288"},
        {128, 12288, 12288, "gpt3-prefill", "128x12288x12288"},
        {1, 8192, 28672, "llama-MLP-d", "1x8192x28672"},
        {128, 8192, 28672, "llama-MLP-p", "128x8192x28672"},
        {1, 1024, 3072, "qwen3-MLP-d", "1x1024x3072"},
        {128, 1024, 3072, "qwen3-MLP-p", "128x1024x3072"},
        {1, 1024, 1024, "qwen3-attn-d", "1x1024x1024"},
        {128, 1024, 1024, "qwen3-attn-p", "128x1024x1024"},
        {0, 0, 0, NULL, NULL},
    };

    printf("%-25s %-16s %10s %10s %10s %8s %10s\n",
           "Shape", "Category", "Custom(ms)", "cuBLAS(ms)", "Speedup",
           "CustomTF", "cuBLASTF");
    printf("%-25s %-16s %10s %10s %10s %8s %10s\n",
           "-----", "--------", "---------", "----------", "-------",
           "--------", "--------");

    int n = sizeof(tests)/sizeof(tests[0]) - 1;
    double tot_custom = 0, tot_cublas = 0;
    int wins = 0, losses = 0;

    for (int i = 0; i < n; i++) {
        int M = tests[i].M, N = tests[i].N, K = tests[i].K;

        int warmup = 3, iters = 20;
        long long flops = 2LL * M * N * K;
        if (flops > 50000000LL) { warmup = 2; iters = 10; }
        if (flops > 500000000LL) { warmup = 1; iters = 5; }

        double cus_ms = bench_custom(M, N, K, warmup, iters);
        double cublas_ms = bench_cublas(handle, M, N, K, warmup, iters);

        double tflops_cus = (2.0 * M * N * K) / (cus_ms / 1000.0) / 1e12;
        double tflops_cublas = (2.0 * M * N * K) / (cublas_ms / 1000.0) / 1e12;
        double speedup = cublas_ms / cus_ms;

        tot_custom += cus_ms;
        tot_cublas += cublas_ms;
        if (speedup > 1.0) wins++; else losses++;

        char desc[32];
        snprintf(desc, sizeof(desc), "%dx%dx%d", M, N, K);
        printf("%-25s %-16s %10.4f %10.4f %9.2fx %8.1f %10.1f\n",
               desc, tests[i].cat,
               cus_ms, cublas_ms, speedup, tflops_cus, tflops_cublas);
    }

    printf("%-25s %-16s %10s %10s %10s %8s %10s\n",
           "-----", "--------", "---------", "----------", "-------",
           "--------", "--------");
    printf("%-25s %-16s %10.2f %10.2f %9.2fx\n",
           "TOTAL", "", tot_custom, tot_cublas, tot_cublas/tot_custom);
    printf("\nWins: %d/%d (%.1f%%)\n", wins, n, 100.0*wins/n);

    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
