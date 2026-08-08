/**
 * Standalone GEMM benchmark: Custom kernel vs cuBLAS
 * Compile on RTX 5090:
 *   /usr/local/cuda-12.8/bin/nvcc -O3 -o gemm_test gemm_kernel.cu gemm_bench.cpp \
 *     -arch=sm_120 -lcublas --use_fast_math -maxrregcount=128 -std=c++17
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

// ============================================================
// Helper: wall-clock time
// ============================================================
double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Kernel 1: 128x128 tile, 128 threads (4 warps)
// ============================================================
__global__ void gemm_128x128(
    const half* __restrict__ A, const half* __restrict__ B,
    half* __restrict__ C, int M, int N, int K
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

    // ===== Prefetch K-tile 0 =====
    for (int idx = threadIdx.x; idx < 128 * 16 / 2; idx += 128) {
        int pos = idx * 2, r = pos / 16, c = pos % 16;
        int gr = blockIdx.y * 128 + r, gc = c;
        if (gr < M && gc + 1 < K)
            *(short2*)&A_buf[0][r][c] = *(const short2*)&A[gr * K + gc];
    }
    for (int idx = threadIdx.x; idx < 16 * 128 / 2; idx += 128) {
        int pos = idx * 2, r = pos / 128, c = pos % 128;
        int gr = r, gc = blockIdx.x * 128 + c;
        if (gr < K && gc + 1 < N)
            *(short2*)&B_buf[0][r][c] = *(const short2*)&B[gr * N + gc];
    }
    __syncthreads();

    int read_buf = 0;

    // ===== Main K-loop =====
    for (int kb = 16; kb < K; kb += 16) {
        int write_buf = 1 - read_buf;

        // cp.async load A: 128x16 @ column kb
        for (int chunk = threadIdx.x; chunk < 128 * 16 / 8; chunk += 128) {
            int pos = chunk * 8, r = pos / 16, c = pos % 16;
            int gr = blockIdx.y * 128 + r, gc = kb + c;
            if (gr < M && gc + 7 < K) {
                unsigned sa = __cvta_generic_to_shared(&A_buf[write_buf][r][c]);
                const half* ga = &A[gr * K + gc];
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(ga));
            }
        }
        // cp.async load B: 16x128 @ row kb
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

        // WMMA compute from read_buf
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

    // ===== Last tile =====
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

    // ===== Store =====
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
// Kernel 2: 256x128 tile optimized for large M
// ============================================================
__global__ void gemm_256x128(
    const half* __restrict__ A, const half* __restrict__ B,
    half* __restrict__ C, int M, int N, int K
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
// Benchmark runner
// ============================================================

typedef void (*gemm_kernel_t)(const half*, const half*, half*, int, int, int);

typedef struct {
    int M, N, K;
    const char* category;
    const char* description;
} TestCase;


void run_gemm_benchmark(
    const char* name,
    gemm_kernel_t kernel,
    const half* A, const half* B,
    half* C, int M, int N, int K,
    int warmup, int iters,
    double* out_time_ms, double* out_tflops
) {
    dim3 block(128);
    dim3 grid((N + 127) / 128, (M + 127) / 128);

    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel<<<grid, block>>>(A, B, C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    double start = get_time_ms();
    for (int i = 0; i < iters; i++) {
        kernel<<<grid, block>>>(A, B, C, M, N, K);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = get_time_ms() - start;

    double avg_ms = elapsed / iters;
    double flops = 2.0 * M * N * K;
    double tflops = (flops * iters) / (elapsed / 1000.0) / 1e12;

    *out_time_ms = avg_ms;
    *out_tflops = tflops;
}


void run_cublas_benchmark(
    cublasHandle_t handle,
    const half* A, const half* B,
    half* C, int M, int N, int K,
    int warmup, int iters,
    double* out_time_ms, double* out_tflops
) {
    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < warmup; i++) {
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha, B, CUDA_R_16F, N,
            A, CUDA_R_16F, K,
            &beta, C, CUDA_R_16F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    double start = get_time_ms();
    for (int i = 0; i < iters; i++) {
        CUBLAS_CHECK(cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K,
            &alpha, B, CUDA_R_16F, N,
            A, CUDA_R_16F, K,
            &beta, C, CUDA_R_16F, N,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = get_time_ms() - start;

    double avg_ms = elapsed / iters;
    double flops = 2.0 * M * N * K;
    double tflops = (flops * iters) / (elapsed / 1000.0) / 1e12;

    *out_time_ms = avg_ms;
    *out_tflops = tflops;
}


void run_kernel_256x128(
    const half* A, const half* B, half* C, int M, int N, int K,
    int warmup, int iters, double* out_time, double* out_tflops
) {
    dim3 block(256);
    dim3 grid((N + 127) / 128, (M + 255) / 256);

    for (int i = 0; i < warmup; i++)
        gemm_256x128<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    double start = get_time_ms();
    for (int i = 0; i < iters; i++)
        gemm_256x128<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = get_time_ms() - start;

    *out_time = elapsed / iters;
    double flops = 2.0 * M * N * K;
    *out_tflops = (flops * iters) / (elapsed / 1000.0) / 1e12;
}


// ============================================================
// Main
// ============================================================
int main() {
    // Get GPU info
    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    printf("============================================================\n");
    printf("GEMM Benchmark: Custom WMMA Kernel vs cuBLAS\n");
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("SMs: %d, Max Threads/Block: %d\n", prop.multiProcessorCount, prop.maxThreadsPerBlock);
    printf("Shared Memory/Block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("============================================================\n\n");

    // Init cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    TestCase tests[] = {
        // ===== Square =====
        {512, 512, 512, "square-small", "512x512x512"},
        {1024, 1024, 1024, "square-medium", "1024x1024x1024"},
        {2048, 2048, 2048, "square-large", "2048x2048x2048"},
        {4096, 4096, 4096, "square-xl", "4096x4096x4096"},

        // ===== Large K =====
        {256, 256, 8192, "largeK-smallMN", "256x256x8192"},
        {512, 512, 16384, "largeK-medMN", "512x512x16384"},
        {1024, 1024, 32768, "largeK-xl", "1024x1024x32768"},

        // ===== LLM Decode (small M) =====
        {1, 4096, 4096, "llm-decode-1", "1x4096x4096"},
        {1, 8192, 8192, "llm-decode-large", "1x8192x8192"},
        {4, 4096, 4096, "llm-decode-b4", "4x4096x4096"},
        {8, 4096, 4096, "llm-decode-b8", "8x4096x4096"},
        {16, 4096, 4096, "llm-decode-b16", "16x4096x4096"},
        {32, 4096, 4096, "llm-decode-b32", "32x4096x4096"},
        {64, 4096, 4096, "llm-decode-b64", "64x4096x4096"},
        {128, 4096, 4096, "llm-decode-b128", "128x4096x4096"},

        // ===== LLM Prefill =====
        {512, 4096, 4096, "llm-prefill-512", "512x4096x4096"},
        {1024, 4096, 4096, "llm-prefill-1K", "1024x4096x4096"},
        {2048, 4096, 4096, "llm-prefill-2K", "2048x4096x4096"},

        // ===== Flat (small K) =====
        {4096, 4096, 64, "flat-smallK", "4096x4096x64"},
        {8192, 8192, 128, "flat-medK", "8192x8192x128"},
        {4096, 1024, 256, "flat-wide", "4096x1024x256"},

        // ===== Unaligned =====
        {1000, 2000, 500, "unaligned-1", "1000x2000x500"},
        {1023, 2047, 511, "unaligned-2", "1023x2047x511"},
        {100, 100, 100, "unaligned-sm", "100x100x100"},

        // ===== Real models =====
        {1, 12288, 12288, "gpt3-decode", "1x12288x12288"},
        {128, 12288, 12288, "gpt3-prefill", "128x12288x12288"},
        {1, 8192, 28672, "llama-MLP-d", "1x8192x28672"},
        {128, 8192, 28672, "llama-MLP-p", "128x8192x28672"},
        {1, 1024, 3072, "qwen3-MLP-d", "1x1024x3072"},
        {128, 1024, 3072, "qwen3-MLP-p", "128x1024x3072"},
        {1, 1024, 1024, "qwen3-attn-d", "1x1024x1024"},
        {128, 1024, 1024, "qwen3-attn-p", "128x1024x1024"},

        // Sentinel
        {0, 0, 0, NULL, NULL},
    };

    printf("%-30s %-20s %12s %12s %12s %8s\n",
           "Shape", "Category", "Custom(ms)", "cuBLAS(ms)", "Custom TF", "Speedup");
    printf("%-30s %-20s %12s %12s %12s %8s\n",
           "-----", "--------", "---------", "----------", "---------", "-------");

    int num_tests = sizeof(tests) / sizeof(tests[0]) - 1;
    double total_custom = 0, total_cublas = 0;

    for (int i = 0; i < num_tests; i++) {
        int M = tests[i].M, N = tests[i].N, K = tests[i].K;

        // Allocate
        half *d_A, *d_B, *d_C_custom, *d_C_cublas;
        CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C_custom, M * N * sizeof(half)));
        CUDA_CHECK(cudaMalloc(&d_C_cublas, M * N * sizeof(half)));

        // Random init
        half* h_A = (half*)malloc(M * K * sizeof(half));
        half* h_B = (half*)malloc(K * N * sizeof(half));
        for (int j = 0; j < M * K; j++) h_A[j] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 2.0f);
        for (int j = 0; j < K * N; j++) h_B[j] = __float2half(((float)rand() / RAND_MAX - 0.5f) * 2.0f);
        CUDA_CHECK(cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice));
        free(h_A); free(h_B);

        // Warmup / iters based on size
        int warmup = 3, iters = 20;
        if ((long long)M * N * K > 50000000LL) { warmup = 2; iters = 10; }
        if ((long long)M * N * K > 500000000LL) { warmup = 1; iters = 5; }

        // Run custom kernel (auto-select variant)
        double custom_ms, custom_tflops;
        if (M >= 256) {
            run_kernel_256x128(d_A, d_B, d_C_custom, M, N, K, warmup, iters, &custom_ms, &custom_tflops);
        } else {
            run_gemm_benchmark("128x128", gemm_128x128, d_A, d_B, d_C_custom, M, N, K, warmup, iters, &custom_ms, &custom_tflops);
        }

        // Run cuBLAS
        double cublas_ms, cublas_tflops;
        run_cublas_benchmark(handle, d_A, d_B, d_C_cublas, M, N, K, warmup, iters, &cublas_ms, &cublas_tflops);

        double speedup = cublas_ms / custom_ms;
        total_custom += custom_ms;
        total_cublas += cublas_ms;

        printf("%-30s %-20s %12.4f %12.4f %12.2f %7.2fx\n",
               tests[i].description, tests[i].category,
               custom_ms, cublas_ms, custom_tflops, speedup);

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C_custom));
        CUDA_CHECK(cudaFree(d_C_cublas));
    }

    printf("%-30s %-20s %12s %12s %12s %8s\n",
           "-----", "--------", "---------", "----------", "---------", "-------");
    printf("%-30s %-20s %12.2f %12.2f %12s %7.2fx\n",
           "TOTAL", "", total_custom, total_cublas, "", total_cublas / total_custom);

    CUBLAS_CHECK(cublasDestroy(handle));
    printf("\nDone!\n");
    return 0;
}
