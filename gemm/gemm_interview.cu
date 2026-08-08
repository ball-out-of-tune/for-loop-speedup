/**
 * Interview-ready SGEMM: 32x32 tiling + 2x2 register blocking
 *
 * Key concepts demonstrated:
 *   1. Shared memory tiling (data reuse)
 *   2. Register blocking (less smem reads per FMA)
 *   3. Cooperative loading (all threads fill the tile)
 *
 * nvcc -o gemm_interview gemm_interview.cu -arch=sm_86 && gemm_interview
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK(c) do { cudaError_t e=c; if(e!=cudaSuccess) \
    {fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(e));exit(1);}} while(0)

#define TILE 32

__global__ void sgemm(int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    // This thread computes 4 outputs: (row0,col0), (row0,col1),
    //                                 (row1,col0), (row1,col1)
    int row0 = blockIdx.y * TILE + ty;
    int row1 = row0 + 16;
    int col0 = blockIdx.x * TILE + tx;
    int col1 = col0 + 16;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float c00 = 0.0f, c01 = 0.0f;
    float c10 = 0.0f, c11 = 0.0f;

    for (int k_block = 0; k_block < K; k_block += TILE) {
        // Cooperative load: each thread loads 4 A elements + 4 B elements
        int ka0 = k_block + tx;
        int ka1 = k_block + tx + 16;
        int kb0 = k_block + ty;
        int kb1 = k_block + ty + 16;

        As[ty][tx]             = (row0 < M && ka0 < K) ? A[row0 * K + ka0] : 0.0f;
        As[ty][tx + 16]        = (row0 < M && ka1 < K) ? A[row0 * K + ka1] : 0.0f;
        As[ty + 16][tx]        = (row1 < M && ka0 < K) ? A[row1 * K + ka0] : 0.0f;
        As[ty + 16][tx + 16]   = (row1 < M && ka1 < K) ? A[row1 * K + ka1] : 0.0f;

        Bs[ty][tx]             = (kb0 < K && col0 < N) ? B[kb0 * N + col0] : 0.0f;
        Bs[ty][tx + 16]        = (kb0 < K && col1 < N) ? B[kb0 * N + col1] : 0.0f;
        Bs[ty + 16][tx]        = (kb1 < K && col0 < N) ? B[kb1 * N + col0] : 0.0f;
        Bs[ty + 16][tx + 16]   = (kb1 < K && col1 < N) ? B[kb1 * N + col1] : 0.0f;

        __syncthreads();

        // Inner product: read from shared memory, accumulate
        for (int k = 0; k < TILE; k++) {
            float a0 = As[ty][k];
            float a1 = As[ty + 16][k];
            float b0 = Bs[k][tx];
            float b1 = Bs[k][tx + 16];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        __syncthreads();
    }

    // Write back results
    if (row0 < M && col0 < N) C[row0 * N + col0] = c00;
    if (row0 < M && col1 < N) C[row0 * N + col1] = c01;
    if (row1 < M && col0 < N) C[row1 * N + col0] = c10;
    if (row1 < M && col1 < N) C[row1 * N + col1] = c11;
}

__global__ void gemm_my_own(int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    int thread_row_id = threadIdx.y;
    int thread_col_id = threadIdx.x;

    int cur_row = blockIdx.y * TILE + thread_row_id;
    int cur_col = blockIdx.x * TILE + thread_col_id;
    int cur_row_next = cur_row + TILE / 2;
    int cur_col_next = cur_col + TILE / 2;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    float s00 = 0, s01 = 0, s02 = 0, s03 = 0;

    for (int k_block = 0; k_block < K; k_block += TILE) {
        int ka0 = k_block + thread_col_id;
        int ka1 = k_block + thread_col_id + TILE / 2;
        int kb0 = k_block + thread_row_id;
        int kb1 = k_block + thread_row_id + TILE / 2;

        // Cooperative load A: each thread loads 4 elements
        As[thread_row_id][thread_col_id]
            = (cur_row < M && ka0 < K) ? A[cur_row * K + ka0] : 0.0f;
        As[thread_row_id][thread_col_id + TILE / 2]
            = (cur_row < M && ka1 < K) ? A[cur_row * K + ka1] : 0.0f;
        As[thread_row_id + TILE / 2][thread_col_id]
            = (cur_row_next < M && ka0 < K) ? A[cur_row_next * K + ka0] : 0.0f;
        As[thread_row_id + TILE / 2][thread_col_id + TILE / 2]
            = (cur_row_next < M && ka1 < K) ? A[cur_row_next * K + ka1] : 0.0f;

        // Cooperative load B: each thread loads 4 elements
        Bs[thread_row_id][thread_col_id]
            = (kb0 < K && cur_col < N) ? B[kb0 * N + cur_col] : 0.0f;
        Bs[thread_row_id][thread_col_id + TILE / 2]
            = (kb0 < K && cur_col_next < N) ? B[kb0 * N + cur_col_next] : 0.0f;
        Bs[thread_row_id + TILE / 2][thread_col_id]
            = (kb1 < K && cur_col < N) ? B[kb1 * N + cur_col] : 0.0f;
        Bs[thread_row_id + TILE / 2][thread_col_id + TILE / 2]
            = (kb1 < K && cur_col_next < N) ? B[kb1 * N + cur_col_next] : 0.0f;

        __syncthreads();

        // Inner product: 2x2 register blocking
        for (int k = 0; k < TILE; k++) {
            float a0 = As[thread_row_id][k];
            float a1 = As[thread_row_id + TILE / 2][k];
            float b0 = Bs[k][thread_col_id];
            float b1 = Bs[k][thread_col_id + TILE / 2];

            s00 += a0 * b0;
            s01 += a0 * b1;
            s02 += a1 * b0;
            s03 += a1 * b1;
        }

        __syncthreads();
    }

    // Write back with boundary checks
    if (cur_row < M && cur_col < N)           C[cur_row * N + cur_col] = s00;
    if (cur_row < M && cur_col_next < N)      C[cur_row * N + cur_col_next] = s01;
    if (cur_row_next < M && cur_col < N)      C[cur_row_next * N + cur_col] = s02;
    if (cur_row_next < M && cur_col_next < N) C[cur_row_next * N + cur_col_next] = s03;
}

// Verify against CPU reference, return max error
float verify_gemm(int M, int N, int K, const float* hA, const float* hB, const float* hC) {
    float max_err = 0.0f;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += hA[i * K + k] * hB[k * N + j];
            float err = fabsf(hC[i * N + j] - sum);
            if (err > max_err) max_err = err;
        }
    }
    return max_err;
}

// Run a single GEMM kernel and return elapsed ms
float bench_kernel(void (*kernel)(int,int,int,const float*,const float*,float*),
                   int M, int N, int K,
                   float *dA, float *dB, float *dC,
                   dim3 grid, dim3 block, int warmup, int iters) {
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < warmup; i++) {
        kernel<<<grid, block>>>(M, N, K, dA, dB, dC);
    }
    CHECK(cudaDeviceSynchronize());

    // Timed runs
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        kernel<<<grid, block>>>(M, N, K, dA, dB, dC);
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));

    return ms / iters;
}

int main() {
    // Test sizes — includes irregular ones that need boundary checks
    struct { int M, N, K; const char* desc; } configs[] = {
        { 512,  512,  512,  "512x512x512  (all 32x)"    },
        {1024, 1024, 1024,  "1024x1024x1024 (all 32x)"  },
        { 520,  520,  520,  "520x520x520  (NOT 32x)"    },
        {1000, 1000, 1000,  "1000x1000x1000 (NOT 32x)"  },
        {2048, 2048, 2048,  "2048x2048x2048 (all 32x)"  },
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    dim3 block(16, 16);
    int warmup = 5, iters = 20;

    printf("%-30s %10s %10s %10s %9s %9s\n",
           "Config", "sgemm(ms)", "my_own(ms)", "Speedup",
           "sgemm_OK", "my_OK");
    printf("%s\n", "-------------------------------------------------------------------------------");

    for (int c = 0; c < n_configs; c++) {
        int M = configs[c].M, N = configs[c].N, K = configs[c].K;

        size_t bytes_A = M * K * sizeof(float);
        size_t bytes_B = K * N * sizeof(float);
        size_t bytes_C = M * N * sizeof(float);

        float *hA = (float*)malloc(bytes_A);
        float *hB = (float*)malloc(bytes_B);
        float *hC_sgemm  = (float*)malloc(bytes_C);
        float *hC_myown  = (float*)malloc(bytes_C);

        for (int i = 0; i < M * K; i++) hA[i] = (float)rand() / RAND_MAX - 0.5f;
        for (int i = 0; i < K * N; i++) hB[i] = (float)rand() / RAND_MAX - 0.5f;

        float *dA, *dB, *dC;
        CHECK(cudaMalloc(&dA, bytes_A));
        CHECK(cudaMalloc(&dB, bytes_B));
        CHECK(cudaMalloc(&dC, bytes_C));

        CHECK(cudaMemcpy(dA, hA, bytes_A, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(dB, hB, bytes_B, cudaMemcpyHostToDevice));

        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

        // Benchmark sgemm
        float t_sgemm = bench_kernel(sgemm, M, N, K, dA, dB, dC, grid, block, warmup, iters);
        CHECK(cudaMemcpy(hC_sgemm, dC, bytes_C, cudaMemcpyDeviceToHost));
        float err_sgemm = verify_gemm(M, N, K, hA, hB, hC_sgemm);

        // Benchmark gemm_my_own
        float t_myown = bench_kernel(gemm_my_own, M, N, K, dA, dB, dC, grid, block, warmup, iters);
        CHECK(cudaMemcpy(hC_myown, dC, bytes_C, cudaMemcpyDeviceToHost));
        float err_myown = verify_gemm(M, N, K, hA, hB, hC_myown);

        float speedup = t_sgemm / t_myown;

        printf("%-30s %10.4f %10.4f %9.2fx  %9s %9s\n",
               configs[c].desc, t_sgemm, t_myown, speedup,
               err_sgemm  < 1e-3f ? "PASS" : "FAIL",
               err_myown  < 1e-3f ? "PASS" : "FAIL");

        cudaFree(dA); cudaFree(dB); cudaFree(dC);
        free(hA); free(hB); free(hC_sgemm); free(hC_myown);
    }

    printf("\nDone.\n");
    return 0;
}
