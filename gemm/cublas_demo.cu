/**
 * cuBLAS demo: 三种调用方式
 *   1. cublasSgemm — 经典 FP32 (CUDA core)
 *   2. cublasGemmEx — 精确控制 compute type (可以指定 TF32)
 *   3. cublasLtMatmul — cuBLASLt, PyTorch 实际使用的路径
 *
 * 编译: nvcc -o cublas_demo cublas_demo.cu -lcublas
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#define M 4096
#define N 4096
#define K 4096
#define WARMUP 5
#define RUNS 20

#define CHECK_CUDA(call) do { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(e)); exit(1); \
    } \
} while(0)

#define CHECK_CUBLAS(call) do { \
    cublasStatus_t s = call; \
    if (s != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__, s); \
        exit(1); \
    } \
} while(0)

float ms(cudaEvent_t s, cudaEvent_t e) {
    float t;
    cudaEventElapsedTime(&t, s, e);
    return t;
}

int main() {
    float *hA = (float*)malloc(M * K * sizeof(float));
    float *hB = (float*)malloc(K * N * sizeof(float));
    float *hC = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * K; i++) hA[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; i++) hB[i] = (float)rand() / RAND_MAX - 0.5f;

    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc(&dA, M * K * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dB, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dC, M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float alpha = 1.0f, beta = 0.0f;

    // ================================================================
    // Method 1: cublasSgemm (经典 FP32, CUDA core)
    // ================================================================
    printf("===== Method 1: cublasSgemm (FP32 compute, CUDA core) =====\n");
    {
        cublasHandle_t handle;
        CHECK_CUBLAS(cublasCreate(&handle));

        // warmup
        for (int i = 0; i < WARMUP; i++)
            CHECK_CUBLAS(cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,          // cuBLAS 是列优先! A=K×M, B=N×K → C=N×M
                &alpha, dB, N,    // B^T 作为 "A"  (N×K, leading dim = N)
                dA, K,            // A^T 作为 "B"  (K×M, leading dim = K)
                &beta, dC, N));
        CHECK_CUDA(cudaDeviceSynchronize());

        float best = 1e9f;
        for (int r = 0; r < RUNS; r++) {
            CHECK_CUDA(cudaEventRecord(start, 0));
            CHECK_CUBLAS(cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
            CHECK_CUDA(cudaEventRecord(stop, 0));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float t = ms(start, stop);
            if (t < best) best = t;
        }

        double flops = 2.0 * M * N * K;
        double tflops = flops / (best / 1000.0) / 1e12;
        printf("  Best: %.3f ms  →  %.3f TFLOPS\n", best, tflops);
        printf("  Note: cuBLAS is column-major, so C = A*B becomes C^T col-major\n");
        CHECK_CUBLAS(cublasDestroy(handle));
    }

    // ================================================================
    // Method 2: cublasGemmEx (精确控制 compute type)
    // ================================================================
    printf("\n===== Method 2: cublasGemmEx (explicit compute type) =====\n");
    {
        cublasHandle_t handle;
        CHECK_CUBLAS(cublasCreate(&handle));

        // TF32 compute: CUBLAS_COMPUTE_32F_FAST_TF32
        // FP32 compute:  CUBLAS_COMPUTE_32F (pedantic, slower)
        // FP16 compute:  CUBLAS_COMPUTE_32F_FAST_16F (input FP16, compute FP32)

        cublasComputeType_t types[] = {
            CUBLAS_COMPUTE_32F,              // 纯 FP32, CUDA core
            CUBLAS_COMPUTE_32F_FAST_TF32,    // TF32 on Tensor Core ← allow_tf32=True
        };
        const char *names[] = {"FP32 (CUDA core)", "TF32 (Tensor Core) ← allow_tf32=True"};

        for (int t = 0; t < 2; t++) {
            cublasSetMathMode(handle, types[t] == CUBLAS_COMPUTE_32F ?
                              CUBLAS_DEFAULT_MATH : CUBLAS_TF32_TENSOR_OP_MATH);

            for (int i = 0; i < WARMUP; i++)
                CHECK_CUBLAS(cublasGemmEx(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    dB, CUDA_R_32F, N,
                    dA, CUDA_R_32F, K,
                    &beta, dC, CUDA_R_32F, N,
                    types[t], CUBLAS_GEMM_DEFAULT));
            CHECK_CUDA(cudaDeviceSynchronize());

            float best = 1e9f;
            for (int r = 0; r < RUNS; r++) {
                CHECK_CUDA(cudaEventRecord(start, 0));
                CHECK_CUBLAS(cublasGemmEx(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    dB, CUDA_R_32F, N,
                    dA, CUDA_R_32F, K,
                    &beta, dC, CUDA_R_32F, N,
                    types[t], CUBLAS_GEMM_DEFAULT));
                CHECK_CUDA(cudaEventRecord(stop, 0));
                CHECK_CUDA(cudaEventSynchronize(stop));
                float t = ms(start, stop);
                if (t < best) best = t;
            }

            double tflops = 2.0 * M * N * K / (best / 1000.0) / 1e12;
            printf("  %-40s  %.3f ms  →  %.3f TFLOPS\n", names[t], best, tflops);
        }
        CHECK_CUBLAS(cublasDestroy(handle));
    }

    // ================================================================
    // Method 3: cublasLtMatmul (PyTorch 实际用的高级 API)
    // ================================================================
    printf("\n===== Method 3: cublasLtMatmul (what PyTorch uses) =====\n");
    {
        cublasLtHandle_t handle;
        CHECK_CUBLAS(cublasLtCreate(&handle));

        cublasLtMatmulDesc_t matmulDesc;
        cublasLtMatrixLayout_t layoutA, layoutB, layoutC;

        CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F, K, M, K));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F, K, N, K));
        CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, M, N, M));

        // Try with TF32
        cublasLtMatmulHeuristicResult_t heuristicResult = {};
        cublasLtMatmulPreference_t pref;
        CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
        int returnedResults = 0;
        CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc,
            layoutA, layoutB, layoutC, layoutC, pref, 1, &heuristicResult, &returnedResults));

        for (int i = 0; i < WARMUP; i++)
            CHECK_CUBLAS(cublasLtMatmul(handle, matmulDesc,
                &alpha, dA, layoutA, dB, layoutB, &beta, dC, layoutC, dC, layoutC,
                &heuristicResult.algo, nullptr, 0, nullptr));
        CHECK_CUDA(cudaDeviceSynchronize());

        float best = 1e9f;
        for (int r = 0; r < RUNS; r++) {
            CHECK_CUDA(cudaEventRecord(start, 0));
            CHECK_CUBLAS(cublasLtMatmul(handle, matmulDesc,
                &alpha, dA, layoutA, dB, layoutB, &beta, dC, layoutC, dC, layoutC,
                &heuristicResult.algo, nullptr, 0, nullptr));
            CHECK_CUDA(cudaEventRecord(stop, 0));
            CHECK_CUDA(cudaEventSynchronize(stop));
            float t = ms(start, stop);
            if (t < best) best = t;
        }
        double tflops = 2.0 * M * N * K / (best / 1000.0) / 1e12;
        printf("  cublasLt (default, heuristic picks TF32): %.3f ms  →  %.3f TFLOPS\n", best, tflops);

        CHECK_CUBLAS(cublasLtMatmulPreferenceDestroy(pref));
        CHECK_CUBLAS(cublasLtMatmulDescDestroy(matmulDesc));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutA));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutB));
        CHECK_CUBLAS(cublasLtMatrixLayoutDestroy(layoutC));
        CHECK_CUBLAS(cublasLtDestroy(handle));
    }

    printf("\n===== Summary =====\n");
    printf("  cublasSgemm:           FP32 input, FP32 compute → CUDA Cores\n");
    printf("  cublasGemmEx + TF32:   FP32 input, TF32 compute → Tensor Cores (2x throughput)\n");
    printf("  cublasLtMatmul:        Auto-heuristic, PyTorch uses this internally\n");
    printf("  allow_tf32=True  =  cuBLAS may use CUBLAS_COMPUTE_32F_FAST_TF32\n");
    printf("  allow_tf32=False =  cuBLAS must use CUBLAS_COMPUTE_32F (pedantic FP32)\n");

    CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
    free(hA); free(hB); free(hC);
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));
    return 0;
}
