/**
 * Minimal test: can we use WMMA TF32 on this system?
 * nvcc -o test_tf32 test_tf32.cu -arch=sm_86
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Try 1: TF32 with precision tag
__global__ void test_tf32_precision(const float* A, const float* B, float* C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, wmma::precision::tf32, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, wmma::precision::tf32, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::load_matrix_sync(a, A, 16);
    wmma::load_matrix_sync(b, B, 16);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
}

// Try 2: float type (should use TF32 implicitly on sm_80+)
__global__ void test_tf32_float(const float* A, const float* B, float* C) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, float, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, float, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::load_matrix_sync(a, A, 16);
    wmma::load_matrix_sync(b, B, 16);
    wmma::mma_sync(c, a, b, c);
    wmma::store_matrix_sync(C, c, 16, wmma::mem_row_major);
}

int main() {
    printf("Testing WMMA TF32 compilation...\n");
    printf("If this compiles, TF32 tensor core is available.\n");

    float *dA, *dB, *dC;
    cudaMalloc(&dA, 256*4); cudaMalloc(&dB, 256*4); cudaMalloc(&dC, 256*4);

    test_tf32_precision<<<1,32>>>(dA, dB, dC);
    cudaDeviceSynchronize();
    printf("precision::tf32: OK\n");

    test_tf32_float<<<1,32>>>(dA, dB, dC);
    cudaDeviceSynchronize();
    printf("float: OK\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
