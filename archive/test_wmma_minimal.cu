#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
using namespace nvcuda;

__global__ void test_wmma(float* C, int N) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    wmma::store_matrix_sync(C, c, N, wmma::mem_row_major);
}

int main() {
    float* d_C;
    cudaMalloc(&d_C, 256 * sizeof(float));
    test_wmma<<<1, 32>>>(d_C, 16);
    cudaDeviceSynchronize();
    printf("OK\n");
    return 0;
}
