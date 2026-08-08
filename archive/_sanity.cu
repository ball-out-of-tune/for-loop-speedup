
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void add_one(float *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}

int main() {
    printf("GPU test starting...
");
    int n = 1024;
    float *dx;
    cudaError_t err = cudaMalloc(&dx, n * sizeof(float));
    if (err != cudaSuccess) { printf("cudaMalloc failed: %s
", cudaGetErrorString(err)); return 1; }
    cudaMemset(dx, 0, n * sizeof(float));

    add_one<<<(n+255)/256, 256>>>(dx, n);
    err = cudaGetLastError();
    if (err != cudaSuccess) { printf("kernel launch failed: %s
", cudaGetErrorString(err)); return 1; }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("sync failed: %s
", cudaGetErrorString(err)); return 1; }

    float host[1];
    cudaMemcpy(host, dx, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Result: %f (expected 1.0)
", host[0]);
    printf("GPU working OK
");
    cudaFree(dx);
    return 0;
}
