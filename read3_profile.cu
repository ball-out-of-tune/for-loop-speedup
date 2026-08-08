/**
 * Minimal test for Nsight Compute profiling:
 * read-3-only kernel at hidden=4096 vs hidden=8192.
 *
 * Run:
 *   nvcc -o read3_profile read3_profile.cu
 *   ncu --metrics \
 *     l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
 *     l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,\
 *     lts__t_sectors_srcunit_tex_op_read.sum,\
 *     lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,\
 *     dram__sectors_read.sum ./read3_profile
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N_ROWS         120      // 1 full wave (20 SMs × 6 blocks/SM)
#define BLOCK_SIZE     256
#define HIDDEN_4096    4096
#define HIDDEN_8192    8192

__global__ void read_three_times(const float *x, float *sink, int n_rows, int hidden_size) {
    int row = blockIdx.x, tid = threadIdx.x, tt = blockDim.x, hs = hidden_size;
    const float *xr = x + row * hs;
    float dummy = 0.0f;
    for (int i = tid; i < hs; i += tt) dummy += xr[i];
    for (int i = tid; i < hs; i += tt) dummy += xr[i];
    for (int i = tid; i < hs; i += tt) dummy += xr[i];
    if (tid == 0) sink[row] = dummy;
}

void check(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) { fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err)); exit(1); }
}

int main() {
    printf("Nsight Compute: read-3-only profile\n");
    printf("  hidden=4096 (16 KB/row) vs hidden=8192 (32 KB/row)\n\n");

    int sizes[] = {HIDDEN_4096, HIDDEN_8192};
    const char *labels[] = {"4096", "8192"};

    for (int s = 0; s < 2; s++) {
        int hidden = sizes[s];
        long long total_elems = (long long)N_ROWS * hidden;
        long long total_bytes = total_elems * sizeof(float);

        float *hx = (float*)malloc(total_bytes);
        for (long long i = 0; i < total_elems; i++)
            hx[i] = (float)(rand()) / RAND_MAX - 0.5f;

        float *dx, *dsink;
        check(cudaMalloc(&dx, total_bytes), "dx alloc");
        check(cudaMalloc(&dsink, N_ROWS * sizeof(float)), "dsink alloc");
        check(cudaMemcpy(dx, hx, total_bytes, cudaMemcpyHostToDevice), "H2D");

        dim3 grid(N_ROWS), block(BLOCK_SIZE);

        // One launch, the profiler catches it
        printf("=== Launch hidden=%s (%d rows, %.1f KB/row) ===\n",
               labels[s], N_ROWS, hidden * 4.0f / 1024);

        read_three_times<<<grid, block>>>(dx, dsink, N_ROWS, hidden);
        check(cudaDeviceSynchronize(), "sync");

        // Verify
        float *hsink = (float*)malloc(N_ROWS * sizeof(float));
        check(cudaMemcpy(hsink, dsink, N_ROWS * sizeof(float), cudaMemcpyDeviceToHost), "D2H");
        printf("  sink[0] = %.2f (dummy value to prevent optimization)\n", hsink[0]);

        cudaFree(dx); cudaFree(dsink); free(hx); free(hsink);
    }

    printf("\nDone. Check ncu output above for cache metrics.\n");
    return 0;
}
