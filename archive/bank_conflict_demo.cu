/**
 * bank_conflict_demo.cu — concrete code to explain Shared Memory Bank Conflict
 *
 * Build: nvcc -o bank_conflict_demo bank_conflict_demo.cu
 *
 * ========================================
 * What is a Bank Conflict?
 * ========================================
 *
 * GPU Shared Memory is divided into 32 banks, each 4 bytes wide.
 * In one clock cycle, each bank can serve only ONE thread.
 *
 * - 32 threads accessing 32 different banks → no conflict, 1 transaction
 * - Multiple threads accessing same bank, different addresses → BANK CONFLICT
 * - Multiple threads accessing same bank, same address → broadcast (no conflict)
 *
 * Bank index (for 4-byte elements like float):
 *   bank_index = (byte_address / 4) % 32
 *
 * ========================================
 * Visualization (float array in shared memory):
 * ========================================
 *
 *  Index:  0   1   2   3  ...  31  32  33
 *  Bank:   0   1   2   3  ...  31   0   1
 *
 *  Thread 0 reads s[0]  → bank 0
 *  Thread 1 reads s[1]  → bank 1
 *  ...
 *  Thread 31 reads s[31] → bank 31  ✓  No conflict (stride=1)
 *
 *  Thread 0 reads s[0]  → bank 0
 *  Thread 1 reads s[32] → bank 0  ← Same bank as thread 0!
 *  Thread 2 reads s[64] → bank 0  ✗ 32-way bank conflict (stride=32)
 *
 * Common scenario: matrix transpose (column read after row write)
 * Solution: add padding to break the stride alignment (32 → 33)
 */

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024
#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define TILE_DIM 32
#define REPEAT 1024  // Repeat accesses to amplify bank conflict effect

// ============================================================
// Kernel 1: No Bank Conflict — stride=1 access
// Thread i reads s[i], consecutive threads → consecutive banks
// ============================================================
__global__ void no_bank_conflict(float *out) {
    __shared__ float s[BLOCK_SIZE];
    volatile float *vs = s;  // volatile forces real memory access each iteration

    int tid = threadIdx.x;
    s[tid] = (float)tid;
    __syncthreads();

    float val = 0.0f;
    for (int r = 0; r < REPEAT; r++) {
        val += vs[tid];  // stride=1: all 32 banks used → NO conflict
    }

    out[tid] = val;
}

// ============================================================
// Kernel 2: 2-way Bank Conflict — stride=2 access
// Thread i reads s[i*2], every two threads share a bank
// ============================================================
__global__ void two_way_bank_conflict(float *out) {
    __shared__ float s[BLOCK_SIZE * 2];
    volatile float *vs = s;

    int tid = threadIdx.x;
    s[tid * 2] = (float)tid;
    __syncthreads();

    // Thread 0 reads s[0] → bank 0. Thread 1 reads s[2] → bank 2.
    // Thread 16 reads s[32] → bank 0  ← same bank as thread 0!
    float val = 0.0f;
    for (int r = 0; r < REPEAT; r++) {
        val += vs[tid * 2];  // 2-way bank conflict
    }

    out[tid] = val;
}

// ============================================================
// Kernel 3: 32-way Bank Conflict (worst case) — stride=32 access
// All threads in a warp hit the SAME bank
// ============================================================
__global__ void worst_bank_conflict(float *out) {
    __shared__ float s[BLOCK_SIZE * 32];
    volatile float *vs = s;

    int tid = threadIdx.x;
    s[tid * 32] = (float)tid;
    __syncthreads();

    // Thread i reads s[i*32]: ALL map to bank 0 → 32-way conflict!
    float val = 0.0f;
    for (int r = 0; r < REPEAT; r++) {
        val += vs[tid * 32];  // 32-way bank conflict
    }

    out[tid] = val;
}

// ============================================================
// Kernel 4: Column access (classic matrix transpose problem)
// Thread (x,y) reads tile[x][y] — column-major after row-major write
// Row stride = 32 = multiple of 32 → ALL same bank! 32-way conflict!
// ============================================================
__global__ void column_access_conflict(float *in, float *out) {
    __shared__ float tile[TILE_DIM][TILE_DIM];  // 32x32 tile

    int x = threadIdx.x % TILE_DIM;
    int y = threadIdx.x / TILE_DIM;

    // Load: thread (x,y) writes tile[y][x], contiguous in x → no conflict
    tile[y][x] = in[y * TILE_DIM + x];
    __syncthreads();

    // TRANSPOSE READ (column-major): tile[x][y]
    // Since each row is 32 floats wide (= multiple of 32 banks),
    // all elements in the same column are in the SAME bank → 32-way conflict!
    volatile float *vtile = (volatile float*)tile;
    float val = 0.0f;
    for (int r = 0; r < REPEAT; r++) {
        val += vtile[x * TILE_DIM + y];  // 32-way bank conflict!
    }

    out[y * TILE_DIM + x] = val;
}

// ============================================================
// Kernel 5: Padding eliminates Bank Conflict
// Add 1 element of padding per row: 32→33 breaks the alignment
// 33 and 32 are coprime → no bank overlap!
// ============================================================
__global__ void padded_no_conflict(float *in, float *out) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // KEY: 32x33 NOT 32x32!

    int x = threadIdx.x % TILE_DIM;
    int y = threadIdx.x / TILE_DIM;

    // Load into padded tile (only use first 32 columns)
    tile[y][x] = in[y * TILE_DIM + x];
    __syncthreads();

    // TRANSPOSE READ with padding: row width = 33, coprime with 32
    volatile float *vtile = (volatile float*)tile;
    float val = 0.0f;
    for (int r = 0; r < REPEAT; r++) {
        val += vtile[x * (TILE_DIM + 1) + y];  // stride=33 → NO conflict!
    }

    out[y * TILE_DIM + x] = val;
}

int main() {
    float *d_out, *d_in;
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMalloc(&d_in, TILE_DIM * TILE_DIM * sizeof(float));

    // Initialize input data
    float *h_in = (float*)malloc(TILE_DIM * TILE_DIM * sizeof(float));
    for (int i = 0; i < TILE_DIM * TILE_DIM; i++) h_in[i] = (float)i;
    cudaMemcpy(d_in, h_in, TILE_DIM * TILE_DIM * sizeof(float), cudaMemcpyHostToDevice);
    free(h_in);

    int iterations = 10000;
    cudaEvent_t start, stop;
    float ms;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("============================================\n");
    printf("  Bank Conflict Performance Demo\n");
    printf("  (each kernel runs %d times, averaged)\n", iterations);
    printf("============================================\n\n");
    printf("%-28s | %-12s | Time(us)\n", "Access Pattern", "Expected");
    printf("------------------------------|--------------|----------\n");

    // Kernel 1: No conflict, stride=1
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++)
        no_bank_conflict<<<1, BLOCK_SIZE>>>(d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%-28s | %-12s | %7.2f\n", "stride=1 (sequential)", "no conflict", ms / iterations * 1000);

    // Kernel 2: 2-way conflict, stride=2
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++)
        two_way_bank_conflict<<<1, BLOCK_SIZE>>>(d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%-28s | %-12s | %7.2f\n", "stride=2", "2-way", ms / iterations * 1000);

    // Kernel 3: 32-way conflict, stride=32 (worst)
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++)
        worst_bank_conflict<<<1, BLOCK_SIZE>>>(d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%-28s | %-12s | %7.2f\n", "stride=32 (worst)", "32-way", ms / iterations * 1000);

    // Kernel 4: Column access without padding → bank conflict
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++)
        column_access_conflict<<<1, 1024>>>(d_in, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%-28s | %-12s | %7.2f\n", "col-access (no pad)", "32-way", ms / iterations * 1000);

    // Kernel 5: Column access with padding → no conflict
    cudaEventRecord(start);
    for (int i = 0; i < iterations; i++)
        padded_no_conflict<<<1, 1024>>>(d_in, d_out);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("%-28s | %-12s | %7.2f\n", "col-access (pad=1)", "no conflict", ms / iterations * 1000);

    printf("\n============================================\n");
    printf("  Summary\n");
    printf("============================================\n\n");

    printf("Key takeaways:\n");
    printf("  1. Shared Memory has 32 banks, each 4 bytes wide\n");
    printf("  2. Consecutive 4-byte words map to consecutive banks\n");
    printf("  3. Same bank + different addresses + same warp\n");
    printf("     → bank conflict, requests serialized\n");
    printf("  4. Same bank + same address → broadcast (OK)\n");
    printf("  5. stride = divisor of 32 → some degree of conflict\n");
    printf("  6. stride = multiple of 32 → 32-way (worst case)\n");
    printf("  7. Fix: add padding to break alignment (e.g. 32→33)\n\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_out);
    cudaFree(d_in);

    return 0;
}
