/**
 * GEMM M=4 vs M=64: fundamentally different optimization strategies
 *
 * Compile: nvcc -o gemm_smallM gemm_smallM.cu -arch=sm_86 && ./gemm_smallM
 *
 * Core insight:
 *   M=4:  only 4 rows in output → NO parallelism along M
 *         → must tile along N (wide tiles) or K (split-K)
 *   M=64: enough rows for standard tiling → balance M/N/K
 *
 * This is the classic "tall-skinny vs fat-short" GEMM problem.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHK(c) do{cudaError_t _e=c;if(_e!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(_e));exit(1);}}while(0)

// ============================================================================
// M=4: Strategy — TILE ALONG N (wide), NO tiling along M
// ============================================================================
// Problem:  M=4, N=4096, K=4096
//   C[4 x 4096] = A[4 x 4096] @ B[4096 x 4096]
//
// With only 4 rows, we get at most 1 warp (32 threads) per "row-group"
// if we try to tile. Better approach: each block processes ALL 4 rows
// of a column slice. Parallelism comes from the N dimension.
//
// smem usage is tiny for A (4 rows x K_tile) but normal for B (K_tile x N_tile).

__global__ void gemm_M4_wideN(const float *__restrict__ A,
                               const float *__restrict__ B,
                               float *__restrict__ C,
                               int N, int K) {
    // Each block: 4 rows of C, N_TILE columns
    const int N_TILE = 256;
    const int K_TILE = 64;

    int tid = threadIdx.x;       // 0..255
    int col_base = blockIdx.x * N_TILE;  // column slice in C/B
    int ty = tid / 32;           // warp id (0..7)
    int tx = tid % 32;           // lane id (0..31)

    // Registers: each thread accumulates for 4 rows x (N_TILE/32) columns
    float c_vals[4] = {0, 0, 0, 0};
    // Each thread handles 8 columns (256 cols / 32 lanes)
    int col = col_base + tx;  // thread's column within tile

    // A tile: [4 x K_TILE] — tiny! only 4*64*4 = 1KB in smem
    // B tile: [K_TILE x N_TILE] — 64*256*4 = 64KB (this is the big one)
    __shared__ float As[4][K_TILE];
    __shared__ float Bs[K_TILE][N_TILE];

    for (int kb = 0; kb < K; kb += K_TILE) {
        // Cooperative load A tile: 4 rows * K_TILE cols = 256 elements
        // 256 threads = perfect fit, each loads 1 element
        if (tid < 4 * K_TILE) {
            int r = tid / K_TILE;
            int c = tid % K_TILE;
            As[r][c] = (kb + c < K) ? A[r * K + (kb + c)] : 0.0f;
        }

        // Cooperative load B tile: K_TILE rows * N_TILE cols
        // 256 threads, each loads K_TILE*N_TILE/256 elements
        for (int i = tid; i < K_TILE * N_TILE; i += 256) {
            int r = i / N_TILE;
            int c = i % N_TILE;
            int gc = col_base + c;
            Bs[r][c] = ((kb + r) < K && gc < N) ? B[(kb + r) * N + gc] : 0.0f;
        }
        __syncthreads();

        // Compute: each thread does 4 rows x 8 cols x K_TILE inner products
        for (int k = 0; k < K_TILE; k++) {
            float a0 = As[0][k], a1 = As[1][k], a2 = As[2][k], a3 = As[3][k];
            for (int ci = 0; ci < 8; ci++) {
                float bv = Bs[k][tx + ci * 32];
                c_vals[ci % 4] += a0 * bv;  // simplified, see full version below
            }
        }
        __syncthreads();
    }

    // Store
    for (int ci = 0; ci < 8 && col_base + tx + ci*32 < N; ci++) {
        for (int r = 0; r < 4; r++) {
            C[r * N + (col_base + tx + ci*32)] = 0.0f; // placeholder
        }
    }
}

// ============================================================================
// Better M=4 kernel: Each thread handles exactly 1 column of the 4 rows
// ============================================================================
// 4 rows x N columns = 4*N outputs. Launch N_BLOCKS along N.
// Each block: all 4 rows, 256 columns, all of K.
//
// This is the "vector kernel" approach — each thread computes
// a dot product [1xK] @ [Kx1] for one output element.

__global__ void gemm_M4_vector(const float *__restrict__ A,
                                const float *__restrict__ B,
                                float *__restrict__ C,
                                int N, int K) {
    const int N_TILE = 256;
    int col = blockIdx.x * N_TILE + threadIdx.x;
    if (col >= N) return;

    // Each thread computes 4 elements of C: C[0..3][col]
    // A[0..3][:] is just 4 rows → tiny, stays in registers entirely
    // BUT: we need B[:, col] which is K elements → stream through

    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;

    // Key: A rows are SO SMALL (4xK) they can be loaded once to smem
    // Then reused N_TILE times as different threads process different cols
    // This is the REVERSE of normal GEMM tiling!

    // Actually for M=4, the best strategy: load A ONCE to smem,
    // then EACH THREAD loads ALL of B[:, col] from global
    __shared__ float As[4][256];  // load K in chunks

    for (int kb = 0; kb < K; kb += 256) {
        int k_tile = min(256, K - kb);

        // Load A slice: 4 rows x 256 cols
        for (int i = threadIdx.x; i < 4 * k_tile; i += blockDim.x) {
            int r = i / k_tile, c = i % k_tile;
            As[r][c] = A[r * K + (kb + c)];
        }
        __syncthreads();

        // Each thread loads its B column and dot-products
        float b_val;
        for (int k = 0; k < k_tile; k++) {
            b_val = B[(kb + k) * N + col];
            c0 += As[0][k] * b_val;
            c1 += As[1][k] * b_val;
            c2 += As[2][k] * b_val;
            c3 += As[3][k] * b_val;
        }
        __syncthreads();
    }

    C[0 * N + col] = c0;
    C[1 * N + col] = c1;
    C[2 * N + col] = c2;
    C[3 * N + col] = c3;
}

// ============================================================================
// M=4 ALTERNATIVE: Split-K
// ============================================================================
// Since we can't parallelize along M, parallelize along K instead!
// Multiple blocks compute PARTIAL sums for the same C[0..3][col],
// then a reduction kernel merges them.

__global__ void gemm_M4_splitK(const float *__restrict__ A,
                                const float *__restrict__ B,
                                float *__restrict__ C_partial,
                                int N, int K, int K_per_block) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int kb_start = blockIdx.y * K_per_block;
    if (col >= N) return;

    float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    int k_end = min(kb_start + K_per_block, K);

    // No smem! Each thread streams A rows from L2 cache (they're tiny)
    // and reads B[:, col] once.
    for (int k = kb_start; k < k_end; k++) {
        float b_val = B[k * N + col];
        c0 += A[0 * K + k] * b_val;
        c1 += A[1 * K + k] * b_val;
        c2 += A[2 * K + k] * b_val;
        c3 += A[3 * K + k] * b_val;
    }

    // Write partial results
    int out_idx = blockIdx.y * 4 * N + 0 * N + col;
    C_partial[out_idx + 0 * N] = c0;
    C_partial[out_idx + 1 * N] = c1;
    C_partial[out_idx + 2 * N] = c2;
    C_partial[out_idx + 3 * N] = c3;
}

// ============================================================================
// M=64: Strategy — STANDARD TILING (balanced M, N, K)
// ============================================================================
// Problem:  M=64, N=4096, K=4096
//   C[64 x 4096] = A[64 x 4096] @ B[4096 x 4096]
//
// Now we have enough rows to use standard tiling. Key differences vs M=4:
//   1. Can tile along BOTH M and N (64x128 tiles work well)
//   2. smem is full-sized for both A and B tiles
//   3. Multiple warps per block is beneficial
//   4. Register blocking (2x2 or 4x4) amortizes smem reads

// Standard tiled GEMM for M=64
__global__ void gemm_M64_tiled(const float *__restrict__ A,
                                const float *__restrict__ B,
                                float *__restrict__ C,
                                int N, int K) {
    const int M_TILE = 64, N_TILE = 64, K_TILE = 32;

    int row = blockIdx.y * M_TILE + threadIdx.y;
    int col = blockIdx.x * N_TILE + threadIdx.x;

    __shared__ float As[M_TILE][K_TILE];
    __shared__ float Bs[K_TILE][N_TILE];

    float acc = 0;

    for (int kb = 0; kb < K; kb += K_TILE) {
        // Cooperative load A
        int a_idx = threadIdx.y * K_TILE + threadIdx.x;
        if (a_idx < M_TILE * K_TILE) {
            int r = a_idx / K_TILE, c = a_idx % K_TILE;
            As[r][c] = ((blockIdx.y * M_TILE + r) < 64 && (kb + c) < K)
                       ? A[(blockIdx.y * M_TILE + r) * K + (kb + c)] : 0;
        }
        // Cooperative load B
        int b_idx = threadIdx.y * N_TILE + threadIdx.x;
        if (b_idx < K_TILE * N_TILE) {
            int r = b_idx / N_TILE, c = b_idx % N_TILE;
            Bs[r][c] = ((kb + r) < K && (blockIdx.x * N_TILE + c) < N)
                       ? B[(kb + r) * N + (blockIdx.x * N_TILE + c)] : 0;
        }
        __syncthreads();

        for (int k = 0; k < K_TILE; k++) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < 64 && col < N) C[row * N + col] = acc;
}

// ============================================================================
// M=64 OPTIMIZED: 2x2 register blocking
// ============================================================================
// Each thread computes 2x2 outputs, halving smem reads per FMA.
// With blockDim=(32, 32) → 1024 threads (4 warps)

__global__ void gemm_M64_rblock2x2(const float *__restrict__ A,
                                    const float *__restrict__ B,
                                    float *__restrict__ C,
                                    int N, int K) {
    const int M_TILE = 64, N_TILE = 128, K_TILE = 32;

    int tid_x = threadIdx.x; // 0..31
    int tid_y = threadIdx.y; // 0..31

    // This thread computes C[row0][col0], C[row0][col1], C[row1][col0], C[row1][col1]
    int row0 = blockIdx.y * M_TILE + tid_y;
    int row1 = row0 + 32;  // 2x2 register blocking
    int col0 = blockIdx.x * N_TILE + tid_x;
    int col1 = col0 + 32;

    __shared__ float As[M_TILE][K_TILE];
    __shared__ float Bs[K_TILE][N_TILE];

    float c00 = 0, c01 = 0, c10 = 0, c11 = 0;

    for (int kb = 0; kb < K; kb += K_TILE) {
        // Load A tile: 64 x 8 = 512 floats. 1024 threads split it.
        for (int i = tid_y * 32 + tid_x; i < M_TILE * K_TILE; i += 1024) {
            int r = i / K_TILE, c = i % K_TILE;
            As[r][c] = ((blockIdx.y * M_TILE + r) < 64 && (kb + c) < K)
                       ? A[(blockIdx.y * M_TILE + r) * K + (kb + c)] : 0;
        }
        // Load B tile
        for (int i = tid_y * 32 + tid_x; i < K_TILE * N_TILE; i += 1024) {
            int r = i / N_TILE, c = i % N_TILE;
            Bs[r][c] = ((kb + r) < K && (blockIdx.x * N_TILE + c) < N)
                       ? B[(kb + r) * N + (blockIdx.x * N_TILE + c)] : 0;
        }
        __syncthreads();

        // Compute: each thread reads 2 A values + 2 B values per k → 4 FMAs
        for (int k = 0; k < K_TILE; k++) {
            float a0 = As[tid_y][k], a1 = As[tid_y + 32][k];
            float b0 = Bs[k][tid_x], b1 = Bs[k][tid_x + 32];
            c00 += a0 * b0; c01 += a0 * b1;
            c10 += a1 * b0; c11 += a1 * b1;
        }
        __syncthreads();
    }

    if (row0 < 64 && col0 < N) C[row0 * N + col0] = c00;
    if (row0 < 64 && col1 < N) C[row0 * N + col1] = c01;
    if (row1 < 64 && col0 < N) C[row1 * N + col0] = c10;
    if (row1 < 64 && col1 < N) C[row1 * N + col1] = c11;
}

// ============================================================================
// Benchmark & comparison
// ============================================================================
float millis(cudaEvent_t s, cudaEvent_t e) {
    float ms; cudaEventElapsedTime(&ms, s, e); return ms;
}

#define WU 10
#define RN 20

float bench(const char *label, void(*kernel)(const float*,const float*,float*,int,int),
            dim3 grid, dim3 block, const float *dA, const float *dB, float *dC,
            int N, int K, cudaEvent_t s, cudaEvent_t e) {
    for (int w = 0; w < WU; w++) kernel<<<grid, block>>>(dA, dB, dC, N, K);
    CHK(cudaDeviceSynchronize());
    float best = 1e9;
    for (int r = 0; r < RN; r++) {
        cudaEventRecord(s);
        kernel<<<grid, block>>>(dA, dB, dC, N, K);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float t = millis(s, e);
        if (t < best) best = t;
    }
    float tflops = 2.0 * 4 * N * K / (best / 1000.0) / 1e12;
    printf("  %-35s %7.4f ms  %6.2f TFLOPS\n", label, best, tflops);
    return best;
}

int main() {
    cudaEvent_t s, e;
    CHK(cudaEventCreate(&s, 0)); CHK(cudaEventCreate(&e, 0));
    cudaDeviceProp p;
    CHK(cudaGetDeviceProperties(&p, 0));
    printf("GPU: %s | SM: %d | MaxThr/SM: %d | smem/SM: %dKB\n\n",
           p.name, p.multiProcessorCount, p.maxThreadsPerMultiProcessor,
           p.sharedMemPerMultiprocessor/1024);

    // Common K dimension for both tests
    const int K = 4096;
    const int N = 4096;

    // ========================================================================
    // Test 1: M = 4
    // ========================================================================
    {
        const int M = 4;
        printf("========================================\n");
        printf("  M=%d, N=%d, K=%d\n", M, N, K);
        printf("  Output: %d x %d = %d floats (%.1f KB)\n",
               M, N, M*N, M*N*4.0f/1024);
        printf("  FLOPs: 2*M*N*K = %.1f MFLOPs\n", 2.0*M*N*K/1e6);
        printf("  Key challenge: ONLY %d rows → no M-parallelism\n", M);
        printf("========================================\n");

        size_t size_A = M * K * sizeof(float);       // 64 KB
        size_t size_B = K * N * sizeof(float);       // 64 MB
        size_t size_C = M * N * sizeof(float);       // 64 KB

        float *dA, *dB, *dC, *dC_partial;
        CHK(cudaMalloc(&dA, size_A));
        CHK(cudaMalloc(&dB, size_B));
        CHK(cudaMalloc(&dC, size_C));

        // Split-K partial results
        int splitK = 8;  // 8 blocks along K
        size_t size_partial = splitK * M * N * sizeof(float);
        CHK(cudaMalloc(&dC_partial, size_partial));

        // Approach 1: vector kernel (1 thread = 1 column of 4 rows)
        {
            dim3 grid((N + 255) / 256), block(256);
            bench("M=4: vector (1 col/thread)", gemm_M4_vector,
                  grid, block, dA, dB, dC, N, K, s, e);
        }

        // Approach 2: wide-N tiled
        // Note: this kernel has issues, keeping for illustration
        // {
        //     dim3 grid((N + 255) / 256), block(256);
        //     bench("M=4: wide-N tile", gemm_M4_wideN,
        //           grid, block, dA, dB, dC, N, K, s, e);
        // }

        printf("\n  M=4 STRATEGY SUMMARY:\n");
        printf("    - A: stays in L2 cache (only %d KB)\n", (int)(size_A/1024));
        printf("    - B: stream through, one column per thread\n");
        printf("    - Parallelism ONLY from N and K dimensions\n");
        printf("    - smem: load A once, reuse for N columns\n");
        printf("    - Split-K: further parallelize along K\n\n");

        CHK(cudaFree(dA)); CHK(cudaFree(dB));
        CHK(cudaFree(dC)); CHK(cudaFree(dC_partial));
    }

    // ========================================================================
    // Test 2: M = 64
    // ========================================================================
    {
        const int M = 64;
        printf("========================================\n");
        printf("  M=%d, N=%d, K=%d\n", M, N, K);
        printf("  Output: %d x %d = %d floats (%d KB)\n",
               M, N, M*N, (int)(M*N*4/1024));
        printf("  FLOPs: 2*M*N*K = %.1f GFLOPs\n", 2.0*M*N*K/1e9);
        printf("  Key difference: %d rows → can tile along M\n", M);
        printf("========================================\n");

        size_t size_A = M * K * sizeof(float);       // 1 MB
        size_t size_B = K * N * sizeof(float);       // 64 MB
        size_t size_C = M * N * sizeof(float);       // 1 MB

        float *dA, *dB, *dC;
        CHK(cudaMalloc(&dA, size_A));
        CHK(cudaMalloc(&dB, size_B));
        CHK(cudaMalloc(&dC, size_C));

        // Approach 1: standard tiled (64x64 tile, 8x8 thread block)
        {
            dim3 grid((N + 63) / 64, (M + 63) / 64), block(8, 8);
            bench("M=64: standard tile (64x64)", gemm_M64_tiled,
                  grid, block, dA, dB, dC, N, K, s, e);
        }

        // Approach 2: 2x2 register blocking (64x128 tile, 32x32 threads)
        {
            dim3 grid((N + 127) / 128, (M + 63) / 64), block(32, 32);
            bench("M=64: 2x2 rblock (64x128)", gemm_M64_rblock2x2,
                  grid, block, dA, dB, dC, N, K, s, e);
        }

        printf("\n  M=64 STRATEGY SUMMARY:\n");
        printf("    - A: 1 MB (fits in L2), can tile normally\n");
        printf("    - Standard M-tiling works (64-row tiles)\n");
        printf("    - smem: full tiles for both A and B\n");
        printf("    - Register blocking (2x2, 4x4) amortizes smem reads\n");
        printf("    - Multiple warps per block → hide latency\n\n");

        CHK(cudaFree(dA)); CHK(cudaFree(dB)); CHK(cudaFree(dC));
    }

    // ========================================================================
    printf("==========================================================\n");
    printf("  KEY DIFFERENCES: M=4 vs M=64\n");
    printf("==========================================================\n");
    printf("  %-25s %-25s %-25s\n", "Aspect", "M=4", "M=64");
    printf("  %-25s %-25s %-25s\n", "-----", "---", "----");
    printf("  %-25s %-25s %-25s\n", "M parallelism", "NONE (4 rows)", "64 rows → 2 tiles");
    printf("  %-25s %-25s %-25s\n", "N parallelism", "PRIMARY source", "Balanced with M");
    printf("  %-25s %-25s %-25s\n", "A in smem", "Tiny (4xK_tile)", "Normal (64xK_tile)");
    printf("  %-25s %-25s %-25s\n", "A reuse", "1x (broadcast to N)", "N_tile/64 times");
    printf("  %-25s %-25s %-25s\n", "Reg blocking", "Not needed", "2x2 or 4x4 helps");
    printf("  %-25s %-25s %-25s\n", "Split-K viable?", "YES (main trick!)", "Sometimes");
    printf("  %-25s %-25s %-25s\n", "Occupancy concern", "Filling SMs is hard", "Easy");
    printf("  %-25s %-25s %-25s\n", "Best smem use", "Load A once, reuse", "Standard tiling");
    printf("  %-25s %-25s %-25s\n", "Bottleneck", "N paral. or K streaming", "smem bandwidth");
    printf("==========================================================\n");

    {cudaError_t err = cudaEventDestroy(s); if(err!=cudaSuccess) printf("WARN\n");}
    {cudaError_t err = cudaEventDestroy(e); if(err!=cudaSuccess) printf("WARN\n");}
    return 0;
}
