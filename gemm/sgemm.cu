/**
 * SGEMM: naive → tiled → register blocking
 * Benchmark vs cuBLAS
 *
 * Compile: nvcc -o sgemm sgemm.cu -lcublas -Xptxas -v
 * Run:     ./sgemm
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(c)  do { cudaError_t e = c; if (e != cudaSuccess) { \
    fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); }} while(0)
#define CUBLAS_CHECK(c) do { cublasStatus_t s = c; if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr,"cuBLAS %s:%d: %d\n",__FILE__,__LINE__,s); exit(1); }} while(0)

float ms(cudaEvent_t s, cudaEvent_t e) { float t; cudaEventElapsedTime(&t,s,e); return t; }

// ============================================================
// V1: Naive — global memory, no tiling
// ============================================================
__global__ void sgemm_naive(int M, int N, int K,
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

// ============================================================
// V2: Tiled shared memory, 16x16 tile, 16x16 threads
// ============================================================
__global__ void sgemm_tiled(int M, int N, int K,
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
    const int TILE = 16;
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float sum = 0.0f;
    for (int k_block = 0; k_block < K; k_block += TILE) {
        // Load tile of A: A[row, k_block + tx]
        int ka = k_block + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && ka < K) ? A[row * K + ka] : 0.0f;
        // Load tile of B: B[k_block + ty, col]
        int kb = k_block + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (kb < K && col < N) ? B[kb * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; i++)
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];

        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}

// ============================================================
// V3: 32x32 tile, 16x16 threads, 4 outputs per thread (2x2)
// ============================================================
__global__ void sgemm_tiled_32(int M, int N, int K,
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
    const int TILE = 32;
    int tx = threadIdx.x, ty = threadIdx.y;

    // This thread computes output (row0,col0), (row0,col1), (row1,col0), (row1,col1)
    int row0 = blockIdx.y * TILE + ty;
    int row1 = row0 + 16;
    int col0 = blockIdx.x * TILE + tx;
    int col1 = col0 + 16;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;

    for (int k_block = 0; k_block < K; k_block += TILE) {
        // BUGFIX: 16×16 threads must fill 32×32 shared memory tile
        // Each thread loads 4 elements for A (2 rows × 2 cols)
        // and 4 elements for B (2 rows × 2 cols)
        int ka0 = k_block + tx;
        int ka1 = k_block + tx + 16;
        int kb0 = k_block + ty;
        int kb1 = k_block + ty + 16;

        float a00 = (row0 < M && ka0 < K) ? A[row0 * K + ka0] : 0.0f;
        float a01 = (row0 < M && ka1 < K) ? A[row0 * K + ka1] : 0.0f;
        float a10 = (row1 < M && ka0 < K) ? A[row1 * K + ka0] : 0.0f;
        float a11 = (row1 < M && ka1 < K) ? A[row1 * K + ka1] : 0.0f;
        As[ty][tx] = a00;
        As[ty][tx + 16] = a01;
        As[ty + 16][tx] = a10;
        As[ty + 16][tx + 16] = a11;

        float b00 = (kb0 < K && col0 < N) ? B[kb0 * N + col0] : 0.0f;
        float b01 = (kb0 < K && col1 < N) ? B[kb0 * N + col1] : 0.0f;
        float b10 = (kb1 < K && col0 < N) ? B[kb1 * N + col0] : 0.0f;
        float b11 = (kb1 < K && col1 < N) ? B[kb1 * N + col1] : 0.0f;
        Bs[ty][tx] = b00;
        Bs[ty][tx + 16] = b01;
        Bs[ty + 16][tx] = b10;
        Bs[ty + 16][tx + 16] = b11;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE; i++) {
            float a_lo = As[ty][i];
            float a_hi = As[ty + 16][i];
            float b_lo = Bs[i][tx];
            float b_hi = Bs[i][tx + 16];
            c00 += a_lo * b_lo;
            c01 += a_lo * b_hi;
            c10 += a_hi * b_lo;
            c11 += a_hi * b_hi;
        }

        __syncthreads();
    }

    if (row0 < M && col0 < N) C[row0 * N + col0] = c00;
    if (row0 < M && col1 < N) C[row0 * N + col1] = c01;
    if (row1 < M && col0 < N) C[row1 * N + col0] = c10;
    if (row1 < M && col1 < N) C[row1 * N + col1] = c11;
}

// ============================================================
// V4: 64x64 tile, 16x16 threads, 4x4 output per thread, float4 loads
// ============================================================
#define TILE64 64
__global__ void sgemm_tiled_64(int M, int N, int K,
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
    const int T = TILE64;
    int tx = threadIdx.x, ty = threadIdx.y;

    // This thread computes a 4x4 sub-block of the 64x64 tile
    int row0 = blockIdx.y * T + ty * 4;
    int col0 = blockIdx.x * T + tx * 4;

    __shared__ float As[T][T];
    __shared__ float Bs[T][T];

    float c[4][4] = {{0}};

    for (int k_block = 0; k_block < K; k_block += T) {
        // Load A: float4 per row (4 cols at once), 4 rows = 4 float4 per thread
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int row = row0 + r;
            int kc = k_block + tx * 4;
            float4 v;
            if (row < M && kc + 3 < K)
                v = __ldg((const float4*)(&A[row * K + kc]));
            else {
                v = make_float4(0,0,0,0);
                if (row < M) for (int i=0;i<4;i++) { int kk=kc+i; if(kk<K)(&v.x)[i]=A[row*K+kk]; }
            }
            ((float4*)&As[ty * 4 + r])[tx] = v;
        }

        // Load B: float4 per row (4 cols at once), 4 rows = 4 float4 per thread
        // NOTE: col0 already = bx*64 + tx*4, so col0 is the correct base column
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int k_row = k_block + ty * 4 + r;
            float4 v;
            if (k_row < K && col0 + 3 < N)
                v = __ldg((const float4*)(&B[k_row * N + col0]));
            else {
                v = make_float4(0,0,0,0);
                if (k_row < K) for (int i=0;i<4;i++) { int cc=col0+i; if(cc<N)(&v.x)[i]=B[k_row*N+cc]; }
            }
            ((float4*)&Bs[ty * 4 + r])[tx] = v;
        }

        __syncthreads();

        // Compute: 4x4 accumulators
        #pragma unroll
        for (int k = 0; k < T; k++) {
            float a0 = As[ty * 4 + 0][k];
            float a1 = As[ty * 4 + 1][k];
            float a2 = As[ty * 4 + 2][k];
            float a3 = As[ty * 4 + 3][k];
            float b0 = Bs[k][tx * 4 + 0];
            float b1 = Bs[k][tx * 4 + 1];
            float b2 = Bs[k][tx * 4 + 2];
            float b3 = Bs[k][tx * 4 + 3];

            c[0][0] += a0 * b0; c[0][1] += a0 * b1; c[0][2] += a0 * b2; c[0][3] += a0 * b3;
            c[1][0] += a1 * b0; c[1][1] += a1 * b1; c[1][2] += a1 * b2; c[1][3] += a1 * b3;
            c[2][0] += a2 * b0; c[2][1] += a2 * b1; c[2][2] += a2 * b2; c[2][3] += a2 * b3;
            c[3][0] += a3 * b0; c[3][1] += a3 * b1; c[3][2] += a3 * b2; c[3][3] += a3 * b3;
        }

        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int row = row0 + r;
        if (row >= M) continue;
        #pragma unroll
        for (int col = 0; col < 4; col++) {
            int cc = col0 + col;
            if (cc < N) C[row * N + cc] = c[r][col];
        }
    }
}

// ============================================================
// V6: 64x64 tile, 16x8 threads, 4x8 output (32 accums), float4
// Rebalanced: more accumulators for better ILP
// ============================================================
#define T64 64
__global__ void __launch_bounds__(128, 6)
sgemm_tiled_64_32acc(int M, int N, int K,
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
    const int T = T64;
    int tx = threadIdx.x, ty = threadIdx.y;  // 16×8 = 128 threads

    // Thread (ty,tx) computes 4 rows × 8 cols = 32 outputs
    int row0 = blockIdx.y * T + ty * 4;
    int col0 = blockIdx.x * T + tx * 8;

    __shared__ float As[T][T];
    __shared__ float Bs[T][T];

    float c00=0,c01=0,c02=0,c03=0,c04=0,c05=0,c06=0,c07=0;
    float c10=0,c11=0,c12=0,c13=0,c14=0,c15=0,c16=0,c17=0;
    float c20=0,c21=0,c22=0,c23=0,c24=0,c25=0,c26=0,c27=0;
    float c30=0,c31=0,c32=0,c33=0,c34=0,c35=0,c36=0,c37=0;

    for (int k_block = 0; k_block < K; k_block += T) {
        // Load A: 4 rows × 1 float4 per row
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int row = row0 + r;
            int kc = k_block + tx * 4;
            float4 v = make_float4(0,0,0,0);
            if (row < M && kc + 3 < K)
                v = *(const float4*)(&A[row * K + kc]);
            else if (row < M) {
                for (int i=0;i<4;i++) { int kk=kc+i; if(kk<K)(&v.x)[i]=A[row*K+kk]; }
            }
            // Store to As[ty*4+r][tx*4..tx*4+3]
            ((float4*)&As[ty * 4 + r])[tx] = v;
        }

        // Load B: 8 rows × 2 float4 per row (8 cols per thread)
        #pragma unroll
        for (int r = 0; r < 8; r++) {
            int k_row = k_block + ty * 8 + r;
            int col = col0 + tx * 8;  // tx ∈ [0,15], each covers 8 cols at tx*8
            // Wait: 16 threads × 8 cols each = 128 cols. But tile is 64 cols!
            // Need: 8 threads × 8 cols = 64. So only 8 of 16 threads load B?
            // Redesign: use 8×8 thread block instead
            (void)k_row; (void)col;
        }
        // BUG: this mapping is wrong — 16×8 threads × 8 cols = too many

        // Redesign needed: 8×8 thread block, 8×8 output, or 16×16 with 4×8
        // Let me fix this properly...

        __syncthreads();
        __syncthreads();
    }

    // Store output...
    // (incomplete implementation — see V7 for the actual fix)
    if (row0 < M && col0 < N) C[row0 * N + col0] = c00;
}
// ============================================================
// V5: 32x32 tile, 16x16 threads, 2x2 output, max occupancy
//     ~8KB shared memory → 6 blocks/SM → 100% theoretical occupancy
// ============================================================
__global__ void sgemm_tiled_32_occ(int M, int N, int K,
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
    const int T = 32;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row0 = blockIdx.y * T + ty * 2;
    int col0 = blockIdx.x * T + tx * 2;

    __shared__ float As[T][T];
    __shared__ float Bs[T][T];

    float c00=0,c01=0,c10=0,c11=0;

    for (int k_block = 0; k_block < K; k_block += T) {
        int ka = k_block + tx * 2;
        float2 av = make_float2(0,0);
        if (row0 < M && ka+1 < K) av = *(const float2*)(&A[row0 * K + ka]);
        else if (row0 < M) { if (ka<K) av.x=A[row0*K+ka]; if (ka+1<K) av.y=A[row0*K+ka+1]; }
        As[ty*2][tx*2] = av.x; As[ty*2][tx*2+1] = av.y;

        float2 av2 = make_float2(0,0);
        if (row0+1 < M && ka+1 < K) av2 = *(const float2*)(&A[(row0+1)*K + ka]);
        else if (row0+1 < M) { if (ka<K) av2.x=A[(row0+1)*K+ka]; if (ka+1<K) av2.y=A[(row0+1)*K+ka+1]; }
        As[ty*2+1][tx*2] = av2.x; As[ty*2+1][tx*2+1] = av2.y;

        int kb = k_block + ty * 2;
        float2 bv = make_float2(0,0);
        if (kb < K && col0+1 < N) bv = *(const float2*)(&B[kb * N + col0]);
        else if (kb < K) { if (col0<N) bv.x=B[kb*N+col0]; if (col0+1<N) bv.y=B[kb*N+col0+1]; }
        Bs[ty*2][tx*2] = bv.x; Bs[ty*2][tx*2+1] = bv.y;

        float2 bv2 = make_float2(0,0);
        if (kb+1 < K && col0+1 < N) bv2 = *(const float2*)(&B[(kb+1)*N + col0]);
        else if (kb+1 < K) { if (col0<N) bv2.x=B[(kb+1)*N+col0]; if (col0+1<N) bv2.y=B[(kb+1)*N+col0+1]; }
        Bs[ty*2+1][tx*2] = bv2.x; Bs[ty*2+1][tx*2+1] = bv2.y;

        __syncthreads();

        #pragma unroll
        for (int k=0;k<T;k++) {
            float a0=As[ty*2][k], a1=As[ty*2+1][k];
            float b0=Bs[k][tx*2], b1=Bs[k][tx*2+1];
            c00+=a0*b0; c01+=a0*b1; c10+=a1*b0; c11+=a1*b1;
        }
        __syncthreads();
    }

    if (row0<M && col0<N) C[row0*N+col0]=c00;
    if (row0<M && col0+1<N) C[row0*N+col0+1]=c01;
    if (row0+1<M && col0<N) C[(row0+1)*N+col0]=c10;
    if (row0+1<M && col0+1<N) C[(row0+1)*N+col0+1]=c11;
}
//     Inner loop: 2 float4 smem reads → 16 FMAs (vs V4's 5 reads)
// ============================================================
__global__ void __launch_bounds__(256, 4)
sgemm_tiled_64_trans(int M, int N, int K,
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C)
{
    const int T = TILE64;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row0 = blockIdx.y * T + ty * 4;
    int col0 = blockIdx.x * T + tx * 4;

    // As_t: K-major with padding to avoid 4-way bank conflicts on scatter writes
    // Stride = 72 (64+8) — banks: 0,8,16,24 for the 4 scattered stores
    // Bs:   row-major, no padding needed (contiguous float4 stores)
    __shared__ float As_t[T][T + 8];  // pad 8 cols to avoid bank conflicts
    __shared__ float Bs[T][T];

    float c[4][4] = {{0}};

    for (int k_block = 0; k_block < K; k_block += T) {
        // Load A: float4 from global, scatter to transposed As_t
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int row = row0 + r;
            int kc = k_block + tx * 4;
            float4 v = make_float4(0,0,0,0);
            if (row < M && kc + 3 < K)
                v = *(const float4*)(&A[row * K + kc]);
            else if (row < M) {
                for (int i = 0; i < 4; i++) {
                    int kk = kc + i;
                    if (kk < K) (&v.x)[i] = A[row * K + kk];
                }
            }
            // Scatter to transposed As_t: As_t[K_idx][row] = A[row][K_idx]
            As_t[tx * 4 + 0][ty * 4 + r] = v.x;
            As_t[tx * 4 + 1][ty * 4 + r] = v.y;
            As_t[tx * 4 + 2][ty * 4 + r] = v.z;
            As_t[tx * 4 + 3][ty * 4 + r] = v.w;
        }

        // Load B: same as V4, float4 from global, contiguous store to Bs
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            int k_row = k_block + ty * 4 + r;
            float4 v;
            if (k_row < K && col0 + 3 < N)
                v = __ldg((const float4*)(&B[k_row * N + col0]));
            else {
                v = make_float4(0,0,0,0);
                if (k_row < K) for (int i=0;i<4;i++) { int cc=col0+i; if(cc<N)(&v.x)[i]=B[k_row*N+cc]; }
            }
            ((float4*)&Bs[ty * 4 + r])[tx] = v;
        }

        __syncthreads();

        // Inner loop: 2 float4 reads, 16 FMAs
        #pragma unroll
        for (int k = 0; k < T; k++) {
            float4 av = *(const float4*)(&As_t[k][ty * 4]);
            float4 bv = *(const float4*)(&Bs[k][tx * 4]);
            float a0 = av.x, a1 = av.y, a2 = av.z, a3 = av.w;
            float b0 = bv.x, b1 = bv.y, b2 = bv.z, b3 = bv.w;

            c[0][0] += a0 * b0; c[0][1] += a0 * b1; c[0][2] += a0 * b2; c[0][3] += a0 * b3;
            c[1][0] += a1 * b0; c[1][1] += a1 * b1; c[1][2] += a1 * b2; c[1][3] += a1 * b3;
            c[2][0] += a2 * b0; c[2][1] += a2 * b1; c[2][2] += a2 * b2; c[2][3] += a2 * b3;
            c[3][0] += a3 * b0; c[3][1] += a3 * b1; c[3][2] += a3 * b2; c[3][3] += a3 * b3;
        }

        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int row = row0 + r;
        if (row >= M) continue;
        #pragma unroll
        for (int cc = 0; cc < 4; cc++) {
            int col = col0 + cc;
            if (col < N) C[row * N + col] = c[r][cc];
        }
    }
}

// ============================================================
// Main benchmark
// ============================================================
int main() {
    int sizes[] = {512, 1024, 2048, 4096};
    int n_sizes = 4;
    const int WARM = 5, RUN = 20;

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));
    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Increase shared memory for V4: Ampere supports up to 100KB/SM
    // Default is 48KB, V4 uses 32KB/block → only 1 block/SM
    // With 100KB → up to 3 blocks/SM → much better occupancy
    cudaFuncSetAttribute(sgemm_tiled_64, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    typedef void (*kfn_t)(int,int,int,const float*,const float*,float*);
    cudaFuncSetAttribute(sgemm_tiled_64_trans, cudaFuncAttributePreferredSharedMemoryCarveout, 100);

    kfn_t kernels[] = {sgemm_naive, sgemm_tiled, sgemm_tiled_32, sgemm_tiled_64, sgemm_tiled_64_trans};
    const char* kname[] = {"V1_naive", "V2_tiled_16", "V3_tiled_32", "V4_tiled_64", "V5_trans"};
    dim3 blocks[] = {dim3(16,16), dim3(16,16), dim3(16,16), dim3(16,16), dim3(16,16)};
    int tiles[]  = {16, 16, 32, 64, 64};

    printf("GEMM Benchmark: our kernels vs cuBLAS (FP32)\n");
    printf("=============================================\n\n");
    printf("%6s | %15s %8s %8s %6s | %15s %8s %8s | %8s\n",
           "size", "kernel", "ms", "TFLOPS", "%peak",
           "cuBLAS", "ms", "TFLOPS", "max_err");
    printf("-------+--------------------------------------------------+---------------------------------+----------\n");

    for (int si = 0; si < n_sizes; si++) {
        int M = sizes[si], N = M, K = M;
        size_t bA = (size_t)M * K * 4, bB = (size_t)K * N * 4, bC = (size_t)M * N * 4;
        double flops = 2.0 * (double)M * N * K;

        float *hA = (float*)malloc(bA), *hB = (float*)malloc(bB);
        float *hC = (float*)malloc(bC), *hRef = (float*)malloc(bC);
        for (size_t i = 0; i < (size_t)M*K; i++) { hA[i] = (float)rand()/RAND_MAX - 0.5f; hB[i] = (float)rand()/RAND_MAX - 0.5f; }

        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, bA)); CUDA_CHECK(cudaMalloc(&dB, bB)); CUDA_CHECK(cudaMalloc(&dC, bC));
        CUDA_CHECK(cudaMemcpy(dA, hA, bA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, bB, cudaMemcpyHostToDevice));

        // cuBLAS reference
        float t_cublas, tf_cublas;
        {
            for (int i = 0; i < WARM; i++)
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
            CUDA_CHECK(cudaDeviceSynchronize());
            float best = 1e9f;
            for (int r = 0; r < RUN; r++) {
                CUDA_CHECK(cudaEventRecord(start, 0));
                CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
                CUDA_CHECK(cudaEventRecord(stop, 0));
                CUDA_CHECK(cudaEventSynchronize(stop));
                float t = ms(start, stop); if (t < best) best = t;
            }
            t_cublas = best;
            tf_cublas = flops / (best/1000.0) / 1e12;
            CUDA_CHECK(cudaMemcpy(hRef, dC, bC, cudaMemcpyDeviceToHost));
        }

        // Our kernels
        for (int k = 0; k < 5; k++) {
            int tsz = tiles[k];
            dim3 grid((N + tsz - 1) / tsz, (M + tsz - 1) / tsz);

            for (int i = 0; i < WARM; i++)
                kernels[k]<<<grid, blocks[k]>>>(M, N, K, dA, dB, dC);
            CUDA_CHECK(cudaDeviceSynchronize());

            float best = 1e9f;
            for (int r = 0; r < RUN; r++) {
                CUDA_CHECK(cudaEventRecord(start, 0));
                kernels[k]<<<grid, blocks[k]>>>(M, N, K, dA, dB, dC);
                CUDA_CHECK(cudaEventRecord(stop, 0));
                CUDA_CHECK(cudaEventSynchronize(stop));
                float t = ms(start, stop); if (t < best) best = t;
            }

            float tflops = flops / (best/1000.0) / 1e12;
            // FP32 peak: ~8.1 TFLOPS at 1.59 GHz (computed from CUDA cores × 2 × clock)
            // Use 8.0 as rough estimate for the percentages
            float pct = tflops / 8.0f * 100;
            CUDA_CHECK(cudaMemcpy(hC, dC, bC, cudaMemcpyDeviceToHost));

            double max_err = 0.0;
            for (size_t i = 0; i < (size_t)M*N; i++) {
                double e = fabs((double)hC[i] - (double)hRef[i]);
                if (e > max_err) max_err = e;
            }

            printf("%6d | %15s %8.3f %8.3f %5.1f%% | %15s %8.3f %8.3f | %8.1e\n",
                   M, kname[k], best, tflops, pct,
                   "cuBLAS", t_cublas, tf_cublas, max_err);
        }

        printf("-------+--------------------------------------------------+---------------------------------+----------\n");
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        free(hA); free(hB); free(hC); free(hRef);
    }

    CUBLAS_CHECK(cublasDestroy(cublas));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}
