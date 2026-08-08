"""
SGEMM kernels with Numba CUDA: naive → tiled → register blocking
Benchmark vs PyTorch (cuBLAS)
"""
import torch
import numpy as np
from numba import cuda
import math
import time

TILE = 16
TILE32 = 32

# ============================================================
# V1: Naive — each thread one output element, global memory
# ============================================================
@cuda.jit
def sgemm_naive(A, B, C, M, N, K):
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if row >= M or col >= N:
        return
    s = 0.0
    for k in range(K):
        s += A[row, k] * B[k, col]
    C[row, col] = s

# ============================================================
# V2: Tiled shared memory, 16x16 tile, 16x16 threads
# ============================================================
@cuda.jit
def sgemm_tiled(A, B, C, M, N, K):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row = cuda.blockIdx.y * TILE + ty
    col = cuda.blockIdx.x * TILE + tx

    As = cuda.shared.array((TILE, TILE), dtype='f4')
    Bs = cuda.shared.array((TILE, TILE), dtype='f4')

    s = 0.0
    for k_block in range(0, K, TILE):
        # Load tile of A
        ka = k_block + tx
        As[ty, tx] = A[row, ka] if (row < M and ka < K) else 0.0
        # Load tile of B
        kb = k_block + ty
        Bs[ty, tx] = B[kb, col] if (kb < K and col < N) else 0.0

        cuda.syncthreads()

        for i in range(TILE):
            s += As[ty, i] * Bs[i, tx]

        cuda.syncthreads()

    if row < M and col < N:
        C[row, col] = s

# ============================================================
# V3: 32x32 tile, 16x16 threads, 4 outputs per thread
# ============================================================
@cuda.jit
def sgemm_tiled_32(A, B, C, M, N, K):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    row0 = cuda.blockIdx.y * TILE32 + ty
    row1 = row0 + 16
    col0 = cuda.blockIdx.x * TILE32 + tx
    col1 = col0 + 16

    As = cuda.shared.array((TILE32, TILE32), dtype='f4')
    Bs = cuda.shared.array((TILE32, TILE32), dtype='f4')

    c00 = 0.0; c01 = 0.0; c10 = 0.0; c11 = 0.0

    for k_block in range(0, K, TILE32):
        # BUG FIX: each thread loads 2 cols for A + 2 rows for B = 4 elements each
        ka0 = k_block + tx
        ka1 = k_block + tx + 16
        a00 = A[row0, ka0] if (row0 < M and ka0 < K) else 0.0
        a01 = A[row0, ka1] if (row0 < M and ka1 < K) else 0.0
        a10 = A[row1, ka0] if (row1 < M and ka0 < K) else 0.0
        a11 = A[row1, ka1] if (row1 < M and ka1 < K) else 0.0
        As[ty, tx] = a00
        As[ty, tx + 16] = a01
        As[ty + 16, tx] = a10
        As[ty + 16, tx + 16] = a11

        kb0 = k_block + ty
        kb1 = k_block + ty + 16
        b00 = B[kb0, col0] if (kb0 < K and col0 < N) else 0.0
        b01 = B[kb0, col1] if (kb0 < K and col1 < N) else 0.0
        b10 = B[kb1, col0] if (kb1 < K and col0 < N) else 0.0
        b11 = B[kb1, col1] if (kb1 < K and col1 < N) else 0.0
        Bs[ty, tx] = b00
        Bs[ty, tx + 16] = b01
        Bs[ty + 16, tx] = b10
        Bs[ty + 16, tx + 16] = b11

        cuda.syncthreads()

        for i in range(TILE32):
            a_lo = As[ty, i]
            a_hi = As[ty + 16, i]
            b_lo = Bs[i, tx]
            b_hi = Bs[i, tx + 16]
            c00 += a_lo * b_lo
            c01 += a_lo * b_hi
            c10 += a_hi * b_lo
            c11 += a_hi * b_hi

        cuda.syncthreads()

    if row0 < M and col0 < N: C[row0, col0] = c00
    if row0 < M and col1 < N: C[row0, col1] = c01
    if row1 < M and col0 < N: C[row1, col0] = c10
    if row1 < M and col1 < N: C[row1, col1] = c11


# ============================================================
# Benchmark
# ============================================================
def benchmark():
    sizes = [512, 1024, 2048, 4096]
    WARM, RUN = 5, 20

    print("SGEMM Benchmark: Numba CUDA vs PyTorch (cuBLAS)")
    print("================================================\n")
    print(f"{'size':>6s} | {'kernel':>15s} {'ms':>8s} {'TFLOPS':>8s} {'%bandw':>7s} | {'cuBLAS':>15s} {'ms':>8s} {'TFLOPS':>8s} | {'err':>8s}")
    print("-" * 90)

    for M in sizes:
        N, K = M, M
        flops = 2.0 * M * N * K
        mem_traffic = (M*K + K*N + M*N) * 4  # semantic DRAM bytes

        # Generate on GPU directly
        A_t = torch.randn(M, K, device='cuda', dtype=torch.float32)
        B_t = torch.randn(K, N, device='cuda', dtype=torch.float32)

        # PyTorch reference
        torch.backends.cuda.matmul.allow_tf32 = True
        for _ in range(WARM): A_t @ B_t
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(RUN): A_t @ B_t
        e.record()
        torch.cuda.synchronize()
        t_torch = s.elapsed_time(e) / RUN
        tf_torch = flops / (t_torch / 1000) / 1e12
        C_torch = A_t @ B_t

        # Numba kernels: need numpy arrays
        A_np = A_t.cpu().numpy()
        B_np = B_t.cpu().numpy()
        C_np = np.zeros((M, N), dtype=np.float32)
        A_d = cuda.to_device(A_np)
        B_d = cuda.to_device(B_np)
        C_d = cuda.device_array((M, N), dtype=np.float32)

        kernels = [
            (sgemm_naive,    (16, 16), (math.ceil(N/16), math.ceil(M/16)), "V1_naive"),
            (sgemm_tiled,    (16, 16), (math.ceil(N/TILE), math.ceil(M/TILE)), "V2_tiled_16"),
            (sgemm_tiled_32, (16, 16), (math.ceil(N/TILE32), math.ceil(M/TILE32)), "V3_tiled_32"),
        ]

        for kern_fn, block, grid, name in kernels:
            C_d.copy_to_device(np.zeros((M, N), dtype=np.float32))
            cuda.synchronize()

            # Warmup
            for _ in range(WARM):
                kern_fn[grid, block](A_d, B_d, C_d, M, N, K)
            cuda.synchronize()

            # Measure
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(RUN):
                kern_fn[grid, block](A_d, B_d, C_d, M, N, K)
            e.record()
            torch.cuda.synchronize()
            t_ms = s.elapsed_time(e) / RUN
            tflops = flops / (t_ms / 1000) / 1e12
            bw = mem_traffic / (t_ms / 1000) / 1e9

            # Error check
            C_out = C_d.copy_to_host()
            err = np.max(np.abs(C_out - C_torch.cpu().numpy()))

            print(f"{M:6d} | {name:>15s} {t_ms:8.3f} {tflops:8.3f} {bw:6.1f} | {'torch':>15s} {t_torch:8.3f} {tf_torch:8.3f} | {err:8.1e}")

        print("-" * 90)

    # Print GFLOPS summary
    print()
    print("Summary:")
    print("  FP32 peak @ ~1.6 GHz: ~8.2 TFLOPS")
    print("  Bandwidth:           192 GB/s")
    print("  Ridge point:         ~42.7 FLOPs/byte")
    print("  AI (semantic):       M/6 FLOPs/byte")
    print("  AI=512: 85, 1024:171, 2048:341, 4096:683 (all >> 43 = compute-bound)")

if __name__ == "__main__":
    benchmark()
