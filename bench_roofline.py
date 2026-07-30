"""
Measure machine compute peak and memory bandwidth peak
"""
import time
import numpy as np
import numba
import statistics

N = 1_000_000  # 1M float64 = 8 MB


# ============================================================
#  Memory bandwidth: STREAM triad  a[i] = b[i] + c[i] * d[i]
#  Per element: 3 reads + 1 write = 32 bytes, 2 FLOPs
#  Arithmetic intensity extremely low -> pure memory bottleneck
# ============================================================
@numba.njit
def stream_triad(a, b, c, d):
    for i in range(len(a)):
        a[i] = b[i] + c[i] * d[i]


# ============================================================
#  Compute peak: unrolled ops to feed the FP pipeline
#  Per element: many FLOPs, only 1 read + 1 write = 16 bytes
#  Arithmetic intensity very high -> pure compute bottleneck
# ============================================================
@numba.njit
def compute_peak(x):
    for i in range(len(x)):
        v = x[i]
        # 4 independent chains to keep the FP pipeline full
        a = v + 1.0
        b = v * 0.5
        c = v - 0.3
        d = v / 2.0
        for _ in range(200):
            a = a * 0.999 + 0.001
            b = b * 1.001 - 0.001
            c = c * 0.998 + 0.002
            d = d * 1.002 - 0.002
        x[i] = a + b + c + d


if __name__ == '__main__':
    print("=" * 55)
    print("  Machine Roofline Benchmark")
    print("=" * 55)
    print()

    a = np.ones(N, dtype=np.float64)
    b = np.ones(N, dtype=np.float64)
    c = np.ones(N, dtype=np.float64)
    d_arr = np.ones(N, dtype=np.float64)

    # --- Memory bandwidth ---
    stream_triad(a, b, c, d_arr)  # warmup + compile

    times_mem = []
    for _ in range(10):
        t0 = time.perf_counter()
        stream_triad(a, b, c, d_arr)
        times_mem.append(time.perf_counter() - t0)

    avg_time = statistics.mean(times_mem)
    total_bytes = N * 32  # 3 reads + 1 write, 8 bytes each
    bw_gb_s = total_bytes / avg_time / 1e9

    print(f"[Memory Bandwidth] STREAM triad")
    print(f"  time:        {avg_time:.6f} s")
    print(f"  data:        {total_bytes / 1e9:.4f} GB")
    print(f"  bandwidth:   {bw_gb_s:.2f} GB/s")
    print()

    # --- Compute peak ---
    x = np.ones(N, dtype=np.float64)
    compute_peak(x)  # warmup + compile

    times_cpu = []
    for _ in range(5):
        t0 = time.perf_counter()
        compute_peak(x)
        times_cpu.append(time.perf_counter() - t0)

    avg_time = statistics.mean(times_cpu)
    # 4 chains x 200 iterations x 2 FLOPs = 1600 FLOPs per element
    total_flops = N * 4 * 200 * 2
    gflops = total_flops / avg_time / 1e9

    print(f"[Compute Peak] 4-way unrolled x 200")
    print(f"  time:        {avg_time:.6f} s")
    print(f"  work:        {total_flops / 1e9:.2f} GFLOPs")
    print(f"  throughput:  {gflops:.2f} GFLOPs/s")
    print()

    # --- Roofline ---
    ridge_point = gflops / bw_gb_s

    print("=" * 55)
    print("  Roofline Model")
    print("=" * 55)
    print(f"  Compute peak:     {gflops:.1f} GFLOPs/s")
    print(f"  Memory bandwidth: {bw_gb_s:.1f} GB/s")
    print(f"  Ridge point:      {ridge_point:.1f} FLOPs/Byte")
    print()
    print("  How to read this:")
    print(f"  - AI < {ridge_point:.1f} FLOPs/Byte -> memory-bound")
    print(f"  - AI > {ridge_point:.1f} FLOPs/Byte -> compute-bound")
    print()
    print("  Your hot_loop.py functions:")
    light_ai = 2
    heavy_ai = 200
    print(f"  - light:  AI ~{light_ai}   FLOPs/Byte -> {'memory-bound' if light_ai < ridge_point else 'compute-bound'}")
    print(f"  - heavy:  AI ~{heavy_ai}  FLOPs/Byte -> {'memory-bound' if heavy_ai < ridge_point else 'compute-bound'}")
