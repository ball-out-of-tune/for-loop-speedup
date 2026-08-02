"""Minimal kernel for profiling with ncu"""
from numba import cuda
import numpy as np
import math

N = 1_000_000
data_np = np.arange(N, dtype=np.float64) * 0.001

@cuda.jit
def kernel(data, ans, n):
    i = cuda.grid(1)
    if i < n:
        x = data[i]
        ans[i] = math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x))

threads = 256
blocks = (N + threads - 1) // threads
d = cuda.to_device(data_np)
a = cuda.device_array(N, dtype=np.float64)

kernel[blocks, threads](d, a, N)
cuda.synchronize()
