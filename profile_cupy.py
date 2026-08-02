"""Minimal script for profiling CuPy kernel launches"""
import cupy as cp
import numpy as np

N = 1_000_000
data_np = np.arange(N, dtype=np.float64) * 0.001

d = cp.asarray(data_np)
_ = cp.sin(d) ** 2 + cp.cos(d) ** 2 + cp.sqrt(cp.abs(d))
cp.cuda.Stream.null.synchronize()
