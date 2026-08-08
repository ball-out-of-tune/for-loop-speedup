"""
GEMM benchmark: PyTorch (cuBLAS) baseline
Compares torch.matmul / @ / einsum across sizes.
"""
import torch
import time

torch.backends.cuda.matmul.allow_tf32 = True  # default on Ampere+

SIZES = [256, 512, 1024, 2048, 4096, 8192]
WARMUP = 10
RUNS = 50
DTYPE = torch.float32


def benchmark(name: str, fn, a, b):
    """Run fn(a, b) and report best time + TFLOPS."""
    # warmup
    for _ in range(WARMUP):
        fn(a, b)
    torch.cuda.synchronize()

    # measure
    start = time.perf_counter()
    for _ in range(RUNS):
        fn(a, b)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / RUNS

    m, k = a.shape
    n = b.shape[1]
    flops = 2 * m * n * k  # multiply-add = 2 ops
    tflops = flops / elapsed / 1e12

    bw = (a.numel() + b.numel() + m * n) * a.element_size()
    bw_gb = bw / elapsed / 1e9

    print(f"  {name:12s}  {elapsed*1e6:8.1f} us  {tflops:6.3f} TFLOPS  {bw_gb:6.1f} GB/s")
    return elapsed


print("=" * 70)
print("GEMM Benchmark: PyTorch (cuBLAS) on RTX 3050 Ti")
print(f"dtype={DTYPE}, tf32={torch.backends.cuda.matmul.allow_tf32}")
print(f"max theoretical: 5.3 TFLOPS (FP32), 192 GB/s bandwidth")
print(f"ridge point: 27.6 FLOPs/byte → AI > 27.6 = compute-bound")
print("=" * 70)

for size in SIZES:
    m = n = k = size
    a = torch.randn(m, k, device="cuda", dtype=DTYPE)
    b = torch.randn(k, n, device="cuda", dtype=DTYPE)

    data_mb = (a.numel() + b.numel() + m * n) * a.element_size() / 1e6
    flops = 2 * m * n * k
    ai = flops / ((a.numel() + b.numel() + m * n) * a.element_size())

    print(f"\n--- M=N=K={size} | {data_mb:.0f} MB traffic | AI={ai:.1f} FLOPs/byte ---")

    # @ operator (same as torch.matmul)
    benchmark("@ (matmul)", lambda x, y: x @ y, a, b)

    # verify einsum gives same result
    benchmark("einsum", lambda x, y: torch.einsum("ik,kj->ij", x, y), a, b)

    # torch.mm (strict 2D matmul, no broadcasting)
    benchmark("torch.mm", lambda x, y: torch.mm(x, y), a, b)

print("\nDone.")
