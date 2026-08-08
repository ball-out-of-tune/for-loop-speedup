"""Benchmark: Custom CUDA GEMM vs PyTorch (cuBLAS) on RTX 5090"""
import ctypes
import torch
import time
import sys
import os

# Load compiled shared library
lib = ctypes.CDLL("/tmp/libgemm.so")

# gemm_launch(const half* A, const half* B, float* C, int M, int N, int K) -> int
lib.gemm_launch.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
lib.gemm_launch.restype = ctypes.c_int

def custom_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Run our custom WMMA GEMM kernel"""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    # Ensure fp16 input (our kernel uses half)
    if a.dtype != torch.float16:
        a = a.half()
    if b.dtype != torch.float16:
        b = b.half()

    # Our kernel produces float32 output
    c = torch.empty(M, N, dtype=torch.float32, device=a.device)

    err = lib.gemm_launch(
        a.data_ptr(), b.data_ptr(), c.data_ptr(),
        M, N, K
    )
    if err != 0:
        raise RuntimeError(f"Kernel failed with error {err}")

    return c


def pytorch_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PyTorch matmul (uses cuBLAS under the hood)"""
    # torch.matmul with fp16 input, fp32 accumulate (matching our kernel's precision)
    return torch.matmul(a.half(), b.half()).float()


def benchmark(fn, a, b, warmup=5, iters=50):
    """Run benchmark, return avg time in ms and TFLOPS"""
    for _ in range(warmup):
        fn(a, b)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    M, K = a.shape
    N = b.shape[1]
    flops = 2.0 * M * N * K
    avg_ms = (elapsed / iters) * 1000
    tflops = (flops * iters) / elapsed / 1e12
    return avg_ms, tflops


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print()

    test_cases = [
        (512, 512, 512, "square-512"),
        (1024, 1024, 1024, "square-1K"),
        (2048, 2048, 2048, "square-2K"),
        (4096, 4096, 4096, "square-4K"),
        (256, 256, 8192, "largeK-8K"),
        (512, 512, 16384, "largeK-16K"),
        (1024, 1024, 32768, "largeK-32K"),
        (1, 4096, 4096, "decode-1"),
        (1, 8192, 8192, "decode-L"),
        (4, 4096, 4096, "decode-4"),
        (8, 4096, 4096, "decode-8"),
        (16, 4096, 4096, "decode-16"),
        (32, 4096, 4096, "decode-32"),
        (64, 4096, 4096, "decode-64"),
        (128, 4096, 4096, "decode-128"),
        (512, 4096, 4096, "prefill-512"),
        (1024, 4096, 4096, "prefill-1K"),
        (2048, 4096, 4096, "prefill-2K"),
        (4096, 4096, 64, "flat-64"),
        (8192, 8192, 128, "flat-128"),
        (1000, 2000, 500, "unaligned"),
        (1, 12288, 12288, "gpt3-decode"),
        (128, 12288, 12288, "gpt3-prefill"),
        (1, 8192, 28672, "llama-MLP-decode"),
        (128, 8192, 28672, "llama-MLP-prefill"),
        (1, 1024, 3072, "qwen3-MLP-decode"),
        (128, 1024, 3072, "qwen3-MLP-prefill"),
    ]

    print(f"{'Shape':<22} {'Category':<18} {'Custom ms':>10} {'PyTorch ms':>10} {'Speedup':>8} {'CusTF':>10} {'TorchTF':>10}")
    print("-" * 96)

    wins = 0
    total_custom = 0
    total_torch = 0

    for M, N, K, cat in test_cases:
        a = torch.randn(M, K, dtype=torch.float16, device="cuda")
        b = torch.randn(K, N, dtype=torch.float16, device="cuda")

        warmup = 3
        iters = 20
        flops = 2 * M * N * K
        if flops > 1e8:
            warmup = 2; iters = 10
        if flops > 5e8:
            warmup = 1; iters = 5

        # Run custom kernel
        try:
            custom_ms, custom_tf = benchmark(custom_gemm, a, b, warmup, iters)
        except Exception as e:
            print(f"{f'{M}x{N}x{K}':<22} {cat:<18} CUSTOM ERROR: {str(e)[:40]}")
            continue

        # Run PyTorch
        torch_ms, torch_tf = benchmark(pytorch_gemm, a, b, warmup, iters)

        speedup = torch_ms / custom_ms
        if speedup >= 1.0:
            wins += 1
        total_custom += custom_ms
        total_torch += torch_ms

        print(f"{f'{M}x{N}x{K}':<22} {cat:<18} {custom_ms:10.4f} {torch_ms:10.4f} {speedup:7.2f}x {custom_tf:10.1f} {torch_tf:10.1f}")

    print("-" * 96)
    overall = total_torch / total_custom if total_custom > 0 else 0
    print(f"{'TOTAL':<22} {'':<18} {total_custom:10.2f} {total_torch:10.2f} {overall:7.2f}x")
    print(f"\nWins (Custom > PyTorch): {wins}/{len(test_cases)}")

    if wins > 0:
        print("We beat PyTorch on some test cases!")
    else:
        print("PyTorch/cuBLAS faster on all cases. Need further optimization.")

if __name__ == "__main__":
    main()
