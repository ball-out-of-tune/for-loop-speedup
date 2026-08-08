#!/bin/bash
# RTX 5090 Baseline Script
PY=/root/miniconda3/bin/python

echo "=== System ==="
$PY --version
$PY -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" 2>&1
nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm,clocks.max.memory --format=csv,noheader 2>&1

echo ""
echo "=== GPU Properties ==="
$PY -c "
import torch
g = torch.cuda.get_device_properties(0)
print(f'Name: {g.name}')
print(f'CC: {g.major}.{g.minor}')
print(f'SMs: {g.multi_processor_count}')
print(f'Shared mem/block (default): {g.shared_mem_per_block / 1024:.0f} KB')
print(f'Registers/SM: {g.regs_per_multiprocessor}')
print(f'Max blocks/SM: {g.max_blocks_per_multi_processor}')
print(f'Clock: {g.clock_rate / 1000:.0f} MHz')
mbw = g.memory_clock_rate * g.memory_bus_width * 2 / 8 / 1e6
print(f'Bandwidth: {mbw:.0f} GB/s')
print(f'Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}')
print(f'Mem-Eff SDP enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}')
" 2>&1

echo ""
echo "=== GEMM FP16 Baseline ==="
$PY -c "
import torch, time
torch.backends.cuda.matmul.allow_tf32 = True
for N in [1024, 2048, 4096, 8192]:
    a = torch.randn(N, N, device='cuda', dtype=torch.float16)
    b = torch.randn(N, N, device='cuda', dtype=torch.float16)
    for _ in range(30): torch.mm(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    reps = 100
    for _ in range(reps): torch.mm(a, b)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / reps * 1000
    flops = 2.0 * N * N * N
    tflops = flops / (t / 1000) / 1e12
    print(f'GEMM {N:5d}: {t:8.3f}ms | {tflops:8.1f} TFLOPS')
" 2>&1

echo ""
echo "=== Flash Attn FP16 Baseline ==="
$PY -c "
import torch, torch.nn.functional as F, time
for N in [1024, 2048, 4096, 8192]:
    q = torch.randn(1, 8, N, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 8, N, 64, device='cuda', dtype=torch.float16)
    v = torch.randn(1, 8, N, 64, device='cuda', dtype=torch.float16)
    for _ in range(20): F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    reps = 100
    for _ in range(reps): F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / reps * 1000
    flops = 4.0 * 8 * N * N * 64
    tflops = flops / (t / 1000) / 1e12
    print(f'Attn {N:5d}: {t:8.3f}ms | {tflops:8.1f} TFLOPS')
" 2>&1

echo ""
echo "=== Check nvcc ==="
which nvcc 2>&1 || echo "nvcc not found"
nvcc --version 2>&1 | head -2 || echo "no nvcc"

echo ""
echo "=== Disk ==="
df -h / | tail -1
