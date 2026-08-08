#!/usr/bin/env python3
"""Check RTX 5090 specs via SSH"""
import paramiko, sys

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('connect.weste.seetacloud.com', port=35845,
               username='root', password='I37sjQ3GftP2', timeout=30)

def run(cmd):
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if err:
        print("ERR:", err[:200])
    return out

# GPU info
print("=== GPU INFO ===")
print(run("nvidia-smi --query-gpu=name,compute_cap,memory.total,clocks.max.sm,clocks.max.memory --format=csv,noheader"))

# PyTorch info
print("\n=== PyTorch ===")
print(run("python3 -c 'import torch; print(torch.__version__); print(torch.version.cuda)'"))

# Detailed GPU properties
print("\n=== GPU Properties ===")
props_script = """
import torch
g = torch.cuda.get_device_properties(0)
print(f'Compute Capability: {g.multi_processor_count} SMs, CC {g.major}.{g.minor}')
print(f'Max threads per SM: {g.max_threads_per_multi_processor}')
print(f'Shared memory per SM: {g.shared_mem_per_block_optin / 1024:.0f} KB (default: {g.shared_mem_per_block / 1024:.0f} KB)')
print(f'Registers per SM: {g.regs_per_multiprocessor}')
print(f'Max blocks per SM: {g.max_blocks_per_multi_processor}')
print(f'Clock rate: {g.clock_rate / 1000:.0f} MHz')
print(f'Memory: {g.total_memory / 1e9:.1f} GB, {g.memory_clock_rate / 1000:.0f} MHz, {g.memory_bus_width}-bit')
bw = g.memory_clock_rate * g.memory_bus_width * 2 / 8 / 1e6
print(f'Bandwidth: {bw:.0f} GB/s')

# SDPA backends
print(f'Flash SDP: {torch.backends.cuda.flash_sdp_enabled()}')
print(f'Mem-Eff SDP: {torch.backends.cuda.mem_efficient_sdp_enabled()}')
print(f'Math SDP: {torch.backends.cuda.math_sdp_enabled()}')
"""
print(run(f"python3 -c '{props_script}'"))

# GEMM baseline at different sizes
print("\n=== GEMM Baseline (FP16, TF32 enabled) ===")
gemm_script = """
import torch, time
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

for N in [1024, 2048, 4096, 8192]:
    a = torch.randn(N, N, device='cuda', dtype=torch.float16)
    b = torch.randn(N, N, device='cuda', dtype=torch.float16)
    # warmup
    for _ in range(10): torch.mm(a, b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        torch.mm(a, b)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / 100 * 1000
    flops = 2.0 * N * N * N
    tflops = flops / (t / 1000) / 1e12
    print(f'{N}: {t:.3f}ms, {tflops:.1f} TFLOPS')
"""
print(run(f"python3 -c '{gemm_script}'"))

# Flash Attention baseline
print("\n=== Flash Attn Baseline (FP16) ===")
attn_script = """
import torch, torch.nn.functional as F, time

for N in [1024, 2048, 4096, 8192]:
    q = torch.randn(1, 8, N, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 8, N, 64, device='cuda', dtype=torch.float16)
    v = torch.randn(1, 8, N, 64, device='cuda', dtype=torch.float16)
    for _ in range(10): F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    reps = 50
    for _ in range(reps): F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    t = (time.perf_counter() - t0) / reps * 1000
    flops = 4.0 * 8 * N * N * 64
    tflops = flops / (t / 1000) / 1e12
    print(f'{N}: {t:.3f}ms, {tflops:.1f} TFLOPS')
"""
print(run(f"python3 -c '{attn_script}'"))

client.close()
print("\nDone.")
