#!/usr/bin/env python3
"""Run GEMM + Flash Attn baselines on RTX 5090"""
import paramiko, sys, json

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('connect.weste.seetacloud.com', port=35845,
               username='root', password='I37sjQ3GftP2', timeout=30)

def ssh(cmd):
    stdin, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode()
    err = stderr.read().decode()
    if err: print("ERR:", err[:200])
    return out.strip()

# Check python and PyTorch
print("=== System ===")
print("Python:", ssh("which python && python --version 2>&1"))
print("PyTorch:", ssh("python -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>&1"))

# GPU properties
props = ssh("""python -c "
import torch
g = torch.cuda.get_device_properties(0)
print(f'Name: {g.name}')
print(f'CC: {g.major}.{g.minor}')
print(f'SMs: {g.multi_processor_count}')
print(f'Max thr/SM: {g.max_threads_per_multi_processor}')
print(f'Max shmem/SM (default): {g.shared_mem_per_block / 1024:.0f} KB')
print(f'Regs/SM: {g.regs_per_multiprocessor}')
print(f'Max blk/SM: {g.max_blocks_per_multi_processor}')
print(f'Clock: {g.clock_rate / 1000:.0f} MHz')
print(f'Mem: {g.total_memory / 1e9:.1f} GB')
mbw = g.memory_clock_rate * g.memory_bus_width * 2 / 8 / 1e6
print(f'BW: {mbw:.0f} GB/s')
print(f'SDP backend flash: {torch.backends.cuda.flash_sdp_enabled()}')
print(f'SDP backend memeff: {torch.backends.cuda.mem_efficient_sdp_enabled()}')
" 2>&1""")
print(props)

# GEMM baseline
print("\n=== GEMM FP16 ===")
gemm = ssh("""python -c "
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
    print(f'GEMM {N}: {t:.3f}ms | {tflops:.1f} TFLOPS')
" 2>&1""")
print(gemm)

# Flash Attention baseline
print("\n=== Flash Attn FP16 ===")
attn = ssh("""python -c "
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
    print(f'Attn {N}: {t:.3f}ms | {tflops:.1f} TFLOPS')
" 2>&1""")
print(attn)

# Check CUDA toolkit
print("\n=== CUDA Toolkit ===")
print(ssh("nvcc --version 2>&1 | head -3"))
print(ssh("which nvcc 2>&1"))

# Check if filesystem is writable
print("\n=== Workspace ===")
print(ssh("ls /root/ 2>&1 | head -5"))
print(ssh("df -h / | tail -1"))

client.close()
print("\nDone.")
