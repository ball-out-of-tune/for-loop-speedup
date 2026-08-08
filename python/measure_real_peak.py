"""
同时测: (1) matmul TFLOPS  (2) SM clock
一次运行，两个数据，没有假设。
"""
import torch
import subprocess
import threading
import time

# 同时测 1024 和 4096
sizes = [1024, 4096]
torch.backends.cuda.matmul.allow_tf32 = True

print(f"{'size':>6s}  {'t_ms':>8s}  {'TFLOPS':>8s}  {'clock_avg':>10s}  {'clock_min':>10s}  {'clock_max':>10s}  {'peak_TF32':>10s}  {'util%':>6s}")
print("-" * 80)

for m in sizes:
    n = k = m
    a = torch.randn(m, k, device="cuda", dtype=torch.float32)
    b = torch.randn(k, n, device="cuda", dtype=torch.float32)

    # 预热
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()

    # 开始采集 clock
    clocks = []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2)
                clocks.append(int(out.stdout.strip()))
            except:
                pass
            time.sleep(0.02)  # 每 20ms 采一次

    t = threading.Thread(target=poll, daemon=True)
    t.start()

    # 计时 + 跑 matmul
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10):
        a @ b
    e.record()
    torch.cuda.synchronize()
    elapsed = s.elapsed_time(e) / 10  # ms per call

    stop.set()
    t.join()

    # 计算
    flops = 2 * m * n * k
    tflops = flops / (elapsed / 1000) / 1e12
    if clocks:
        avg_clock = sum(clocks) / len(clocks)
        min_clock = min(clocks)
        max_clock = max(clocks)
    else:
        avg_clock = min_clock = max_clock = None

    # 理论峰值 @ 实测频率
    # 2560 CUDA cores × 2 FMA × clock_GHz = GFLOPS
    cores = 2560
    if avg_clock:
        peak_tf32 = cores * 2 * (avg_clock / 1000) * 2 / 1000  # TF32: 2x FP32
        util = tflops / peak_tf32 * 100
        print(f"{m:6d}  {elapsed:8.3f}  {tflops:8.3f}  {avg_clock:8.0f} MHz  {min_clock:8d} MHz  {max_clock:8d} MHz  {peak_tf32:10.3f}  {util:5.1f}%")
    else:
        print(f"{m:6d}  {elapsed:8.3f}  {tflops:8.3f}  (clock poll failed)")

print()

# 单独测 TF32 OFF
print("--- TF32 OFF ---")
torch.backends.cuda.matmul.allow_tf32 = False
for m in sizes:
    a = torch.randn(m, m, device="cuda", dtype=torch.float32)
    b = torch.randn(m, m, device="cuda", dtype=torch.float32)
    for _ in range(5): a @ b
    torch.cuda.synchronize()

    clocks = []
    stop = threading.Event()
    def poll():
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2)
                clocks.append(int(out.stdout.strip()))
            except:
                pass
            time.sleep(0.02)
    t = threading.Thread(target=poll, daemon=True)
    t.start()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10): a @ b
    e.record()
    torch.cuda.synchronize()
    elapsed = s.elapsed_time(e) / 10
    stop.set()
    t.join()

    flops = 2 * m * m * m  # n=k=m for square
    tflops = flops / (elapsed / 1000) / 1e12
    avg_clock = sum(clocks) / len(clocks) if clocks else None
    if avg_clock:
        peak_fp32 = cores * 2 * (avg_clock / 1000) / 1000  # FP32: 1x
        util = tflops / peak_fp32 * 100
        print(f"{m:6d}  {elapsed:8.3f}  {tflops:8.3f}  {avg_clock:8.0f} MHz  FP32_peak={peak_fp32:.3f}T  util={util:.1f}%")
