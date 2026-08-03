"""PyTorch softmax benchmark at multiple hidden sizes"""
import torch
import torch.nn.functional as F
import statistics

N_ROWS = 1000
WARMUP = 10
RUNS = 30

def bench(fn, warmup=WARMUP):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(RUNS):
        start.record()
        y = fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)

print("===== PyTorch softmax at different hidden sizes =====\n")
print(f"{'HIDDEN':<12} {'time(ms)':<12} {'BW(GB/s)':<12}")
print("-" * 36)

for hidden in [4096, 8192, 16384, 32768]:
    x = torch.randn(N_ROWS, hidden, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        t = bench(lambda: F.softmax(x, dim=-1))
    bw = 2 * N_ROWS * hidden * 4 / (t/1000) / 1e9  # 1R+1W model
    print(f"{hidden:<12} {t:<12.4f} {bw:<12.1f}")
