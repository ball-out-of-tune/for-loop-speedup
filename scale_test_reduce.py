"""Scale test v4 — match original pytorch_reduce.py pattern exactly"""
import torch
import statistics

sizes = [1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000]

@torch.compile(fullgraph=True)
def compiled_sum(x):
    return torch.sum(x)

# Pre-compile
dummy = torch.arange(1000000, dtype=torch.float32, device="cuda") * 0.001
for _ in range(10):
    compiled_sum(dummy)
torch.cuda.synchronize()
del dummy
print("Compiled. Starting...\n")

WARMUP = 10
RUNS = 30

def bench(fn, x, warmup=True):
    if warmup:
        for _ in range(WARMUP):
            fn(x)
        torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(RUNS):
        start.record()
        y = fn(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    t = statistics.mean(times)
    data_mb = x.numel() * x.element_size() / (1024 * 1024)
    bw = data_mb / t * 1000  # MB / ms * 1000 = MB/s -> GB/s...

    # bw = data_mb * 1e6 bytes / (t * 1e-3 seconds) = data_mb / t * 1e3 bytes/s
    # data_mb / t * 1000 -> GB/s?
    # Actually: data_mb MB = data_mb * 1e6 bytes
    # t ms = t/1000 seconds
    # bw = data_mb * 1e6 / (t/1000) = data_mb / t * 1e9 bytes/s = data_mb / t GB/s
    # So: bw = data_mb / t (since data_mb is in MB and t is in ms)
    # 4 MB / 0.032 ms = 125 GB/s ✓
    bw = data_mb / t
    return t, bw, y.item()

print(f"{'N':>12s}  {'eager ms':>10s}  {'eager BW':>10s}  {'comp ms':>10s}  {'comp BW':>10s}  {'ratio':>8s}")
print("-" * 65)

for n in sizes:
    x = torch.arange(n, dtype=torch.float32, device="cuda") * 0.001

    t1, bw1, r1 = bench(torch.sum, x, warmup=True)
    t2, bw2, r2 = bench(compiled_sum, x, warmup=True)

    print(f"  {n:>10,d}  {t1:>10.4f}  {bw1:>9.1f}  {t2:>10.4f}  {bw2:>9.1f}  {t2/t1:>7.2f}x")

    # check first result against our baseline
    if n == 1_000_000 and bw1 > 200:
        print(f"  WARNING: BW={bw1:.0f} impossibly high (>192 theoretical) -> timing broken!")
    del x
