"""Diagnose: why is torch.compile reduce 13x slower?"""
import torch
import statistics

N = 1_000_000
x = torch.arange(N, dtype=torch.float32, device="cuda") * 0.001

WARMUP = 10
RUNS = 50

def bench(name, fn):
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

    t_avg = statistics.mean(times)
    t_min = min(times)
    print(f"  {name:20s}  avg={t_avg:.4f} ms  min={t_min:.4f} ms  result={y.item():.4f}")


print("=" * 60)
print("  torch.compile reduce: diagnose")
print("=" * 60)
print()

# ---- 1. Eager baseline ----
print("[1] Eager mode (torch.sum directly)")
bench("eager.sum", lambda x: torch.sum(x))


# ---- 2. torch.compile with fullgraph=True ----
# fullgraph=True -> errors if graph breaks (no silent fallback)
print("\n[2] torch.compile(fullgraph=True)")

@torch.compile(fullgraph=True)
def compiled_fullgraph(x):
    return torch.sum(x)

# extra warmup for compilation
for _ in range(10):
    compiled_fullgraph(x)
torch.cuda.synchronize()

bench("compile.fullgraph", lambda x: compiled_fullgraph(x))


# ---- 3. torch.compile with mode="reduce-overhead" ----
# reduce-overhead mode uses CUDA graphs to amortize launch overhead
print("\n[3] torch.compile(mode='reduce-overhead')")

@torch.compile(mode="reduce-overhead", fullgraph=True)
def compiled_reduce_oh(x):
    return torch.sum(x)

for _ in range(10):
    compiled_reduce_oh(x)
torch.cuda.synchronize()

bench("compile.reduce-oh", lambda x: compiled_reduce_oh(x))


# ---- 4. Larger input: maybe 1M is too small for compile? ----
print("\n[4] Larger input (10M elements)")

N2 = 10_000_000
x2 = torch.arange(N2, dtype=torch.float32, device="cuda") * 0.001

bench("eager.sum 10M", lambda x2: torch.sum(x2))

for _ in range(10):
    compiled_fullgraph(x2)
torch.cuda.synchronize()

bench("compile 10M", lambda x2: compiled_fullgraph(x2))


# ---- 5. torch.compile on a MANUAL reduction loop ----
# This is where compile SHOULD help -- fusing multiple ops
print("\n[5] Manual reduction (multiple ops -> fusion should help)")

def manual_sum(x):
    # Simulate what a naive user might write:
    # split into chunks, sum each, then sum chunks
    result = torch.tensor(0.0, device=x.device)
    chunk_size = 10000
    for i in range(0, len(x), chunk_size):
        result = result + x[i:i+chunk_size].sum()
    return result

@torch.compile(fullgraph=True)
def manual_sum_compiled(x):
    result = torch.tensor(0.0, device=x.device)
    chunk_size = 10000
    for i in range(0, len(x), chunk_size):
        result = result + x[i:i+chunk_size].sum()
    return result

bench("manual.eager", lambda x: manual_sum(x))

for _ in range(10):
    manual_sum_compiled(x)
torch.cuda.synchronize()

bench("manual.compile", lambda x: manual_sum_compiled(x))


# ---- 6. Profiler check: what kernels does compile launch? ----
print("\n[6] Kernel trace (first 3 runs after compile)")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    for _ in range(3):
        compiled_fullgraph(x)
        torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
