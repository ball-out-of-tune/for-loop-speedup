"""PyTorch Softmax benchmark — baseline for CUDA kernel comparison

Softmax:  exp(x_i - max) / sum(exp(x_j - max))

Same category as RMSNorm (reduce + element-wise fusion), but:
  - Needs TWO reduces (max then sum) — not one
  - exp() is a transcendental (~4-8x more expensive than multiply)
  - Higher AI than RMSNorm (more FLOPs per byte), but still mostly BW-bound
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics

N_ROWS = 1000
HIDDEN_SIZE = 4096
WARMUP = 10
RUNS = 50


class SoftmaxModel(nn.Module):
    """Wrapper so torch.compile can trace F.softmax cleanly."""
    def forward(self, x):
        return F.softmax(x, dim=-1)


def bench(fn, warmup_runs=WARMUP):
    """CUDA-event based benchmark, returns (mean_ms, bw_gb_s)."""
    for _ in range(warmup_runs):
        fn()
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

    t = statistics.mean(times)
    # Read x once, write out once (intermediate reduce results are tiny)
    total_bytes = N_ROWS * HIDDEN_SIZE * 4 * 2
    bw = total_bytes / (t / 1000) / 1e9
    return t, bw


if __name__ == '__main__':
    print("=" * 60)
    print(f"  PyTorch Softmax: {N_ROWS} rows x {HIDDEN_SIZE} hidden")
    print("=" * 60)
    print()

    x = torch.randn(N_ROWS, HIDDEN_SIZE, dtype=torch.float32, device="cuda")

    model = SoftmaxModel()

    # --- Verify correctness (row 0 vs manual) ---
    with torch.no_grad():
        y = model(x)
    x0 = x[0]
    x0_max = x0.max()
    exp_sum = (x0 - x0_max).exp().sum()
    expected_row0 = (x0 - x0_max).exp() / exp_sum
    max_diff = (y[0] - expected_row0).abs().max().item()
    print(f"  Verification (row 0): max_diff = {max_diff:.2e}")
    print()

    # --- Eager mode ---
    print("--- Eager mode ---")
    with torch.no_grad():
        t_eager, bw_eager = bench(lambda: model(x))
    print(f"  {t_eager:.4f} ms  |  {bw_eager:.1f} GB/s")

    # --- torch.compile ---
    print("\n--- torch.compile ---")
    try:
        model_compiled = torch.compile(model, fullgraph=True)
        with torch.no_grad():
            for _ in range(10):
                model_compiled(x)
            torch.cuda.synchronize()
            t_comp, bw_comp = bench(lambda: model_compiled(x))
        print(f"  {t_comp:.4f} ms  |  {bw_comp:.1f} GB/s")
    except Exception as e:
        print(f"  SKIP: torch.compile failed ({type(e).__name__}).")
        print(f"  This is expected on Windows — Triton not available.")
        t_comp, bw_comp = -1, -1

    # --- Summary ---
    print()
    print("=" * 60)
    print(f"  Eager:      {t_eager:.4f} ms  ({bw_eager:.1f} GB/s)")
    if t_comp > 0:
        print(f"  torch.comp: {t_comp:.4f} ms  ({bw_comp:.1f} GB/s)")
        if t_comp < t_eager:
            print(f"  speedup:    {t_eager / t_comp:.1f}x")
        else:
            print(f"  compile didn't help much — reduce ops can't be fused")
    else:
        print(f"  torch.comp: not available (no Triton on Windows)")
    print()
    print(f"  Run './softmax_cuda' for our CUDA kernel comparison")
