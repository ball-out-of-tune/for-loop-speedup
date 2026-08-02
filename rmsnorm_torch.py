"""PyTorch RMSNorm benchmark — baseline for CUDA kernel comparison"""
import torch
import torch.nn as nn
import statistics

N_ROWS = 1000
HIDDEN_SIZE = 4096
WARMUP = 10
RUNS = 50


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


if __name__ == '__main__':
    print("=" * 55)
    print(f"  PyTorch RMSNorm: {N_ROWS} rows x {HIDDEN_SIZE} hidden")
    print("=" * 55)
    print()

    x = torch.randn(N_ROWS, HIDDEN_SIZE, dtype=torch.float32, device="cuda")

    model = RMSNorm(HIDDEN_SIZE).cuda()

    # --- verify correctness (first row vs manual) ---
    with torch.no_grad():
        y = model(x)
    rms_ref = torch.sqrt(torch.mean(x[0] ** 2) + 1e-5)
    expected_row0 = x[0] / rms_ref * model.weight[0]
    max_diff = (y[0] - expected_row0).abs().max().item()
    print(f"  Verification (row 0): max_diff = {max_diff:.2e}")
    print()

    # --- Benchmark helper ---
    def bench(fn, warmup_runs=WARMUP):
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
        total_bytes = N_ROWS * HIDDEN_SIZE * 4 * 2  # read x + write out
        bw = total_bytes / (t / 1000) / 1e9
        return t, bw

    # --- Eager mode ---
    print("--- Eager mode ---")
    with torch.no_grad():
        t_eager, bw_eager = bench(lambda: model(x))
    print(f"  {t_eager:.4f} ms  |  {bw_eager:.1f} GB/s")

    # --- torch.compile ---
    print("\n--- torch.compile ---")
    model_compiled = torch.compile(model, fullgraph=True)
    with torch.no_grad():
        for _ in range(10):
            model_compiled(x)
        torch.cuda.synchronize()
        t_comp, bw_comp = bench(lambda: model_compiled(x), warmup_runs=0)
    print(f"  {t_comp:.4f} ms  |  {bw_comp:.1f} GB/s")

    # --- Summary ---
    print()
    print("=" * 55)
    print(f"  Eager:      {t_eager:.4f} ms  ({bw_eager:.1f} GB/s)")
    print(f"  torch.comp: {t_comp:.4f} ms  ({bw_comp:.1f} GB/s)")
    if t_comp < t_eager:
        print(f"  speedup: {t_eager / t_comp:.1f}x")
    else:
        print(f"  compile didn't help (already fused in ATen)")
    print(f"\n  Our CUDA kernel: run './rms_norm_cuda'")
