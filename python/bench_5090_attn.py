"""Flash Attention benchmark for RTX 5090"""
import torch, torch.nn.functional as F, time
from torch.nn.attention import sdpa_kernel, SDPBackend
torch.manual_seed(42)

print("=== SDPA Backend Check ===")
q = torch.randn(1, 8, 1024, 64, device='cuda', dtype=torch.float16)
k = torch.randn(1, 8, 1024, 64, device='cuda', dtype=torch.float16)
v = torch.randn(1, 8, 1024, 64, device='cuda', dtype=torch.float16)

for name, backends in [
    ("FLASH", [SDPBackend.FLASH_ATTENTION]),
    ("MEM_EFF", [SDPBackend.EFFICIENT_ATTENTION]),
    ("MATH", [SDPBackend.MATH]),
]:
    try:
        with sdpa_kernel(backends):
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        print(f"  {name}: OK")
    except RuntimeError as e:
        print(f"  {name}: FAILED - {str(e)[:80]}")

print(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
print(f"MemEff enabled: {torch.backends.cuda.mem_efficient_sdp_enabled()}")

def bench(fn, q, k, v, warm=10, reps=50, **kw):
    for _ in range(warm): fn(q, k, v, **kw)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): fn(q, k, v, **kw)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000

def manual(Q, K, V, is_causal=True, scale=None):
    d = Q.size(-1)
    if scale is None: scale = d ** 0.5
    attn = (Q @ K.transpose(-2, -1)) / scale
    if is_causal:
        m = torch.triu(torch.ones(attn.size(-2), attn.size(-1), device=Q.device), diagonal=1).bool()
        attn = attn.masked_fill(m, float('-inf'))
    return F.softmax(attn, dim=-1) @ V

print()
print("=== Performance: Flash vs Manual ===")
print(f"{'N':>6s}  {'Flash ms':>9s}  {'Manual ms':>9s}  {'Speedup':>7s}  {'Flash TF':>8s}")
print("-" * 50)

for N in [512, 1024, 2048, 4096, 8192]:
    q = torch.randn(1, 8, N, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 8, N, 64, device='cuda', dtype=torch.float16)
    v = torch.randn(1, 8, N, 64, device='cuda', dtype=torch.float16)
    tf = bench(F.scaled_dot_product_attention, q, k, v)
    if N <= 4096:
        tm = bench(manual, q, k, v)
    else:
        try:
            tm = bench(manual, q, k, v, warm=3, reps=10)
        except:
            tm = -1
    flops = 4.0 * 8 * N * N * 64
    tflops = flops / (tf / 1000) / 1e12
    if tm > 0:
        print(f"{N:6d}  {tf:8.3f}ms  {tm:8.1f}ms  {tm/tf:5.1f}x  {tflops:7.1f}T")
    else:
        print(f"{N:6d}  {tf:8.3f}ms     OOM      N/A  {tflops:7.1f}T")

print()
print("=== Accuracy Check ===")
q = torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16)
k = torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16)
v = torch.randn(1, 8, 512, 64, device='cuda', dtype=torch.float16)
ref = manual(q.float(), k.float(), v.float())
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print(f"Max error (FP16 vs FP32 ref): {(out.float()-ref).abs().max().item():.6f}")

print()
print("=== GEMM vs Flash Attn ===")
for N in [1024, 2048, 4096]:
    a = torch.randn(N, N, device='cuda', dtype=torch.float16)
    b = torch.randn(N, N, device='cuda', dtype=torch.float16)
    tg = bench(torch.mm, a, b, warm=20, reps=100)
    gf = 2.0 * N * N * N
    gt = gf / (tg / 1000) / 1e12

    q = torch.randn(1, 8, N, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 8, N, 64, device='cuda', dtype=torch.float16)
    v = torch.randn(1, 8, N, 64, device='cuda', dtype=torch.float16)
    ta = bench(F.scaled_dot_product_attention, q, k, v)
    af = 4.0 * 8 * N * N * 64
    atf = af / (ta / 1000) / 1e12

    print(f"  {N}: GEMM={gt:.0f}T | FlashAttn={atf:.0f}T | ratio={atf/gt:.2f}x")

print("\nDone.")
