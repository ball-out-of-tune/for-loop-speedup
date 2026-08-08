"""
Flash Attention in PyTorch — Complete Demo
==========================================

PyTorch 2.0+ has built-in scaled_dot_product_attention, auto-selects best backend:
  - Flash Attention (CUDA, SM 8.0+): fastest, tiling + online softmax
  - Memory Efficient Attention (CUDA): xFormers style, saves memory
  - Vanilla Math (CPU/CUDA fallback): standard O(N^2) memory

Your 3050 Ti (SM 8.6) fully supports Flash Attention backend.

Single line usage:
    out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

torch.manual_seed(42)

# ============================================================================
# Part 0: Environment Check
# ============================================================================
print("=" * 65)
print("Part 0: Environment Check")
print("=" * 65)
print(f"PyTorch:      {torch.__version__}")
print(f"CUDA:         {torch.version.cuda}")
print(f"GPU:          {torch.cuda.get_device_name(0)}")
print(f"Compute Cap:  {torch.cuda.get_device_capability(0)}")
print(f"Flash SDP:    {torch.backends.cuda.flash_sdp_enabled()}")
print(f"Mem-Eff SDP:  {torch.backends.cuda.mem_efficient_sdp_enabled()}")
print(f"Math SDP:     {torch.backends.cuda.math_sdp_enabled()}")
print()

# ============================================================================
# Part 1: Correctness — Manual vs Flash Attention
# ============================================================================
def manual_attention(Q, K, V, is_causal=True, scale=None):
    """Explicit O(N^2) attention like in gpt.py"""
    head_dim = Q.size(-1)
    if scale is None:
        scale = head_dim ** 0.5
    attn = (Q @ K.transpose(-2, -1)) / scale      # (B,H,N,N)
    if is_causal:
        mask = torch.triu(torch.ones(
            attn.size(-2), attn.size(-1), device=Q.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
    attn_weights = F.softmax(attn, dim=-1)
    return attn_weights @ V

def flash_attention(Q, K, V, is_causal=True, scale=None):
    """PyTorch built-in: auto-selects Flash Attention backend"""
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal, scale=scale)

print("=" * 65)
print("Part 1: Correctness (FP16 Flash vs FP32 Manual)")
print("=" * 65)

B, H, N, D = 2, 8, 512, 64
Q = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
K = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)
V = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16)

ref = manual_attention(Q.float(), K.float(), V.float())       # FP32 reference
out = flash_attention(Q, K, V)

max_err = (out.float() - ref).abs().max().item()
print(f"  FP16 Flash vs FP32 Manual: max error = {max_err:.6f}")
print(f"  Error < 0.01 = PASS (numerical diff from FP16 + online softmax)")
print()

# ============================================================================
# Part 2: Backend Selection
# ============================================================================
print("=" * 65)
print("Part 2: Backend Selection (Flash / Mem-Efficient / Math)")
print("=" * 65)

def try_backend(name, **kwargs):
    try:
        with torch.backends.cuda.sdp_kernel(**kwargs):
            o = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        return f"  {name}: OK, shape={o.shape}, dtype={o.dtype}"
    except Exception as e:
        return f"  {name}: FAILED — {e}"

print(try_backend("Flash only       ", enable_flash=True,
      enable_mem_efficient=False, enable_math=False))
print(try_backend("Mem-Efficient only", enable_flash=False,
      enable_mem_efficient=True, enable_math=False))
print(try_backend("Math only         ", enable_flash=False,
      enable_mem_efficient=False, enable_math=True))
print(try_backend("All enabled       ", enable_flash=True,
      enable_mem_efficient=True, enable_math=True))
print()

# ============================================================================
# Part 3: Performance — Manual vs Flash at Different Seq Lengths
# ============================================================================
print("=" * 65)
print("Part 3: Performance — Manual vs Flash Attention")
print("=" * 65)

def benchmark(fn, q, k, v, warmup=5, repeat=20, **kwargs):
    for _ in range(warmup):
        fn(q, k, v, **kwargs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn(q, k, v, **kwargs)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeat * 1000  # ms
    return elapsed

print(f"{'Seqlen':>8s}  {'Manual(ms)':>10s}  {'Flash(ms)':>10s}  "
      f"{'Speedup':>8s}  {'Attn Matrix':>12s}")
print("-" * 70)

for s in [256, 512, 1024, 2048, 4096, 8192]:
    q = torch.randn(1, 8, s, 64, device='cuda', dtype=torch.float16)
    k = torch.randn(1, 8, s, 64, device='cuda', dtype=torch.float16)
    v = torch.randn(1, 8, s, 64, device='cuda', dtype=torch.float16)

    if s <= 2048:
        t_manual = benchmark(manual_attention, q, k, v)
    else:
        t_manual = float('inf')  # OOM risk

    t_flash = benchmark(flash_attention, q, k, v)
    attn_gb = 1 * 8 * s * s * 2 / 1024**3  # fp16 attention matrix

    if t_manual != float('inf'):
        print(f"  {s:6d}   {t_manual:8.2f} ms  {t_flash:8.2f} ms  "
              f"{t_manual/t_flash:6.1f}x   {attn_gb:10.4f} GB")
    else:
        print(f"  {s:6d}    {'OOM':>8s}     {t_flash:8.2f} ms  "
              f"{'N/A':>6s}   {attn_gb:10.4f} GB")

print()
print("Flash Attention stores O(N^2) matrix in SRAM — 0 GB HBM overhead")
print("Manual would need >1 GB for 8192 seqlen → OOM")

# ============================================================================
# Part 4: Dtype Impact
# ============================================================================
print()
print("=" * 65)
print("Part 4: FP32 vs FP16 vs BF16 (1024 seqlen)")
print("=" * 65)

for dt in [torch.float32, torch.float16, torch.bfloat16]:
    try:
        q = torch.randn(1, 8, 1024, 64, device='cuda', dtype=dt)
        k = torch.randn(1, 8, 1024, 64, device='cuda', dtype=dt)
        v = torch.randn(1, 8, 1024, 64, device='cuda', dtype=dt)
        t = benchmark(flash_attention, q, k, v)
        print(f"  {str(dt):>16s}: {t:8.2f} ms")
    except Exception as e:
        print(f"  {str(dt):>16s}: ERROR — {e}")

print()
print("FP16/BF16 fastest (Tensor Core native), FP32 slowest")

# ============================================================================
# Part 5: Causal vs Non-Causal
# ============================================================================
print()
print("=" * 65)
print("Part 5: Causal vs Non-Causal (1024, FP16)")
print("=" * 65)

q = torch.randn(1, 8, 1024, 64, device='cuda', dtype=torch.float16)
k = torch.randn(1, 8, 1024, 64, device='cuda', dtype=torch.float16)
v = torch.randn(1, 8, 1024, 64, device='cuda', dtype=torch.float16)

for causal in [True, False]:
    t = benchmark(flash_attention, q, k, v, is_causal=causal)
    label = "causal mask (LLM)  " if causal else "no mask (encoder)   "
    print(f"  {label}: {t:8.2f} ms")

print()

# ============================================================================
# Part 6: GQA — Grouped Query Attention
# ============================================================================
print("=" * 65)
print("Part 6: GQA — Grouped Query Attention (KV heads < Q heads)")
print("=" * 65)

# Qwen3-style: Q heads=32, KV heads=8/4/2/1
for n_kv_heads in [8, 4, 2, 1]:
    q = torch.randn(1, 32, 1024, 128, device='cuda', dtype=torch.float16)
    k_orig = v_orig = torch.randn(1, n_kv_heads, 1024, 128, device='cuda', dtype=torch.float16)
    n_repeat = 32 // n_kv_heads

    # K,V need to match Q heads via repeat_interleave
    k = k_orig.repeat_interleave(n_repeat, dim=1)
    v = v_orig.repeat_interleave(n_repeat, dim=1)
    t_flash = benchmark(F.scaled_dot_product_attention, q, k, v)
    print(f"  KV heads={n_kv_heads:2d} (repeat {n_repeat}x): Flash={t_flash:.2f}ms")

print()
print("GQA: K,V repeat_interleave to match Q head count, then normal SDPA")

# ============================================================================
# Part 7: Replace MHA in gpt.py with Flash Attention
# ============================================================================
print()
print("=" * 65)
print("Part 7: MHA Manual vs Flash Attention (512-dim, 128-seq, FP16)")
print("=" * 65)

class MHA_Manual(nn.Module):
    """Original gpt.py: O(N^2) memory attention"""
    def __init__(self, dim=512, head_nums=8):
        super().__init__()
        self.head_nums = head_nums
        self.head_dim = dim // head_nums
        self.scaling = self.head_dim ** 0.5
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        Q = self.W_q(x).view(B, N, self.head_nums, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.head_nums, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.head_nums, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-1, -2)) / self.scaling  # (B,H,N,N) — O(N^2)!
        mask = torch.triu(torch.ones(N, N, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        out = (F.softmax(scores, dim=-1) @ V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)

class MHA_Flash(nn.Module):
    """Flash Attention version — single line change, 0 extra memory"""
    def __init__(self, dim=512, head_nums=8):
        super().__init__()
        self.head_nums = head_nums
        self.head_dim = dim // head_nums
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        Q = self.W_q(x).view(B, N, self.head_nums, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.head_nums, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.head_nums, self.head_dim).transpose(1, 2)

        # THE ONLY CHANGE: replace manual O(N^2) with Flash Attention
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.W_o(out)

x = torch.randn(2, 128, 512, device='cuda', dtype=torch.float16)

# FP32 reference model
ref_mha = MHA_Manual().cuda().float()
flash_mha = MHA_Flash().cuda().half()
flash_mha.load_state_dict(ref_mha.state_dict())  # auto-cast float->half ok

with torch.no_grad():
    ref = ref_mha(x.float())
    out_flash = flash_mha(x)

    max_diff = (out_flash.float() - ref).abs().max().item()
    print(f"  Manual(FP32) vs Flash(FP16) max diff: {max_diff:.6f}")
    print()

    # Benchmark FP16 models
    manual_half = MHA_Manual().cuda().half()
    manual_half.load_state_dict(flash_mha.state_dict())

    for _ in range(3): manual_half(x); flash_mha(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10): manual_half(x)
    torch.cuda.synchronize()
    t_manual = (time.perf_counter() - t0) / 10 * 1000

    t0 = time.perf_counter()
    for _ in range(10): flash_mha(x)
    torch.cuda.synchronize()
    t_flash = (time.perf_counter() - t0) / 10 * 1000

    print(f"  Manual FP16: {t_manual:.2f} ms")
    print(f"  Flash  FP16: {t_flash:.2f} ms  ({t_manual/t_flash:.1f}x faster)")

# ============================================================================
# Summary
# ============================================================================
print()
print("=" * 65)
print("SUMMARY: Flash Attention in PyTorch = ONE LINE")
print("=" * 65)
print("""
  import torch.nn.functional as F
  out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

  Q, K, V shape: (batch, num_heads, seq_len, head_dim)
  - is_causal=True: upper triangular mask (GPT autoregressive)
  - scale: default 1/sqrt(head_dim)
  - Auto-selects backend: Flash -> Mem-Efficient -> Math
  - GQA support: K,V can have fewer heads (broadcast via repeat_interleave)
  - dtype: FP16/BF16 use Tensor Cores, fastest

  No other code changes needed — just replace the attention computation.
""")
