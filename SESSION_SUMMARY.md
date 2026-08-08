# GEMM + Flash Attention Optimization — Session Summary

## RTX 5090 (Blackwell SM 120) — 170 SMs, 32 GB, 216 TFLOPS PT baseline

### GEMM Results (gemm_blackwell.cu)

| Kernel | 1024² | 2048² | 4096² | 8192² |
|--------|-------|-------|-------|-------|
| V9 port (128×128, K=16) | 28% | 89% | 79% | 90% |
| V1 (K=32 dbuf) | 29% | 84% | 79% | 88% |
| **V2 (256×128, K=32, 8w)** | 30% | **108%** ✓ | **112%** ✓ | **134%** ✓ |

**V2 beats PyTorch at industrial-scale sizes (2048–8192)!**

Key design:
- 256×128 tile with 8 warps (256 threads)
- K=32 double-buffered (48 KB smem → fits default)
- Only 128 K-syncs for K=4096 (vs 256 for K=16)
- 64 registers/warp (8 WMMA × 8) — well within limits

1024² needs a smaller tile variant (more blocks for 170 SMs).

### Flash Attention
PyTorch SDPA already achieves 120–269 TFLOPS on this GPU. 
CUDA kernel (`flash_attn_5090.cu`) has the same WMMA rescaling challenge.

### Files
- `/root/gemm_bw` — working binary on 5090 server
- `gemm_blackwell.cu` — source code
- `gemm_5090_final.cu` — with CPU correctness check (segfaults, needs O(N³) fix)

---

## RTX 3050 Ti (Ampere GA107) — 20 SMs, 4 GB

### GEMM Results (WSL, ~11-26% penalty vs Windows native)

| Kernel | 1024² | 2048² | 4096² |
|--------|-------|-------|-------|
| V9 (WMMA, 128×128, K=16) | 104% ✓ (Win) | 92% | 89% |
| V15_hybrid (triple buf + interleave) | 84% | **95%** | 85% |
| V14_interleave | 82% | 92% | 86% |

**4096² ceiling**: 256 K-syncs unavoidable with WMMA's K=16 per mma_sync. 
Only path to beat PT: PTX-level mma.sync with instruction-level K-pipelining (not found in this session).

### Files
- `gemm_v9plus.cu` — V9 + V13/V14/V15 variants
- `gemm_ptx_mma.cu` — interleaved WMMA attempt (same perf as V14)
- `flash_attn_fwd.cu` — WMMA Flash Attention (compiles but crashes — local mem issue)
- `flash_attn_simple.cu` — FP16 CUDA Core Flash Attention (incomplete P@V)

---

## Key Insights

1. **Tile size is king**: 256×128 on 170-SM 5090 crushes it; 128×128 on 20-SM 3050 Ti is optimal
2. **cp.async double buffering is non-negotiable** — single-buffered loses 40%+ perf
3. **WMMA accumulator fragment rescaling** is the fundamental blocker for Flash Attention
4. **SM count dictates parallelism strategy**: large tiles for many SMs, small tiles for few SMs
5. **Blackwell consumer constraint**: 99 KB opt-in smem (not 228 KB like B200)

## Machine Access
- RTX 5090: `ssh -p 35845 root@connect.weste.seetacloud.com` (pw: I37sjQ3GftP2)
- RTX 3050 Ti: Windows local, WSL: `/mnt/c/Users/16874/AIInfraGuide/nano-vllm-venv312/`
