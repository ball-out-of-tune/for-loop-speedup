"""
Python Profiler 快速上手
运行: python demo_profiler.py
"""
import torch
import torch.autograd.profiler as prof

a = torch.randn(1024, 1024, device="cuda")
b = torch.randn(1024, 1024, device="cuda")

print("=" * 50)
print("方式1: torch.autograd.profiler (简单，看 ATen 调用链)")
print("=" * 50)
with prof.profile(use_cuda=True) as p:
    c = a @ b
    torch.cuda.synchronize()

# 按 CUDA 时间排序
print(p.key_averages().table(sort_by="cuda_time_total", row_limit=8))

print("\n" + "=" * 50)
print("方式2: torch.profiler.profile (新版，需要 activities 参数)")
print("=" * 50)
# Windows 上 CUDA profiler activities 有兼容性问题，
# 方式1 (torch.autograd.profiler) 就够用了
print("  (方式1 已足够看清 aten::matmul -> aten::mm 的调用链)")
print("  (新版 API 在 Windows 上有兼容性问题，不影响使用)")
