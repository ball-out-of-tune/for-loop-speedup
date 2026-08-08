"""
Profile script for ncu: see exactly which CUDA kernels PyTorch calls.

Usage:
  ncu --print-summary python profile_kernels.py
  ncu --set full python profile_kernels.py
  ncu --kernel-name regex:ampere python profile_kernels.py
  ncu --page details python profile_kernels.py
"""
import torch

m = n = k = 1024
a = torch.randn(m, k, device="cuda", dtype=torch.float32)
b = torch.randn(k, n, device="cuda", dtype=torch.float32)

torch.backends.cuda.matmul.allow_tf32 = True

# Simple matmul
c = a @ b
torch.cuda.synchronize()

print("Done. Check ncu output above for kernel names.")
print("Look for lines containing 'ampere_' or 'sm' in the kernel column.")
