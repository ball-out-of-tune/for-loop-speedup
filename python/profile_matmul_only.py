"""
Only matmul, no randn. Pre-computed data.
ncu will show exactly one kernel.
"""
import torch

m = n = k = 1024
# Pre-generate on CPU, move to GPU — no randn kernel
a_cpu = torch.randn(m, k)
b_cpu = torch.randn(k, n)
a = a_cpu.cuda()
b = b_cpu.cuda()
torch.cuda.synchronize()  # H2D complete

# NOW run matmul — this is the ONLY kernel ncu should see
c = a @ b
torch.cuda.synchronize()
print("Done. The single CUDA kernel in ncu output = matmul.")
