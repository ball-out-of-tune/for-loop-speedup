"""
ncu 测试脚本: 只做 matmul，方便看清楚 kernel 名字
"""
import torch

# 在 CPU 上生成数据，然后传到 GPU（避免 randn kernel 干扰）
a = torch.randn(1024, 1024).cuda()
b = torch.randn(1024, 1024).cuda()
torch.cuda.synchronize()

# 就这一行计算
c = a @ b
torch.cuda.synchronize()
print("Done.")
