import torch
a = torch.randn(4096, 4096).cuda()
b = torch.randn(4096, 4096).cuda()
torch.cuda.synchronize()
c = a @ b
torch.cuda.synchronize()
print("Done.")
