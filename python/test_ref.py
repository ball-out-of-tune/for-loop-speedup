import torch, torch.nn.functional as F
N=64; D=64; scale=1.0/D**0.5
q=torch.randn(1,1,N,D,device="cuda",dtype=torch.float16)
k=torch.randn(1,1,N,D,device="cuda",dtype=torch.float16)
v=torch.randn(1,1,N,D,device="cuda",dtype=torch.float16)
ref = F.scaled_dot_product_attention(q,k,v,scale=scale).float()
S=q.float()@k.float().transpose(-2,-1)*scale
P=F.softmax(S,dim=-1)
man=P@v.float()
print(f"SDPA vs Manual max diff: {(ref-man).abs().max().item():.8f}")
