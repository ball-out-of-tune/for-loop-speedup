import torch, torch.nn.functional as F  
torch.manual_seed(42)
N=64; D=64; scale=1.0/D**0.5
q=torch.randn(1,1,N,D,device='cuda',dtype=torch.float16)
k=torch.randn(1,1,N,D,device='cuda',dtype=torch.float16)  
v=torch.randn(1,1,N,D,device='cuda',dtype=torch.float16)
# Manual FP32 reference
S=q.float()@k.float().transpose(-2,-1)*scale
P=F.softmax(S,dim=-1)
ref=P@v.float()
# SDPA
out=F.scaled_dot_product_attention(q,k,v,scale=scale).float()
print(f'SDPA vs Manual max diff: {(out-ref).abs().max().item():.8f}')
print(f'First row of O (ref):  {ref[0,0,:8].tolist()}')
print(f'First row of O (sdpa): {out[0,0,:8].tolist()}')
print(f'First S values (row 0): {S[0,0,:8].tolist()}')
