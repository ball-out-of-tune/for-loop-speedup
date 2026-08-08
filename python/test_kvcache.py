"""验证 forward_kv_cache 和原始 forward 输出是否一致"""
import torch
torch.manual_seed(42)
from qwen3_naive import Qwen3Config, Qwen3Attention

config = Qwen3Config()
attn = Qwen3Attention(config)
attn.eval()
attn_kv = Qwen3Attention(config)
attn_kv.load_state_dict(attn.state_dict())
attn_kv.allocate_cache(batch_size=1, max_len=128)

x = torch.randn(1, 3, 1024)

with torch.no_grad():
    # ===== Prefill 对比 =====
    out1 = attn.forward(x)       # positions: [0,1,2] (算在 forward 里)
    out2 = attn_kv.forward_kv_cache(
        x,
        positions=torch.tensor([0, 1, 2]),
        is_prefill=True,
        cache_pos=[0, 1, 2],
    )
    diff = (out1 - out2).abs().max().item()

    print(f"  Out1 range: [{out1.min():.6f}, {out1.max():.6f}]")
    print(f"  Out2 range: [{out2.min():.6f}, {out2.max():.6f}]")
    print(f"  Prefill diff: {diff:.10f}")
    print(f"  Out1[0,0,:5]: {out1[0,0,:5].tolist()}")
    print(f"  Out2[0,0,:5]: {out2[0,0,:5].tolist()}")
    print()

# ===== Decode 对比 =====
x_new = torch.randn(1, 1, 1024)
with torch.no_grad():
    # 原始: forward 全部 4 个 token
    out1_d = attn.forward(torch.cat([x, x_new], dim=1))[:, -1:, :]

    # kv cache: prefill 3 + decode 1
    attn_kv2 = Qwen3Attention(config)
    attn_kv2.load_state_dict(attn.state_dict())
    attn_kv2.allocate_cache(batch_size=1, max_len=128)
    attn_kv2.forward_kv_cache(
        x,
        positions=torch.tensor([0, 1, 2]),
        is_prefill=True,
        cache_pos=[0, 1, 2],
    )
    out2_d = attn_kv2.forward_kv_cache(
        x_new,
        positions=torch.tensor([3]),
        is_prefill=False,
        cache_pos=[3],
    )
    diff_d = (out1_d - out2_d).abs().max().item()

    print(f"  Out1_d range: [{out1_d.min():.6f}, {out1_d.max():.6f}]")
    print(f"  Out2_d range: [{out2_d.min():.6f}, {out2_d.max():.6f}]")
    print(f"  Decode diff:  {diff_d:.10f}")
    print(f"  Out1_d[0,0,:5]: {out1_d[0,0,:5].tolist()}")
    print(f"  Out2_d[0,0,:5]: {out2_d[0,0,:5].tolist()}")
