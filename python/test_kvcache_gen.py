"""KV-Cache 正确性 & 速度验证"""
import time
import torch
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device('cuda')

model_path = '/mnt/c/Users/16874/Downloads/Qwen3-0.6B'
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
formatted = tokenizer.apply_chat_template(
    [{'role': 'user', 'content': '你好'}],
    tokenize=False, add_generation_prompt=True,
)
prompt_ids = tokenizer.encode(formatted, return_tensors='pt').to('cuda')
plen = prompt_ids.shape[1]

from qwen3_naive import Qwen3Config, Qwen3ForCausalLM, load_weights
config = Qwen3Config.from_json(f'{model_path}/config.json')
model = Qwen3ForCausalLM(config)
load_weights(model, f'{model_path}/model.safetensors')

# ===== 正确性: 对比 logits =====
torch.manual_seed(42)
logits = model.forward(prompt_ids)[:, -1, :] / 0.6
_, topk = torch.topk(logits, 20)
masked = torch.full_like(logits, float('-inf')).scatter(-1, topk, logits.gather(-1, topk))
next_tok = torch.multinomial(torch.softmax(masked, -1), 1)

# Naive
h_naive = model.model(torch.cat([prompt_ids, next_tok], dim=1))
logit_n = model.lm_head(h_naive[0, -1].unsqueeze(0))[0] / 0.6

# KV-Cache
model.allocate_kv_cache(1, plen + 100)
model.model.forward_kvcache(prompt_ids, torch.arange(0, plen, device='cuda'), True,
                             torch.arange(0, plen, device='cuda'))
h_kv = model.model.forward_kvcache(next_tok, torch.tensor([plen], device='cuda'), False,
                                    torch.tensor([plen], device='cuda'))
logit_k = model.lm_head(h_kv[0, 0].unsqueeze(0))[0] / 0.6

diff = (logit_n - logit_k).abs().max().item()
same_top5 = set(logit_n.topk(5).indices.tolist()) == set(logit_k.topk(5).indices.tolist())
print(f"[正确性] logits max diff: {diff:.4f}, top-5 match: {same_top5}")

# ===== 速度: 长序列生成 =====
max_new = 500
torch.manual_seed(42)
torch.cuda.synchronize(); t0 = time.time()
out_n = model.generate_naive(prompt_ids, max_new_tokens=max_new, temperature=0.6, top_k=20)
torch.cuda.synchronize(); t_n = time.time() - t0
n_tok = out_n.shape[1] - plen

torch.manual_seed(42)
torch.cuda.synchronize(); t0 = time.time()
out_k = model.generate_kvcache(prompt_ids, max_new_tokens=max_new, temperature=0.6, top_k=20)
torch.cuda.synchronize(); t_k = time.time() - t0
k_tok = out_k.shape[1] - plen

print(f"\n[速度] max_new={max_new}:")
print(f"  Naive:    {n_tok} tokens / {t_n:.1f}s = {n_tok/t_n:.1f} tok/s")
print(f"  KV-Cache: {k_tok} tokens / {t_k:.1f}s = {k_tok/t_k:.1f} tok/s")
print(f"  提速:     {t_n/t_k:.1f}x")

# ===== nano-vllm: 需要独立进程，python test_kvcache_gen.py --nano 单独跑 =====
if "--nano" in __import__('sys').argv:
    import os, sys as _sys
    _sys.path.insert(0, '/mnt/c/Users/16874/AIInfraGuide/nano-vllm')
    import torch._dynamo; torch._dynamo.config.suppress_errors = True
    from nanovllm import LLM, SamplingParams
    nano_model_path = '/mnt/c/Users/16874/AIInfraGuide/nano-vllm/Qwen3-0.6B'
    llm = LLM(nano_model_path, enforce_eager=True, tensor_parallel_size=1)
    sp = SamplingParams(temperature=0.6, max_tokens=max_new)
    formatted_long = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': '你好，请介绍一下你自己'}],
        tokenize=False, add_generation_prompt=True,
    )
    llm.generate(["warmup"], SamplingParams(max_tokens=1))
    torch.cuda.synchronize(); t0 = time.time()
    nano_out = llm.generate([formatted_long], sp)
    torch.cuda.synchronize(); t_nano = time.time() - t0
    print(f"  nano-vllm: {t_nano:.1f}s, {max_new/t_nano:.1f} tok/s")
    print(f"\n{'='*60}")
    print("nano-vllm output:")
    print(nano_out[0]['text'])

print(f"\n{'='*60}")
print("Naive output:")
print(tokenizer.decode(out_n[0].tolist(), skip_special_tokens=True))
print(f"\n{'='*60}")
print("KV-Cache output:")
print(tokenizer.decode(out_k[0].tolist(), skip_special_tokens=True))
