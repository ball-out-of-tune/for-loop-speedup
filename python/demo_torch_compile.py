"""torch.compile 演示：chunk + silu + multiply 的 kernel fusion"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class SwiGLU_NoCompile(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, -1)
        return F.silu(a) * b

class SwiGLU_Compile(nn.Module):
    @torch.compile
    def forward(self, x):
        a, b = x.chunk(2, -1)
        return F.silu(a) * b

def main():
    x = torch.randn(128, 6144, device='cuda', dtype=torch.bfloat16)

    no_compile = SwiGLU_NoCompile()
    compile_mod = SwiGLU_Compile()

    # 预热（触发 compile）
    print("预热中（触发 torch.compile JIT 编译）...")
    for _ in range(3):
        no_compile(x)
        compile_mod(x)
    torch.cuda.synchronize()

    print()
    print("融合之前：")
    print("  读 x → chunk → 写 a,b → 读 a → silu → 写 s → 读 s,b → mul → 写 y")
    print("  3 步，每步都要写回显存再读出来")
    print()
    print("融合之后：")
    print("  读 x → chunk → silu → mul → 写 y")
    print("  1 个 CUDA kernel，中间值留在寄存器里，不碰显存")
    print()

    # 性能对比
    for name, mod in [('No Compile', no_compile), ('Compile   ', compile_mod)]:
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10000):
            mod(x)
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        us_per_iter = elapsed / 10000 * 1e6
        print(f"  {name}: 10000 次 = {elapsed:.3f}s ({us_per_iter:.1f} us/次)")

    print()
    print("compile 后的模块类型:", type(compile_mod._forward_module).__name__)

if __name__ == "__main__":
    main()
