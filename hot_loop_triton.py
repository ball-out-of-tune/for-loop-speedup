"""
Triton 实现 — OpenAI 推出的 GPU kernel 语言
安装: pip install triton
运行: python hot_loop_triton.py

Triton vs Numba CUDA vs Raw CUDA C++:

  抽象层级:
    CUDA C++    → 最底层，直接写 thread/block/grid，手动管理一切
    Numba CUDA  → 在 Python 里写 CUDA，体验接近 CUDA C++，但有 Python 开销
    Triton      → 不写 thread，写 "program" (block 级别)，编译器自动分配线程

  关键区别:
    CUDA/Numba:    你控制每个 thread 做什么
    Triton:        你控制每个 program (block) 做什么，block 内部的并行由编译器处理

  所以 Triton 代码看起来更像 NumPy — 用 arange + mask 的模式，
  不用手动写 cuda.grid(1) 和边界检查。
"""
import triton
import triton.language as tl
import torch
import statistics


# ---------------------------------------------------------------------------
# Triton kernel (GPU 代码)
# ---------------------------------------------------------------------------
@triton.jit
def triton_hot_loop(
    data_ptr,          # 输入数组的 GPU 指针
    ans_ptr,           # 输出数组的 GPU 指针
    n,                 # 数组总长度
    BLOCK_SIZE: tl.constexpr,  # 编译期常量 — 每个 program 处理多少元素
):
    """
    Triton kernel 模型:
      - 你写的是 "一个 program 做什么"
      - Triton 自动把这个 program 复制到所有 block 上
      - 每个 program 内部，用 tl.arange(0, BLOCK_SIZE) 来向量化处理

    对比 CUDA:
      cuda.grid(1) → pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
      你不需要手动分配 thread，Triton 编译器帮你展开。
    """
    # 当前 program (block) 的 ID
    pid = tl.program_id(0)

    # 这个 program 负责的元素偏移量
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # 边界 mask — 自动处理尾部不足 BLOCK_SIZE 的情况
    mask = offsets < n

    # 从 GPU 显存加载数据 (合并访存，自动向量化)
    x = tl.load(data_ptr + offsets, mask=mask)

    # 计算 — 和 Python 一样写就行，Triton 编译成 GPU 指令
    result = tl.sin(x) * tl.sin(x) + tl.cos(x) * tl.cos(x) + tl.sqrt(tl.abs(x))

    # 写回显存
    tl.store(ans_ptr + offsets, result, mask=mask)


# ---------------------------------------------------------------------------
# CPU 端 benchmark
# ---------------------------------------------------------------------------
def bench_triton(n: int, block_size: int, warmup=3, runs=10):
    # 创建输入数据 (在 CPU 上)
    data_cpu = torch.arange(n, dtype=torch.float32) * 0.001
    ans_cpu = torch.empty(n, dtype=torch.float32)

    # 搬到 GPU
    data_gpu = data_cpu.cuda()
    ans_gpu = ans_cpu.cuda()

    # Triton 需要的参数: 几个 program (即 grid 大小)
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    # 预热
    for _ in range(warmup):
        triton_hot_loop[grid](data_gpu, ans_gpu, n, BLOCK_SIZE=block_size)
    torch.cuda.synchronize()

    # 计时
    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(runs):
        start.record()
        triton_hot_loop[grid](data_gpu, ans_gpu, n, BLOCK_SIZE=block_size)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    t = statistics.mean(times)
    data_size = n * 4 * 2  # 读 + 写，各 4 bytes × N
    bandwidth = data_size / (t / 1000) / 1e9
    return t, bandwidth, ans_gpu


if __name__ == '__main__':
    N = 1_000_000

    print("=" * 55)
    print("  Triton GPU Kernel: hot_loop")
    print("=" * 55)
    print()

    # 扫一下 block size
    for bs in [128, 256, 512, 1024]:
        t, bw, ans = bench_triton(N, bs)
        pct = bw / 192 * 100
        print(f"  BLOCK_SIZE={bs:4d}  |  {t:.4f} ms  |  {bw:.1f} GB/s  ({pct:.1f}%)")

    # 数值验证
    data_cpu = torch.arange(N, dtype=torch.float32) * 0.001
    ans_gpu = ans.cpu()
    expected = torch.sin(data_cpu)**2 + torch.cos(data_cpu)**2 + torch.sqrt(torch.abs(data_cpu))
    max_diff = (ans_gpu - expected).abs().max().item()
    print(f"\n  数值验证: max_diff = {max_diff:.2e} {'✅' if max_diff < 1e-4 else '❌'}")

    print()
    print("Triton vs Numba CUDA — 关键差异:")
    print("  ┌─────────────────────┬───────────────────────┐")
    print("  │ Numba CUDA          │ Triton                │")
    print("  ├─────────────────────┼───────────────────────┤")
    print("  │ 你写 thread 级代码   │ 你写 block 级代码     │")
    print("  │ cuda.grid(1) 手动算  │ tl.program_id 自动    │")
    print("  │ if i < n 手动边界    │ mask=mask 自动处理    │")
    print("  │ JIT: LLVM → PTX      │ JIT: MLIR → PTX       │")
    print("  │ 编译器: LLVM         │ 编译器: Triton (更现代)│")
    print("  │ 适合教学、简单 kernel│ 适合复杂 tensor 操作   │")
    print("  └─────────────────────┴───────────────────────┘")

@triton.jit
def loop_triton_my_own(
    data,
    ans,
    N,
    block_size:tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * block_size + tl.arange(block_size)

    x = tl.load(data + offsets, mask=(offsets < N))
    y = tl.sin(x) * tl.sin(x) + tl.cos(x) * tl.cos(x) + tl.sqrt(tl.abs(x))
    tl.store(ans + offsets, y, mask=(offsets<N))