"""Float4 向量化：每个线程处理 4 个元素，减少线程开销，提高带宽利用率"""
from numba import cuda
import numpy as np
import math
import statistics

N = 1_000_000

@cuda.jit
def kernel_baseline(data, ans, n):
    """原始版本：1 线程 = 1 元素"""
    i = cuda.grid(1)
    if i < n:
        x = data[i]
        ans[i] = math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x))


@cuda.jit
def kernel_float4(data, ans, n):
    """向量化版本：1 线程 = 4 元素 (float4)"""
    i = cuda.grid(1)         # 线程号
    base = i * 4             # 这个线程负责的第 0 个元素
    if base + 3 < n:
        # 读 4 个连续的 float → LLVM 会自动合并成 128-bit 向量化加载
        x0 = data[base]
        x1 = data[base + 1]
        x2 = data[base + 2]
        x3 = data[base + 3]

        ans[base]     = math.sin(x0) ** 2 + math.cos(x0) ** 2 + math.sqrt(abs(x0))
        ans[base + 1] = math.sin(x1) ** 2 + math.cos(x1) ** 2 + math.sqrt(abs(x1))
        ans[base + 2] = math.sin(x2) ** 2 + math.cos(x2) ** 2 + math.sqrt(abs(x2))
        ans[base + 3] = math.sin(x3) ** 2 + math.cos(x3) ** 2 + math.sqrt(abs(x3))
    else:
        # 尾部：处理不满 4 个的剩余元素
        for j in range(base, n):
            x = data[j]
            ans[j] = math.sin(x) ** 2 + math.cos(x) ** 2 + math.sqrt(abs(x))


def bench(name, kernel_fn, threads, blocks, d_data, d_ans, n, warmup=3, runs=10):
    # 预热
    for _ in range(warmup):
        kernel_fn[blocks, threads](d_data, d_ans, n)
    cuda.synchronize()

    # 计时
    times = []
    for _ in range(runs):
        start = cuda.event(); end = cuda.event()
        start.record()
        kernel_fn[blocks, threads](d_data, d_ans, n)
        end.record()
        end.synchronize()
        times.append(cuda.event_elapsed_time(start, end))

    t = statistics.mean(times)
    bandwith = (n * 4 * 2) / (t / 1000) / 1e9  # 读 + 写各 4 bytes × n
    print(f'{name:20s}  {t:.4f} ms  |  带宽: {bandwith:.1f} GB/s')
    return t


if __name__ == '__main__':
    data_np = np.arange(N, dtype=np.float32) * 0.001
    d_data = cuda.to_device(data_np)
    d_ans = cuda.device_array(N, dtype=np.float32)

    print('=' * 55)
    print('  float4 向量化 — 每个线程处理 4 个元素')
    print('=' * 55)
    print()

    # 基线：N 个线程
    t_base = bench(
        'baseline (1 elem/thr)',
        kernel_baseline, 256, (N + 255) // 256,
        d_data, d_ans, N
    )

    # float4: N/4 个线程，每线程 4 元素
    t_vec = bench(
        'float4 (4 elem/thr)',
        kernel_float4, 256, (N + 4 * 256 - 1) // (4 * 256),
        d_data, d_ans, N
    )

    print()
    print(f'加速: {t_base / t_vec:.2f}x')
    print(f'理论最快 (带宽): 0.040 ms  ({7.72 / 192 * 1000:.3f} 的数据 ÷ 192 GB/s)')
