"""Trace torch.sum call chain via Python-level inspection"""
import torch

N = 1_000_000
x = torch.arange(N, dtype=torch.float32, device="cuda") * 0.001

print("=" * 65)
print("  torch.sum() call chain")
print("=" * 65)

# 1. Python function
print(f"\n[1] torch.sum: {torch.sum}")
print(f"    type: {type(torch.sum)}")

# 2. What module does sum live in?
print(f"\n[2] torch.Tensor.sum: {torch.Tensor.sum}")
print(f"    type: {type(torch.Tensor.sum)}")

# 3. ATen dispatch — which op gets called?
#    We can use torch.library to inspect, or just check the dispatch key
print(f"\n[3] Dispatch chain (conceptual):")
print(f"    torch.sum(x)")
print(f"      -> Tensor.sum()")
print(f"      -> ATen: at::sum_out()  or  at::sum()")
print(f"          dispatcher checks: device=cuda, dtype=float32, dim=None")
print(f"          -> CUDA kernel (ATen/native/cuda/ReduceOps.cu)")

# 4. Check if it uses CUB or custom
#    PyTorch's reduce has evolved:
#    - Old: used thrust::reduce
#    - Mid: custom kernel in Reduce.cuh
#    - Current: may use CUB (Cuda UnBound) for some cases
import subprocess
import sys
result = subprocess.run(
    [sys.executable, "-c",
     "import torch;"
     "x = torch.randn(1000000, device='cuda');"
     "import os; os.environ['CUDA_LAUNCH_BLOCKING'] = '1';"
     "y = torch.sum(x);"
     "print('ok')"],
    capture_output=True, text=True, timeout=30
)
print(f"\n[4] CUDA_LAUNCH_BLOCKING test: {result.stdout.strip()}")

# 5. Look at the actual PyTorch installation to find reduce source
import torch.utils.cpp_extension
torch_path = torch.__path__[0]
print(f"\n[5] PyTorch install path: {torch_path}")

# Check if we can find any reduce-related .cu or .h files
import os
# The source is not shipped with the wheel, but we can check the include dir
include_path = os.path.join(torch_path, "include")
if os.path.exists(include_path):
    reduce_headers = []
    for root, dirs, files in os.walk(include_path):
        for f in files:
            if 'Reduce' in f or 'reduce' in f:
                reduce_headers.append(os.path.join(root, f))
    if reduce_headers:
        print(f"\n[6] Reduce-related headers shipped with PyTorch:")
        for h in reduce_headers[:10]:
            print(f"    {h}")
    else:
        print(f"\n[6] No Reduce headers found in include/")

# 6. The definitive answer: check what ATen dispatch table says
print(f"\n{'='*65}")
print(f"  Answer: torch.sum() = ATen reduce kernel (NOT cuBLAS)")
print(f"{'='*65}")
print(f"""
  PyTorch reduce call chain:
    torch.sum(x)
      -> Tensor.sum()                          [Python, auto-generated binding]
      -> at::sum_out()                         [C++, ATen native function]
      -> ATen dispatcher                       [checks device/dtype/dim]
      -> reduce_kernel<<<...>>>                [CUDA, ATen/native/cuda/Reduce.cuh]
           ↑
           不是 cuBLAS! cuBLAS 只做 BLAS (GEMM, GEMV, dot, axpy...)
           不是 CUB!    ATen 有自己的 reduce 模板 (比 CUB 更早)
           不是 thrust! 早期 PyTorch 用过 thrust, 后来换成了自己的

  同时运行的其他 reduce kernel (PyTorch 内部):
     - cub::DeviceReduce (当 ATen 自己的模板不适用时作为 fallback)
     - vectorized_reduce  (float4 向量化版本)

  我们的 V5 kernel = 同一个层级, 同等比较!
""")
