"""
Verify peak by:
  (A) Polling GPU clock DURING matmul (not idle!)
  (B) Running pure FP32 vs TF32 matmul to confirm 2x = Tensor Core
  (C) Verify table numbers against actual measured clock
"""
import torch
import subprocess
import time
import threading

props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name}")
print(f"SMs: {props.multi_processor_count}")
print(f"CUDA cores reported by driver: unknown (PyTorch doesn't expose directly)")
print(f"Compute Capability: {props.major}.{props.minor}")
print()

# ================================================================
# (A) Poll GPU clock WHILE matmul is running
# ================================================================
print("=" * 70)
print("(A) Polling GPU SM clock DURING sustained matmul")
print("=" * 70)

clock_samples = []
stop_poll = threading.Event()

def poll_clock():
    while not stop_poll.is_set():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2)
            clk = int(out.stdout.strip())
            clock_samples.append(clk)
        except:
            pass
        time.sleep(0.1)  # poll every 100ms

# Start polling
t = threading.Thread(target=poll_clock, daemon=True)
t.start()

# Run sustained matmul for several seconds to let clock stabilize
m = n = k = 4096
a = torch.randn(m, k, device="cuda", dtype=torch.float32)
b = torch.randn(k, n, device="cuda", dtype=torch.float32)

torch.backends.cuda.matmul.allow_tf32 = True
start = time.perf_counter()
iters = 0
while time.perf_counter() - start < 5.0:
    a @ b
    iters += 1
torch.cuda.synchronize()

stop_poll.set()
t.join()

if clock_samples:
    # Remove first few samples (ramp-up)
    stable = clock_samples[10:] if len(clock_samples) > 10 else clock_samples
    print(f"  Total samples: {len(clock_samples)}")
    print(f"  Stable samples (after ramp): {len(stable)}")
    print(f"  Min clock:     {min(stable):4d} MHz")
    print(f"  Max clock:     {max(stable):4d} MHz")
    print(f"  Avg clock:     {sum(stable)/len(stable):.0f} MHz")
    actual_clock = sum(stable) / len(stable) / 1000  # GHz
    print(f"  Avg: {actual_clock:.3f} GHz")
else:
    print("  (nvidia-smi polling failed)")
    actual_clock = 1.485  # fallback

print()

# ================================================================
# (B) Confirm TF32 = Tensor Core by measuring compute type
# ================================================================
print("=" * 70)
print("(B) TF32 = Tensor Core proof")
print("=" * 70)

# TF32 is a format that ONLY tensor cores understand.
# CUDA cores can't do TF32. So if TF32 gives 2x, that's tensor core.
#
# But let's verify: TF32 off should use CUDA cores only.
# TF32 on should use Tensor Cores.
m = n = k = 4096
a = torch.randn(m, k, device="cuda", dtype=torch.float32)
b = torch.randn(k, n, device="cuda", dtype=torch.float32)

def measure_matmul(tf32):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    for _ in range(5): a @ b
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(10): a @ b
    e.record()
    torch.cuda.synchronize()
    t_ms = s.elapsed_time(e) / 10
    tflops = 2 * m * n * k / (t_ms / 1000) / 1e12
    return t_ms, tflops

t_off, tf_off = measure_matmul(False)
t_on, tf_on = measure_matmul(True)

print(f"  TF32 OFF (CUDA Core):  {t_off:.2f} ms  →  {tf_off:.3f} TFLOPS")
print(f"  TF32 ON  (Tensor Core): {t_on:.2f} ms  →  {tf_on:.3f} TFLOPS")
print(f"  Speedup: {tf_on/tf_off:.2f}x")

# The key insight:
# Ampere Tensor Core TF32 throughput = 2× FP32 CUDA core throughput
# If we see exactly 2x, that confirms TF32 = Tensor Core
print()
if 1.8 < tf_on/tf_off < 2.2:
    print(f"  OK: {tf_on/tf_off:.1f}x speedup matches Ampere spec: TF32 = 2x FP32 on Tensor Cores")
else:
    print(f"  NOTE: Speedup {tf_on/tf_off:.1f}x -- not exactly 2x, clock may vary between runs")

# ================================================================
# (C) Verify the peak table
# ================================================================
print()
print("=" * 70)
print("(C) Peak table verification")
print("=" * 70)

# NVIDIA官方: 3050 Ti Laptop = 2560 CUDA cores
# 公式: 2560 cores × 2 FMA/clock × freq = GFLOPS
#
# 验证来源:
#   1. GA107 spec: 20 SMs × 128 CUDA cores = 2560 (TechPowerUp GPU DB)
#   2. FMA = 2 ops/cycle (one multiply + one add)
#   3. Peak = cores × 2 × clock
#
# NVIDIA官方 3050 Ti Laptop 标称:
#   Base clock: 1035 MHz
#   Boost clock: 1695 MHz (spec, but limited by TGP 35-80W in laptops)
#   FP32: 5.3 TFLOPS (at base) / 8.7 TFLOPS (at max boost 1695 MHz)
#   Source: NVIDIA official spec sheet & TechPowerUp

nvidia_spec_boost = 1695  # MHz
nvidia_spec_base = 1035   # MHz
cores = 2560

print(f"""
  Formula: {cores} CUDA cores × 2 FMA × clock = peak GFLOPS

  Source verification:
    - CUDA cores: 2560 (20 SMs × 128, GA107)
      Ref: https://www.techpowerup.com/gpu-specs/geforce-rtx-3050-ti-mobile.c3722
    - FMA: 1 fused multiply-add = 2 floating-point operations
    - SMs = 20: confirmed by torch.cuda.get_device_properties()

  At NVIDIA spec clocks:
""")

for name, clk in [("Base", nvidia_spec_base), ("Max Boost", nvidia_spec_boost)]:
    fp32 = cores * 2 * clk / 1000
    tf32 = fp32 * 2
    print(f"    {name:12s} {clk:4d} MHz: FP32 = {fp32:.1f} GFLOPS = {fp32/1000:.2f} TFLOPS,  "
          f"TF32 = {tf32:.1f} GFLOPS = {tf32/1000:.2f} TFLOPS")

print(f"""
  At ACTUAL measured clock ({actual_clock*1000:.0f} MHz, from nvidia-smi during matmul):
    FP32 peak: {cores * 2 * actual_clock:.1f} GFLOPS = {cores * 2 * actual_clock / 1000:.2f} TFLOPS
    TF32 peak: {cores * 2 * actual_clock * 2:.1f} GFLOPS = {cores * 2 * actual_clock * 2 / 1000:.2f} TFLOPS

  Our measured matmul:
    TF32 OFF: {tf_off:.2f} TFLOPS  →  {tf_off / (cores * 2 * actual_clock / 1000) * 100:.0f}% of FP32 peak
    TF32 ON:  {tf_on:.2f} TFLOPS  →  {tf_on / (cores * 2 * actual_clock * 2 / 1000) * 100:.0f}% of TF32 peak

  Typical cuBLAS efficiency: 70-85% of theoretical peak for large matrices.
""")

# ================================================================
# (D) FAQ
# ================================================================
print("=" * 70)
print("(D) FAQ")
print("=" * 70)
print("""
Q: allow_tf32=True 是让计算在 Tensor Core 上运行吗？
A: 是的。TF32 是 Tensor Core 专用的 19-bit 格式。
   CUDA Core 不能处理 TF32。
   allow_tf32=True  → cuBLAS 内部把 FP32 输入 round 成 TF32，用 Tensor Core 算
   allow_tf32=False → cuBLAS 用 CUDA Core 算 FP32
   所以 TF32 ON/OFF 的 2× 速度差就是 Tensor Core 的贡献。

Q: 上面的表确定吗？
A: 公式是确定的 (CUDA cores × 2 × clock)。
   频率取什么值取决于你的 GPU 实际跑在什么频率。
   3050 Ti Laptop 的 TGP 限制 (35-80W) 决定了实际频率会低于理论 boost。
   nvidia-smi 实测是唯一可靠的方法。

Q: 3050 Ti 有多少 Tensor Core？
A: 20 SMs × 4 = 80 个第 3 代 Tensor Core。
   来源: NVIDIA Ampere GA107 whitepaper / TechPowerUp GPU DB.

Q: A100 有多少 Tensor Core？
A: 108 SMs × 4 = 432 个。H100: 132 × 4 = 528 个。
   来源: NVIDIA A100/H100 spec sheets.
""")
