#!/bin/bash
PY=/root/miniconda3/bin/python
set -e

echo "=== GPU Details ==="
$PY -c "
import torch
g = torch.cuda.get_device_properties(0)
print(f'SMs: {g.multi_processor_count}')
print(f'Max thr/SM: {g.max_threads_per_multi_processor}')
print(f'Shmem default: {g.shared_memory_per_block / 1024:.0f} KB')
print(f'Shmem optin: {g.shared_memory_per_block_optin / 1024:.0f} KB')
print(f'Regs/SM: {g.regs_per_multiprocessor}')
print(f'Max blk/SM: {g.max_blocks_per_multi_processor}')
print(f'Warp size: {g.warp_size}')
print(f'Flash SDP: {torch.backends.cuda.flash_sdp_enabled()}')
" 2>&1

echo ""
echo "=== OS Info ==="
cat /etc/os-release | head -3
uname -m

echo ""
echo "=== Installing CUDA Toolkit ==="
# Check if nvcc exists
if command -v nvcc &> /dev/null; then
    echo "nvcc already available: $(which nvcc)"
    nvcc --version | head -2
else
    echo "Installing CUDA 12.8 toolkit..."
    # Try pip install first (nvidia-cuda-nvcc)
    pip install nvidia-cuda-nvcc-cu12 2>/dev/null || true
    # Check if conda has cuda
    if [ -d /root/miniconda3 ]; then
        conda install -y -c nvidia cuda-nvcc 2>/dev/null || true
    fi
    # Try apt
    if ! command -v nvcc &> /dev/null; then
        apt-get update -qq 2>/dev/null
        apt-get install -y -qq nvidia-cuda-toolkit 2>/dev/null || true
    fi
fi

echo ""
echo "=== nvcc status ==="
which nvcc 2>&1 || echo "nvcc still not found"
nvcc --version 2>&1 | head -2 || true

echo ""
echo "=== Check gcc ==="
which gcc && gcc --version | head -1

echo ""
echo "=== Workspace ==="
mkdir -p /root/gemm_work
ls /root/
"
