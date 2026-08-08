@echo off
"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.2\ncu.bat" --print-summary per-kernel python profile_matmul_only.py
