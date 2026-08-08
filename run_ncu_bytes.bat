@echo off
"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.2\ncu.bat" --print-summary per-kernel --metrics l1tex__t_bytes.sum,lts__t_bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum python demo_ncu.py
