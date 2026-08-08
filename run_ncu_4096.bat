@echo off
"C:\Program Files\NVIDIA Corporation\Nsight Compute 2024.3.2\ncu.bat" --print-summary per-kernel --metrics ^
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,^
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,^
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,^
dram__bytes_read.sum,^
dram__bytes_write.sum,^
gpu__time_duration.sum ^
python demo_ncu_4096.py
