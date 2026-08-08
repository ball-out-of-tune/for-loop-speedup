@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >/dev/null 2>&1
nvcc -o nosmem_faster.exe nosmem_faster.cu -arch=sm_80 2>&1
