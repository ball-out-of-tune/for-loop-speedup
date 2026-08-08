@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64 > nul 2>&1
set "MSVC=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.44.35207"
set "SDK=C:\Program Files (x86)\Windows Kits\10"
set "PATH=%MSVC%\bin\HostX64\x64;%PATH%"
set "INCLUDE=%MSVC%\include;%SDK%\Include\10.0.26100.0\ucrt;%SDK%\Include\10.0.26100.0\shared;%SDK%\Include\10.0.26100.0\um;%SDK%\Include\10.0.26100.0\winrt;%SDK%\Include\10.0.26100.0\cppwinrt;%INCLUDE%"
set "LIB=%MSVC%\lib\x64;%SDK%\Lib\10.0.26100.0\um\x64;%SDK%\Lib\10.0.26100.0\ucrt\x64;%LIB%"
cd /d C:\Users\16874\codingPractice\numba-benchmark
nvcc -o sgemm_fp16.exe sgemm_fp16.cu -lcublas -arch=sm_86 -Xptxas -v
if %ERRORLEVEL% equ 0 (
    echo === BUILD OK ===
    PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%
    sgemm_fp16.exe
) else (
    echo === BUILD FAILED ===
)
