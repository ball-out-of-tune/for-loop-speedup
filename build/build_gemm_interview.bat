@echo off
set "MSVC=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.44.35207"
set "SDK=C:\Program Files (x86)\Windows Kits\10"
set "PATH=%MSVC%\bin\HostX64\x64;%PATH%"
set "INCLUDE=%MSVC%\include;%SDK%\Include\10.0.26100.0\ucrt;%SDK%\Include\10.0.26100.0\shared;%SDK%\Include\10.0.26100.0\um;%SDK%\Include\10.0.26100.0\winrt;%SDK%\Include\10.0.26100.0\cppwinrt;%INCLUDE%"
set "LIB=%MSVC%\lib\x64;%SDK%\Lib\10.0.26100.0\um\x64;%SDK%\Lib\10.0.26100.0\ucrt\x64;%LIB%"

cd /d C:\Users\16874\codingPractice\numba-benchmark
echo Building gemm_interview...
nvcc -o gemm_interview.exe gemm_interview.cu -arch=sm_86 -O3 -use_fast_math --fmad=true --restrict
if %ERRORLEVEL% equ 0 (
    echo === BUILD OK ===
    gemm_interview.exe
) else (
    echo === BUILD FAILED ===
)
