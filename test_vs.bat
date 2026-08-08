@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
echo.
echo === PATH check ===
echo %PATH%
echo.
echo === Direct cl.exe check ===
dir "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.44.35207\bin\HostX64\x64\cl.exe"
