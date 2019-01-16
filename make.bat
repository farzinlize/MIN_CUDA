@echo off
if "%1" == "clean" goto :clean

nvcc -o MIN_CUDA.exe kernel.cu helper_functions.c 
goto :eof

:clean
del MIN_CUDA.exe MIN_CUDA.exp MIN_CUDA.lib