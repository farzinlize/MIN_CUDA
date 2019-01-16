@echo off
if "%1" == "clean" goto :clean
if "%1" == "debug" goto :debug

nvcc -o MIN_CUDA.exe kernel.cu helper_functions.c 
goto :eof

:debug
nvcc -o MIN_CUDA_DEBUG.exe kernel.cu helper_functions.c -D DEBUG
goto :eof

:clean
del MIN_CUDA.exe MIN_CUDA.exp MIN_CUDA.lib MIN_CUDA_DEBUG.exe MIN_CUDA_DEBUG.exp MIN_CUDA_DEBUG.lib