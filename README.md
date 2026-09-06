# CUDALearn - Learn and Practice CUDA

This repository serves as my notebook and documents CUDA concepts and learning during the GPU Programming Courses provided by NHR@FAU.

## Benefits of GPU

Higher instruction throughput and memory bandwidth than CPU within similar price
and power envelope. 

CPU is designed to excel at executing a serial sequence of operations (called a thread) as fast as possible and can execute a few tens of these threads in parallel, a GPU is designed to excel at executing thousands of threads in parallel, trading off lower single-thread performance to achieve much greater total throughput.

GPUs are specialized for highly parallel computations and devote more transistors to data processing units, while CPUs dedicate more transistors to data caching and flow control.

An ever-growing collection of algorithms and routines from a variety of domains is available through specialized libraries. When a library has already been implemented—especially those provided by NVIDIA—using it is often more productive and performant than reimplementing algorithms from scratch. Libraries like cuBLAS, cuFFT, cuDNN, and CUTLASS are just a few examples of libraries that help developers avoid reimplementing well-established algorithms. These libraries have the added benefit of being optimized for each GPU architecture, providing an ideal mix of productivity, performance, and portability.

## Programming Model Abstract

Heterogenous Systems: Systems with both CPU (Host) and GPU (Device)

CUDA applications start execution on the CPU, copy data to GPU to execute 
some part of the code, the CPU waits for data copies or the GPU code to complete.

The CPU and GPU can both be executing code simultaneously, and best performance is usually found by maximizing utilization of both CPUs and GPUs.


## Programming GPUs in CUDA

