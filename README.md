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

Device Code: The code an application executes on the GPU
Kernel: A function that is invoked for execution on the GPU

The act of starting a kernel running is called launching the kernel. A kernel launch can be thought of as starting many threads executing the kernel code in parallel on the GPU.

### GPU Hardware Model

For the purposes of CUDA programming, the GPU can be considered to be a collection of Streaming Multiprocessors (SMs) which are organized into groups called Graphics Processing Clusters (GPCs).

Each SM contains a local register file, a unified data cache, and a number of functional units that perform computations. The unified data cache provides the physical resources for shared memory and L1 cache.

The allocation of L1 cache and shared memory from the unified data cache can be
configured at runtime.

The sizes of different types of memory and the number of functional units within an SM can vary across GPU architectures.

The actual hardware layout of a GPU or the way it physically carries out the execution of the programming model may vary. These differences do not affect correctness of software written using the CUDA programming model.

<p align="center">
  <img src="images/gpu-cpu-system-diagram.png" alt="GPU and CPU System" width="900"/>
</p>

A GPU has many streaming multiprocessors (SMs), each of which contains many functional units. Graphics processing clusters (GPCs) are collections of SMs. A GPU is a set of GPCs connected to the GPU memory. A CPU typically has several cores and a memory controller which connects to the system memory. A CPU and a GPU are connected by an interconnect such as PCIe or NVLINK.


## Programming GPUs in CUDA

