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

<p align="center" style="background-color: white; padding: 20px;">
  <img src="images/gpu-cpu-system-diagram.png" alt="GPU and CPU System" width="900"/>
</p>

A GPU has many streaming multiprocessors (SMs), each of which contains many functional units. Graphics processing clusters (GPCs) are collections of SMs. A GPU is a set of GPCs connected to the GPU memory. A CPU typically has several cores and a memory controller which connects to the system memory. A CPU and a GPU are connected by an interconnect such as PCIe or NVLINK.

### Thread Blocks and Grids

When an application launches a kernel, it does so with many threads, often millions of threads organized into blocks.

Thread Block: Block of threads
Grid: Thread blocks are organized into a grid.

All the thread blocks in a grid have the same size and dimensions.

<p align="center" style="background-color: white; padding: 20px;">
  <img src="images/grid-of-thread-blocks.png" alt="Grid of Thread Blocks" width="900"/>
</p>

Thread blocks and grids may be 1, 2, or 3 dimensional. These dimensions can simplify mapping of individual threads to units of work or data items.

Kernel is launched with a specific "execution configuration" that specifies the grid and thread block dimensions. It can also include optional parms such as cluster size, stream, and SM configuration settings.

A grid may have millions of thread blocks, while a GPU executing the grid may have far less, maybe 10 or 100 SMs.

All threads of a thread block will be executed in a single SM and mostly complete in the same SM. The thread blocks cannot rely on any result from another thread blocks, so no dependencies between threads of different yhread blocks.

A thread should not depend on results from or synchronize with a thread in a different thread block of the same grid. All the threads within a thread block run on the same SM at the same time. Different thread blocks within the grid are scheduled among the available SMs and may be executed in any order. In short, the CUDA programming model requires that it be possible to execute thread blocks in any order, in parallel or in series.



## Programming GPUs in CUDA

