#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cuda/cmath>
#include <util.h>

__global__ void kernel() {
    int start = blockDim.x * blockIdx.x + threadIdx.x; //# TODO: calculate global thread index
    int stride = blockDim.x * gridDim.x; //# TODO: calculate total number of threads

    for (int i = start; i < 10; i += stride)
        printf("Hello world from iteration %d\n", i);
}

int main(int argc, char *argv[]) {
    int numBlocks = 1; //# TODO: choose number of blocks
    int numThreadsPerBlock = 32; //# TODO: choose number of threads per block

    kernel<<<numBlocks, numThreadsPerBlock>>>();
    checkCudaError(cudaDeviceSynchronize(), true);
}
