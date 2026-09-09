#include "../stencil-2d-util.h"


__global__ inline void stencil2D(const double *__restrict__ u, double *__restrict__ uNew, size_t globalNumCellsX, size_t globalNumCellsY) {
    const size_t idx1 = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t stride1 = blockDim.y * gridDim.y;
    const size_t idx0 = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride0 = blockDim.x * gridDim.x;
    for (size_t i1 = idx1; i1 < globalNumCellsY - 1; i1+=stride1) {
        for (size_t i0 = idx0; i0 < globalNumCellsX - 1; i0+=stride0) {
            uNew[i0 + i1 * globalNumCellsX] = u[i0 + i1 * globalNumCellsX]
                + alpha * (
                          u[(i0 - 1) +  i1      * globalNumCellsX]
                    +     u[(i0 + 1) +  i1      * globalNumCellsX]
                    +     u[ i0      + (i1 + 1) * globalNumCellsX]
                    +     u[ i0      + (i1 - 1) * globalNumCellsX]
                    - 4 * u[ i0      +  i1      * globalNumCellsX]
                );
        }
    }
}


int main(int argc, char *argv[]) {
    // determine application parameters
    size_t globalNumCellsX, globalNumCellsY, numItWarmUp, numItTimed, printInterval;
    parseCLA_2d(argc, argv, globalNumCellsX, globalNumCellsY, numItWarmUp, numItTimed, printInterval);

    // allocation
    double *u, *uNew;
    cudaMallocManaged(&u, sizeof(double[globalNumCellsX * globalNumCellsY]));
    cudaMallocManaged(&uNew, sizeof(double[globalNumCellsX * globalNumCellsY]));

    // init temperature fields including their boundaries
    initTemperature(u, uNew, globalNumCellsX, globalNumCellsY);

    // Prefetch the u and uNew arrays to the GPU
    // prefetch to GPU
    int deviceId = 0;
    checkCudaError(cudaGetDevice(&deviceId));

    cudaMemLocation deviceID{cudaMemLocationTypeDevice, deviceId};
    cudaMemLocation hostID{cudaMemLocationTypeHost, 0};
    checkCudaError(cudaMemPrefetchAsync(u, globalNumCellsX * globalNumCellsY * sizeof(double), deviceID, 0));
    checkCudaError(cudaMemPrefetchAsync(uNew, globalNumCellsX * globalNumCellsY * sizeof(double), deviceID, 0));

    /// print function
    auto print = [&](size_t it) {
        std::cout << "  Completed iteration " << it << std::endl;

        std::string idx = std::to_string(it);
        if (idx.size() < 6) idx = std::string(6 - idx.size(), '0') + idx;

        writeTemperatureNpy("../output/temperature_" + idx + ".npy", u, globalNumCellsX, globalNumCellsY);
    };

    /// work function    
    auto work = [&](size_t it) {
        // stencil2D(u, uNew, globalNumCellsX, globalNumCellsY);
        // std::swap(u, uNew);

        if (printInterval > 0 && 0 == (it % printInterval))
            print(it);
    };

    // warm-up
    for (size_t i = 0; i < numItWarmUp; ++i)
        work(i);

    // measurement
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < numItTimed; ++i)
            work(i + numItWarmUp);    // account for warm-up iterations in the print interval computation

    auto end = std::chrono::steady_clock::now();

    // print stats and diagnostic result
    printStats(end - start, numItTimed, globalNumCellsX * globalNumCellsY, sizeof(double) + sizeof(double), 7);

    auto totalTemperature = accumulateTemperature(u, globalNumCellsX, globalNumCellsY);
    std::cout << "  Total temperature is " << totalTemperature << std::endl;

    // clean up
    delete[] u;
    delete[] uNew;

    return 0;
}
