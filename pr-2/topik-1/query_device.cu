#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);  // Ambil device aktif
    cudaGetDeviceProperties(&prop, device);

    printf("Device: %s\n", prop.name);
    printf("Max threads per block      : %d\n", prop.maxThreadsPerBlock);
    printf("Max threads dim (x, y, z)  : %d x %d x %d\n",
           prop.maxThreadsDim[0],
           prop.maxThreadsDim[1],
           prop.maxThreadsDim[2]);
    printf("Max grid size (x, y, z)    : %d x %d x %d\n",
           prop.maxGridSize[0],
           prop.maxGridSize[1],
           prop.maxGridSize[2]);
    printf("Shared memory per block    : %zu bytes\n", prop.sharedMemPerBlock);
    printf("Total global memory        : %zu bytes\n", prop.totalGlobalMem);
    return 0;
}
