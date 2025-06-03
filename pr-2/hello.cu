#include <iostream>

// Fungsi kernel CUDA (dieksekusi di GPU)
__global__ void hello_from_gpu() {
    printf("Hello from GPU!\n");
}

int main() {
    // Panggil fungsi kernel (1 block, 1 thread)
    hello_from_gpu<<<1, 1>>>();

    // Tunggu semua thread GPU selesai
    cudaDeviceSynchronize();

    // Cetak dari CPU
    std::cout << "Hello from CPU!" << std::endl;

    return 0;
}
