#include <stdio.h>
#include <cuda_runtime.h>

__global__ void inc_gpu(int *a_d, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a_d[idx] = a_d[idx] + 1;
    }
}

void print_array(const char *label, int *a, int N) {
    printf("%s: ", label);
    for (int i = 0; i < N; ++i)
        printf("%d ", a[i]);
    printf("\n");
}

int main() {
    const int N = 1024;  // Ukuran array bisa divariasikan
    int *a_h = new int[N];
    int *a_d;

    cudaMalloc((void**)&a_d, N * sizeof(int));

    // Event untuk pengukuran waktu
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Variasi konfigurasi
    int blockSizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    int numConfigs = sizeof(blockSizes) / sizeof(int);

    printf("Eksperimen Increment Array CUDA (N = %d)\n", N);
    printf("-------------------------------------------------------------\n");
    printf("| BlockSize | GridSize | Time (ms) |\n");
    printf("-------------------------------------------------------------\n");

    for (int i = 0; i < numConfigs; ++i) {
        int blockSize = blockSizes[i];
        int gridSize = (N + blockSize - 1) / blockSize;

        // Inisialisasi ulang data host
        for (int j = 0; j < N; ++j) a_h[j] = j;

        // Salin ke device
        cudaMemcpy(a_d, a_h, N * sizeof(int), cudaMemcpyHostToDevice);

        // Ukur waktu kernel
        cudaEventRecord(start);
        inc_gpu<<<gridSize, blockSize>>>(a_d, N);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);

        printf("| %9d | %8d | %9.4f |\n", blockSize, gridSize, ms);

        // Optional: verifikasi hasil
        cudaMemcpy(a_h, a_d, N * sizeof(int), cudaMemcpyDeviceToHost);
        for (int j = 0; j < N; ++j) {
            if (a_h[j] != j + 1) {
                printf("ERROR at index %d: expected %d, got %d\n", j, j + 1, a_h[j]);
                break;
            }
        }
    }

    printf("-------------------------------------------------------------\n");

    // Bersihkan
    delete[] a_h;
    cudaFree(a_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
