#include <stdio.h>
#include <cuda_runtime.h>

__global__ void inc_gpu(int *a_d, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a_d[idx] = a_d[idx] + 1;
    }
}

int main() {
    // Variasi ukuran array N
    int arraySizes[] = {256, 1024, 4096, 8192};
    int numArraySizes = sizeof(arraySizes) / sizeof(int);

    // Variasi block size (termasuk yang melebihi batas untuk uji kegagalan)
    int blockSizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
    int numBlockSizes = sizeof(blockSizes) / sizeof(int);

    // CUDA event untuk waktu eksekusi
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Eksperimen Increment Array CUDA\n");
    printf("=====================================================================\n");
    printf("|    N    | BlockSize | GridSize | Time (ms) |     Status          |\n");
    printf("=====================================================================\n");

    for (int nidx = 0; nidx < numArraySizes; ++nidx) {
        int N = arraySizes[nidx];
        int *a_h = new int[N];
        int *a_d;
        cudaMalloc((void**)&a_d, N * sizeof(int));

        for (int bidx = 0; bidx < numBlockSizes; ++bidx) {
            int blockSize = blockSizes[bidx];
            int gridSize = (N + blockSize - 1) / blockSize;

            // Inisialisasi ulang array host
            for (int j = 0; j < N; ++j) a_h[j] = j;
            cudaMemcpy(a_d, a_h, N * sizeof(int), cudaMemcpyHostToDevice);

            // Jalankan kernel dan ukur waktu
            cudaEventRecord(start);
            inc_gpu<<<gridSize, blockSize>>>(a_d, N);
            cudaEventRecord(stop);

            cudaEventSynchronize(stop);
            float ms = 0;
            cudaEventElapsedTime(&ms, start, stop);

            // Salin hasil kembali
            cudaMemcpy(a_h, a_d, N * sizeof(int), cudaMemcpyDeviceToHost);

            // Validasi
            bool correct = true;
            for (int j = 0; j < N; ++j) {
                if (a_h[j] != j + 1) {
                    correct = false;
                    break;
                }
            }

            printf("| %7d | %9d | %8d | %9.4f | %s\n", 
                N, blockSize, gridSize, ms,
                correct ? "✅ correct" : "❌ ERROR");

            // (Opsional) flush error CUDA
            cudaGetLastError();
        }

        cudaFree(a_d);
        delete[] a_h;
    }

    printf("=====================================================================\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
