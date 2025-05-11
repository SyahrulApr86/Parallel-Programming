#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

__global__ void incrementArrayLoop(float *a, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < N; i += gridDim.x * blockDim.x) {
        a[i] += 1.0f;
    }
}

void incrementArrayOnHost(float *a, size_t N) {
    for (size_t i = 0; i < N; i++) {
        a[i] += 1.0f;
    }
}

int main() {
    // Ukuran array yang divariasikan
    size_t sizes[] = {
        16,
        1000,
        1 << 20,         // 1M
        1 << 24,         // 16M
        1 << 28,         // 268M (1GB)
        1 << 30          // 1B (4GB)
    };
    int numSizes = sizeof(sizes) / sizeof(size_t);

    // Variasi jumlah block
    int nBlocksOptions[] = {32, 64, 128, 256};
    int numBlockOptions = sizeof(nBlocksOptions) / sizeof(int);

    // Tetapkan blockSize tetap
    int blockSize = 256;

    // Siapkan event CUDA untuk pengukuran waktu
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int s = 0; s < numSizes; ++s) {
        size_t N = sizes[s];
        size_t size = N * sizeof(float);
        double sizeMB = size / (1024.0 * 1024.0);

        for (int b = 0; b < numBlockOptions; ++b) {
            int nBlocks = nBlocksOptions[b];
            printf("➡️  N = %zu (%.2f MB) | nBlocks = %d | blockSize = %d\n", N, sizeMB, nBlocks, blockSize);

            float *a_h = (float *)malloc(size);
            float *b_h = (float *)malloc(size);
            float *a_d = NULL;

            if (!a_h || !b_h) {
                printf("❌ Gagal malloc host untuk N = %zu\n\n", N);
                free(a_h); free(b_h);
                continue;
            }

            cudaError_t err = cudaMalloc((void **)&a_d, size);
            if (err != cudaSuccess) {
                printf("❌ Gagal cudaMalloc untuk N = %zu: %s\n\n", N, cudaGetErrorString(err));
                free(a_h); free(b_h);
                continue;
            }

            for (size_t i = 0; i < N; i++) a_h[i] = (float)i;
            cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

            // Jalankan versi host
            incrementArrayOnHost(a_h, N);

            // Jalankan kernel
            cudaEventRecord(start);
            incrementArrayLoop<<<nBlocks, blockSize>>>(a_d, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float gpuTimeMs = 0;
            cudaEventElapsedTime(&gpuTimeMs, start, stop);

            cudaMemcpy(b_h, a_d, size, cudaMemcpyDeviceToHost);

            // Verifikasi
            bool correct = true;
            for (size_t i = 0; i < N; i++) {
                if (a_h[i] != b_h[i]) {
                    correct = false;
                    printf("❌ ERROR at index %zu: expected %.1f, got %.1f\n", i, a_h[i], b_h[i]);
                    break;
                }
            }

            printf("✅ N = %-12zu | %8.2f MB | nBlocks = %4d | Time = %8.4f ms | Status: %s\n\n",
                   N, sizeMB, nBlocks, gpuTimeMs, correct ? "correct" : "incorrect");

            free(a_h); free(b_h); cudaFree(a_d);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
