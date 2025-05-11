#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>

// Kernel TANPA loop → hanya satu elemen per thread
__global__ void incrementArrayNoLoop(float *a, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] += 1.0f;
    }
}

void incrementArrayOnHost(float *a, size_t N) {
    for (size_t i = 0; i < N; i++) {
        a[i] += 1.0f;
    }
}

int main() {
    // Ukuran array besar
    size_t sizes[] = {
        16,
        1000,
        1 << 20,         // 1M
        1 << 24,         // 16M
        1 << 28          // 268M (1GB)
    };
    int numSizes = sizeof(sizes) / sizeof(size_t);

    // Variasi nBlocks
    int nBlocksOptions[] = {32, 64};
    int numBlockOptions = sizeof(nBlocksOptions) / sizeof(int);

    int blockSize = 256;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int s = 0; s < numSizes; ++s) {
        size_t N = sizes[s];
        size_t size = N * sizeof(float);
        double sizeMB = size / (1024.0 * 1024.0);

        for (int b = 0; b < numBlockOptions; ++b) {
            int nBlocks = nBlocksOptions[b];
            size_t totalThreads = nBlocks * blockSize;

            printf("➡️  N = %zu (%.2f MB) | Threads = %zu | nBlocks = %d\n", N, sizeMB, totalThreads, nBlocks);

            float *a_h = (float *)malloc(size);
            float *b_h = (float *)malloc(size);
            float *a_d = NULL;

            if (!a_h || !b_h) {
                printf("❌ Gagal malloc host\n\n");
                continue;
            }

            if (cudaMalloc((void **)&a_d, size) != cudaSuccess) {
                printf("❌ Gagal cudaMalloc untuk N = %zu\n\n", N);
                free(a_h); free(b_h);
                continue;
            }

            for (size_t i = 0; i < N; i++) a_h[i] = (float)i;
            cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

            incrementArrayOnHost(a_h, N);

            cudaEventRecord(start);
            incrementArrayNoLoop<<<nBlocks, blockSize>>>(a_d, N);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            float gpuTimeMs = 0;
            cudaEventElapsedTime(&gpuTimeMs, start, stop);

            cudaMemcpy(b_h, a_d, size, cudaMemcpyDeviceToHost);

            // Verifikasi hasil — akan gagal jika totalThreads < N
            int errorCount = 0;
            for (size_t i = 0; i < N; i++) {
                if (a_h[i] != b_h[i]) {
                    if (errorCount < 10)
                        printf("❌ ERROR at %zu: expected %.1f, got %.1f\n", i, a_h[i], b_h[i]);
                    errorCount++;
                }
            }

            printf("🧪 N = %-12zu | %8.2f MB | nBlocks = %3d | Threads = %7zu | Time = %8.4f ms | Errors: %d\n\n",
                   N, sizeMB, nBlocks, totalThreads, gpuTimeMs, errorCount);

            free(a_h); free(b_h); cudaFree(a_d);
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
