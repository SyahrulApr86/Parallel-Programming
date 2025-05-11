#include <stdio.h>
#include <assert.h>
#include <cuda.h>

// Fungsi di host
void incrementArrayOnHost(float *a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = a[i] + 1.f;
    }
}

// Fungsi kernel di device (GPU)
__global__ void incrementArrayOnDevice(float *a, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        a[idx] = a[idx] + 1.f;
    }
}

int main(void) {
    float *a_h, *b_h; // pointer ke memori host
    float *a_d;       // pointer ke memori device
    int i, N = 10;
    size_t size = N * sizeof(float);

    // Alokasi array di host
    a_h = (float *)malloc(size);
    b_h = (float *)malloc(size);

    // Alokasi array di device
    cudaMalloc((void **)&a_d, size);

    // Inisialisasi data di host
    for (i = 0; i < N; i++) {
        a_h[i] = (float)i;
    }

    // Salin data dari host ke device
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

    // Jalankan kalkulasi di host
    incrementArrayOnHost(a_h, N);

    // Jalankan kalkulasi di device
    int blockSize = 4;
    int nBlocks = (N + blockSize - 1) / blockSize;

    incrementArrayOnDevice<<<nBlocks, blockSize>>>(a_d, N);

    // Sinkronisasi untuk memastikan kernel selesai
    cudaDeviceSynchronize();

    // Ambil hasil dari device ke host
    cudaMemcpy(b_h, a_d, size, cudaMemcpyDeviceToHost);

    // Verifikasi hasil
    for (i = 0; i < N; i++) {
        assert(a_h[i] == b_h[i]);
    }

    printf("✅ Semua elemen array berhasil ditambahkan +1 dan diverifikasi benar.\n");

    // Bersihkan memori
    free(a_h);
    free(b_h);
    cudaFree(a_d);

    return 0;
}
