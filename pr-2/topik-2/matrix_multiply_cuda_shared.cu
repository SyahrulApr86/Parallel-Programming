#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

// Kernel CUDA dengan shared memory
__global__ void matrix_multiply_cuda_shared(float *A, float *B, float *C, int N, int TILE_DIM) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    extern __shared__ float shared_mem[];
    float *As = shared_mem;
    float *Bs = shared_mem + TILE_DIM * TILE_DIM;

    float sum = 0.0f;

    // Jumlah tile yang perlu diproses
    int numTiles = (N + TILE_DIM - 1) / TILE_DIM;

    for (int t = 0; t < numTiles; t++) {
        // Load data ke shared memory
        if (row < N && t * TILE_DIM + tx < N)
            As[ty * TILE_DIM + tx] = A[row * N + t * TILE_DIM + tx];
        else
            As[ty * TILE_DIM + tx] = 0.0f;

        if (t * TILE_DIM + ty < N && col < N)
            Bs[ty * TILE_DIM + tx] = B[(t * TILE_DIM + ty) * N + col];
        else
            Bs[ty * TILE_DIM + tx] = 0.0f;

        __syncthreads();

        // Hitung dot product untuk tile ini
        for (int k = 0; k < TILE_DIM; k++) {
            sum += As[ty * TILE_DIM + k] * Bs[k * TILE_DIM + tx];
        }

        __syncthreads();
    }

    // Tulis hasil
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Perkalian matriks sekuensial di CPU (untuk verifikasi)
void matrix_multiply_sequential(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i*N + k] * B[k*N + j];
            }
            C[i*N + j] = sum;
        }
    }
}

// Fungsi untuk menghasilkan matriks acak
void generate_random_matrix(float *M, int N) {
    for (int i = 0; i < N*N; i++) {
        M[i] = (float)rand() / RAND_MAX;
    }
}

// Fungsi untuk memeriksa kesamaan dua matriks dengan toleransi
int compare_matrices(float *A, float *B, int N, float tolerance) {
    for (int i = 0; i < N*N; i++) {
        if (fabsf(A[i] - B[i]) > tolerance) {
            printf("Mismatch at index %d: %f vs %f\n", i, A[i], B[i]);
            return 0;
        }
    }
    return 1;
}

// Fungsi untuk menyimpan matriks ke file (untuk verifikasi)
void save_matrix(float *M, int N, const char *filename) {
    FILE *fp = fopen(filename, "w");
    if (!fp) {
        fprintf(stderr, "Error: Tidak dapat membuka file %s\n", filename);
        return;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(fp, "%.6f ", M[i*N + j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
}

int main(int argc, char *argv[]) {
    // Variabel untuk menyimpan parameter
    int N = 1024; // Default size
    int blockSize = 16; // Default block size
    int gridSize = 0;   // Will be calculated based on N and blockSize
    int verify_result = 1; // Verifikasi hasil secara default
    int save_result = 0;   // Jangan simpan hasil secara default

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-N") == 0 && i + 1 < argc) {
            N = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
            blockSize = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-g") == 0 && i + 1 < argc) {
            gridSize = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-noverify") == 0) {
            verify_result = 0;
        } else if (strcmp(argv[i], "-save") == 0) {
            save_result = 1;
        }
    }

    // Calculate grid size if not specified
    if (gridSize == 0) {
        gridSize = (N + blockSize - 1) / blockSize;
    }

    // Menampilkan informasi eksekusi
    printf("Matrix Multiplication CUDA Shared Memory: N=%d, blockSize=%d, gridSize=%d\n",
           N, blockSize, gridSize);

    // Alokasi memori host
    float *A = (float*)malloc(N*N*sizeof(float));
    float *B = (float*)malloc(N*N*sizeof(float));
    float *C = (float*)malloc(N*N*sizeof(float));
    float *C_ref = NULL;

    if (verify_result) {
        C_ref = (float*)malloc(N*N*sizeof(float));
    }

    if (!A || !B || !C || (verify_result && !C_ref)) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }

    // Alokasi memori device
    float *d_A, *d_B, *d_C;
    cudaError_t err;

    err = cudaMalloc((void**)&d_A, N*N*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaMalloc d_A failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    err = cudaMalloc((void**)&d_B, N*N*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaMalloc d_B failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    err = cudaMalloc((void**)&d_C, N*N*sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaMalloc d_C failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Inisialisasi seed untuk angka acak
    srand(42); // Gunakan seed tetap untuk konsistensi

    // Generate matriks acak
    generate_random_matrix(A, N);
    generate_random_matrix(B, N);

    // Reset matriks hasil
    memset(C, 0, N*N*sizeof(float));
    if (verify_result) {
        memset(C_ref, 0, N*N*sizeof(float));
    }

    // Variabel untuk menyimpan waktu
    float computation_time = 0.0f;
    float communication_time = 0.0f;

    // CUDA Events untuk pengukuran waktu
    cudaEvent_t start, stop, start_comm, stop_comm;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start_comm);
    cudaEventCreate(&stop_comm);

    // Ukur waktu komunikasi (transfer data host ke device)
    cudaEventRecord(start_comm);

    cudaMemcpy(d_A, A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaEventRecord(stop_comm);
    cudaEventSynchronize(stop_comm);
    float comm_time1;
    cudaEventElapsedTime(&comm_time1, start_comm, stop_comm);

    // Ukur waktu komputasi
    cudaEventRecord(start);

    // Konfigurasi eksekusi kernel
    dim3 blockDim(blockSize, blockSize);
    dim3 gridDim(gridSize, gridSize);

    // Calculate shared memory size
    int sharedMemSize = 2 * blockSize * blockSize * sizeof(float);

    matrix_multiply_cuda_shared<<<gridDim, blockDim, sharedMemSize>>>(d_A, d_B, d_C, N, blockSize);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float comp_time;
    cudaEventElapsedTime(&comp_time, start, stop);

    // Check for errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Kernel execution failed: %s\n", cudaGetErrorString(err));
        exit(1);
    }

    // Ukur waktu transfer data kembali ke host
    cudaEventRecord(start_comm);

    cudaMemcpy(C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_comm);
    cudaEventSynchronize(stop_comm);
    float comm_time2;
    cudaEventElapsedTime(&comm_time2, start_comm, stop_comm);

    computation_time = comp_time;
    communication_time = comm_time1 + comm_time2;

    printf("CUDA Shared Memory: Computation Time = %.4f ms, Communication Time = %.4f ms\n",
           computation_time, communication_time);

    // Verifikasi hasil jika diminta
    if (verify_result) {
        matrix_multiply_sequential(A, B, C_ref, N);

        if (compare_matrices(C, C_ref, N, 1e-2)) {
            printf("Verification: Results match!\n");
        } else {
            printf("Verification: Results do NOT match!\n");
        }
    }

    // Format for CSV output
    printf("CSV,2,%d,%d,%d,%.4f,%.4f\n",
           N, blockSize, gridSize, computation_time, communication_time);

    // Simpan hasil jika diminta
    if (save_result) {
        save_matrix(A, N, "matrix_A.txt");
        save_matrix(B, N, "matrix_B.txt");
        save_matrix(C, N, "matrix_C_cuda_shared.txt");
    }

    // Bersihkan
    free(A);
    free(B);
    free(C);
    if (verify_result) free(C_ref);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start_comm);
    cudaEventDestroy(stop_comm);

    return 0;
}
