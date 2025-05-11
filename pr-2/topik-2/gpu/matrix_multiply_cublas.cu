#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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
    int verify_result = 1; // Verifikasi hasil secara default
    int save_result = 0;   // Jangan simpan hasil secara default

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-N") == 0 && i + 1 < argc) {
            N = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-noverify") == 0) {
            verify_result = 0;
        } else if (strcmp(argv[i], "-save") == 0) {
            save_result = 1;
        }
    }

    // Menampilkan informasi eksekusi
    printf("Matrix Multiplication cuBLAS: N=%d\n", N);

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

    // Inisialisasi cuBLAS
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Error: cuBLAS initialization failed\n");
        exit(1);
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

    // Perform matrix multiplication using cuBLAS
    // Note: cuBLAS expects column-major matrices, but we're using row-major
    // So we compute C = B*A instead of C = A*B
    float alpha = 1.0f;
    float beta = 0.0f;

    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                         N, N, N,
                         &alpha,
                         d_B, N,  // In column-major: first matrix is B
                         d_A, N,  // Second matrix is A
                         &beta,
                         d_C, N);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "Error: cuBLAS SGEMM failed\n");
        exit(1);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float comp_time;
    cudaEventElapsedTime(&comp_time, start, stop);

    // Ukur waktu transfer data kembali ke host
    cudaEventRecord(start_comm);

    cudaMemcpy(C, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop_comm);
    cudaEventSynchronize(stop_comm);
    float comm_time2;
    cudaEventElapsedTime(&comm_time2, start_comm, stop_comm);

    computation_time = comp_time;
    communication_time = comm_time1 + comm_time2;

    printf("cuBLAS: Computation Time = %.4f ms, Communication Time = %.4f ms\n",
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
    printf("CSV,3,%d,0,0,%.4f,%.4f\n",
           N, computation_time, communication_time);

    // Simpan hasil jika diminta
    if (save_result) {
        save_matrix(A, N, "matrix_A.txt");
        save_matrix(B, N, "matrix_B.txt");
        save_matrix(C, N, "matrix_C_cublas.txt");
    }

    // Bersihkan
    free(A);
    free(B);
    free(C);
    if (verify_result) free(C_ref);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cublasDestroy(handle);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start_comm);
    cudaEventDestroy(stop_comm);

    return 0;
}
