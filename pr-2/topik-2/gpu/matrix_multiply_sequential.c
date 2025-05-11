#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

// Perkalian matriks sekuensial di CPU
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
    int save_result = 0; // Jangan simpan hasil secara default

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-N") == 0 && i + 1 < argc) {
            N = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-save") == 0) {
            save_result = 1;
        }
    }

    // Menampilkan informasi eksekusi
    printf("Matrix Multiplication Sequential: N=%d\n", N);

    // Alokasi memori host
    float *A = (float*)malloc(N*N*sizeof(float));
    float *B = (float*)malloc(N*N*sizeof(float));
    float *C = (float*)malloc(N*N*sizeof(float));

    if (!A || !B || !C) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        exit(1);
    }

    // Inisialisasi seed untuk angka acak
    srand(42); // Gunakan seed tetap untuk konsistensi

    // Generate matriks acak
    generate_random_matrix(A, N);
    generate_random_matrix(B, N);

    // Reset matriks hasil
    memset(C, 0, N*N*sizeof(float));

    // Ukur waktu eksekusi
    clock_t start, end;
    double cpu_time_used;

    start = clock();
    matrix_multiply_sequential(A, B, C, N);
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0; // Convert to ms

    printf("Sequential CPU: Computation Time = %.4f ms, Communication Time = 0.0000 ms\n",
           cpu_time_used);

    // Format for CSV output
    printf("CSV,0,%d,0,0,%.4f,0.0000\n", N, cpu_time_used);

    // Simpan hasil jika diminta
    if (save_result) {
        save_matrix(A, N, "matrix_A.txt");
        save_matrix(B, N, "matrix_B.txt");
        save_matrix(C, N, "matrix_C_sequential.txt");
    }

    // Bersihkan
    free(A);
    free(B);
    free(C);

    return 0;
}
