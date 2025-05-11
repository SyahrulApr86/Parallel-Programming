#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <string.h>
#include <time.h>

// Function to generate random matrices
void generate_random_matrices(double *A, double *B, int N) {
    srand(12345);  // Fixed seed for reproducibility
    
    // Generate random values between 0 and 1
    for (int i = 0; i < N*N; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }
}

// Function to verify the result of matrix multiplication
int verify_result(double *A, double *B, double *C, int N, double tolerance) {
    // Compute C_ref = A*B sequentially
    double *C_ref = (double*)malloc(N*N*sizeof(double));
    
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C_ref[i*N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                C_ref[i*N + j] += A[i*N + k] * B[k*N + j];
            }
        }
    }
    
    // Compare with parallel result
    int correct = 1;
    for (int i = 0; i < N*N; i++) {
        if (fabs(C[i] - C_ref[i]) > tolerance) {
            printf("Verification failed at index %d: %f vs %f\n", i, C[i], C_ref[i]);
            correct = 0;
            break;
        }
    }
    
    free(C_ref);
    return correct;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 1024;  // Default matrix size
    int verify = 0;  // Verification is off by default
    
    // MPI initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse command line arguments
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    if (argc > 2 && strcmp(argv[2], "verify") == 0) {
        verify = 1;
    }
    
    if (rank == 0) {
        printf("Matrix Multiplication using MPI\n");
        printf("Matrix size: %d x %d\n", N, N);
        printf("Number of processes: %d\n", size);
    }
    
    // Timing variables
    double start_time, end_time;
    double compute_time = 0.0;
    double comm_time = 0.0;
    
    // Calculate local problem size
    int local_N = N / size;
    int remainder = N % size;
    
    // Calculate local size and starting position
    int local_rows = (rank < remainder) ? local_N + 1 : local_N;
    int local_start = rank * local_N + ((rank < remainder) ? rank : remainder);
    
    // Allocate memory for matrices
    double *A = NULL;       // Full matrix A (only on rank 0)
    double *B = NULL;       // Full matrix B (broadcast to all)
    double *C = NULL;       // Full result matrix (gathered on rank 0)
    
    // Local portions
    double *local_A = (double*)malloc(local_rows * N * sizeof(double));
    double *local_C = (double*)malloc(local_rows * N * sizeof(double));
    
    // Matrix B is needed in full by all processes
    B = (double*)malloc(N * N * sizeof(double));
    
    if (rank == 0) {
        A = (double*)malloc(N * N * sizeof(double));
        C = (double*)malloc(N * N * sizeof(double));
        
        if (A == NULL || C == NULL) {
            printf("Memory allocation failed on rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // Generate random matrices
        generate_random_matrices(A, B, N);
    }
    
    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    double comm_start, comm_end;
    
    // Broadcast matrix B to all processes
    comm_start = MPI_Wtime();
    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Prepare arrays for scatter operation
    int *sendcounts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));
    
    if (sendcounts == NULL || displs == NULL) {
        printf("Memory allocation failed on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    // Calculate send counts and displacements for matrix distribution
    int disp = 0;
    for (int i = 0; i < size; i++) {
        int rows = (i < remainder) ? local_N + 1 : local_N;
        sendcounts[i] = rows * N;
        displs[i] = disp;
        disp += sendcounts[i];
    }
    
    // Scatter matrix A
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, 
                 local_A, local_rows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    comm_end = MPI_Wtime();
    comm_time += comm_end - comm_start;
    
    // Matrix multiplication
    double compute_start = MPI_Wtime();
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i*N + j] = 0.0;
            for (int k = 0; k < N; k++) {
                local_C[i*N + j] += local_A[i*N + k] * B[k*N + j];
            }
        }
    }
    double compute_end = MPI_Wtime();
    compute_time += compute_end - compute_start;
    
    // Gather the result
    comm_start = MPI_Wtime();
    MPI_Gatherv(local_C, local_rows * N, MPI_DOUBLE,
                C, sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    comm_end = MPI_Wtime();
    comm_time += comm_end - comm_start;
    
    // End timing
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    // Verify the result if requested
    int verified = 0;
    if (rank == 0 && verify) {
        verified = verify_result(A, B, C, N, 1e-10);
    }
    
    // Print results from rank 0
    if (rank == 0) {
        double total_time = end_time - start_time;
        printf("Matrix multiplication completed.\n");
        printf("Total time: %f seconds\n", total_time);
        printf("Compute time: %f seconds\n", compute_time);
        printf("Communication time: %f seconds\n", comm_time);
        printf("Compute/Comm ratio: %f/%f\n", compute_time, comm_time);
        
        if (verify) {
            printf("Result verification: %s\n", verified ? "PASSED" : "FAILED");
        }
    }
    
    // Clean up
    free(sendcounts);
    free(displs);
    free(local_A);
    free(local_C);
    free(B);
    
    if (rank == 0) {
        if (A != NULL) free(A);
        if (C != NULL) free(C);
    }
    
    MPI_Finalize();
    return 0;
}
