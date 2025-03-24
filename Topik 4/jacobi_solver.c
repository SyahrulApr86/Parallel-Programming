#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

// Function to generate a diagonally dominant matrix
void generate_matrix(double *A, double *b, int N) {
    int i, j;
    srand(12345);  // Fixed seed for reproducibility
    
    for (i = 0; i < N; i++) {
        double row_sum = 0.0;
        for (j = 0; j < N; j++) {
            if (i != j) {
                // Generate random value between -1 and 1
                A[i*N + j] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
                row_sum += fabs(A[i*N + j]);
            } else {
                A[i*N + j] = 0.0;  // Will set diagonal later
            }
        }
        // Make diagonally dominant
        A[i*N + i] = row_sum + 1.0;
        
        // Generate right-hand side
        b[i] = ((double)rand() / RAND_MAX) * 10.0;
    }
}

// Function to verify solution (calculate residual norm ||Ax - b||)
double compute_residual(double *A, double *b, double *x, int N) {
    double *residual = (double*)malloc(N * sizeof(double));
    double norm = 0.0;
    int i, j;
    
    for (i = 0; i < N; i++) {
        residual[i] = b[i];
        for (j = 0; j < N; j++) {
            residual[i] -= A[i*N + j] * x[j];
        }
        norm += residual[i] * residual[i];
    }
    
    free(residual);
    return sqrt(norm);
}

int main(int argc, char *argv[]) {
    int rank, size;
    int N = 128;  // Default matrix size
    int max_iter = 1000;  // Maximum iterations
    double tol = 1e-6;    // Convergence tolerance
    
    // MPI initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse command line arguments
    if (argc > 1) {
        N = atoi(argv[1]);
    }
    
    if (rank == 0) {
        printf("Jacobi method for solving linear systems\n");
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
    int local_size;
    
    if (rank < remainder) {
        local_size = local_N + 1;
    } else {
        local_size = local_N;
    }
    
    // Calculate displacement
    int *counts = (int*)malloc(size * sizeof(int));
    int *displs = (int*)malloc(size * sizeof(int));
    
    int disp = 0;
    for (int i = 0; i < size; i++) {
        if (i < remainder) {
            counts[i] = local_N + 1;
        } else {
            counts[i] = local_N;
        }
        displs[i] = disp;
        disp += counts[i];
    }
    
    // Allocate memory for matrices and vectors
    double *A = NULL;       // Full matrix (only on rank 0)
    double *b = NULL;       // Full right-hand side (only on rank 0)
    double *x = (double*)malloc(N * sizeof(double));     // Current solution (all ranks)
    double *x_new = (double*)malloc(N * sizeof(double)); // New solution (all ranks)
    
    // Local portions
    double *local_A = (double*)malloc(local_size * N * sizeof(double));
    double *local_b = (double*)malloc(local_size * sizeof(double));
    double *local_x_new = (double*)malloc(local_size * sizeof(double));
    
    // Initialize solution vector with zeros
    for (int i = 0; i < N; i++) {
        x[i] = 0.0;
    }
    
    // Create matrix on rank 0
    if (rank == 0) {
        A = (double*)malloc(N * N * sizeof(double));
        b = (double*)malloc(N * sizeof(double));
        
        generate_matrix(A, b, N);
    }
    
    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    double comm_start, comm_end;
    
    // Distribute matrix and vector
    comm_start = MPI_Wtime();
    
    // Create counts and displacements for matrix distribution
    int *matrix_counts = (int*)malloc(size * sizeof(int));
    int *matrix_displs = (int*)malloc(size * sizeof(int));
    
    for (int i = 0; i < size; i++) {
        matrix_counts[i] = counts[i] * N;
        matrix_displs[i] = displs[i] * N;
    }
    
    // Distribute matrix A and vector b
    MPI_Scatterv(A, matrix_counts, matrix_displs, MPI_DOUBLE,
                 local_A, local_size * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    MPI_Scatterv(b, counts, displs, MPI_DOUBLE,
                 local_b, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    comm_end = MPI_Wtime();
    comm_time += comm_end - comm_start;
    
    // Jacobi iteration
    int iter;
    double residual = tol + 1.0;
    double local_residual;
    
    for (iter = 0; iter < max_iter && residual > tol; iter++) {
        double compute_start = MPI_Wtime();
        
        // Compute new values for local portion
        for (int i = 0; i < local_size; i++) {
            double sum = 0.0;
            int global_row = displs[rank] + i;
            
            for (int j = 0; j < N; j++) {
                if (j != global_row) {
                    sum += local_A[i*N + j] * x[j];
                }
            }
            
            local_x_new[i] = (local_b[i] - sum) / local_A[i*N + global_row];
        }
        
        double compute_end = MPI_Wtime();
        compute_time += compute_end - compute_start;
        
        // Gather new solution and check convergence
        comm_start = MPI_Wtime();
        
        MPI_Allgatherv(local_x_new, local_size, MPI_DOUBLE,
                       x_new, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        
        // Compute local maximum difference
        local_residual = 0.0;
        for (int i = 0; i < N; i++) {
            double diff = fabs(x_new[i] - x[i]);
            if (diff > local_residual) {
                local_residual = diff;
            }
        }
        
        // Find global maximum difference
        MPI_Allreduce(&local_residual, &residual, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        
        // Update solution
        memcpy(x, x_new, N * sizeof(double));
        
        comm_end = MPI_Wtime();
        comm_time += comm_end - comm_start;
    }
    
    // End timing
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    // Print results from rank 0
    if (rank == 0) {
        double total_time = end_time - start_time;
        printf("Iterations: %d\n", iter);
        printf("Final residual: %e\n", residual);
        printf("Total time: %f seconds\n", total_time);
        printf("Compute time: %f seconds\n", compute_time);
        printf("Communication time: %f seconds\n", comm_time);
        printf("Compute/Comm ratio: %f/%f\n", compute_time, comm_time);
        
        // Calculate residual norm ||Ax - b|| for verification
        double final_residual = compute_residual(A, b, x, N);
        printf("Final error ||Ax - b||: %e\n", final_residual);
    }
    
    // Clean up
    free(counts);
    free(displs);
    free(matrix_counts);
    free(matrix_displs);
    free(x);
    free(x_new);
    free(local_A);
    free(local_b);
    free(local_x_new);
    
    if (rank == 0) {
        free(A);
        free(b);
    }
    
    MPI_Finalize();
    return 0;
}