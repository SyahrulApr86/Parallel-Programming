#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

// Function to generate a symmetric positive-definite matrix
void generate_spd_matrix(double *A, double *b, int N) {
    int i, j;
    srand(12345);  // Fixed seed for reproducibility
    
    // First generate a random matrix
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            // Generate random value between -0.5 and 0.5
            A[i*N + j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }
    
    // Make it symmetric: A = 0.5 * (A + A^T)
    for (i = 0; i < N; i++) {
        for (j = 0; j < i; j++) {  // Only need to process lower triangle
            double avg = (A[i*N + j] + A[j*N + i]) * 0.5;
            A[i*N + j] = A[j*N + i] = avg;
        }
    }
    
    // Make it diagonally dominant to ensure positive definiteness
    for (i = 0; i < N; i++) {
        double row_sum = 0.0;
        for (j = 0; j < N; j++) {
            if (i != j) {
                row_sum += fabs(A[i*N + j]);
            }
        }
        // Make diagonal elements larger than sum of other elements in row
        A[i*N + i] = row_sum + 1.0;
    }
    
    // Generate right-hand side vector b
    for (i = 0; i < N; i++) {
        b[i] = ((double)rand() / RAND_MAX) * 10.0;
    }
}

// Function to compute matrix-vector product y = A*x
void matrix_vector_mult(double *local_A, double *x, double *y, int local_start, int local_size, int N) {
    for (int i = 0; i < local_size; i++) {
        y[i] = 0.0;
        for (int j = 0; j < N; j++) {
            y[i] += local_A[i*N + j] * x[j];
        }
    }
}

// Function to compute dot product of two vectors
double dot_product(double *a, double *b, int local_size) {
    double local_dot = 0.0;
    
    for (int i = 0; i < local_size; i++) {
        local_dot += a[i] * b[i];
    }
    
    return local_dot;
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
    int max_iter = 5000;  // Maximum iterations
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
        printf("Conjugate Gradient method for solving linear systems\n");
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
    int local_size = (rank < remainder) ? local_N + 1 : local_N;
    int local_start = rank * local_N + ((rank < remainder) ? rank : remainder);
    
    // Allocate memory for matrices and vectors
    double *A = NULL;       // Full matrix (only on rank 0)
    double *b = NULL;       // Full right-hand side (only on rank 0)
    double *x = (double*)malloc(N * sizeof(double));  // Solution vector
    
    // Local portions
    double *local_A = (double*)malloc(local_size * N * sizeof(double));
    double *local_b = (double*)malloc(local_size * sizeof(double));
    
    // CG algorithm vectors
    double *r = (double*)malloc(N * sizeof(double));
    double *p = (double*)malloc(N * sizeof(double));
    double *Ap = (double*)malloc(N * sizeof(double));
    
    // Local portions of CG vectors
    double *local_r = (double*)malloc(local_size * sizeof(double));
    double *local_Ap = (double*)malloc(local_size * sizeof(double));
    
    // Initialize solution vector with zeros
    for (int i = 0; i < N; i++) {
        x[i] = 0.0;
    }
    
    // Create matrix on rank 0
    if (rank == 0) {
        A = (double*)malloc(N * N * sizeof(double));
        b = (double*)malloc(N * sizeof(double));
        
        if (A == NULL || b == NULL) {
            printf("Memory allocation failed on rank 0\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        generate_spd_matrix(A, b, N);
    }
    
    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    
    double comm_start, comm_end;
    
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
    
    // Scatter matrix A and vector b
    comm_start = MPI_Wtime();
    
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE, 
                 local_A, local_size * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Recalculate for vector b
    disp = 0;
    for (int i = 0; i < size; i++) {
        int rows = (i < remainder) ? local_N + 1 : local_N;
        sendcounts[i] = rows;
        displs[i] = disp;
        disp += rows;
    }
    
    MPI_Scatterv(b, sendcounts, displs, MPI_DOUBLE,
                 local_b, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Broadcast x (initially zero) to all processes
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    comm_end = MPI_Wtime();
    comm_time += comm_end - comm_start;
    
    // Compute initial residual r = b - A*x
    double compute_start = MPI_Wtime();
    double compute_end;
    
    // Since x is zero initially, r = b
    for (int i = 0; i < local_size; i++) {
        local_r[i] = local_b[i];
    }
    
    // Gather the complete residual vector
    comm_start = MPI_Wtime();
    MPI_Allgatherv(local_r, local_size, MPI_DOUBLE,
                  r, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    comm_end = MPI_Wtime();
    comm_time += comm_end - comm_start;
    
    // Set initial search direction p = r
    for (int i = 0; i < N; i++) {
        p[i] = r[i];
    }
    
    compute_end = MPI_Wtime();
    compute_time += compute_end - compute_start;
    
    // CG iteration
    int iter;
    double r_dot_r, r_dot_r_new, p_dot_Ap;
    double alpha, beta;
    double residual_norm;
    
    // Compute initial r_dot_r (r·r)
    comm_start = MPI_Wtime();
    double local_r_dot_r = dot_product(local_r, local_r, local_size);
    MPI_Allreduce(&local_r_dot_r, &r_dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    // Compute initial residual norm
    residual_norm = sqrt(r_dot_r);
    comm_end = MPI_Wtime();
    comm_time += comm_end - comm_start;
    
    if (rank == 0) {
        printf("Initial residual: %e\n", residual_norm);
    }
    
    for (iter = 0; iter < max_iter && residual_norm > tol; iter++) {
        // Compute Ap
        compute_start = MPI_Wtime();
        matrix_vector_mult(local_A, p, local_Ap, local_start, local_size, N);
        compute_end = MPI_Wtime();
        compute_time += compute_end - compute_start;
        
        // Gather all Ap pieces
        comm_start = MPI_Wtime();
        MPI_Allgatherv(local_Ap, local_size, MPI_DOUBLE,
                      Ap, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        comm_end = MPI_Wtime();
        comm_time += comm_end - comm_start;
        
        // Compute p_dot_Ap (p·Ap)
        compute_start = MPI_Wtime();
        double local_p_dot_Ap = 0.0;
        for (int i = 0; i < local_size; i++) {
            local_p_dot_Ap += p[local_start + i] * local_Ap[i];
        }
        compute_end = MPI_Wtime();
        compute_time += compute_end - compute_start;
        
        // Global sum for p_dot_Ap
        comm_start = MPI_Wtime();
        MPI_Allreduce(&local_p_dot_Ap, &p_dot_Ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        comm_end = MPI_Wtime();
        comm_time += comm_end - comm_start;
        
        // Compute alpha = (r·r)/(p·Ap)
        compute_start = MPI_Wtime();
        alpha = r_dot_r / p_dot_Ap;
        
        // Update x and r
        for (int i = 0; i < N; i++) {
            x[i] = x[i] + alpha * p[i];
        }
        
        for (int i = 0; i < local_size; i++) {
            local_r[i] = local_r[i] - alpha * local_Ap[i];
        }
        compute_end = MPI_Wtime();
        compute_time += compute_end - compute_start;
        
        // Gather updated residual
        comm_start = MPI_Wtime();
        MPI_Allgatherv(local_r, local_size, MPI_DOUBLE,
                      r, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        comm_end = MPI_Wtime();
        comm_time += comm_end - comm_start;
        
        // Compute new r_dot_r (r_new·r_new)
        compute_start = MPI_Wtime();
        double local_r_dot_r_new = dot_product(local_r, local_r, local_size);
        compute_end = MPI_Wtime();
        compute_time += compute_end - compute_start;
        
        comm_start = MPI_Wtime();
        MPI_Allreduce(&local_r_dot_r_new, &r_dot_r_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        comm_end = MPI_Wtime();
        comm_time += comm_end - comm_start;
        
        // Compute beta = (r_new·r_new)/(r·r)
        compute_start = MPI_Wtime();
        beta = r_dot_r_new / r_dot_r;
        
        // Update search direction p
        for (int i = 0; i < N; i++) {
            p[i] = r[i] + beta * p[i];
        }
        
        // Update r_dot_r for next iteration
        r_dot_r = r_dot_r_new;
        
        // Update residual norm
        residual_norm = sqrt(r_dot_r);
        compute_end = MPI_Wtime();
        compute_time += compute_end - compute_start;
        
        // Print progress periodically
        if (rank == 0 && iter % 100 == 0) {
            printf("Iteration %d: residual = %e\n", iter, residual_norm);
        }
    }
    
    // End timing
    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    
    // Print results from rank 0
    if (rank == 0) {
        double total_time = end_time - start_time;
        printf("Iterations: %d\n", iter);
        printf("Final residual: %e\n", residual_norm);
        printf("Total time: %f seconds\n", total_time);
        printf("Compute time: %f seconds\n", compute_time);
        printf("Communication time: %f seconds\n", comm_time);
        printf("Compute/Comm ratio: %f/%f\n", compute_time, comm_time);
        
        // Calculate residual norm ||Ax - b|| for verification
        if (A != NULL && b != NULL) {
            double final_residual = compute_residual(A, b, x, N);
            printf("Final error ||Ax - b||: %e\n", final_residual);
        }
    }
    
    // Clean up
    free(sendcounts);
    free(displs);
    free(x);
    free(local_A);
    free(local_b);
    free(local_r);
    free(local_Ap);
    free(r);
    free(p);
    free(Ap);
    
    if (rank == 0) {
        if (A != NULL) free(A);
        if (b != NULL) free(b);
    }
    
    MPI_Finalize();
    return 0;
}