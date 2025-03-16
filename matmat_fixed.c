#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// Matrix-Matrix multiplication program with timing for computation and communication
// Run with: mpirun -np <num_processes> ./matmat <matrix_size>

int main(int argc, char *argv[]) {
    int rank, size, n;
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A = NULL, *local_C = NULL;
    int rows_per_proc, start_row, end_row;
    double comp_time = 0.0, comm_time = 0.0;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get matrix size from command line
    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: %s <matrix_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    n = atoi(argv[1]);
    
    // Calculate rows per process - ensure equal distribution
    rows_per_proc = n / size;
    if (rows_per_proc * size != n) {
        if (rank == 0) {
            printf("Error: Matrix size (%d) must be divisible by number of processes (%d)\n", n, size);
        }
        MPI_Finalize();
        return 1;
    }
    
    start_row = rank * rows_per_proc;
    end_row = start_row + rows_per_proc;
    
    // ALL processes allocate memory for B
    B = (double*)malloc(n * n * sizeof(double));
    if (!B) {
        printf("Process %d: Failed to allocate memory for matrix B\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    // Allocate memory for local matrices
    local_A = (double*)malloc(rows_per_proc * n * sizeof(double));
    local_C = (double*)malloc(rows_per_proc * n * sizeof(double));
    
    if (!local_A || !local_C) {
        printf("Process %d: Failed to allocate memory for local matrices\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    // Process 0 initializes the full matrices
    if (rank == 0) {
        A = (double*)malloc(n * n * sizeof(double));
        C = (double*)malloc(n * n * sizeof(double));
        
        if (!A || !C) {
            printf("Process 0: Failed to allocate memory for matrices A or C\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // Initialize matrices with random values
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i * n + j] = (double)rand() / RAND_MAX;
                B[i * n + j] = (double)rand() / RAND_MAX;
            }
        }
    }
    
    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Broadcast matrix B to all processes
    start_time = MPI_Wtime();
    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    comm_time += end_time - start_time;
    
    // Distribute matrix A rows
    start_time = MPI_Wtime();
    MPI_Scatter(A, rows_per_proc * n, MPI_DOUBLE, 
                local_A, rows_per_proc * n, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    comm_time += end_time - start_time;
    
    // Perform local matrix-matrix multiplication
    start_time = MPI_Wtime();
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < n; j++) {
            local_C[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                local_C[i * n + j] += local_A[i * n + k] * B[k * n + j];
            }
        }
    }
    end_time = MPI_Wtime();
    comp_time = end_time - start_time;
    
    // Gather results
    start_time = MPI_Wtime();
    MPI_Gather(local_C, rows_per_proc * n, MPI_DOUBLE,
               C, rows_per_proc * n, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    comm_time += end_time - start_time;
    
    // Reduce computation and communication times
    double global_comp_time, global_comm_time;
    MPI_Reduce(&comp_time, &global_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &global_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Print results
    if (rank == 0) {
        printf("Matrix-Matrix Multiplication: N=%d, NP=%d\n", n, size);
        printf("Computation time: %.6f s\n", global_comp_time);
        printf("Communication time: %.6f s\n", global_comm_time);
        printf("Total time: %.6f s\n", global_comp_time + global_comm_time);
        printf("Result summary (%.6f/%.6f)\n", global_comp_time, global_comm_time);
    }
    
    // Free memory
    if (rank == 0) {
        if (A) free(A);
        if (C) free(C);
    }
    if (B) free(B);
    if (local_A) free(local_A);
    if (local_C) free(local_C);
    
    MPI_Finalize();
    return 0;
}
