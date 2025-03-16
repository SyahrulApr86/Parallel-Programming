#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// Matrix-Vector multiplication program with timing for computation and communication
// Run with: mpirun -np <num_processes> ./matvec <matrix_size>

int main(int argc, char *argv[]) {
    int rank, size, n;
    double *matrix = NULL, *vector = NULL, *result = NULL, *local_result = NULL;
    double *local_matrix = NULL;
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
    
    // All processes allocate memory for the vector
    vector = (double*)malloc(n * sizeof(double));
    if (!vector) {
        printf("Process %d: Failed to allocate memory for vector\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    // Allocate memory for local matrix portion and result
    local_matrix = (double*)malloc(rows_per_proc * n * sizeof(double));
    local_result = (double*)malloc(rows_per_proc * sizeof(double));
    
    if (!local_matrix || !local_result) {
        printf("Process %d: Failed to allocate memory for local data\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    // Process 0 initializes the full matrix and vector
    if (rank == 0) {
        matrix = (double*)malloc(n * n * sizeof(double));
        result = (double*)malloc(n * sizeof(double));
        
        if (!matrix || !result) {
            printf("Process 0: Failed to allocate memory for matrix or result\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // Initialize matrix and vector with random values
        srand(time(NULL));
        for (int i = 0; i < n; i++) {
            vector[i] = (double)rand() / RAND_MAX;
            for (int j = 0; j < n; j++) {
                matrix[i * n + j] = (double)rand() / RAND_MAX;
            }
        }
    }
    
    // Start timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Broadcast vector to all processes
    start_time = MPI_Wtime();
    MPI_Bcast(vector, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    comm_time += end_time - start_time;
    
    // Distribute matrix rows
    start_time = MPI_Wtime();
    MPI_Scatter(matrix, rows_per_proc * n, MPI_DOUBLE, 
                local_matrix, rows_per_proc * n, MPI_DOUBLE, 
                0, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    comm_time += end_time - start_time;
    
    // Perform local matrix-vector multiplication
    start_time = MPI_Wtime();
    for (int i = 0; i < rows_per_proc; i++) {
        local_result[i] = 0.0;
        for (int j = 0; j < n; j++) {
            local_result[i] += local_matrix[i * n + j] * vector[j];
        }
    }
    end_time = MPI_Wtime();
    comp_time = end_time - start_time;
    
    // Gather results
    start_time = MPI_Wtime();
    MPI_Gather(local_result, rows_per_proc, MPI_DOUBLE,
               result, rows_per_proc, MPI_DOUBLE,
               0, MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    comm_time += end_time - start_time;
    
    // Reduce computation and communication times
    double global_comp_time, global_comm_time;
    MPI_Reduce(&comp_time, &global_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &global_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Print results
    if (rank == 0) {
        printf("Matrix-Vector Multiplication: N=%d, NP=%d\n", n, size);
        printf("Computation time: %.6f s\n", global_comp_time);
        printf("Communication time: %.6f s\n", global_comm_time);
        printf("Total time: %.6f s\n", global_comp_time + global_comm_time);
        printf("Result summary (%.6f/%.6f)\n", global_comp_time, global_comm_time);
    }
    
    // Free memory
    if (rank == 0) {
        if (matrix) free(matrix);
        if (result) free(result);
    }
    if (vector) free(vector);
    if (local_matrix) free(local_matrix);
    if (local_result) free(local_result);
    
    MPI_Finalize();
    return 0;
}
