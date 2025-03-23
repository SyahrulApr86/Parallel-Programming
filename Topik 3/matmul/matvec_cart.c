#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// Matrix-Vector multiplication using Cartesian topology
// Run with: mpirun -np <num_processes> ./matvec_cart <matrix_size>

int main(int argc, char *argv[]) {
    int rank, size, n;
    double *matrix = NULL, *vector = NULL, *result = NULL, *local_result = NULL;
    double *local_matrix = NULL;
    int rows_per_proc, start_row, end_row;
    double comp_time = 0.0, comm_time = 0.0;
    double start_time, end_time;
    
    // Cartesian topology variables
    MPI_Comm cart_comm;
    int ndims = 1;         // 1D Cartesian grid for matrix-vector mult
    int dims[1];           // Array for number of processes in each dimension
    int periods[1] = {0};  // Non-periodic boundaries
    int reorder = 1;       // Allow MPI to reorder processes for optimization
    int coords[1];         // Process coordinates in Cartesian grid
    int source, dest;      // Neighbor ranks
    
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
    
    // Create 1D Cartesian topology for processes
    dims[0] = size;  // Use all processes in a single dimension
    
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &cart_comm);
    
    // Get new rank and coordinates in Cartesian topology
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, ndims, coords);
    end_time = MPI_Wtime();
    comm_time += end_time - start_time;
    
    // Calculate row ranges based on Cartesian coordinates
    start_row = coords[0] * rows_per_proc;
    end_row = start_row + rows_per_proc;
    
    // All processes allocate memory for the vector
    vector = (double*)malloc(n * sizeof(double));
    if (!vector) {
        printf("Process %d: Failed to allocate memory for vector\n", rank);
        MPI_Abort(cart_comm, 1);
        return 1;
    }
    
    // Allocate memory for local matrix portion and result
    local_matrix = (double*)malloc(rows_per_proc * n * sizeof(double));
    local_result = (double*)malloc(rows_per_proc * sizeof(double));
    
    if (!local_matrix || !local_result) {
        printf("Process %d: Failed to allocate memory for local data\n", rank);
        MPI_Abort(cart_comm, 1);
        return 1;
    }
    
    // Root process initializes the full matrix and vector
    if (rank == 0) {
        matrix = (double*)malloc(n * n * sizeof(double));
        result = (double*)malloc(n * sizeof(double));
        
        if (!matrix || !result) {
            printf("Process 0: Failed to allocate memory for matrix or result\n");
            MPI_Abort(cart_comm, 1);
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
    MPI_Barrier(cart_comm);
    
    // Broadcast vector to all processes in the Cartesian grid
    start_time = MPI_Wtime();
    MPI_Bcast(vector, n, MPI_DOUBLE, 0, cart_comm);
    end_time = MPI_Wtime();
    comm_time += end_time - start_time;
    
    // Distribute matrix rows using Cartesian coordinates
    start_time = MPI_Wtime();
    MPI_Scatter(matrix, rows_per_proc * n, MPI_DOUBLE, 
                local_matrix, rows_per_proc * n, MPI_DOUBLE, 
                0, cart_comm);
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
    
    // Demonstrate use of Cart_shift to find neighbor processes
    // (Just for demonstration - not used in actual computation for matvec)
    start_time = MPI_Wtime();
    MPI_Cart_shift(cart_comm, 0, 1, &source, &dest);
    
    // Optional: Print neighbor information
    /*
    printf("Process %d (coords %d): Source = %d, Dest = %d\n", 
           rank, coords[0], source, dest);
    */
    
    // Gather results using Cartesian communicator
    MPI_Gather(local_result, rows_per_proc, MPI_DOUBLE,
               result, rows_per_proc, MPI_DOUBLE,
               0, cart_comm);
    end_time = MPI_Wtime();
    comm_time += end_time - start_time;
    
    // Reduce computation and communication times
    double global_comp_time, global_comm_time;
    MPI_Reduce(&comp_time, &global_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    MPI_Reduce(&comm_time, &global_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    
    // Print results
    if (rank == 0) {
        printf("Matrix-Vector Multiplication (Cartesian): N=%d, NP=%d\n", n, size);
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
    
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}