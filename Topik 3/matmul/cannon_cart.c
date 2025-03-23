#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

// Cannon's algorithm for matrix multiplication with 2D Cartesian topology
// Run with: mpirun -np <perfect_square> ./cannon_cart <matrix_size>

int main(int argc, char *argv[]) {
    int rank, size, n;
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A = NULL, *local_B = NULL, *local_C = NULL;
    double *temp_A = NULL, *temp_B = NULL;
    int grid_size, block_size;
    double comp_time = 0.0, comm_time = 0.0;
    double start_time, end_time;
    
    // Cartesian topology variables
    MPI_Comm cart_comm;
    int ndims = 2;                   // 2D Cartesian grid for Cannon's algorithm
    int dims[2] = {0, 0};            // Will be set by MPI_Dims_create
    int periods[2] = {1, 1};         // Periodic boundaries for shifting
    int reorder = 1;                 // Allow MPI to reorder for optimization
    int coords[2];                   // Process coordinates in grid
    int row_source, row_dest, col_source, col_dest;
    MPI_Comm row_comm, col_comm;     // Row and column communicators
    
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
    
    // Check if number of processes is a perfect square
    grid_size = (int)sqrt(size);
    if (grid_size * grid_size != size) {
        if (rank == 0) {
            printf("Error: Number of processes (%d) must be a perfect square\n", size);
        }
        MPI_Finalize();
        return 1;
    }
    
    // Check if matrix size is divisible by grid size
    if (n % grid_size != 0) {
        if (rank == 0) {
            printf("Error: Matrix size (%d) must be divisible by grid size (%d)\n", n, grid_size);
        }
        MPI_Finalize();
        return 1;
    }
    
    // Create 2D Cartesian topology
    dims[0] = dims[1] = grid_size;  // Equal number of processes in each dimension
    
    start_time = MPI_Wtime();
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &cart_comm);
    
    // Get new rank and coordinates in Cartesian topology
    MPI_Comm_rank(cart_comm, &rank);
    MPI_Cart_coords(cart_comm, rank, ndims, coords);
    
    // Create row and column communicators
    int remain_dims[2];
    
    // Row communicator: processes with same coords[0]
    remain_dims[0] = 0;  // Don't keep row dimension
    remain_dims[1] = 1;  // Keep column dimension
    MPI_Cart_sub(cart_comm, remain_dims, &row_comm);
    
    // Column communicator: processes with same coords[1]
    remain_dims[0] = 1;  // Keep row dimension
    remain_dims[1] = 0;  // Don't keep column dimension
    MPI_Cart_sub(cart_comm, remain_dims, &col_comm);
    end_time = MPI_Wtime();
    comm_time += end_time - start_time;
    
    // Calculate block size for each process
    block_size = n / grid_size;
    
    // Allocate memory for local blocks
    local_A = (double*)malloc(block_size * block_size * sizeof(double));
    local_B = (double*)malloc(block_size * block_size * sizeof(double));
    local_C = (double*)malloc(block_size * block_size * sizeof(double));
    temp_A = (double*)malloc(block_size * block_size * sizeof(double));
    temp_B = (double*)malloc(block_size * block_size * sizeof(double));
    
    if (!local_A || !local_B || !local_C || !temp_A || !temp_B) {
        printf("Process %d: Failed to allocate memory for local blocks\n", rank);
        MPI_Abort(cart_comm, 1);
        return 1;
    }
    
    // Root process initializes matrices A and B
    if (rank == 0) {
        A = (double*)malloc(n * n * sizeof(double));
        B = (double*)malloc(n * n * sizeof(double));
        C = (double*)malloc(n * n * sizeof(double));
        
        if (!A || !B || !C) {
            printf("Process 0: Failed to allocate memory for matrices\n");
            MPI_Abort(cart_comm, 1);
            return 1;
        }
        
        // Initialize A and B with random values
        srand(time(NULL));
        for (int i = 0; i < n * n; i++) {
            A[i] = (double)rand() / RAND_MAX;
            B[i] = (double)rand() / RAND_MAX;
        }
    }
    
    // Scatter blocks of A and B to all processes
    start_time = MPI_Wtime();
    
    // Used for scattering blocks
    double *temp_A_blocks = NULL, *temp_B_blocks = NULL;
    if (rank == 0) {
        temp_A_blocks = (double*)malloc(size * block_size * block_size * sizeof(double));
        temp_B_blocks = (double*)malloc(size * block_size * block_size * sizeof(double));
        
        // Arrange A and B in blocks for scattering
        for (int p = 0; p < size; p++) {
            int p_coords[2];
            MPI_Cart_coords(cart_comm, p, ndims, p_coords);
            int p_row = p_coords[0];
            int p_col = p_coords[1];
            
            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j < block_size; j++) {
                    // Global coordinates
                    int global_i = p_row * block_size + i;
                    int global_j = p_col * block_size + j;
                    
                    // Linear index in block
                    int block_idx = i * block_size + j;
                    
                    // Linear index in global array
                    int global_idx = global_i * n + global_j;
                    
                    // Linear index in temp arrays
                    int temp_idx = p * block_size * block_size + block_idx;
                    
                    temp_A_blocks[temp_idx] = A[global_idx];
                    temp_B_blocks[temp_idx] = B[global_idx];
                }
            }
        }
    }
    
    // Scatter blocks to all processes
    MPI_Scatter(temp_A_blocks, block_size * block_size, MPI_DOUBLE,
                local_A, block_size * block_size, MPI_DOUBLE,
                0, cart_comm);
    MPI_Scatter(temp_B_blocks, block_size * block_size, MPI_DOUBLE,
                local_B, block_size * block_size, MPI_DOUBLE,
                0, cart_comm);
    
    // Free temporary arrays
    if (rank == 0) {
        free(temp_A_blocks);
        free(temp_B_blocks);
    }
    end_time = MPI_Wtime();
    comm_time += end_time - start_time;
    
    // Initial alignment of blocks for Cannon's algorithm
    start_time = MPI_Wtime();
    
    // Shift local_A left by coords[0] positions (along row)
    MPI_Cart_shift(cart_comm, 1, -coords[0], &row_source, &row_dest);
    MPI_Sendrecv(local_A, block_size * block_size, MPI_DOUBLE, row_dest, 0,
                 temp_A, block_size * block_size, MPI_DOUBLE, row_source, 0,
                 cart_comm, MPI_STATUS_IGNORE);
    
    // Swap local_A and temp_A
    double *swap = local_A;
    local_A = temp_A;
    temp_A = swap;
    
    // Shift local_B up by coords[1] positions (along column)
    MPI_Cart_shift(cart_comm, 0, -coords[1], &col_source, &col_dest);
    MPI_Sendrecv(local_B, block_size * block_size, MPI_DOUBLE, col_dest, 0,
                 temp_B, block_size * block_size, MPI_DOUBLE, col_source, 0,
                 cart_comm, MPI_STATUS_IGNORE);
    
    // Swap local_B and temp_B
    swap = local_B;
    local_B = temp_B;
    temp_B = swap;
    end_time = MPI_Wtime();
    comm_time += end_time - start_time;
    
    // Initialize local_C to zeros
    for (int i = 0; i < block_size * block_size; i++) {
        local_C[i] = 0.0;
    }
    
    // Main computation loop for Cannon's algorithm
    MPI_Cart_shift(cart_comm, 1, -1, &row_source, &row_dest);
    MPI_Cart_shift(cart_comm, 0, -1, &col_source, &col_dest);
    
    for (int k = 0; k < grid_size; k++) {
        // Perform local matrix multiplication
        start_time = MPI_Wtime();
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                for (int p = 0; p < block_size; p++) {
                    local_C[i * block_size + j] += 
                        local_A[i * block_size + p] * local_B[p * block_size + j];
                }
            }
        }
        end_time = MPI_Wtime();
        comp_time += end_time - start_time;
        
        if (k < grid_size - 1) {  // Skip communication in last iteration
            start_time = MPI_Wtime();
            // Shift A left by 1
            MPI_Sendrecv(local_A, block_size * block_size, MPI_DOUBLE, row_dest, 0,
                         temp_A, block_size * block_size, MPI_DOUBLE, row_source, 0,
                         cart_comm, MPI_STATUS_IGNORE);
            
            // Shift B up by 1
            MPI_Sendrecv(local_B, block_size * block_size, MPI_DOUBLE, col_dest, 0,
                         temp_B, block_size * block_size, MPI_DOUBLE, col_source, 0,
                         cart_comm, MPI_STATUS_IGNORE);
            
            // Update local blocks
            swap = local_A;
            local_A = temp_A;
            temp_A = swap;
            
            swap = local_B;
            local_B = temp_B;
            temp_B = swap;
            end_time = MPI_Wtime();
            comm_time += end_time - start_time;
        }
    }
    
    // Gather results to root process
    start_time = MPI_Wtime();
    double *temp_C_blocks = NULL;
    if (rank == 0) {
        temp_C_blocks = (double*)malloc(size * block_size * block_size * sizeof(double));
    }
    
    MPI_Gather(local_C, block_size * block_size, MPI_DOUBLE,
               temp_C_blocks, block_size * block_size, MPI_DOUBLE,
               0, cart_comm);
    
    // Reorganize blocks into final matrix C
    if (rank == 0) {
        for (int p = 0; p < size; p++) {
            int p_coords[2];
            MPI_Cart_coords(cart_comm, p, ndims, p_coords);
            int p_row = p_coords[0];
            int p_col = p_coords[1];
            
            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j < block_size; j++) {
                    // Global coordinates
                    int global_i = p_row * block_size + i;
                    int global_j = p_col * block_size + j;
                    
                    // Linear index in block
                    int block_idx = i * block_size + j;
                    
                    // Linear index in global array
                    int global_idx = global_i * n + global_j;
                    
                    // Linear index in temp array
                    int temp_idx = p * block_size * block_size + block_idx;
                    
                    C[global_idx] = temp_C_blocks[temp_idx];
                }
            }
        }
        free(temp_C_blocks);
    }
    end_time = MPI_Wtime();
    comm_time += end_time - start_time;
    
    // Reduce computation and communication times
    double global_comp_time, global_comm_time;
    MPI_Reduce(&comp_time, &global_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    MPI_Reduce(&comm_time, &global_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    
    // Print results
    if (rank == 0) {
        printf("Cannon's Matrix Multiplication (Cartesian): N=%d, NP=%d\n", n, size);
        printf("Computation time: %.6f s\n", global_comp_time);
        printf("Communication time: %.6f s\n", global_comm_time);
        printf("Total time: %.6f s\n", global_comp_time + global_comm_time);
        printf("Result summary (%.6f/%.6f)\n", global_comp_time, global_comm_time);
    }
    
    // Free memory
    if (rank == 0) {
        if (A) free(A);
        if (B) free(B);
        if (C) free(C);
    }
    free(local_A);
    free(local_B);
    free(local_C);
    free(temp_A);
    free(temp_B);
    
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}