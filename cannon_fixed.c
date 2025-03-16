#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

// Cannon's algorithm for matrix multiplication with timing
// Run with: mpirun -np <num_processes> ./cannon <matrix_size>
// Note: num_processes must be a perfect square (e.g., 4, 9, 16)

int main(int argc, char *argv[]) {
    int rank, size, n;
    int dims[2], periods[2], coords[2];
    int grid_size, block_size;
    int up, down, left, right;
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A = NULL, *local_B = NULL, *local_C = NULL;
    MPI_Comm cart_comm;
    double comp_time = 0.0, comm_time = 0.0;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Check if number of processes is a perfect square
    grid_size = (int)sqrt(size);
    if (grid_size * grid_size != size) {
        if (rank == 0) {
            printf("Number of processes must be a perfect square (got %d)\n", size);
        }
        MPI_Finalize();
        return 1;
    }
    
    // Get matrix size from command line
    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: %s <matrix_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    n = atoi(argv[1]);
    
    // Check if matrix size is divisible by grid size
    if (n % grid_size != 0) {
        if (rank == 0) {
            printf("Matrix size (%d) must be divisible by grid size (%d)\n", n, grid_size);
        }
        MPI_Finalize();
        return 1;
    }
    
    // Set up 2D Cartesian topology
    dims[0] = dims[1] = grid_size;
    periods[0] = periods[1] = 1;  // Enable wraparound for shifts
    
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    
    // Get neighbors
    MPI_Cart_shift(cart_comm, 0, 1, &up, &down);
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);
    
    // Calculate block size
    block_size = n / grid_size;
    
    // Allocate memory for local blocks
    local_A = (double*)malloc(block_size * block_size * sizeof(double));
    local_B = (double*)malloc(block_size * block_size * sizeof(double));
    local_C = (double*)calloc(block_size * block_size, sizeof(double));
    
    if (!local_A || !local_B || !local_C) {
        printf("Process %d: Failed to allocate memory for local blocks\n", rank);
        MPI_Abort(cart_comm, 1);
        return 1;
    }
    
    // Process 0 initializes the full matrices
    if (rank == 0) {
        A = (double*)malloc(n * n * sizeof(double));
        B = (double*)malloc(n * n * sizeof(double));
        C = (double*)malloc(n * n * sizeof(double));
        
        if (!A || !B || !C) {
            printf("Process 0: Failed to allocate memory for matrices\n");
            MPI_Abort(cart_comm, 1);
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
    MPI_Barrier(cart_comm);
    
    // Distribute matrix blocks
    start_time = MPI_Wtime();
    if (rank == 0) {
        // Fill local blocks for process 0
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                local_A[i * block_size + j] = A[i * n + j];
                local_B[i * block_size + j] = B[i * n + j];
            }
        }
        
        // Send blocks to other processes
        for (int p = 1; p < size; p++) {
            int p_coords[2];
            MPI_Cart_coords(cart_comm, p, 2, p_coords);
            int row = p_coords[0];
            int col = p_coords[1];
            
            // Extract and send block A
            double *temp_A = (double*)malloc(block_size * block_size * sizeof(double));
            if (!temp_A) {
                printf("Process 0: Failed to allocate memory for temp block A\n");
                MPI_Abort(cart_comm, 1);
                return 1;
            }
            
            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j < block_size; j++) {
                    temp_A[i * block_size + j] = A[(row * block_size + i) * n + (col * block_size + j)];
                }
            }
            MPI_Send(temp_A, block_size * block_size, MPI_DOUBLE, p, 0, cart_comm);
            
            // Extract and send block B
            double *temp_B = (double*)malloc(block_size * block_size * sizeof(double));
            if (!temp_B) {
                printf("Process 0: Failed to allocate memory for temp block B\n");
                MPI_Abort(cart_comm, 1);
                return 1;
            }
            
            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j < block_size; j++) {
                    temp_B[i * block_size + j] = B[(row * block_size + i) * n + (col * block_size + j)];
                }
            }
            MPI_Send(temp_B, block_size * block_size, MPI_DOUBLE, p, 1, cart_comm);
            
            free(temp_A);
            free(temp_B);
        }
    } else {
        // Receive blocks
        MPI_Recv(local_A, block_size * block_size, MPI_DOUBLE, 0, 0, cart_comm, MPI_STATUS_IGNORE);
        MPI_Recv(local_B, block_size * block_size, MPI_DOUBLE, 0, 1, cart_comm, MPI_STATUS_IGNORE);
    }
    end_time = MPI_Wtime();
    comm_time += end_time - start_time;
    
    // Initial alignment of A and B for Cannon's algorithm
    start_time = MPI_Wtime();
    MPI_Status status;
    double *temp = (double*)malloc(block_size * block_size * sizeof(double));
    if (!temp) {
        printf("Process %d: Failed to allocate memory for temp block\n", rank);
        MPI_Abort(cart_comm, 1);
        return 1;
    }
    
    // Shift A left by coords[0] positions
    int source_col = (coords[1] + coords[0]) % grid_size;
    int dest_col = (coords[1] - coords[0] + grid_size) % grid_size;
    
    int source_A_rank, dest_A_rank;
    int source_coords[2] = {coords[0], source_col};
    int dest_coords[2] = {coords[0], dest_col};
    
    MPI_Cart_rank(cart_comm, source_coords, &source_A_rank);
    MPI_Cart_rank(cart_comm, dest_coords, &dest_A_rank);
    
    MPI_Sendrecv(local_A, block_size * block_size, MPI_DOUBLE, dest_A_rank, 2,
                 temp, block_size * block_size, MPI_DOUBLE, source_A_rank, 2,
                 cart_comm, &status);
    
    // Copy temp to local_A
    for (int i = 0; i < block_size * block_size; i++) {
        local_A[i] = temp[i];
    }
    
    // Shift B up by coords[1] positions
    int source_row = (coords[0] + coords[1]) % grid_size;
    int dest_row = (coords[0] - coords[1] + grid_size) % grid_size;
    
    int source_B_rank, dest_B_rank;
    int source_B_coords[2] = {source_row, coords[1]};
    int dest_B_coords[2] = {dest_row, coords[1]};
    
    MPI_Cart_rank(cart_comm, source_B_coords, &source_B_rank);
    MPI_Cart_rank(cart_comm, dest_B_coords, &dest_B_rank);
    
    MPI_Sendrecv(local_B, block_size * block_size, MPI_DOUBLE, dest_B_rank, 3,
                 temp, block_size * block_size, MPI_DOUBLE, source_B_rank, 3,
                 cart_comm, &status);
    
    // Copy temp to local_B
    for (int i = 0; i < block_size * block_size; i++) {
        local_B[i] = temp[i];
    }
    
    end_time = MPI_Wtime();
    comm_time += end_time - start_time;
    
    // Perform Cannon's algorithm
    for (int k = 0; k < grid_size; k++) {
        // Multiply local blocks
        start_time = MPI_Wtime();
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                for (int l = 0; l < block_size; l++) {
                    local_C[i * block_size + j] += local_A[i * block_size + l] * local_B[l * block_size + j];
                }
            }
        }
        end_time = MPI_Wtime();
        comp_time += end_time - start_time;
        
        if (k < grid_size - 1) {
            // Shift A left by one position
            start_time = MPI_Wtime();
            MPI_Sendrecv_replace(local_A, block_size * block_size, MPI_DOUBLE,
                                left, 4, right, 4, cart_comm, &status);
            
            // Shift B up by one position
            MPI_Sendrecv_replace(local_B, block_size * block_size, MPI_DOUBLE,
                                up, 5, down, 5, cart_comm, &status);
            end_time = MPI_Wtime();
            comm_time += end_time - start_time;
        }
    }
    
    // Gather results
    start_time = MPI_Wtime();
    if (rank == 0) {
        // Place local results in the result matrix
        for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
                C[i * n + j] = local_C[i * block_size + j];
            }
        }
        
        // Receive results from other processes
        for (int p = 1; p < size; p++) {
            int p_coords[2];
            MPI_Cart_coords(cart_comm, p, 2, p_coords);
            int row = p_coords[0];
            int col = p_coords[1];
            
            double *temp_C = (double*)malloc(block_size * block_size * sizeof(double));
            if (!temp_C) {
                printf("Process 0: Failed to allocate memory for temp block C\n");
                MPI_Abort(cart_comm, 1);
                return 1;
            }
            
            MPI_Recv(temp_C, block_size * block_size, MPI_DOUBLE, p, 6, cart_comm, MPI_STATUS_IGNORE);
            
            // Place received block in the result matrix
            for (int i = 0; i < block_size; i++) {
                for (int j = 0; j < block_size; j++) {
                    C[(row * block_size + i) * n + (col * block_size + j)] = temp_C[i * block_size + j];
                }
            }
            
            free(temp_C);
        }
    } else {
        // Send local results to process 0
        MPI_Send(local_C, block_size * block_size, MPI_DOUBLE, 0, 6, cart_comm);
    }
    end_time = MPI_Wtime();
    comm_time += end_time - start_time;
    
    // Reduce computation and communication times
    double global_comp_time, global_comm_time;
    MPI_Reduce(&comp_time, &global_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    MPI_Reduce(&comm_time, &global_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    
    // Print results
    if (rank == 0) {
        printf("Cannon's Algorithm Matrix Multiplication: N=%d, NP=%d\n", n, size);
        printf("Computation time: %.6f s\n", global_comp_time);
        printf("Communication time: %.6f s\n", global_comm_time);
        printf("Total time: %.6f s\n", global_comp_time + global_comm_time);
        printf("Result summary (%.6f/%.6f)\n", global_comp_time, global_comm_time);
    }
    
    // Free memory
    if (local_A) free(local_A);
    if (local_B) free(local_B);
    if (local_C) free(local_C);
    if (temp) free(temp);
    
    if (rank == 0) {
        if (A) free(A);
        if (B) free(B);
        if (C) free(C);
    }
    
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
