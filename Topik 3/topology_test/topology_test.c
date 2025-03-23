#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>

// Function to test basic Cartesian topology creation
void test_cart_create(int scenario, int size) {
    int rank, dims[2], periods[2], reorder, cart_rank;
    MPI_Comm cart_comm;
    int coords[2];
    double start_time, end_time;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Different scenarios for topology configuration
    switch(scenario) {
        case 1: // Non-periodic grid
            dims[0] = dims[1] = (int)sqrt(size);
            periods[0] = periods[1] = 0;
            reorder = 0;
            break;
        case 2: // Periodic grid
            dims[0] = dims[1] = (int)sqrt(size);
            periods[0] = periods[1] = 1;
            reorder = 0;
            break;
        case 3: // Optimized mapping
            dims[0] = dims[1] = (int)sqrt(size);
            periods[0] = periods[1] = 0;
            reorder = 1;
            break;
        case 4: // Mixed periodicity
            dims[0] = dims[1] = (int)sqrt(size);
            periods[0] = 1; periods[1] = 0;
            reorder = 0;
            break;
        default:
            if (rank == 0) printf("Invalid scenario\n");
            return;
    }
    
    // Ensure we have perfect square processes
    if (dims[0] * dims[1] != size) {
        if (rank == 0) {
            printf("Number of processes must be a perfect square for this test\n");
        }
        return;
    }
    
    start_time = MPI_Wtime();
    
    // Create the Cartesian topology
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    
    // Get the new rank and coordinates
    MPI_Comm_rank(cart_comm, &cart_rank);
    MPI_Cart_coords(cart_comm, cart_rank, 2, coords);
    
    end_time = MPI_Wtime();
    
    // Print information for a few processes to avoid excessive output
    if (rank % (size/4 + 1) == 0) {
        printf("Process: %d, Cart Rank: %d, Coords: (%d,%d)\n", rank, cart_rank, coords[0], coords[1]);
    }
    
    if (rank == 0) {
        printf("Create Cartesian topology time: %f seconds\n", end_time - start_time);
        printf("Scenario %d: %s\n", scenario, 
               scenario == 1 ? "Non-periodic grid" : 
               scenario == 2 ? "Periodic grid" : 
               scenario == 3 ? "Optimized mapping" : "Mixed periodicity");
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_free(&cart_comm);
}

// Function to test neighbor communication in a Cartesian topology
void test_neighbor_comm(int dimension, int size) {
    int rank, dims[2], periods[2], reorder;
    MPI_Comm cart_comm;
    int coords[2], source, dest;
    double start_time, end_time, total_time;
    double send_val, recv_val;
    MPI_Status status;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Setup grid topology
    dims[0] = dims[1] = (int)sqrt(size);
    periods[0] = periods[1] = 1; // Use periodic boundaries for simplicity
    reorder = 0;
    
    // Ensure we have perfect square processes
    if (dims[0] * dims[1] != size) {
        if (rank == 0) {
            printf("Number of processes must be a perfect square for this test\n");
        }
        return;
    }
    
    // Create the Cartesian topology
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    
    // Get coordinates in the grid
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    
    // Determine neighbor ranks
    MPI_Cart_shift(cart_comm, dimension, 1, &source, &dest);
    
    MPI_Barrier(cart_comm);
    start_time = MPI_Wtime();
    
    // Perform communication - each process sends its rank and receives from neighbor
    send_val = (double)rank;
    
    // Perform multiple exchanges to get better timing
    for (int i = 0; i < 100; i++) {
        MPI_Sendrecv(&send_val, 1, MPI_DOUBLE, dest, 0,
                      &recv_val, 1, MPI_DOUBLE, source, 0,
                      cart_comm, &status);
    }
    
    MPI_Barrier(cart_comm);
    end_time = MPI_Wtime();
    
    // Collect timing information
    double local_time = (end_time - start_time) / 100.0;
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    
    if (rank == 0) {
        printf("Dimension %d neighbor communication time: %f seconds\n", dimension, total_time);
    }
    
    // Print communication pattern for a few processes
    if (rank % (size/4 + 1) == 0) {
        printf("Process %d at (%d,%d) sends to %d, receives from %d\n", 
               rank, coords[0], coords[1], dest, source);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_free(&cart_comm);
}

// Function to test partitioning a grid using MPI_Cart_sub
void test_cart_sub(int size) {
    int rank, dims[3], periods[3], reorder;
    int remain_dims[3], sub_rank;
    MPI_Comm cart_comm, sub_comm;
    int coords[3];
    double start_time, end_time;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Try to set up a 3D grid
    if (size < 8) {
        if (rank == 0) {
            printf("Need at least 8 processes for 3D grid test\n");
        }
        return;
    }
    
    // Find reasonable dimensions for the 3D grid
    dims[0] = dims[1] = dims[2] = 0;
    MPI_Dims_create(size, 3, dims);
    
    periods[0] = periods[1] = periods[2] = 0;
    reorder = 0;
    
    start_time = MPI_Wtime();
    
    // Create the 3D Cartesian topology
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, reorder, &cart_comm);
    
    if (cart_comm == MPI_COMM_NULL) {
        if (rank == 0) {
            printf("Failed to create 3D Cartesian communicator\n");
        }
        return;
    }
    
    // Get coordinates in the 3D grid
    MPI_Cart_coords(cart_comm, rank, 3, coords);
    
    // Create 2D sub-grid by keeping the first and third dimensions
    remain_dims[0] = 1;
    remain_dims[1] = 0;
    remain_dims[2] = 1;
    
    MPI_Cart_sub(cart_comm, remain_dims, &sub_comm);
    
    if (sub_comm != MPI_COMM_NULL) {
        MPI_Comm_rank(sub_comm, &sub_rank);
    } else {
        sub_rank = -1;
    }
    
    end_time = MPI_Wtime();
    
    // Print information for some processes
    if (rank % (size/4 + 1) == 0) {
        printf("Process %d: 3D coords (%d,%d,%d), sub_rank: %d\n", 
               rank, coords[0], coords[1], coords[2], sub_rank);
    }
    
    if (rank == 0) {
        printf("3D grid dimensions: %d x %d x %d\n", dims[0], dims[1], dims[2]);
        printf("Cart_sub time: %f seconds\n", end_time - start_time);
    }
    
    if (sub_comm != MPI_COMM_NULL) {
        MPI_Comm_free(&sub_comm);
    }
    MPI_Comm_free(&cart_comm);
}

// Function to test a 5-point stencil communication pattern using a Cartesian topology
void test_stencil_comm(int size) {
    int rank, dims[2], periods[2], reorder;
    MPI_Comm cart_comm;
    int coords[2];
    int neighbors[4]; // north, east, south, west
    MPI_Status status[8];
    MPI_Request request[8];
    double local_data = 0.0;
    double neighbor_data[4];
    double start_time, end_time, total_time;
    int iterations = 100;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Setup grid topology
    dims[0] = dims[1] = (int)sqrt(size);
    periods[0] = periods[1] = 0; // Non-periodic for stencil
    reorder = 0;
    
    // Ensure we have perfect square processes
    if (dims[0] * dims[1] != size) {
        if (rank == 0) {
            printf("Number of processes must be a perfect square for this test\n");
        }
        return;
    }
    
    // Create the Cartesian topology
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    
    // Get coordinates in the grid
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    
    // Find the ranks of neighboring processes in each direction
    int source, dest;
    
    // North neighbor (negative y-direction)
    MPI_Cart_shift(cart_comm, 0, -1, &source, &dest);
    neighbors[0] = dest;
    
    // East neighbor (positive x-direction)
    MPI_Cart_shift(cart_comm, 1, 1, &source, &dest);
    neighbors[1] = dest;
    
    // South neighbor (positive y-direction)
    MPI_Cart_shift(cart_comm, 0, 1, &source, &dest);
    neighbors[2] = dest;
    
    // West neighbor (negative x-direction)
    MPI_Cart_shift(cart_comm, 1, -1, &source, &dest);
    neighbors[3] = dest;
    
    // Initialize local data with the process rank
    local_data = (double)rank;
    
    MPI_Barrier(cart_comm);
    start_time = MPI_Wtime();
    
    // Perform stencil communication iterations
    for (int iter = 0; iter < iterations; iter++) {
        int req_count = 0;
        
        // Send data to all valid neighbors
        for (int i = 0; i < 4; i++) {
            if (neighbors[i] != MPI_PROC_NULL) {
                MPI_Isend(&local_data, 1, MPI_DOUBLE, neighbors[i], 
                          0, cart_comm, &request[req_count++]);
            }
        }
        
        // Receive data from all valid neighbors
        for (int i = 0; i < 4; i++) {
            if (neighbors[i] != MPI_PROC_NULL) {
                MPI_Irecv(&neighbor_data[i], 1, MPI_DOUBLE, neighbors[i], 
                          0, cart_comm, &request[req_count++]);
            } else {
                neighbor_data[i] = 0.0;
            }
        }
        
        // Wait for all communications to complete
        MPI_Waitall(req_count, request, status);
        
        // Update local data (simple average of neighbors and self)
        if (req_count > 0) {
            double sum = local_data;
            int count = 1;
            
            for (int i = 0; i < 4; i++) {
                if (neighbors[i] != MPI_PROC_NULL) {
                    sum += neighbor_data[i];
                    count++;
                }
            }
            
            local_data = sum / count;
        }
    }
    
    MPI_Barrier(cart_comm);
    end_time = MPI_Wtime();
    
    // Collect timing information
    double local_time = end_time - start_time;
    MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    
    if (rank == 0) {
        printf("5-point stencil communication time (%d iterations): %f seconds\n", 
               iterations, total_time);
        printf("Time per iteration: %f seconds\n", total_time / iterations);
    }
    
    // Print some results for verification
    if (rank % (size/4 + 1) == 0) {
        printf("Process %d final value: %f\n", rank, local_data);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_free(&cart_comm);
}

int main(int argc, char *argv[]) {
    int rank, size;
    int test_case = 0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: %s <test_case>\n", argv[0]);
            printf("1: Basic Cartesian topology creation (different scenarios)\n");
            printf("2: Neighbor communication in different dimensions\n");
            printf("3: Cartesian grid partitioning with MPI_Cart_sub\n");
            printf("4: 5-point stencil communication pattern\n");
            printf("5: Run all tests\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    test_case = atoi(argv[1]);
    
    if (rank == 0) {
        printf("===== Process Topology Test =====\n");
        printf("Number of processes: %d\n", size);
        printf("Test case: %d\n", test_case);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    switch(test_case) {
        case 1:
            if (rank == 0) printf("\n----- Testing Cartesian Topology Creation -----\n");
            for (int scenario = 1; scenario <= 4; scenario++) {
                test_cart_create(scenario, size);
                MPI_Barrier(MPI_COMM_WORLD);
            }
            break;
        case 2:
            if (rank == 0) printf("\n----- Testing Neighbor Communication -----\n");
            for (int dim = 0; dim < 2; dim++) {
                test_neighbor_comm(dim, size);
                MPI_Barrier(MPI_COMM_WORLD);
            }
            break;
        case 3:
            if (rank == 0) printf("\n----- Testing Cart_sub -----\n");
            test_cart_sub(size);
            break;
        case 4:
            if (rank == 0) printf("\n----- Testing 5-point Stencil Communication -----\n");
            test_stencil_comm(size);
            break;
        case 5:
            if (rank == 0) printf("\n----- Running All Tests -----\n");
            
            if (rank == 0) printf("\n----- Testing Cartesian Topology Creation -----\n");
            for (int scenario = 1; scenario <= 4; scenario++) {
                test_cart_create(scenario, size);
                MPI_Barrier(MPI_COMM_WORLD);
            }
            
            if (rank == 0) printf("\n----- Testing Neighbor Communication -----\n");
            for (int dim = 0; dim < 2; dim++) {
                test_neighbor_comm(dim, size);
                MPI_Barrier(MPI_COMM_WORLD);
            }
            
            if (rank == 0) printf("\n----- Testing Cart_sub -----\n");
            test_cart_sub(size);
            MPI_Barrier(MPI_COMM_WORLD);
            
            if (rank == 0) printf("\n----- Testing 5-point Stencil Communication -----\n");
            test_stencil_comm(size);
            break;
        default:
            if (rank == 0) printf("Invalid test case\n");
    }
    
    MPI_Finalize();
    return 0;
}