#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// Simulation parameters
#define NX 400        // Domain size in x-direction
#define NY 100        // Domain size in y-direction
#define MAX_ITER 1000  // Maximum number of iterations
#define REYNOLDS 100.0 // Reynolds number
#define OUTPUT_FREQ 100 // Output frequency

// D2Q9 LBM parameters
#define Q 9           // Number of discrete velocities
#define RHO0 1.0      // Reference density
#define U0 0.1        // Inlet velocity

// D2Q9 velocity model parameters
const int cx[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int cy[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
const double w[Q] = {4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 
                     1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0};

// Function prototypes
void init_cavity_flow(double *f, double *rho, double *ux, double *uy, 
                     int local_nx, int local_ny, int coords[2]);
void stream_and_collide(double *f, double *f_new, double *rho, double *ux, double *uy, 
                       double omega, int local_nx, int local_ny);
void apply_boundary_conditions(double *f, int local_nx, int local_ny, int coords[2], int dims[2]);
void compute_macroscopic(double *f, double *rho, double *ux, double *uy, 
                        int local_nx, int local_ny);
void exchange_ghost_cells(double *f, int local_nx, int local_ny, 
                         MPI_Comm cart_comm, int north, int south, int east, int west);
void save_velocity_field(double *ux, double *uy, double *rho, int iter, 
                        int local_nx, int local_ny, int coords[2], int dims[2], 
                        MPI_Comm cart_comm);

int main(int argc, char *argv[]) {
    int rank, size, i, iter;
    int dims[2] = {0, 0};      // Let MPI decide the dimensions
    int periods[2] = {0, 0};   // Non-periodic boundaries
    int coords[2];             // Process coordinates
    int reorder = 1;           // Allow MPI to reorder ranks
    int north, south, east, west; // Neighbor ranks
    MPI_Comm cart_comm;        // Cartesian communicator
    
    double *f, *f_new;         // Distribution functions
    double *rho, *ux, *uy;     // Macroscopic variables
    double omega;              // Relaxation parameter
    double nu;                 // Kinematic viscosity
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Create a 2D Cartesian grid
    MPI_Dims_create(size, 2, dims);
    
    if (rank == 0) {
        printf("CFD Simulation using MPI Process Topologies\n");
        printf("Domain size: %d x %d\n", NX, NY);
        printf("Process grid: %d x %d\n", dims[0], dims[1]);
        printf("Reynolds number: %f\n", REYNOLDS);
    }
    
    // Create the Cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    
    // Determine local grid size
    int local_nx = NX / dims[1];
    int local_ny = NY / dims[0];
    
    // Adjust for remainder
    if (coords[1] == dims[1] - 1) local_nx += NX % dims[1];
    if (coords[0] == dims[0] - 1) local_ny += NY % dims[0];
    
    // Allocate memory (+2 for ghost cells in each direction)
    f = (double *)malloc(Q * (local_nx + 2) * (local_ny + 2) * sizeof(double));
    f_new = (double *)malloc(Q * (local_nx + 2) * (local_ny + 2) * sizeof(double));
    rho = (double *)malloc((local_nx + 2) * (local_ny + 2) * sizeof(double));
    ux = (double *)malloc((local_nx + 2) * (local_ny + 2) * sizeof(double));
    uy = (double *)malloc((local_nx + 2) * (local_ny + 2) * sizeof(double));
    
    // Find neighboring processes
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);
    
    // Calculate relaxation parameter
    nu = U0 * (NX/2) / REYNOLDS;  // Kinematic viscosity
    omega = 1.0 / (3.0 * nu + 0.5); // Relaxation parameter
    
    // Initialize the flow field
    init_cavity_flow(f, rho, ux, uy, local_nx + 2, local_ny + 2, coords);
    
    // Main simulation loop
    for (iter = 0; iter < MAX_ITER; iter++) {
        // Exchange ghost cells with neighboring processes
        exchange_ghost_cells(f, local_nx, local_ny, cart_comm, north, south, east, west);
        
        // Streaming and collision step
        stream_and_collide(f, f_new, rho, ux, uy, omega, local_nx + 2, local_ny + 2);
        
        // Swap the distribution functions
        double *temp = f;
        f = f_new;
        f_new = temp;
        
        // Apply boundary conditions
        apply_boundary_conditions(f, local_nx + 2, local_ny + 2, coords, dims);
        
        // Compute macroscopic variables
        compute_macroscopic(f, rho, ux, uy, local_nx + 2, local_ny + 2);
        
        // Output results periodically
        if (iter % OUTPUT_FREQ == 0) {
            if (rank == 0) printf("Iteration %d\n", iter);
            save_velocity_field(ux, uy, rho, iter, local_nx, local_ny, coords, dims, cart_comm);
        }
    }
    
    // Free memory
    free(f);
    free(f_new);
    free(rho);
    free(ux);
    free(uy);
    
    MPI_Finalize();
    return 0;
}

// Initialize the cavity flow
void init_cavity_flow(double *f, double *rho, double *ux, double *uy, 
                     int local_nx, int local_ny, int coords[2]) {
    int x, y, k;
    double feq;
    
    // Initialize macroscopic variables
    for (y = 0; y < local_ny; y++) {
        for (x = 0; x < local_nx; x++) {
            rho[y * local_nx + x] = RHO0;
            ux[y * local_nx + x] = 0.0;
            uy[y * local_nx + x] = 0.0;
            
            // Set top boundary (lid) velocity if this process includes the top boundary
            if (coords[0] == 0 && y == 1) {
                ux[y * local_nx + x] = U0;
            }
        }
    }
    
    // Initialize distribution functions with equilibrium values
    for (y = 0; y < local_ny; y++) {
        for (x = 0; x < local_nx; x++) {
            double rho_local = rho[y * local_nx + x];
            double ux_local = ux[y * local_nx + x];
            double uy_local = uy[y * local_nx + x];
            
            for (k = 0; k < Q; k++) {
                double cu = cx[k] * ux_local + cy[k] * uy_local;
                double usqr = ux_local * ux_local + uy_local * uy_local;
                
                feq = w[k] * rho_local * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usqr);
                f[k * local_nx * local_ny + y * local_nx + x] = feq;
            }
        }
    }
}

// Streaming and collision step
void stream_and_collide(double *f, double *f_new, double *rho, double *ux, double *uy, 
                       double omega, int local_nx, int local_ny) {
    int x, y, k;
    double feq, f_in;
    
    // Loop over the interior cells (excluding ghost cells)
    for (y = 1; y < local_ny - 1; y++) {
        for (x = 1; x < local_nx - 1; x++) {
            double rho_local = rho[y * local_nx + x];
            double ux_local = ux[y * local_nx + x];
            double uy_local = uy[y * local_nx + x];
            
            for (k = 0; k < Q; k++) {
                // Streaming from neighboring cells
                int xn = x - cx[k];
                int yn = y - cy[k];
                
                f_in = f[k * local_nx * local_ny + yn * local_nx + xn];
                
                // Equilibrium distribution
                double cu = cx[k] * ux_local + cy[k] * uy_local;
                double usqr = ux_local * ux_local + uy_local * uy_local;
                
                feq = w[k] * rho_local * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usqr);
                
                // BGK collision operator
                f_new[k * local_nx * local_ny + y * local_nx + x] = f_in * (1.0 - omega) + feq * omega;
            }
        }
    }
}

// Apply boundary conditions
void apply_boundary_conditions(double *f, int local_nx, int local_ny, int coords[2], int dims[2]) {
    int x, y, k, opp;
    
    // Opposite directions for bounce-back
    int opposite[Q] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    
    // Bottom boundary (no-slip) - if this process includes it
    if (coords[0] == dims[0] - 1) {
        y = local_ny - 2;
        for (x = 1; x < local_nx - 1; x++) {
            for (k = 0; k < Q; k++) {
                if (cy[k] > 0) {
                    opp = opposite[k];
                    f[k * local_nx * local_ny + y * local_nx + x] = 
                        f[opp * local_nx * local_ny + y * local_nx + x];
                }
            }
        }
    }
    
    // Top boundary (moving lid) - if this process includes it
    if (coords[0] == 0) {
        y = 1;
        for (x = 1; x < local_nx - 1; x++) {
            for (k = 0; k < Q; k++) {
                if (cy[k] < 0) {
                    opp = opposite[k];
                    f[k * local_nx * local_ny + y * local_nx + x] = 
                        f[opp * local_nx * local_ny + y * local_nx + x] + 
                        6.0 * w[k] * RHO0 * cx[k] * U0;
                }
            }
        }
    }
    
    // Left boundary (no-slip) - if this process includes it
    if (coords[1] == 0) {
        x = 1;
        for (y = 1; y < local_ny - 1; y++) {
            for (k = 0; k < Q; k++) {
                if (cx[k] > 0) {
                    opp = opposite[k];
                    f[k * local_nx * local_ny + y * local_nx + x] = 
                        f[opp * local_nx * local_ny + y * local_nx + x];
                }
            }
        }
    }
    
    // Right boundary (no-slip) - if this process includes it
    if (coords[1] == dims[1] - 1) {
        x = local_nx - 2;
        for (y = 1; y < local_ny - 1; y++) {
            for (k = 0; k < Q; k++) {
                if (cx[k] < 0) {
                    opp = opposite[k];
                    f[k * local_nx * local_ny + y * local_nx + x] = 
                        f[opp * local_nx * local_ny + y * local_nx + x];
                }
            }
        }
    }
}

// Compute macroscopic variables (density and velocity)
void compute_macroscopic(double *f, double *rho, double *ux, double *uy, int local_nx, int local_ny) {
    int x, y, k;
    
    // Loop over interior cells
    for (y = 1; y < local_ny - 1; y++) {
        for (x = 1; x < local_nx - 1; x++) {
            double rho_local = 0.0;
            double momentum_x = 0.0;
            double momentum_y = 0.0;
            
            // Sum over all directions
            for (k = 0; k < Q; k++) {
                double f_local = f[k * local_nx * local_ny + y * local_nx + x];
                rho_local += f_local;
                momentum_x += cx[k] * f_local;
                momentum_y += cy[k] * f_local;
            }
            
            rho[y * local_nx + x] = rho_local;
            ux[y * local_nx + x] = momentum_x / rho_local;
            uy[y * local_nx + x] = momentum_y / rho_local;
        }
    }
}

// Exchange ghost cells with neighboring processes
void exchange_ghost_cells(double *f, int local_nx, int local_ny, 
                         MPI_Comm cart_comm, int north, int south, int east, int west) {
    int k;
    MPI_Status status;
    
    // For each direction in the LBM model
    for (k = 0; k < Q; k++) {
        double *f_k = &f[k * (local_nx + 2) * (local_ny + 2)];
        
        // Create MPI types for rows and columns
        MPI_Datatype row_type, col_type;
        
        MPI_Type_vector(1, local_nx, 1, MPI_DOUBLE, &row_type);
        MPI_Type_commit(&row_type);
        
        MPI_Type_vector(local_ny, 1, local_nx + 2, MPI_DOUBLE, &col_type);
        MPI_Type_commit(&col_type);
        
        // North-South exchange
        if (north >= 0) {
            MPI_Sendrecv(&f_k[1 * (local_nx + 2) + 1], 1, row_type, north, 0,
                         &f_k[0 * (local_nx + 2) + 1], 1, row_type, north, 0,
                         cart_comm, &status);
        }
        
        if (south >= 0) {
            MPI_Sendrecv(&f_k[(local_ny) * (local_nx + 2) + 1], 1, row_type, south, 0,
                         &f_k[(local_ny + 1) * (local_nx + 2) + 1], 1, row_type, south, 0,
                         cart_comm, &status);
        }
        
        // East-West exchange
        if (east >= 0) {
            MPI_Sendrecv(&f_k[1 * (local_nx + 2) + local_nx], local_ny, MPI_DOUBLE, east, 0,
                         &f_k[1 * (local_nx + 2) + (local_nx + 1)], local_ny, MPI_DOUBLE, east, 0,
                         cart_comm, &status);
        }
        
        if (west >= 0) {
            MPI_Sendrecv(&f_k[1 * (local_nx + 2) + 1], local_ny, MPI_DOUBLE, west, 0,
                         &f_k[1 * (local_nx + 2) + 0], local_ny, MPI_DOUBLE, west, 0,
                         cart_comm, &status);
        }
        
        MPI_Type_free(&row_type);
        MPI_Type_free(&col_type);
    }
}

// Save velocity field for visualization
void save_velocity_field(double *ux, double *uy, double *rho, int iter, 
                        int local_nx, int local_ny, int coords[2], int dims[2], 
                        MPI_Comm cart_comm) {
    int rank;
    MPI_Comm_rank(cart_comm, &rank);
    
    char filename[100];
    sprintf(filename, "cavity_flow_%d.dat", iter);
    
    // Prepare local data
    double *local_data = (double *)malloc(3 * local_nx * local_ny * sizeof(double));
    int idx = 0;
    
    for (int y = 1; y <= local_ny; y++) {
        for (int x = 1; x <= local_nx; x++) {
            local_data[idx++] = ux[(y) * (local_nx + 2) + (x)];
            local_data[idx++] = uy[(y) * (local_nx + 2) + (x)];
            local_data[idx++] = rho[(y) * (local_nx + 2) + (x)];
        }
    }
    
    // Gather all data to rank 0
    if (rank == 0) {
        // Write local data to file
        FILE *fp = fopen(filename, "w");
        
        // Header
        fprintf(fp, "X Y UX UY RHO\n");
        
        // Global coordinates for rank 0
        int global_y_offset = coords[0] * (NY / dims[0]);
        int global_x_offset = coords[1] * (NX / dims[1]);
        
        idx = 0;
        for (int y = 0; y < local_ny; y++) {
            for (int x = 0; x < local_nx; x++) {
                fprintf(fp, "%d %d %f %f %f\n", 
                        global_x_offset + x, global_y_offset + y,
                        local_data[idx], local_data[idx+1], local_data[idx+2]);
                idx += 3;
            }
        }
        
        // Receive and write data from other processes
        for (int p = 1; p < dims[0] * dims[1]; p++) {
            int p_coords[2];
            MPI_Cart_coords(cart_comm, p, 2, p_coords);
            
            int p_local_nx = NX / dims[1];
            int p_local_ny = NY / dims[0];
            
            // Adjust for remainder
            if (p_coords[1] == dims[1] - 1) p_local_nx += NX % dims[1];
            if (p_coords[0] == dims[0] - 1) p_local_ny += NY % dims[0];
            
            double *p_data = (double *)malloc(3 * p_local_nx * p_local_ny * sizeof(double));
            
            MPI_Recv(p_data, 3 * p_local_nx * p_local_ny, MPI_DOUBLE, p, 0, cart_comm, MPI_STATUS_IGNORE);
            
            // Global coordinates for process p
            global_y_offset = p_coords[0] * (NY / dims[0]);
            global_x_offset = p_coords[1] * (NX / dims[1]);
            
            idx = 0;
            for (int y = 0; y < p_local_ny; y++) {
                for (int x = 0; x < p_local_nx; x++) {
                    fprintf(fp, "%d %d %f %f %f\n", 
                            global_x_offset + x, global_y_offset + y,
                            p_data[idx], p_data[idx+1], p_data[idx+2]);
                    idx += 3;
                }
            }
            
            free(p_data);
        }
        
        fclose(fp);
    } else {
        // Send data to rank 0
        MPI_Send(local_data, 3 * local_nx * local_ny, MPI_DOUBLE, 0, 0, cart_comm);
    }
    
    free(local_data);
}