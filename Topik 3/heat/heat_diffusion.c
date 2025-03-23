#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

// Program ini menunjukkan penggunaan MPI Process Topologies
// untuk simulasi penyebaran panas 2D sederhana

#define GRID_SIZE 240     // Ukuran grid (harus kelipatan dari sqrt(p) untuk semua p)
#define MAX_ITER 100      // Jumlah iterasi maksimum
#define HEAT_CENTER 100.0 // Nilai panas di pusat grid
#define OUTPUT_ITER 20    // Interval output

int main(int argc, char **argv) {
    int rank, size;
    int dims[2] = {0, 0};        // Dimensi proses (0,0 berarti MPI akan memutuskan)
    int periods[2] = {0, 0};     // Tidak periodik
    int coords[2];               // Koordinat proses saat ini
    int reorder = 1;             // Izinkan MPI untuk mengubah susunan rank
    MPI_Comm cart_comm;          // Communicator untuk topologi Cartesian
    
    int grid_size = GRID_SIZE;   // Ukuran default
    int max_iter = MAX_ITER;     // Iterasi default
    
    double *local_grid, *new_local_grid;
    int local_rows, local_cols;
    int north, south, east, west; // Rank tetangga
    
    // Timing variables
    double comp_time = 0.0, comm_time = 0.0;
    double start_time, end_time;
    
    // Inisialisasi MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Parse argumen baris perintah (opsional)
    if (argc > 1) grid_size = atoi(argv[1]);
    if (argc > 2) max_iter = atoi(argv[2]);
    
    // Membuat grid proses 2D
    MPI_Dims_create(size, 2, dims);
    
    if (rank == 0) {
        printf("Heat Diffusion 2D with MPI Process Topologies\n");
        printf("Grid size: %d x %d\n", grid_size, grid_size);
        printf("Process grid: %d x %d (%d processes)\n", dims[0], dims[1], size);
        printf("Max iterations: %d\n", max_iter);
    }
    
    // Membuat topologi proses Cartesian
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);
    
    // Menemukan rank proses tetangga
    MPI_Cart_shift(cart_comm, 0, 1, &north, &south);
    MPI_Cart_shift(cart_comm, 1, 1, &west, &east);
    
    // Menghitung ukuran grid lokal (+ 2 untuk ghost cells)
    local_rows = grid_size / dims[0];
    local_cols = grid_size / dims[1];
    
    // Alokasi memori untuk grid lokal
    local_grid = (double *)malloc((local_rows + 2) * (local_cols + 2) * sizeof(double));
    new_local_grid = (double *)malloc((local_rows + 2) * (local_cols + 2) * sizeof(double));
    
    if (!local_grid || !new_local_grid) {
        printf("Error: Alokasi memori gagal pada rank %d\n", rank);
        MPI_Abort(cart_comm, 1);
        return 1;
    }
    
    // Inisialisasi grid (0 di mana-mana kecuali pusat grid global)
    for (int i = 0; i <= local_rows + 1; i++) {
        for (int j = 0; j <= local_cols + 1; j++) {
            local_grid[i * (local_cols + 2) + j] = 0.0;
            new_local_grid[i * (local_cols + 2) + j] = 0.0;
        }
    }
    
    // Set sumber panas di pusat grid global
    int global_center_row = grid_size / 2;
    int global_center_col = grid_size / 2;
    
    // Cek apakah pusat grid global berada dalam grid lokal saat ini
    int local_start_row = coords[0] * local_rows;
    int local_start_col = coords[1] * local_cols;
    
    if (global_center_row >= local_start_row && global_center_row < local_start_row + local_rows &&
        global_center_col >= local_start_col && global_center_col < local_start_col + local_cols) {
        int local_center_row = global_center_row - local_start_row + 1; // +1 for ghost cells
        int local_center_col = global_center_col - local_start_col + 1;
        local_grid[local_center_row * (local_cols + 2) + local_center_col] = HEAT_CENTER;
    }
    
    // Sinkronisasi sebelum memulai timer
    MPI_Barrier(cart_comm);
    double total_start_time = MPI_Wtime();
    
    // Loop simulasi utama
    for (int iter = 0; iter < max_iter; iter++) {
        // Pertukaran ghost cells dengan tetangga
        start_time = MPI_Wtime();
        
        MPI_Status status;
        
        // Pertukaran dengan tetangga utara
        if (north >= 0) {
            MPI_Sendrecv(&local_grid[1 * (local_cols + 2) + 1], local_cols, MPI_DOUBLE, north, 0,
                         &local_grid[0 * (local_cols + 2) + 1], local_cols, MPI_DOUBLE, north, 0,
                         cart_comm, &status);
        }
        
        // Pertukaran dengan tetangga selatan
        if (south >= 0) {
            MPI_Sendrecv(&local_grid[local_rows * (local_cols + 2) + 1], local_cols, MPI_DOUBLE, south, 0,
                         &local_grid[(local_rows + 1) * (local_cols + 2) + 1], local_cols, MPI_DOUBLE, south, 0,
                         cart_comm, &status);
        }
        
        // Pertukaran dengan tetangga barat
        if (west >= 0) {
            // Membuat tipe MPI untuk kolom
            MPI_Datatype column_type;
            MPI_Type_vector(local_rows, 1, local_cols + 2, MPI_DOUBLE, &column_type);
            MPI_Type_commit(&column_type);
            
            MPI_Sendrecv(&local_grid[1 * (local_cols + 2) + 1], 1, column_type, west, 0,
                         &local_grid[1 * (local_cols + 2) + 0], 1, column_type, west, 0,
                         cart_comm, &status);
                         
            MPI_Type_free(&column_type);
        }
        
        // Pertukaran dengan tetangga timur
        if (east >= 0) {
            // Membuat tipe MPI untuk kolom
            MPI_Datatype column_type;
            MPI_Type_vector(local_rows, 1, local_cols + 2, MPI_DOUBLE, &column_type);
            MPI_Type_commit(&column_type);
            
            MPI_Sendrecv(&local_grid[1 * (local_cols + 2) + local_cols], 1, column_type, east, 0,
                         &local_grid[1 * (local_cols + 2) + (local_cols + 1)], 1, column_type, east, 0,
                         cart_comm, &status);
                         
            MPI_Type_free(&column_type);
        }
        
        end_time = MPI_Wtime();
        comm_time += end_time - start_time;
        
        // Perhitungan penyebaran panas (stencil 5-titik)
        start_time = MPI_Wtime();
        
        for (int i = 1; i <= local_rows; i++) {
            for (int j = 1; j <= local_cols; j++) {
                // Pusat grid tetap panas konstan
                if (coords[0] * local_rows + i - 1 == grid_size / 2 && 
                    coords[1] * local_cols + j - 1 == grid_size / 2) {
                    new_local_grid[i * (local_cols + 2) + j] = HEAT_CENTER;
                    continue;
                }
                
                // Formula penyebaran panas: rata-rata dari 4 tetangga
                new_local_grid[i * (local_cols + 2) + j] = 0.25 * (
                    local_grid[(i - 1) * (local_cols + 2) + j] + // Utara
                    local_grid[(i + 1) * (local_cols + 2) + j] + // Selatan
                    local_grid[i * (local_cols + 2) + (j - 1)] + // Barat
                    local_grid[i * (local_cols + 2) + (j + 1)]   // Timur
                );
            }
        }
        
        end_time = MPI_Wtime();
        comp_time += end_time - start_time;
        
        // Tukar grid lama dengan grid baru
        double *temp = local_grid;
        local_grid = new_local_grid;
        new_local_grid = temp;
        
        // Output setiap OUTPUT_ITER iterasi
        if (iter % OUTPUT_ITER == 0 && rank == 0) {
            printf("Iteration %d completed\n", iter);
        }
    }
    
    // Waktu total
    double total_end_time = MPI_Wtime();
    double total_time = total_end_time - total_start_time;
    
    // Reduksi waktu untuk mendapatkan nilai maksimum di antara semua proses
    double global_comp_time, global_comm_time, global_total_time;
    MPI_Reduce(&comp_time, &global_comp_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    MPI_Reduce(&comm_time, &global_comm_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    MPI_Reduce(&total_time, &global_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, cart_comm);
    
    // Output hasil timing
    if (rank == 0) {
        printf("\n=== Performance Results ===\n");
        printf("Grid size: %d x %d, Process grid: %d x %d\n", grid_size, grid_size, dims[0], dims[1]);
        printf("Computation time: %.6f s\n", global_comp_time);
        printf("Communication time: %.6f s\n", global_comm_time);
        printf("Total time: %.6f s\n", global_total_time);
        printf("Comm/Comp ratio: %.2f\n", global_comm_time / global_comp_time);
        printf("Result summary (%.6f/%.6f)\n", global_comp_time, global_comm_time);
    }
    
    // Kumpulkan hasil akhir untuk validasi/output (opsional)
    if (rank == 0) {
        printf("\nSimulation completed successfully.\n");
    }
    
    // Pembersihan
    free(local_grid);
    free(new_local_grid);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    
    return 0;
}