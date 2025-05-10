#include <mpi.h>
...
// Misal, kita menggunakan communicator comm_cart dari MPI_Cart_create
int remain_dims[2] = {1, 0};   // Pertahankan dimensi 0 (baris), drop dimensi 1 (kolom)
MPI_Comm sub_comm;

MPI_Cart_sub(comm_cart, remain_dims, &sub_comm);

// sub_comm sekarang mengelompokkan proses berdasarkan baris saja
// Setiap proses dalam baris yang sama akan berada dalam sub-communicator yang sama

// Contoh: Mendapatkan rank dalam subgrid
int sub_rank;
MPI_Comm_rank(sub_comm, &sub_rank);
printf("Rank dalam subgrid: %d\n", sub_rank);
