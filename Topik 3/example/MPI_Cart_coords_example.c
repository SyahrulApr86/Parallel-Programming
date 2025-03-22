#include <mpi.h>
...
int rank = 6;        // Contoh rank yang ingin diterjemahkan
int coords[2];

MPI_Cart_coords(comm_cart, rank, 2, coords);
printf("Rank %d berada pada koordinat: (%d, %d)\n", rank, coords[0], coords[1]);
