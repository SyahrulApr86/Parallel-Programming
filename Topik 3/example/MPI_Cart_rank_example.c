#include <mpi.h>
...
// Misal, kita sudah memiliki communicator comm_cart dari MPI_Cart_create
int coords[2] = {1, 2};  // Koordinat yang diinginkan
int rank;

MPI_Cart_rank(comm_cart, coords, &rank);
printf("Koordinat (1,2) berada pada rank: %d\n", rank);
