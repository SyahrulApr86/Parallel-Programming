#include <mpi.h>
...
// Misal, kita menggunakan communicator comm_cart dari MPI_Cart_create
int source, dest;
int dir = 0;         // Misalnya, kita ingin mencari tetangga pada dimensi 0 (baris)
int displ = 1;       // Jarak perpindahan 1 (tetangga berikutnya di arah baris)

MPI_Cart_shift(comm_cart, dir, displ, &source, &dest);

printf("Tetangga (dimensi %d, displacement %d): source = %d, dest = %d\n", dir, displ, source, dest);
