#include <mpi.h>
...
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    // Misal: total 12 proses akan diatur dalam grid 2 dimensi
    int ndims = 2;
    int dims[2] = {3, 4};      // Grid 3 baris x 4 kolom
    int periods[2] = {0, 0};   // Grid tidak periodik pada kedua dimensi
    int reorder = 0;
    MPI_Comm comm_cart;

    // Membuat communicator dengan topologi Cartesian
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart);

    // Lanjutkan dengan operasi lain menggunakan comm_cart ...
    MPI_Finalize();
    return 0;
}

