#include <mpi.h>
...
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int nodes = 12;          // Total jumlah proses
    int ndims = 2;           // Jumlah dimensi grid
    int dims[2] = {0, 0};      // Nilai 0 berarti MPI akan menentukan pembagian yang seimbang

    // Fungsi ini akan mengatur dims agar pembagian proses merata, misalnya dims menjadi {3,4}
    MPI_Dims_create(nodes, ndims, dims);

    // dims[0] dan dims[1] sekarang berisi nilai yang seimbang
    printf("Grid dimensions: %d x %d\n", dims[0], dims[1]);

    MPI_Finalize();
    return 0;
}
