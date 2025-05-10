#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // rank proses
    MPI_Comm_size(MPI_COMM_WORLD, &size); // total proses

    printf("Hello World from rank %d of %d by Syahrul Apriansyah - 2106078311\n", rank, size);

    MPI_Finalize();
    return 0;
}
