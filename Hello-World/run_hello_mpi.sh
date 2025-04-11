#!/bin/bash
#SBATCH --job-name=hello-mpi-cluster
#SBATCH --partition=batch
#SBATCH --nodes=1-8
#SBATCH --output=hello_cluster_%j.out
#SBATCH --error=hello_cluster_%j.err
#SBATCH --time=01:00:00

# Compile program (pastikan hello_mpi.c ada di folder ini)
mpicc -o hello_mpi hello_mpi.c

# Only test these process counts
PROCS=(2 4 8)

# Directory for result logs
RESULTS_DIR="hello_mpi_cluster_results"
mkdir -p "$RESULTS_DIR"

# Run hello_mpi with selected np values
for np in "${PROCS[@]}"; do
    echo "====================================================="
    echo "Running hello_mpi with np=$np"
    echo "====================================================="

    OUT_FILE="$RESULTS_DIR/hello_mpi_np${np}.out"

    mpirun --mca btl_tcp_if_exclude docker0,lo -np "$np" ./hello_mpi > "$OUT_FILE" 2>&1

    echo "Output saved to $OUT_FILE"
    echo ""
done

echo "Semua pengujian selesai. Hasil ada di folder $RESULTS_DIR"
