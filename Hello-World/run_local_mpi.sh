#!/bin/bash

echo "Menjalankan program hello_mpi di lingkungan multicore..."

echo -e "\n>>> np = 2"
mpirun -np 2 ./hello_mpi

echo -e "\n>>> np = 4"
mpirun -np 4 ./hello_mpi

echo -e "\n>>> np = 8"
mpirun -np 8 ./hello_mpi
