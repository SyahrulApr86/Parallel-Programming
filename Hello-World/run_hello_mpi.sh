#!/bin/bash
#SBATCH -o ~/UTS-Syahrul-Apriansyah/run-hello.out      # File output hasil run
#SBATCH -p batch                                       # Nama partition
#SBATCH -N 2                                           # Jumlah node
#SBATCH --nodelist=node-02,node-06                     # Spesifik node yang digunakan

mpirun --mca btl_tcp_if_exclude docker0,lo -np 4 ~/UTS-Syahrul-Apriansyah/hello_mpi
