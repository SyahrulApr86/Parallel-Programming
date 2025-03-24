#!/bin/bash
#SBATCH -o ~/kelompok_senja/pr1/run-3.out
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --nodelist=node-02,node-06

mpirun --mca btl_tcp_if_exclude docker0,lo -np 4 ~/kelompok_senja/pr1/topik1.o

