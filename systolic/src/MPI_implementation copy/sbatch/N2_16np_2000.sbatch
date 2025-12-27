#!/bin/bash
#SBATCH -J HPC_assignment_ex1
#SBATCH --time=0:04:00
#SBATCH --partition=global
#SBATCH -n 16
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --mail-user=s347092@studenti.polito.it
#SBATCH --mail-type=all
mpicc systolic_array_2000.c -lm -o systolic_array_2000
mpirun ./systolic_array_2000  matrix_A_2000.csv matrix_B_2000.csv matrix_C_2000.csv timing_N2_16np_2000.csv