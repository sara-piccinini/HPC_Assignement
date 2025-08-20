#!/bin/bash
#SBATCH -J HPC_assignment_ex1
#SBATCH --time=0:04:00
#SBATCH --partition=global
#SBATCH -n 4
#SBATCH --nodes=1
#SBATCH --mail-user=s347092@studenti.polito.it
#SBATCH --mail-type=all
mpicc systolic_array_500.c -lm -o systolic_array_500
mpirun ./systolic_array_500  matrix_A_500.csv matrix_B_500.csv matrix_C_500.csv timing_N1_4np_500.csv