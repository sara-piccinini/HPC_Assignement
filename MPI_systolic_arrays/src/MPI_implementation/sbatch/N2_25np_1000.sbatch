#!/bin/bash
#SBATCH -J HPC_assignment_ex1
#SBATCH --time=0:04:00
#SBATCH --partition=global
#SBATCH -n 25
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=13
#SBATCH --mail-user=s347092@studenti.polito.it
#SBATCH --mail-type=all
mpicc systolic_array_1000.c -lm -o systolic_array_1000
mpirun ./systolic_array_1000  matrix_A_1000.csv matrix_B_1000.csv matrix_C_1000.csv timing_N2_25np_1000.csv