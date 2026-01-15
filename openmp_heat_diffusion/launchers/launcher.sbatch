#!/bin/bash
#SBATCH --job-name=Heat_Diff
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s344808@studenti.polito.it
#SBATCH --time=0:45:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=88   
#SBATCH --output=Heat_Diff.log

echo $SLURMD_NODENAME

EXEC="Heat_diff"
THREADS='1 2 4 8 12 16 20 24 28 32 36 40 44 48 52 64 72 80 88'
CONFIG='a b'
ITER='10000 20000 50000'

gcc -fopenmp Heat_diff.c -o ./$EXEC

for it in $ITER 
do
    for th in $THREADS
    do
        for cf in $CONFIG
        do
            OMP_NUM_THREADS=$th ./$EXEC $cf $it
        done
    done 
done
