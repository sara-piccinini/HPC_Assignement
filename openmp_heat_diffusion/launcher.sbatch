#!/bin/bash
#SBATCH --job-name=Heat_Diff
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s344808@studenti.polito.it
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --output=Heat_Diff.log

EXEC="Heat_diff"

THREADS='1 2 4 8 12 16 20 24 28 32 40 52 64 96 128'
CONFIG='a b'
ITER=10000

gcc -fopenmp Heat_diff.c -o $EXEC

rm ./data/time_conf_a
rm ./data/time_conf_b

for th in $THREADS
do
    for cf in $CONFIG
    do
        OMP_NUM_THREADS=$th ./$EXEC $cf $ITER
    done
done \