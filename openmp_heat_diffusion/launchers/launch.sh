EXEC="Heat_diff"
THREADS='1 2 4 8 10 12 14 16 18 20 22 24 26 28 30 32 48 64'
CONFIG='a b'
ITER='10000 20000 50000'

gcc -fopenmp Heat_diff.c -o ./$EXEC

rm ./data/time_conf_a
rm ./data/time_conf_b
# if [ $ITER -gt 99999 ]; then
#     rm ./data/temp_conf_a
#     rm ./data/temp_conf_b
# fi

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