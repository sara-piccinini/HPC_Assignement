EXEC="Heat_diff"
THREADS='1 2 4 8 10 12 14 16'
CONFIG='a b'
ITER=10000

gcc -fopenmp Heat_diff.c -o ./$EXEC

rm ./data/time_conf_a
rm ./data/time_conf_b
if [ $ITER -gt 99999 ]; then
    rm ./data/temp_conf_a
    rm ./data/temp_conf_b
fi

for th in $THREADS
do
    for cf in $CONFIG
    do
        OMP_NUM_THREADS=$th ./$EXEC $cf $ITER
    done
done 