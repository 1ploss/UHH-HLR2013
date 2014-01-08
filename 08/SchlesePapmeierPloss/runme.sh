#!/bin/bash
set -e
EXE=./partdiff-par
#EXE=./partdiff-omp
INTERLINES=100
#METHOD=1
FUNC=2
COND=2
PRES=100
NUM_THREADS=2
rm -rf $EXE*.txt
for METHOD in {1..2}
do
    for i in {1..12}
    do
    	CMDLINE="mpirun -n $i $EXE $NUM_THREADS $METHOD $INTERLINES  $FUNC $COND $PRES"
	echo doing $CMDLINE
	echo $CMDLINE > $EXE-tasks$i-$NUM_THREADS-$METHOD-$INTERLINES-$FUNC-$COND-$PRES.txt 
	$CMDLINE
    done
done
