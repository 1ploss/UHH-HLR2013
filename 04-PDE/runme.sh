#!/bin/bash
test1() \
{
	PARAMS="2 512 2 1 7e-08 3 100"
	for i in {1..12}
	do
		echo "running ./partdiff-seq $i $PARAMS > partdiff-seq-numthreads-${i}.txt" >> t1.txt
		./partdiff-seq            $i $PARAMS > partdiff-seq-numthreads-${i}.txt
		echo "running ./partdiff-openmp-element $i $PARAMS  > partdiff-openmp-element-numthreads-${i}.txt" >> t1.txt
		./partdiff-openmp-element $i $PARAMS  > partdiff-openmp-element-numthreads-${i}.txt
		echo "running ./partdiff-openmp-spalten $i $PARAMS  > partdiff-openmp-spalten-numthreads-${i}.txt" >> t1.txt
		./partdiff-openmp-spalten $i $PARAMS  > partdiff-openmp-spalten-numthreads-${i}.txt
		echo "running ./partdiff-openmp-zeilen  $i $PARAMS  > partdiff-openmp-zeilen-numthreads-${i}.txt" >> t1.txt
		./partdiff-openmp-zeilen  $i $PARAMS  > partdiff-openmp-zeilen-numthreads-${i}.txt
	done
}

test2() \
{
	PARAMS1="12 2 " 
	PARAMS2=" 2 1 7e-08 3 100"
	# für X = 2^i 0<=i<=10
	for i in 1 2 4 8 16 32 64 128 512 1024
	do
		echo "running ./partdiff-seq $PARAMS1 $i $PARAMS2 > partdiff-seq-lines-${i}.txt" >> t2.txt
		./partdiff-seq            $PARAMS1 $i $PARAMS2 > partdiff-seq-lines-${i}.txt
		echo "running ./partdiff-openmp-element $PARAMS1 $i $PARAMS2  > partdiff-openmp-element-liness-${i}.txt" >> t2.txt
		./partdiff-openmp-element $PARAMS1 $i $PARAMS2  > partdiff-openmp-element-liness-${i}.txt
		echo "running ./partdiff-openmp-spalten PARAMS1 $i $PARAMS2  > partdiff-openmp-spalten-lines-${i}.txt" >> t2.txt
		./partdiff-openmp-spalten $PARAMS1 $i $PARAMS2  > partdiff-openmp-spalten-lines-${i}.txt
		echo "running ./partdiff-openmp-zeilen $PARAMS1 $i $PARAMS2  > partdiff-openmp-zeilen-lines-${i}.txt" >> t2.txt
		./partdiff-openmp-zeilen  $PARAMS1 $i $PARAMS2  > partdiff-openmp-zeilen-lines-${i}.txt
	done
}

test1 && echo -----------------------[test1 done]------------------------------------
test2 && echo -----------------------[test2 done]------------------------------------
