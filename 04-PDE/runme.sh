#!/bin/bash
#PRECISION=29e-08
PRECISION=100
test1() \
{
	PARAMS="2 512 2 2 "${PRECISION}" 3 100"
	echo "running ./partdiff-seq 1 $PARAMS > partdiff-seq-numthreads-1.txt"
	echo "running ./partdiff-seq 1 $PARAMS > partdiff-seq-numthreads-1.txt" >> t1.txt
	./partdiff-seq  1 $PARAMS > partdiff-seq-numthreads-1.txt
	for i in {1..12}
	do
		echo "running ./partdiff-openmp-element $i $PARAMS  > partdiff-openmp-element-numthreads-${i}.txt"
		echo "running ./partdiff-openmp-element $i $PARAMS  > partdiff-openmp-element-numthreads-${i}.txt" >> t1.txt
		./partdiff-openmp-element $i $PARAMS  > partdiff-openmp-element-numthreads-${i}.txt
		echo "running ./partdiff-openmp-spalten $i $PARAMS  > partdiff-openmp-spalten-numthreads-${i}.txt"
		echo "running ./partdiff-openmp-spalten $i $PARAMS  > partdiff-openmp-spalten-numthreads-${i}.txt" >> t1.txt
		./partdiff-openmp-spalten $i $PARAMS  > partdiff-openmp-spalten-numthreads-${i}.txt
		echo "running ./partdiff-openmp-zeilen  $i $PARAMS  > partdiff-openmp-zeilen-numthreads-${i}.txt"
		echo "running ./partdiff-openmp-zeilen  $i $PARAMS  > partdiff-openmp-zeilen-numthreads-${i}.txt" >> t1.txt
		./partdiff-openmp-zeilen  $i $PARAMS  > partdiff-openmp-zeilen-numthreads-${i}.txt
	done
}

test_openmp() \
{
	MOD=$1
	INTERL=$2
	SCHED=$3
	CHUNK_SIZE=$4
	#PREFIX="valgrind --leak-check=full"
	PREFIX=
	PARAMS="12 2 512 2 2 "${PRECISION}" "${SCHED}" "${CHUNK_SIZE}
	echo "running ./partdiff-openmp-element $PARAMS  > partdiff-openmp-element-${MOD}-${SCHED}-${CHUNK_SIZE}.txt"
	echo "running ./partdiff-openmp-element $PARAMS  > partdiff-openmp-element-${MOD}-${SCHED}-${CHUNK_SIZE}.txt" >> t1.txt
	${PREFIX} ./partdiff-openmp-element $i $PARAMS  > partdiff-openmp-element-${MOD}-${SCHED}-${CHUNK_SIZE}.txt
	echo "running ./partdiff-openmp-spalten $PARAMS  > partdiff-openmp-spalten-${MOD}-${SCHED}-${CHUNK_SIZE}.txt"
	echo "running ./partdiff-openmp-spalten $PARAMS  > partdiff-openmp-spalten-${MOD}-${SCHED}-${CHUNK_SIZE}.txt" >> t1.txt
	${PREFIX} ./partdiff-openmp-spalten $i $PARAMS  > partdiff-openmp-spalten-${MOD}-${SCHED}-${CHUNK_SIZE}.txt
	echo "running ./partdiff-openmp-zeilen  $PARAMS  > partdiff-openmp-zeilen-${MOD}-${SCHED}-${CHUNK_SIZE}.txt"
	echo "running ./partdiff-openmp-zeilen  $PARAMS  > partdiff-openmp-zeilen-${MOD}-${SCHED}-${CHUNK_SIZE}.txt" >> t1.txt
	${PREFIX}./partdiff-openmp-zeilen  $i $PARAMS  > partdiff-openmp-zeilen-${MOD}-${SCHED}-${CHUNK_SIZE}.txt
}

test2() \
{
	PARAMS1="12 2 " 
	PARAMS2=" 2 2 "${PRECISION}" 3 100"
	# für X = 2^i 0<=i<=10
	for i in 1 2 4 8 16 32 64 128 512 1024
	do
		echo "running ./partdiff-seq $PARAMS1 $i $PARAMS2 > partdiff-seq-lines-${i}.txt"
		echo "running ./partdiff-seq $PARAMS1 $i $PARAMS2 > partdiff-seq-lines-${i}.txt" >> t2.txt
		./partdiff-seq            $PARAMS1 $i $PARAMS2 > partdiff-seq-lines-${i}.txt
		echo "running ./partdiff-openmp-element $PARAMS1 $i $PARAMS2  > partdiff-openmp-element-liness-${i}.txt"
		echo "running ./partdiff-openmp-element $PARAMS1 $i $PARAMS2  > partdiff-openmp-element-liness-${i}.txt" >> t2.txt
		./partdiff-openmp-element $PARAMS1 $i $PARAMS2  > partdiff-openmp-element-liness-${i}.txt
		echo "running ./partdiff-openmp-spalten $PARAMS1 $i $PARAMS2  > partdiff-openmp-spalten-lines-${i}.txt"
		echo "running ./partdiff-openmp-spalten $PARAMS1 $i $PARAMS2  > partdiff-openmp-spalten-lines-${i}.txt" >> t2.txt
		./partdiff-openmp-spalten $PARAMS1 $i $PARAMS2  > partdiff-openmp-spalten-lines-${i}.txt
		echo "running ./partdiff-openmp-zeilen $PARAMS1 $i $PARAMS2  > partdiff-openmp-zeilen-lines-${i}.txt"
		echo "running ./partdiff-openmp-zeilen $PARAMS1 $i $PARAMS2  > partdiff-openmp-zeilen-lines-${i}.txt" >> t2.txt
		./partdiff-openmp-zeilen  $PARAMS1 $i $PARAMS2  > partdiff-openmp-zeilen-lines-${i}.txt
	done
}

test3() \
{
	# {omp_sched_static, omp_sched_dynamic, omp_sched_guided, omp_sched_auto};	
	for CHUNK_SIZE in 1 2 4 16
	do	
		test_openmp "sched" 512 0 ${CHUNK_SIZE}
	done
	for CHUNK_SIZE in 1 4
	do	
		test_openmp "sched" 512 1 ${CHUNK_SIZE}
	done
	test_openmp "sched" 512 2 0		
}


rm partdiff-*.txt
#test1
echo -----------------------[test1 done]------------------------------------
#test2
echo -----------------------[test2 done]------------------------------------
test3
