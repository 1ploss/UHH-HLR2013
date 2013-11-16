#!/bin/bash
# This is a script we used to run tests.
# 1 - presigion, 2 - iterations
METHOD=1
PRECISION=7e-08
#PRECISION=100
INTERLINES=10
test1() \
{
	rm -rf ./partdiff-posix-*.txt
	
	REPORT_FILE="test1_state.txt"
	rm -rf ${REPORT_FILE}
	for NUM_THREADS in {1..12}
	do
		PARAMS="${NUM_THREADS} 2 ${INTERLINES} 2 ${METHOD} ${PRECISION}"
		RESULT_FILE=partdiff-posix-${NUM_THREADS}.txt
		echo "creating ${RESULT_FILE} and running ./partdiff-posix $PARAMS > ${RESULT_FILE}"
		./partdiff-posix $PARAMS > ${RESULT_FILE}
	done
}

test1
