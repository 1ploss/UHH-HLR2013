CC=mpicc
CFLAGS=-std=c99 -Wall -Wextra -Wno-unknown-pragmas -ggdb -fno-strict-aliasing
LIBS=-lm

%-par: %-par.o gauss.o
	$(CC) $(CFLAGS) -ggdb $? $(LIBS) -o $@

%-omp.o: %-omp.c Makefile
	$(CC) $(CFLAGS) -fopenmp -c $< -o $@
	
%-omp: %-omp.o gauss-omp.o
	$(CC) $(CFLAGS)  -fopenmp -ggdb $? $(LIBS) -o $@

%.o : %.c common.h
	$(CC) $(CFLAGS) -c $< -o $@

all: partdiff-par partdiff-omp

clean:
	rm -rf *.o ; rm -f partdiff-par partdiff-omp
	
