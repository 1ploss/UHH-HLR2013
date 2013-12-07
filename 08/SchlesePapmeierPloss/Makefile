CC=mpicc
CFLAGS=-std=c99 -Wall -Wextra -Wno-unknown-pragmas -ggdb -lm

partdiff-par.o: partdiff-par.c Makefile
	$(CC) $(CFLAGS) -c $< -o $@

partdiff-par: partdiff-par.o
	$(CC) $(CFLAGS) -ggdb $< -o $@

partdiff-omp.o: partdiff-omp.c Makefile
	$(CC) $(CFLAGS) -fopenmp -c $< -o $@
	
partdiff-omp: partdiff-omp.o
	$(CC) $(CFLAGS)  -fopenmp -ggdb $< -o $@

all: partdiff-par partdiff-omp

clean:
	rm *.o ; rm -f partdiff-par partdiff-omp
	
