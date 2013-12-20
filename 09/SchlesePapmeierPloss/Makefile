# Common definitions
CC=mpicc
CFLAGS=-std=c99 -Wall -Wextra -O2 -ggdb
LIBS = -lm

OBJS = partdiff-par.o askparams.o displaymatrix-mpi.o

# Rule to create *.o from *.c
.c.o:
	$(CC) -c $(CFLAGS) $*.c

all: partdiff-par

partdiff-par: $(OBJS) Makefile
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LIBS)

partdiff-par.o: partdiff-par.c Makefile

askparams.o: askparams.c Makefile

clean:
	rm *.o *.~ ; rm -f partdiff-par
