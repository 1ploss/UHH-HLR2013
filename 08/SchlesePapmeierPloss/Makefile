CC=mpicc
CFLAGS=-std=c99 -Wall -Wextra -ggdb -lm

OBJS = partdiff-par.o displaymatrix-mpi.o

# Rule to create *.o from *.c
%.o : %.c
	$(CC) $(CFLAGS) -c $< -o $@

partdiff-par: $(OBJS) Makefile
	$(CC) $(CFLAGS) -ggdb $(OBJS) -o $@

all: partdiff-par

clean:
	rm *.o ; rm -f partdiff-par
	
