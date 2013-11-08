# Common definitions
CC = gcc

# Compiler flags, paths and libraries
CFLAGS = -std=c99 -pedantic -Wall -Wextra -O2 -ggdb -fopenmp
LDFLAGS = $(CFLAGS)
LIBS   = -lm
OBJS = partdiff-seq.o askparams.o
PROGS = partdiff-seq partdiff-openmp-element partdiff-openmp-spalten partdiff-openmp-zeilen

all: $(PROGS)

# Rule to create *.o from *.c
%.o : %.c %.h Makefile
	$(CC) -c $(CFLAGS) -o $@ $*.c

partdiff-openmp-element.o: partdiff-seq.c
	$(CC) -c $(CFLAGS) -o $@ -DPAR_VERSION=1 partdiff-seq.c

partdiff-openmp-spalten.o: partdiff-seq.c
	$(CC) -c $(CFLAGS) -o $@ -DPAR_VERSION=2 partdiff-seq.c

partdiff-openmp-zeilen.o: partdiff-seq.c
	$(CC) -c $(CFLAGS) -o $@ -DPAR_VERSION=3 partdiff-seq.c

$(PROGS): % : %.o askparams.o
	$(CC) $(LDFLAGS) -o $@ $@.o askparams.o $(LIBS)

clean:
	$(RM) *.o *~

partdiff-seq.o: partdiff-seq.c Makefile

askparams.o: askparams.c Makefile

#Das Makefile muss noch auf openmp angepasst werden und -D VERSION={omp,columns,nested,static,dynamic,guided} passend beim compilieren gesetzt werden.
