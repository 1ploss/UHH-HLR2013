# Common definitions
CC = gcc

# Compiler flags, paths and libraries
CFLAGS = -std=c99 -pedantic -Wall -Wextra -O2
LDFLAGS = $(CFLAGS)
LIBS   = -lm
PROGS = partdiff-seq partdiff-openmp-element partdiff-openmp-spalten partdiff-openmp-zeilen


all: $(PROGS) partdiff-openmp

partdiff-seq-prog.o: CFLAGS += -DPARVER=0
partdiff-seq-askparams.o: CFLAGS += -DPARVER=0
partdiff-openmp-element: CFLAGS += -DPARVER=1 -fopenmp
partdiff-openmp-element-prog.o: CFLAGS += -DPARVER=1 -fopenmp
partdiff-openmp-element-askparams.o: CFLAGS += -DPARVER=1 -fopenmp
partdiff-openmp-spalten: CFLAGS += -DPARVER=2  -fopenmp
partdiff-openmp-spalten-prog.o: CFLAGS += -DPARVER=2  -fopenmp
partdiff-openmp-spalten-askparams.o: CFLAGS += -DPARVER=2  -fopenmp
partdiff-openmp-zeilen: CFLAGS += -DPARVER=3  -fopenmp
partdiff-openmp-zeilen-prog.o: CFLAGS += -DPARVER=3  -fopenmp
partdiff-openmp-zeilen-askparams.o: CFLAGS += -DPARVER=3  -fopenmp

%-prog.o: partdiff-seq.c
	$(CC) $(CFLAGS) -c $< -o $@
	
%-askparams.o: askparams.c
	$(CC) $(CFLAGS) -c $< -o $@

partdiff-openmp : partdiff-openmp-spalten
	ln -s partdiff-openmp-spalten partdiff-openmp

$(PROGS): % : %-prog.o %-askparams.o
	$(CC) $(LDFLAGS) -o $@ $@-prog.o $@-askparams.o $(LIBS)

clean:
	$(RM) $(PROGS) *.o *~
