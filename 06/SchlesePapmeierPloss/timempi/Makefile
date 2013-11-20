# Common definitions
CC = mpicc

# Compiler flags, paths and libraries
#CFLAGS = -std=c99 -pedantic -Wall -Wextra -O0 -ggdb -fno-strict-aliasing
CFLAGS = -std=c99 -pedantic -Wall -Wextra -O2 -ggdb -fno-strict-aliasing
LDFLAGS = 
LIBS   = -lm -lmpi

OBJS1 = timempi.o
EXE1 = timempi

OBJS2 = timempi2.o
EXE2 = timempi2

# Targets ...
all: $(EXE1) $(EXE2)

$(EXE1): $(OBJS1) Makefile
	$(CC) $(LDFLAGS) -o $@ $(OBJS1) $(LIBS)

$(EXE2): $(OBJS2) Makefile
	$(CC) $(LDFLAGS) -o $@ $(OBJS2) $(LIBS)

# Rule to create *.o from *.c
%.o : %.c %.h
	$(CC) -c $< $(CFLAGS)

clean:
	$(RM) $(OBJS1) $(EXE1) $(OBJS2) $(EXE2) *~

run:
	mpirun -np 4 $(EXE1)
	mpirun -np 4 $(EXE2)

cluster_run:
	srun mpirun -np 24 $(EXE2)
	
