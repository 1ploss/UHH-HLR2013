# Common definitions
CC = mpicc

# Compiler flags, paths and libraries
#CFLAGS = -std=c99 -pedantic -Wall -Wextra -O0 -ggdb -fno-strict-aliasing
CFLAGS = -std=c99 -pedantic -Wall -Wextra -O2 -ggdb -fno-strict-aliasing
LDFLAGS = 
LIBS   = -lm -lmpi

OBJS = timempi.o
EXE = timempi

# Rule to create *.o from *.c
%.o : %.c %.h
	$(CC) -c $< $(CFLAGS)

# Targets ...
all: $(EXE)

$(EXE): $(OBJS) Makefile
	$(CC) $(LDFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	$(RM) $(OBJS) *~ $(EXE)
