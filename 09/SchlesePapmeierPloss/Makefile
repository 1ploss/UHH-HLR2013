# Common definitions
CC = mpicc

# Compiler flags, paths and libraries
CFLAGS = -std=c99 -pedantic -Wall -Wextra -Wno-unknown-pragmas -O0 -ggdb
LFLAGS = $(CFLAGS)
LIBS   = -lm

OBJS = askparams.o partdiff-par.o

# Rule to create *.o from *.c
.c.o:
	$(CC) -c $(CFLAGS) $*.c

# Targets ...
all: partdiff-par

partdiff-par: $(OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	$(RM) *.o *~ partdiff-par.exe

partdiff-seq.o: partdiff-seq.c Makefile

askparams.o: askparams.c Makefile
