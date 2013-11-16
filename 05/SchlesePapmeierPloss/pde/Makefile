# Common definitions
CC = gcc

# Compiler flags, paths and libraries
#CFLAGS = -std=c99 -pedantic -Wall -Wextra -O0 -ggdb -fno-strict-aliasing
CFLAGS = -std=c99 -pedantic -Wall -Wextra -O2 -ggdb -fno-strict-aliasing
LFLAGS = 
LIBS   = -lm -lpthread -lrt

OBJS = thread_pool.o partdiff-posix.o askparams.o

# Rule to create *.o from *.c
%.o : %.c %.h
	$(CC) -c $< $(CFLAGS)

# Targets ...
all: partdiff-posix

partdiff-posix: $(OBJS) Makefile
	$(CC) $(LFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	$(RM) *.o *~ partdiff-posix
