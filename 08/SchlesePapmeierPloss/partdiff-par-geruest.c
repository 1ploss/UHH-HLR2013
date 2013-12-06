#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <limits.h>
#include <mpi.h>
#include <08-displaymatrix/displaymatrix-mpi.h>
#define DEBUG
#ifdef DEBUG
#define LOG(...) fprintf(stderr, __VA_ARGS__);
#else
#define LOG(...)
#endif
#define NUM_CHUNKS 2
#ifndef PI
#define PI 			3.141592653589793
#endif
#define TWO_PI_SQUARE 		(2 * PI * PI)

/**
 * Initializes the matrix.
 */
void init(double** chunk, unsigned first_line, unsigned last_line)
{
	// TODO
	/*TODO wir müssen klären, wie wir die Ränder der globalen Matrix verwalten
	 * Ich gehe für compute erstmal davon aus, dass die Ränder in den Teilmatrizen enthalten sind
	 */
}

/**
 * computes from current into next. TODO Hinsichtlich der Annahmen anpassen
 * @param current ich gehe davon aus, dass die erste Dimension den Zeilen entspricht und die zweite den Spalten
 * @return max residium
 */
double compute(double** current, double** next, unsigned first_line, unsigned last_line, unsigned stoerfunktion, unsigned laenge)
{
	double pih = 0.0;
	double fpisin = 0.0;
	double star = 0.0;

	if(stoerfunktion)
	{
		pih = PI * (1.0/(double)laenge);
		fpisin = 2 * TWO_PI_SQUARE;
	}
	double maxresiduum= 0.0;
	//Hier gehe ich davon aus, dass first_line nie 0 und last_line nie N ist
	for(int i = 0; i < (last_line - first_line);i++)
	{
		double fpisin_i = 0.0;
		double iGlobal = i+first_line;

		if(stoerfunktion)
		{
			fpisin_i = fpisin * sin(pih * iGlobal);
		}
		//Hier gehe ich davon aus, dass die laenge der tatsächlichen Matrix entspricht
		for(int j= 1;j < (int)(laenge-1);j++)
		{
			//Hier gehe ich davon aus, dass die Randwerte von den Nachbarprozessoren im current enthalten sind
			star = 0.25 * (current[i-1][j] + current[i][j-1] + current[i][j+1] + current[i+1][j]);
			if(stoerfunktion)
			{
				star += fpisin_i * sin(pih * (double)j);
			}

			int dummy = 1;//TODO ich brauche eine Aussage dafür, ob ein Residuum berechnet werden soll, oder nicht
			//Bis dahin, wird es immer berechnet

			if(dummy)
			{
				double residuum = current[i][j] - star;
				residuum = (residuum < 0) ? -residuum : residuum;
				maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
			}

			next[i][j] = star;
		}
	}
	return maxresiduum;
}

void communicate(double** current, double** next, unsigned first_line, unsigned last_line)
{
	// TODO
#if 0
	MPI_Request send_request, recv_request;
	MPI_Isend(buffers[send_buff_index],						/* message buffer */
			 buffsz,						/* number of data items */
			 MPI_INT,			/* data item is an integer */
			 right_rank,							/* destination process rank */
			 TAG_SEND_RECEIVE,	/* user chosen message tag */
			 MPI_COMM_WORLD, /* default communicator */
			 &send_request);
	MPI_Irecv(buffers[recv_buff_index],           /* message buffer */
			 buffsz,                 /* one data item */
			 MPI_INT,        /* of type double real */
			 left_rank,    					/* receive from left rank */
			 TAG_SEND_RECEIVE , // use MPI_ANY_TAG for any type of message
			 MPI_COMM_WORLD,    /* default communicator */
			 &recv_request);          /* info about the received message */

	MPI_Wait(&send_request, &status);
	MPI_Wait(&recv_request, &status);
	LOG("%i: iteration %u send recv complete\n", rank, iteration);
#endif
}

void display()
{
	// TODO
}

int main(int argc, char** argv)
{
	int rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS)
	{
		fprintf (stderr, "MPI_Init failed!\n");
		exit (EXIT_FAILURE);
	}

	/**
	 * Parsing input
	 */
	if (argc < 3)
	{
		printf("Arguments error\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	unsigned interlines;
	if (sscanf(argv[1], "%u", &interlines) != 1)
	{
		printf("expecting 1st argument of unsigned type\n");
		MPI_Abort(MPI_COMM_WORLD, 2);
	}
	unsigned N = (interlines * 8)+9;

	unsigned stoerfunktion;
	if (sscanf(argv[3], "%u", &stoerfunktion) != 1)
		{
			printf("expecting 2nd argument of boolean type\n");
			MPI_Abort(MPI_COMM_WORLD, 3);
		}

	unsigned target_iter;
	if (sscanf(argv[3], "%u", &target_iter) != 1)
	{
		printf("expecting 3nd argument of unsigned type\n");
		MPI_Abort(MPI_COMM_WORLD, 4);
	}
	unsigned stop_after_precision_reached = (target_iter == 0);
	double target_residuum;
	if (stop_after_precision_reached)
	{
		if (argc < 4)
		{
			printf("expecting max residuum as a 4rth argument\n");
			MPI_Abort(MPI_COMM_WORLD, 5);
		}

		if (sscanf(argv[4], "%lf", &target_residuum) != 1)
		{
			printf("max residuum should be a double\n");
			MPI_Abort(MPI_COMM_WORLD, 6);
		}
	}



	/**
	 * Main Programm, obligatory mpi queries, calculating chunk sizes
	 */
	int rank, num_tasks;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

	div_t d = div (N, num_tasks);
	const unsigned num_lines_in_chunk = d.quot + (d.rem != 0) - (rank > d.rem);
	const unsigned first_line = rank * num_lines_in_chunk;
	const unsigned last_line = (rank + 1) * num_lines_in_chunk;
	LOG("%d: num_lines_in_chunk: %u\n", rank, num_lines_in_chunk);

	double* chunk[NUM_CHUNKS][num_lines_in_chunk];
	for (unsigned i = 0; i < NUM_CHUNKS; i++)
	{
		unsigned long bytes = N * num_lines_in_chunk * sizeof(double);
		LOG("%d: allocating chunk %u memory of %u bytes\n", rank, i, bytes);
		double* pool = malloc(bytes);
		if (pool == 0)
		{
			printf("matrix chunk memory allocation failed\n");
			MPI_Abort(MPI_COMM_WORLD, 6);
		}
		// fixing line pointers
		for (unsigned j = 0; j < num_lines_in_chunk; j++)
		{
			chunk[i][j] = &pool[j * N];
		}
	}

	LOG("%d: main algorithm\n", rank);
	init(chunk[0], first_line, last_line);
	unsigned curr = 0, next;
	for (unsigned iter = 0;  stop_after_precision_reached || iter < target_iter; iter++)
	{
		next = (curr + 1) % NUM_CHUNKS;
		communicate(chunk[curr], chunk[next], first_line, last_line);//Zeilenaustausch
		double max_residuum = compute(chunk[curr], chunk[next], first_line, last_line, stoerfunktion, N);
		curr = next;
		if (stop_after_precision_reached)
		{
			double reduced_max_residuum;
			MPI_Reduce(&max_residuum, &reduced_max_residuum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

			if (reduced_max_residuum > target_residuum)
			{
				break;
			}
		}
	}

	display();//reicht da nicht einfach das hier:
	char s = "Hallo";
	DisplayMatrix ( *s,chunk,(int)interlines , rank , last_line - first_line, first_line, last_line );
	//TODO ob ich den Aufruf von DisplayMatrix richtig gemacht hab, weiß ich nicht

	LOG("%d: cleanup\n", rank);
	for (unsigned i = 0; i < NUM_CHUNKS; i++)
	{
		free(chunk[i][0]);
	}
	MPI_Finalize();
}
