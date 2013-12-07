#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <limits.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>
#include "displaymatrix-mpi.h"
#define DEBUG
#ifdef DEBUG
#define LOG(...) fprintf(stderr, __VA_ARGS__);
#else
#define LOG(...)
#endif
#define NUM_CHUNKS 2

#ifndef PI
#define PI           3.14159265358979323846
#endif
#define TWO_PI_SQUARE (2.0 * PI * PI)


/**
 * Initializes the matrix.
 * Was ist die pNr und pAnzahl?
 */
void init(double** chunk, unsigned N, unsigned first_line, unsigned last_line, unsigned use_stoerfunktion,
				int pNr, int pAnzahl)
{
	unsigned lines = last_line - first_line;
	double h = 1.0 / (double) N;
	/* Matrix is already initialized with zeros */

	/* initialize borders, depending on function (function 2: nothing to do) */
	if (!use_stoerfunktion)
	{
		for (unsigned i = 0; i < lines; i++)
		{
			unsigned globali = (i + (pNr * lines));
			chunk[i][0] = 1.0 - (h * globali);
			chunk[i][N] = h * globali;
			if (pNr == 0)
			{
				chunk[0][i] = 1.0 - (h * globali);
			}
			if (pNr == pAnzahl - 1)
			{
				chunk[lines - 1][i] = h * globali;
			}
		}
		chunk[N][0] = 0.0;
		chunk[0][N] = 0.0;
		//TODO Es könnte sein, dass die Initialisierung anders ist als die sequentielle, da N irgendwie komisch verwaltet wird
	}
}

/**
 * computes from current into next.
 * @param current ich gehe davon aus, dass die erste Dimension den Zeilen entspricht und die zweite den Spalten
 * @return max residium
 */
double compute(double** current, double** next, unsigned N, unsigned num_lines, unsigned use_stoerfunktion, unsigned first_gobal_line)
{
	int rank, num_tasks;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
	LOG("%i: compute %u lines\n", rank, num_lines);


	double pih = 0.0;
	double fpisin = 0.0;
	double star = 0.0;

	if (use_stoerfunktion)
	{
		pih = PI * (1.0 / (double) N);
		fpisin = 2.0 * TWO_PI_SQUARE;
	}
	double maxresiduum = 0.0;

	for (unsigned i = 1; i < num_lines - 1; i++)
	{
		double fpisin_i = 0.0;
		double global_line_nr = first_gobal_line + i;

		if (use_stoerfunktion)
		{
			fpisin_i = fpisin * sin(pih * (double)global_line_nr);
		}
		//Hier gehe ich davon aus, dass die laenge der tatsächlichen Matrix entspricht
		for (unsigned j = 1; j < (N - 1); j++)
		{
			//Hier gehe ich davon aus, dass die Randwerte von den Nachbarprozessoren im current enthalten sind
			star = 0.25 * (current[i - 1][j] + current[i][j - 1] + current[i][j + 1] + current[i + 1][j]);
			if (use_stoerfunktion)
			{
				star += fpisin_i * sin(pih * (double) j);
			}

			double residuum = current[i][j] - star;
			residuum = (residuum < 0) ? -residuum : residuum;
			maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
			next[i][j] = star;
		}
	}
	return maxresiduum;
}


#define TAG_SEND_RECEIVE 1
void communicate(double** current, double** next, unsigned N, unsigned num_lines)
{
	int rank, num_tasks;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

	int next_rank = (rank == num_tasks - 1) ? 0 : rank + 1;
	int prev_rank = (rank == 0) ? num_tasks - 1 : rank - 1;

	LOG("%i: comm with %u and %u\n", rank, prev_rank, next_rank);


	MPI_Request oben_request[2], unten_request[2];
	MPI_Status status;

	/**
	 * Kommuniziere mit unten (alle ausser dem letzten Rank)
	 */
	if (rank != num_tasks - 1)
	{
		 MPI_Isend(next[num_lines - 2],        N, MPI_DOUBLE, next_rank, TAG_SEND_RECEIVE, MPI_COMM_WORLD, &unten_request[0]);
		 MPI_Irecv(current[num_lines - 1], N, MPI_DOUBLE, next_rank, TAG_SEND_RECEIVE, MPI_COMM_WORLD, &unten_request[1]);
	}

	/**
	 * Kommuniziere mit oben (alle ausser dem 0-ten rank)
	 */
	if (rank != 0)
	{
		MPI_Isend(next[1],    N, MPI_DOUBLE, prev_rank, TAG_SEND_RECEIVE, MPI_COMM_WORLD, &oben_request[0]);
		MPI_Irecv(current[0], N, MPI_DOUBLE, prev_rank, TAG_SEND_RECEIVE, MPI_COMM_WORLD, &oben_request[1]);
	}

	if (rank != num_tasks - 1)
	{
		MPI_Wait(&unten_request[0], &status);
		MPI_Wait(&unten_request[1], &status);
	}
	if (rank != 0)
	{
		MPI_Wait(&oben_request[0], &status);
		MPI_Wait(&oben_request[1], &status);
	}
}

#include "displaymatrix-mpi.h"

void calculate_lines(unsigned N, unsigned* the_first_line, unsigned* the_num_lines)
{
	int rank, num_tasks;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

	div_t d = div(N, num_tasks);
	unsigned num_lines = d.quot;
	if (rank < d.rem)
	{
		num_lines++;
	}

	unsigned first_line_loop = 0;
	for (int i = 0; i < rank; i++)
	{
		first_line_loop ++;
		if (i < d.rem)
		{
			first_line_loop ++;
		}
	}

	//assert(first_line_loop == first_line);

	*the_first_line = first_line_loop;
	*the_num_lines = num_lines;
}

/**
 * Benutzung:
 * arg[0] interlines use_stoerfunktion target_iter target_residuum
 * target_residuum ist optinal, wenn target_iter == 0
 */
int main(int argc, char** argv)
{
	int rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS)
	{
		fprintf(stderr, "MPI_Init failed!\n");
		exit(EXIT_FAILURE);
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
	unsigned N = (interlines * 8) + 9;

	unsigned use_stoerfunktion;
	if (sscanf(argv[3], "%u", &use_stoerfunktion) != 1)
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

	unsigned first_line, num_lines;
	calculate_lines(N, &first_line, &num_lines);

	LOG("%d: first_line: %u, num lines is %u\n", rank, first_line, num_lines);


	/**
	 * Dies ist ein Teil unserer Matrix.
	 * Die erste und die letzten Zeilen sind ränder und werden vom compute() nicht angefasst.
	 */
	double* chunk[NUM_CHUNKS][num_lines];

	//Für die Nachrichtenverschickung müssen alle Matrizen die gleiche Größe haben
	//TODO sich drum kümmern, dass bei kleinerer lokalen Matrix alles glatt läuft!
	unsigned pool_size = N * num_lines;
	for (unsigned i = 0; i < NUM_CHUNKS; i++)
	{
		LOG("%d: allocating chunk %u memory of %u doubles\n", rank, i, pool_size);
		double* pool = calloc(sizeof(double), pool_size);
		if (pool == 0)
		{
			printf("matrix chunk memory allocation failed\n");
			MPI_Abort(MPI_COMM_WORLD, 6);
		}
		// fixing line pointers
		for (unsigned j = 0; j < num_lines; j++)
		{
			chunk[i][j] = &pool[j * N];
		}
	}

	LOG("%d: main algorithm\n", rank);
	init(chunk[0], first_line, first_line + num_lines, N, use_stoerfunktion, rank, num_tasks);
	unsigned curr = 0, next;
	for (unsigned iter = 0; stop_after_precision_reached || iter < target_iter; iter++)
	{
		next = (curr + 1) % NUM_CHUNKS;
		communicate(chunk[curr], chunk[next], N, num_lines);	//Zeilenaustausch
		double max_residuum = compute(chunk[curr], chunk[next], N, num_lines, use_stoerfunktion, first_line);
		curr = next;
		if (stop_after_precision_reached)
		{
			double reduced_max_residuum;
			MPI_Reduce(&max_residuum, &reduced_max_residuum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

			if (reduced_max_residuum < target_residuum)
			{
				break;
			}
		}
	}

	DisplayMatrix ("bla", chunk, (int)interlines , rank , num_lines, first_line, first_line + num_lines);

	LOG("%d: cleanup\n", rank);
	for (unsigned i = 0; i < NUM_CHUNKS; i++)
	{
		free(chunk[i][0]);
	}
	MPI_Finalize();
}

