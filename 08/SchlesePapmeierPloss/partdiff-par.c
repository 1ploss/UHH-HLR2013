#define _BSD_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <limits.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>
#include "gauss.h"
#include <mpi.h>
#ifdef _OPENMP
# include <omp.h>
#endif
//#define DEBUG
#ifdef DEBUG
#define LOG(...) fprintf(stderr, __VA_ARGS__);
#else
#define LOG(...)
#endif
//#define TEST_VALUES

/**
 * Describes how the Matrix is drawn.
 */
typedef struct
{
	unsigned x0; // first column to display
	unsigned x1; // last column to display
	unsigned y0; // first row to display
	unsigned y1; // last row to display
	unsigned advance; // how much to increment the row and column counter after a value is displayed
	int		 show_next_chunk_line; // show a line after the rows of each task (this is mainly to ease debugging)
} Display_Params;

/**
 * Initializes part of a matrix, that is defined by params->first_line to params->first_row + params->num_rows.
 */
static void init_chunk(double** chunk, const Params* params)
{
//#define LOG_INIT(...) fprintf(stderr, "%i: init: " __VA_ARGS__);
#define LOG_INIT(...)

#ifdef TEST_VALUES
	const double border_value = NAN;

	/* Initialize first and last element of a matrix row */
	for (unsigned y = 0; y < params->num_rows; y++)
	{
		unsigned world_y = params->first_row + y;
		if (y == 0 || y == params->num_rows - 1)
		{
			for (unsigned x = 0; x < params->row_len; ++x)
			{
				chunk[y][x] = 0;
			}
		}
		else
		{
			for (unsigned x = 0; x < params->row_len; ++x)
			{
				if (x == 0 || x == params->row_len - 1)
				{
					chunk[y][x] = border_value;
				}
				else
				{
					chunk[y][x] = world_y;
				}
			}
		}
	}

	if (params->rank == 0)
	{
		for (unsigned x = 0; x < params->row_len; ++x)
		{
			chunk[0][x] = border_value;
		}
	}
	else if (params->rank == params->num_tasks - 1)
	{
		for (unsigned x = 0; x < params->row_len; ++x)
		{
			chunk[params->num_rows - 1][x] = border_value;
		}
	}

#else
	const unsigned row_len = params->row_len;
	assert(row_len > 0);

	double h = params->h;
	LOG_INIT("h is %lf\n", params->rank, h);

	// zero out the chunk
	for (unsigned y = 0; y < params->num_rows; y++)
	{
		memset(chunk[y], 0, row_len * sizeof(chunk[y][0]));
	}

	if (!params->use_stoerfunktion)
	{
		/* Initialize first and last element of a matrix row */
		for (unsigned y = 0; y < params->num_rows; y++)
		{
			double world_y = params->first_row + y;
			chunk[y][0] = 1.0 - (h * world_y);
			chunk[y][row_len - 1] = h * world_y;
		}

		if (is_first_rank(params))
		{
			/* initialize top row */
			for (unsigned x = 0; x < row_len; x++)
			{
				chunk[0][x] = 1.0 - (h * x);
				LOG_INIT("chunk[0][%u] is %lf\n", params->rank, x, chunk[0][x]);
			}
			chunk[0][row_len - 1] = 0.0;
		}
		else if (is_last_rank(params))
		{
			/* initialize bottom row */
			for (unsigned x = 0; x < row_len; x++)
			{
				chunk[params->num_rows - 1][x] = h * x;
				LOG_INIT("chunk[%u][%u] is %lf\n", params->rank, params->num_rows - 1, x, chunk[params->num_rows - 1][x]);
			}
			chunk[params->num_rows - 1][0] = 0.0;
		}
	}
#endif
}

/**
 * Solves the equation.
 * This function processes rows as specified in first_row and num_rows (number of rows) parameters.
 * It stores the matrix pointed by the src parameter as source.
 * It stores the result in the matrix pointed to with the dest argument.
 */
double compute(double** const src, double** dest, const Params* params, unsigned first_row, unsigned num_rows)
{
	//#define LOG_COMP(...) fprintf(stderr, __VA_ARGS__);
	#define LOG_COMP(...)
	double max_residuum = 0;

	LOG_COMP("%i:?: comp %u rows [%u:%u] world: [%u:%u]\n", params->rank, num_rows,
			first_row, first_row + num_rows - 1, params->first_row + first_row, params->first_row + first_row + num_rows - 1);

	/* over all rows */
	#pragma omp parallel for
	for (unsigned y = first_row; y < (first_row + num_rows); y++)
	{
		unsigned world_y = params->first_row + y;
		double fpisin_i = 0.0;
		LOG_COMP("%i:?: row %u, world %u\n", params->rank, y, world_y);

		if (params->use_stoerfunktion)
		{
			fpisin_i = params->fpisin * sin(params->pih * (double)world_y);
		}

		/* over all columns, excluding borders */
		for (unsigned x = 1; x < (params->row_len - 1); x++)
		{
			double star = 0.25 * (/* left*/ src[y-1][x] + /* right */ src[y+1][x] +
								  /*bottom*/src[y][x-1] + /* up */ src[y][x+1]);

			if (params->use_stoerfunktion)
			{
				star += fpisin_i * sin(params->pih * (double)x);
			}

			//if (options->termination == TERM_PREC || term_iteration == 1)
			{
				double residuum = fabs(src[y][x] - star);
				#pragma omp critical
				{
					max_residuum = max(residuum, max_residuum);
				}
			}
			dest[y][x] = star;
		}
	}
	return max_residuum;
}

/**
 * Performs communictaion with
 */
void communicate_jacobi(double** chunk, const Params* params)
{
//#define LOG_COMM(...) fprintf(stderr, "%i: comm: " __VA_ARGS__);
#define LOG_COMM(...)
	const int rank = params->rank;
	const int num_tasks = params->num_tasks;
	const unsigned row_length = params->row_len;

	int next_rank = (rank == num_tasks - 1) ? 0 : rank + 1;
	int prev_rank = (rank == 0) ? num_tasks - 1 : rank - 1;

	LOG_COMM("<-> u:%u, d:%u\n", rank, prev_rank, next_rank);


	MPI_Request top_request[2], bottom_request[2];
	MPI_Status status;


	/**
	 * Communicate with the rank above.
	 * Send them our 1st row and receive the 0th.
	 */
	if (rank != 0) /* 0th rank doesn't send or receive anything above */
	{
		MPI_Isend(chunk[1],                     row_length, MPI_DOUBLE, prev_rank, TAG_COMM_ROW_UP, MPI_COMM_WORLD,   &top_request[0]);
		MPI_Irecv(chunk[0],                     row_length, MPI_DOUBLE, prev_rank, TAG_COMM_ROW_DOWN, MPI_COMM_WORLD, &bottom_request[0]);
	}

	/*
	 * Communicate with bottom:
	 * send one before bottom row and receive bottom row.
	 */
	if (rank != num_tasks - 1) /* last rank doesn't send or receive anything from below */
	{
		 MPI_Isend(chunk[params->num_rows - 2], row_length, MPI_DOUBLE, next_rank, TAG_COMM_ROW_DOWN, MPI_COMM_WORLD, &bottom_request[1]);
		 MPI_Irecv(chunk[params->num_rows - 1], row_length, MPI_DOUBLE, next_rank, TAG_COMM_ROW_UP, MPI_COMM_WORLD,   &top_request[1]);
	}

	/**
	 * Wait for completion
	 */
	if (rank != 0)
	{
		MPI_Wait(&top_request[0], &status);
		MPI_Wait(&bottom_request[0], &status);
	}

	if (rank != num_tasks - 1)
	{
		MPI_Wait(&bottom_request[1], &status);
		MPI_Wait(&top_request[1], &status);
	}
}

/**
 * Calculates params->first_row (world y, row number of the whole matrix) and
 * num_rows (number of rows, the tasks has locally in memory)
 */
void calculate_row_offsets(Params* params)
{
	assert(params->num_tasks);

	div_t d = div(params->row_len, params->num_tasks);
	unsigned num_rows = d.quot;

	/**
	 * Give some tasks more rows based on their rank.
	 * TODO rebalance
	 */
	if (params->rank < d.rem)
	{
		num_rows++;
	}

	unsigned first_row = 0;
	for (int i = 0; i < params->rank; i++)
	{
		first_row += d.quot;
		if (i < d.rem)
		{
			first_row ++;
		}
	}

	/**
	 * create overlap.
	 */
	if (params->rank != params->num_tasks - 1)
	{
		num_rows += 2;
	}

	params->first_row = first_row;
	params->num_rows = num_rows;

	LOG("%u: first row: %u, num_rows: %u\n", params->rank, params->first_row, params->num_rows);

	if (params->num_rows < 4)
	{
		MPI_Barrier(MPI_COMM_WORLD); // so the error will be not too hight above
		fprintf(stderr, "set more interlines or less mpi processese, because matrix row distribution is too uneven.\n"
						"this limitation is fixable, and is currently keept that way at the moment to simplify the programm.\n");
		MPI_Abort(MPI_COMM_WORLD, 10);
	}
}

/**
 * Like malloc, only checks for result and exits the program on failure.
 */
static void* allocate_memory (size_t size)
{
	void * mem = malloc(size);
	if (!mem)
	{
		printf("Failed to allocate %lu bytes\n", size);
		MPI_Abort(MPI_COMM_WORLD, 10);
	}
	return mem;
}

/**
 * Allocates memory for the matrix part owned by this task.
 */
void allocate_matrix_chunks (Params* params)
{
	const size_t chunk_size = params->row_len * params->num_rows;
	params->mem_pool = allocate_memory(chunk_size * params->num_chunks * sizeof(double));

	params->chunk = allocate_memory(params->num_chunks * sizeof(double**));
	for (unsigned chunk_num = 0; chunk_num < params->num_chunks; chunk_num++)
	{
		params->chunk[chunk_num] = allocate_memory(params->num_rows * sizeof(double*));
		for (unsigned y = 0; y < params->num_rows; y++)
		{
			params->chunk[chunk_num][y] = params->mem_pool + (chunk_num * chunk_size) + (y * params->row_len);
		}
	}
}

/**
 * Frees memory allocated with the function above.
 */
void clean_up(Params* params)
{
	LOG("%d: cleanup\n", params->rank);
	for (unsigned chunk_num = 0; chunk_num < params->num_chunks; chunk_num++)
	{
		free(params->chunk[chunk_num]);
	}
	free(params->chunk);
	free(params->mem_pool);
}

/**
 * Displays the matrix. Display_Params describe how the displaying is done.
 * Rank 0 receives the rows and displays them.
 *
 * The way the function works is at length documented here:
 * http://wr.informatik.uni-hamburg.de/_media/teaching/wintersemester_2013_2014/hr-1314-uebungsblatt-08-materialien.tar.gz
 */
void display(const Params* params, const Result* result, const Display_Params* dp)
{
	//#define LOG_DISP(...) fprintf(stderr, "%i: disp: " __VA_ARGS__);
	#define LOG_DISP(...)
	FILE * out = stdout;
	const int rank = params->rank;

	if (rank == 0)
	{
		unsigned rank0_last_row = params->first_row + params->num_rows - 1;
		LOG_DISP("rank0_last_row %u\n", rank, rank0_last_row);
		double* row = result->chunk[0]; // first row to display. also reuse to receive next row
		int last_source = 0;
		for (unsigned row_index = dp->y0; row_index < dp->y1; row_index += dp->advance)
		{
			LOG_DISP("row_index %u\n", rank, row_index);
			/**
			 * Do we have this row? If not, try to receive it.
			 */
			if (row_index <= rank0_last_row)
			{
				LOG_DISP("using our row %u\n", rank, row_index);
				row = result->chunk[row_index];
			}
			else
			{
				LOG_DISP("receiving row %u\n", rank, row_index);
				MPI_Status status;
				MPI_Recv(row, params->row_len, MPI_DOUBLE, MPI_ANY_SOURCE, row_index, MPI_COMM_WORLD, &status);
				if (dp->show_next_chunk_line && status.MPI_SOURCE != last_source)
				{
					fprintf(out, "------------------------------[from %i]-------------------------------------------\n", status.MPI_SOURCE);
					last_source = status.MPI_SOURCE;
				}
			}

			for (unsigned x = dp->x0; x < dp->x1; x += dp->advance)
			{
				fprintf(out, "%7.4f ", row[x]);
			}
			fprintf(out, "\n");
		}
		fprintf(out, "max residuum is: %lf, number of iterations done: %lu\n", result->max_residuum, result->num_iterations);
	}
	else
	{
		unsigned begin = params->first_row + 2;
		while (((begin - dp->y0) % dp->advance) != 0)
		{
			begin++;
		}
		const unsigned end = min(dp->y1, (params->first_row + params->num_rows));

		LOG_DISP("begin %u, end %u\n", params->rank, begin, end);
		for (unsigned row_index = begin; row_index < end; row_index += dp->advance)
		{
			/*
			 * Assume the process before us already send 2
			 */
			if (row_index >= (params->first_row + 2) && (row_index < params->first_row + params->num_rows))
			{
				const unsigned local_index = row_index - params->first_row;
				LOG_DISP("sending row %u\n", rank, row_index);
				MPI_Send(result->chunk[local_index], params->row_len, MPI_DOUBLE, 0, row_index, MPI_COMM_WORLD);
			}
		}
	}
}

/**
 * Function that displays stats of the program.
 */
void print_params(const Params* params)
{
	char buff[1024];
	if (is_first_rank(params))
	{
		const char* methods[] = { "unknown", "Gauss-Seidel", "Jacobi" };
		printf("%i: ----[general info]----:\n", params->rank);
		printf("%i: num_tasks : %i\n", params->rank, params->num_tasks);
		printf("%i: num_chunks : %u\n", params->rank, params->num_chunks);
		printf("%i: method : %s\n", params->rank, methods[params->method]);
		printf("%i: interlines : %u\n", params->rank, params->interlines);
		printf("%i: Stoerfunktion : %s\n", params->rank, (params->use_stoerfunktion ? "f(x,y) = 2pi^2*sin(pi*x)sin(pi*y)" : "f(x,y) = 0"));
		printf("%i: Termination condition : %s\n", params->rank, (params->target_iteration ? "Number of iterations" : "Sufficient precision"));
		if (params->target_iteration)
		{
			printf("%i: target_iteration : %lu\n", params->rank, params->target_iteration);
		}
		else
		{
			printf("%i: target_residuum : %lf\n", params->rank, params->target_residuum);
		}
		printf("%i: row_len: %u\n", params->rank, params->row_len);

		printf("%i: ----[per-rank info]----:\n", params->rank);
		printf("%i: first_row : %u\n", params->rank, params->first_row);
		printf("%i: num_rows : %u\n", params->rank, params->num_rows);

		double bytes_used = params->num_chunks * params->num_rows * params->row_len * sizeof(double);
		printf("%i: chunks mem usage: : %6.3lf %cb\n", params->rank,
					((bytes_used > (1024 * 1024)) ? bytes_used / (1024 * 1024) : bytes_used / 1024),
					((bytes_used > (1024 * 1024)) ? 'm' : 'k'));

		MPI_Status status;
		for (int i = 1; i < params->num_tasks; i++)
		{
			for (unsigned j = 0; j < 3; j++)
			{
				MPI_Recv(buff, sizeof(buff), MPI_CHAR, i, 0, MPI_COMM_WORLD, &status);
				printf(buff);
			}
		}
#ifdef _OPENMP
	printf("%i: omp_num_threads : %u\n", params->rank, params->num_threads);
#endif
	}
	else
	{
		snprintf(buff, sizeof(buff), "%i: first_row : %u\n", params->rank, params->first_row);
		MPI_Send(buff, sizeof(buff), MPI_CHAR, 0, 0, MPI_COMM_WORLD);

		snprintf(buff, sizeof(buff), "%i: num_rows : %u\n", params->rank, params->num_rows);
		MPI_Send(buff, sizeof(buff), MPI_CHAR, 0, 0, MPI_COMM_WORLD);

		double bytes_used = params->num_chunks * params->num_rows * params->row_len * sizeof(double);
		snprintf(buff, sizeof(buff), "%i: chunks mem usage: : %6.3lf %cb\n", params->rank,
					((bytes_used > (1024 * 1024)) ? bytes_used / (1024 * 1024) : bytes_used / 1024),
					((bytes_used > (1024 * 1024)) ? 'm' : 'k'));
		MPI_Send(buff, sizeof(buff), MPI_CHAR, 0, 0, MPI_COMM_WORLD);
	}

}

/**
 * Parses input and stores it in params.
 */
void params_init(int argc, char** argv, Params* params)
{
	MPI_Comm_rank(MPI_COMM_WORLD, &params->rank);
	MPI_Comm_size(MPI_COMM_WORLD, &params->num_tasks);
	params->next_rank = (params->rank == params->num_tasks - 1) ? 0 : params->rank + 1;
	params->prev_rank = (params->rank == 0) ? params->num_tasks - 1 : params->rank - 1;
	params->first_row = 0;
	params->num_rows = 0;
	unsigned pos;


	if (argc < 6)
	{
		printf("there should be 6 arguments!\n");
		printf("Usage: partdiff-par [num] [method] [lines] [func] [term] [prec/iter]\n");
		printf("\n");
		printf("  - num:       number of threads (1 .. )\n");
		printf("  - method:    calculation method (1 .. 2)\n");
		printf("                 1: Gauß-Seidel\n");
		printf("                 2: Jacobi\n");
		printf("  - lines:     number of interlines (0 .. depends of how much memory you have)\n");
		printf("                 matrixsize = (interlines * 8) + 9\n");
		printf("  - func:      interference function (1 .. 2)\n");
		printf("                 1: f(x,y) = 0\n");
		printf("                 2: f(x,y) = 2 * pi^2 * sin(pi * x) * sin(pi * y)\n");
		printf("  - term:      termination condition ( 1.. 2)\n");
		printf("                 1: sufficient precision\n");
		printf("                 2: number of iterations\n");
		printf("  - prec/iter: depending on term:\n");
		printf("                 precision:  1e-4 .. 1e-20\n");
		printf("                 iterations:    1 .. \n");
		printf("\n");
		printf("Example: partdiff-par 1 2 100 1 2 100 \n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if ((sscanf(argv[1], "%u%n", &params->num_threads, &pos) != 1) || pos != strlen(argv[1]))
	{
		printf("first argument should have unsigned type\n");
		MPI_Abort(MPI_COMM_WORLD, 2);
	}

#ifdef _OPENMP
	if (params->num_threads)
	{
		omp_set_dynamic(0);     // Explicitly disable dynamic teams
		omp_set_num_threads(params->num_threads);
	}
#endif

	if ((sscanf(argv[2], "%u%n", &params->method, &pos) != 1) || pos != strlen(argv[2]) ||
		params->method == 0 || params->method > 2)
	{
		printf("expecting first argument to be in range of [2:1]\n");
		MPI_Abort(MPI_COMM_WORLD, 2);
	}

	if (sscanf(argv[3], "%u%n", &params->interlines, &pos) != 1 ||  pos != strlen(argv[3]))
	{
		printf("expecting 2st argument of unsigned type\n");
		MPI_Abort(MPI_COMM_WORLD, 3);
	}
	params->row_len = (params->interlines * 8) + 9;

	if ((sscanf(argv[4], "%u%n", &params->use_stoerfunktion, &pos) != 1) ||  pos != strlen(argv[4]) ||
		 params->use_stoerfunktion == 0 || params->use_stoerfunktion > 2)
	{
		printf("expecting 3rd argument of boolean type (0 or 1)\n");
		MPI_Abort(MPI_COMM_WORLD, 4);
	}
	params->use_stoerfunktion--;

	unsigned use_iterations;
	if (sscanf(argv[5], "%u%n", &use_iterations, &pos) != 1 ||  pos != strlen(argv[5]) ||
					use_iterations == 0 || use_iterations > 2)
	{
		printf("expecting 1 or 2 for the 5-th argument\n");
		MPI_Abort(MPI_COMM_WORLD, 5);
	}
	use_iterations--;

	if (use_iterations)
	{
		if (sscanf(argv[6], "%lu%n", &params->target_iteration, &pos) != 1 ||  pos != strlen(argv[6]) ||
						params->target_iteration == 0)
		{
			printf("expecting 6-th argument of 64 bit unsigned type\n");
			MPI_Abort(MPI_COMM_WORLD, 5);
		}
	}
	else
	{
		params->target_iteration = 0;
		if (sscanf(argv[6], "%lf%n", &params->target_residuum, &pos) != 1 ||  pos != strlen(argv[6]))
		{
			printf("max residuum should be a floating point number\n");
			MPI_Abort(MPI_COMM_WORLD, 7);
		}

		if (!isnormal(params->target_residuum) || params->target_residuum <= 0)
		{
			printf("max residuum should be a positive floating point number\n");
			MPI_Abort(MPI_COMM_WORLD, 8);
		}

		if (params->target_residuum > 1e-1 || params->target_residuum < 1e-20)
		{
			printf("max residuum should be a within 1e-4 .. 1e-20\n");
			MPI_Abort(MPI_COMM_WORLD, 9);
		}
	}

	params->num_chunks = 1 + (params->method == JACOBI);
	params->h = 1.0 / (double)params->row_len;
	params->fpisin = 0.25 * TWO_PI_SQUARE * params->h * params->h;//ist das *h*h Absicht?
	params->pih = PI * params->h;
}



void do_jacobi(const Params* params, Result* result)
{
//#define LOG_JACOBI(...) fprintf(stderr, "%i: jacobi: " __VA_ARGS__);
#define LOG_JACOBI(...)
	unsigned stop_after_precision_reached = (params->target_iteration == 0);
	unsigned curr = 0;
	unsigned next = (curr + 1) % params->num_chunks;
	for (result->num_iterations = 0;; result->num_iterations++)
	{
		LOG_JACOBI("iter: %lu\n", params->rank, result->num_iterations);
		double max_residuum = compute(params->chunk[curr], params->chunk[next], params, 1, params->num_rows - 2);
		communicate_jacobi(params->chunk[next], params);
		if (stop_after_precision_reached)
		{
			/* all tasks need max residuum */
			MPI_Allreduce(&max_residuum, &result->max_residuum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			if (result->max_residuum <= params->target_residuum)
			{
				break;
			}
		}
		else if (result->num_iterations >= params->target_iteration)
		{
			/* only rank 0 needs max residuum */
			MPI_Reduce(&max_residuum, &result->max_residuum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			break;
		}
		curr = next;
		next = (next + 1) % params->num_chunks;

		if (params->rank == 0)
		{
			LOG_JACOBI("target_res: %lf, max_res: %lf\n", params->rank, params->target_residuum, result->max_residuum);
		}
	}
	result->chunk = params->chunk[next];
}


void alt_display(const Params* params, const Result* result, const Display_Params* dp)
{
	MPI_Barrier(MPI_COMM_WORLD);

	fflush(stdout);
	fflush(stderr);

	for (int i = 0; i < params->rank; i++)
	{
		MPI_Barrier(MPI_COMM_WORLD);
	}

	for (unsigned world_y = dp->y0; world_y < (params->first_row + params->num_rows); world_y += dp->advance)
	{
		if (world_y > params->first_row && ((world_y + dp->advance - dp->y0) % dp->advance) == 0)
		{
			unsigned y = world_y - params->first_row;
			fprintf(stdout, "%i: y: %u:%u | ", params->rank, y, params->first_row + y);
			for (unsigned x = dp->x0; x < dp->x1; x += dp->advance)
			{
				fprintf(stdout, "%4.4f ", result->chunk[y][x]);
			}
			fprintf(stdout, "\n");
		}
	}
	fflush(stdout);

	for (int i = params->rank; i < params->num_tasks; i++)
	{
		MPI_Barrier(MPI_COMM_WORLD);
	}
}



/**
 * Benutzung:
 * arg[0] interlines use_stoerfunktion target_iter target_residuum
 * target_residuum ist optinal, wenn target_iter == 0
 */
int main(int argc, char** argv)
{
	#define LOG_MAIN(...) fprintf(stderr, "%i: main: " __VA_ARGS__);
	//#define LOG_MAIN(...)

	if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
	{
		fprintf(stderr, "MPI_Init failed!\n");
		exit(EXIT_FAILURE);
	}
	Params params;
	params_init(argc, argv, &params);
	calculate_row_offsets(&params);

	print_params(&params);

	allocate_matrix_chunks(&params);
	for (unsigned i = 0; i < params.num_chunks; ++i)
	{
		init_chunk(params.chunk[i], &params);
	}

	Result result;
	struct timeval start_time, end_time;
	gettimeofday(&start_time, NULL); // measure time it took the the calcualtion to run
	if (params.method == JACOBI)
	{
		do_jacobi(&params, &result);
	}
	else
	{
		do_gauss(&params, &result);
	}

	gettimeofday(&end_time, NULL);

	// show full matrix
	//Display_Params dp = { 0, params.row_len, 0, params.row_len, 1, 1};

	// no borders
	//Display_Params dp = { 1, params.row_len - 1, 1, params.row_len - 1, params.interlines + 1, 0 };


	Display_Params dp = { 0, params.row_len, 0, params.row_len, params.interlines + 1, 0 };
	display(&params, &result, &dp);

#if 0
	alt_display(&params, &result, &dp);
#endif

	if (params.rank == 0)
	{
		printf("time taken: %lf seconds\n", (double)(end_time.tv_sec - start_time.tv_sec) + (((double)(end_time.tv_usec - start_time.tv_usec) * 1e-6 )));
	}


	clean_up(&params);
	MPI_Finalize();
	return EXIT_SUCCESS;
}

