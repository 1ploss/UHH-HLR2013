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


#ifndef PI
#define PI           3.14159265358979323846
#endif
#define TWO_PI_SQUARE (2.0 * PI * PI)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

typedef struct
{
	unsigned x0;
	unsigned x1;
	unsigned y0;
	unsigned y1;
	unsigned advance;
	int		 show_next_chunk_line;
} Display_Params;

typedef struct
{
	unsigned use_stoerfunktion;
	unsigned target_iteration;
	double target_residuum;
	int rank;
	int num_tasks;
	unsigned interlines;
	unsigned row_len;
	unsigned first_row;
	unsigned num_rows;

	unsigned omp_num_threads;

	unsigned num_chunks;
	double* mem_pool;
	double*** chunk;
} Params;

/**
 * Initializes part of a matrix, that is defined by params->first_line to params->first_row + params->num_rows.
 */
void init_chunk(double** chunk, const Params* params)
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
	const int rank = params->rank;
	const int num_tasks = params->num_tasks;
	assert(row_len > 0);

	double h = 1.0 / (double) row_len;
	LOG_INIT("h is %lf\n", params->rank, h);

	for (unsigned y = 0; y < params->num_rows; y++)
	{
		memset(chunk[y], 0, row_len * sizeof(chunk[y][0]));
	}

	if (!params->use_stoerfunktion)
	{
		/* Initialize first and last element of a matrix row */
		for (unsigned y = 0; y < params->num_rows; y++)
		{
			chunk[y][0] = 1.0 - (h * y);
			chunk[y][row_len - 1] = h * y;
		}

		if (rank == 0)
		{
			/* initialize top row */
			for (unsigned x = 0; x < row_len; x++)
			{
				chunk[0][x] = 1.0 - (h * x);
				LOG_INIT("chunk[0][%u] is %lf\n", params->rank, x, chunk[0][x]);
			}
			chunk[0][row_len - 1] = 0.0;
		}
		else if (rank == num_tasks - 1)
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

double compute(double** const src, double** dest, const Params* params)
{
	double maxresiduum = 0;
	const double h = 1.0 / (double)params->row_len;
	const double pih = PI * h;
	const double fpisin = 0.25 * TWO_PI_SQUARE * h * h;//ist das *h*h Absicht?
	//#define LOG_COMP(...) fprintf(stderr, "%i: comp: " __VA_ARGS__);
	#define LOG_COMP(...)

	LOG_COMP("computing %u lines [%u:%u]\n", rank, params->num_rows - 2, params->first_row + 1, params->first_row + params->num_rows - 2);

	/* over all rows */
	#pragma omp parallel for
	for (unsigned y = 1; y < params->num_rows - 1; y++)
	{
		double fpisin_i = 0.0;
		LOG_COMP("calculating row %u\n", rank, y);

		if (params->use_stoerfunktion)
		{
			fpisin_i = fpisin * sin(pih * (double)y);
		}

		/* over all columns, excluding borders */
		for (unsigned x = 1; x < (params->row_len - 1); x++)
		{
			double star = 0.25 * (/* left*/ src[y-1][x] + /* right */ src[y+1][x] +
								  /*bottom*/ src[y][x-1] + /* up */ src[y][x+1]);

			if (params->use_stoerfunktion)
			{
				star += fpisin_i * sin(pih * (double)x);
			}

			//if (options->termination == TERM_PREC || term_iteration == 1)
			{
				double residuum = src[y][x] - star;
				residuum = (residuum < 0) ? -residuum : residuum;

				#pragma omp critical
				{
					maxresiduum = (residuum < maxresiduum) ? maxresiduum : residuum;
				}
			}
			dest[y][x] = star;
		}
	}
	return maxresiduum;
}


#define TAG_COMM_UP 1
#define TAG_COMM_DOWN 2
void communicate_jacobi(double** current, const Params* params)
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
		MPI_Isend(current[1],                     row_length, MPI_DOUBLE, prev_rank, TAG_COMM_UP, MPI_COMM_WORLD,   &top_request[0]);
		MPI_Irecv(current[0],                     row_length, MPI_DOUBLE, prev_rank, TAG_COMM_DOWN, MPI_COMM_WORLD, &bottom_request[0]);
	}

	/*
	 * Communicate with bottom:
	 * send one before bottom row and receive bottom row.
	 */
	if (rank != num_tasks - 1) /* last rank doesn't send or receive anything from below */
	{
		 MPI_Isend(current[params->num_rows - 2], row_length, MPI_DOUBLE, next_rank, TAG_COMM_DOWN, MPI_COMM_WORLD, &bottom_request[1]);
		 MPI_Irecv(current[params->num_rows - 1], row_length, MPI_DOUBLE, next_rank, TAG_COMM_UP, MPI_COMM_WORLD,   &top_request[1]);
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
	//LOG("%u: first row: %u, num_rows: %u\n", params->rank, params->first_row, params->num_rows);
}

/* ************************************************************************ */
/* allocateMemory ()                                                        */
/* allocates memory and quits if there was a memory allocation problem      */
/* ************************************************************************ */
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


void display(double** chunk, const Params* params, Display_Params* dp, double max_residuum, unsigned num_iterations)
{
	//#define LOG_DISP(...) fprintf(stderr, "%i: disp: " __VA_ARGS__);
	#define LOG_DISP(...)
	FILE * out = stderr;
	const int rank = params->rank;

	if (rank == 0)
	{
		unsigned rank0_last_row = params->first_row + params->num_rows - 1;
		LOG_DISP("rank0_last_row %u\n", rank, rank0_last_row);
		double* row = chunk[0]; // first row to display. also reuse to receive next row
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
				row = chunk[row_index];
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
				fprintf(out, "%4.4f ", row[x]);
			}
			fprintf(out, "\n");
		}
		fprintf(out, "max residuum is: %lf, number of iterations done: %u\n", max_residuum, num_iterations);
	}
	else
	{
		unsigned begin = params->first_row + 2;
		while (((begin - dp->y0) % dp->advance) != 0)
		{
			begin++;
		}
		const unsigned end = MIN(dp->y1, (params->first_row + params->num_rows));

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
				MPI_Send(chunk[local_index], params->row_len, MPI_DOUBLE, 0, row_index, MPI_COMM_WORLD);
			}
		}
	}
}

void print_params(const Params* params)
{
	printf("%i: num_tasks : %i\n", params->rank, params->num_tasks);
	printf("%i: num_chunks : %u\n", params->rank, params->num_chunks);
	printf("%i: use_stoerfunktion : %i\n", params->rank, params->use_stoerfunktion);
	printf("%i: target_residuum : %lf\n", params->rank, params->target_residuum);
	printf("%i: target_iteration : %u\n", params->rank, params->target_iteration);
	printf("%i: interlines : %u\n", params->rank, params->interlines);
	printf("%i: row_len: %u\n", params->rank, params->row_len);
	printf("%i: first_row : %u\n", params->rank, params->first_row);
	printf("%i: num_rows : %u\n", params->rank, params->num_rows);
	printf("%i: chunks mem usage: : %6.3lf kb\n", params->rank, ((double)(params->num_chunks * params->num_rows * params->row_len * sizeof(double)) / (double)(1024)));
#ifdef _OPENMP
	printf("%i: omp_num_threads : %u\n", params->rank, params->omp_num_threads);
#endif

}

/**
 * parses input and stores it in params.
 */
void parse_cmd_line(int argc, char** argv, Params* params)
{
	if (argc < 3)
	{
		printf("Arguments error\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if (sscanf(argv[1], "%u", &params->interlines) != 1)
	{
		printf("expecting 1st argument of unsigned type\n");
		MPI_Abort(MPI_COMM_WORLD, 2);
	}
	params->row_len = (params->interlines * 8) + 9;

	if ((sscanf(argv[2], "%u", &params->use_stoerfunktion) != 1) ||
		(params->use_stoerfunktion > 1))
	{
		printf("expecting 2nd argument of boolean type (0 or 1)\n");
		MPI_Abort(MPI_COMM_WORLD, 3);
	}

	if (sscanf(argv[3], "%u", &params->target_iteration) != 1)
	{
		printf("expecting 3nd argument of unsigned type\n");
		MPI_Abort(MPI_COMM_WORLD, 4);
	}

	unsigned stop_after_precision_reached = (params->target_iteration == 0);
	if (stop_after_precision_reached)
	{
		if (argc < 4)
		{
			printf("expecting max residuum as a 4rth argument\n");
			MPI_Abort(MPI_COMM_WORLD, 5);
		}

		if (sscanf(argv[4], "%lf", &params->target_residuum) != 1)
		{
			printf("max residuum should be a double\n");
			MPI_Abort(MPI_COMM_WORLD, 6);
		}
	}

#ifdef _OPENMP
	params->omp_num_threads = omp_get_thread_num();
#else
	params->omp_num_threads = 0;
#endif
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


	int rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS)
	{
		fprintf(stderr, "MPI_Init failed!\n");
		exit(EXIT_FAILURE);
	}
	Params params;
	params.num_chunks = 2;
	MPI_Comm_rank(MPI_COMM_WORLD, &params.rank);
	MPI_Comm_size(MPI_COMM_WORLD, &params.num_tasks);

	parse_cmd_line(argc, argv, &params);
	calculate_row_offsets(&params);

	//print_params(&params);
	//MPI_Barrier(MPI_COMM_WORLD);
	unsigned stop_after_precision_reached = (params.target_iteration == 0);


	allocate_matrix_chunks(&params);
	for (unsigned i = 0; i < params.num_chunks; ++i)
	{
		init_chunk(params.chunk[i], &params);
	}

	double reduced_max_residuum;
	unsigned curr = 0;
	unsigned next = (curr + 1) % params.num_chunks;
	unsigned iter = 0;
	for (;; iter++)
	{
		communicate_jacobi(params.chunk[curr], &params);
		double max_residuum = compute(params.chunk[curr], params.chunk[next], &params);
		if (stop_after_precision_reached)
		{
			/* all prozesses need max residuum */
			MPI_Allreduce(&max_residuum, &reduced_max_residuum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			if (reduced_max_residuum < params.target_residuum)
			{
				break;
			}
		}
		else if (iter >= params.target_iteration)
		{
			/* only rank 0 needs max residuum */
			MPI_Reduce(&max_residuum, &reduced_max_residuum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
			break;
		}
		curr = next;
		next = (next + 1) % params.num_chunks;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	fflush(stdout);
	fflush(stderr);
	usleep(500);

	// show full matrix
	// Display_Params dp = { 0, params.row_len, 0, params.row_len, 1, 1};

	Display_Params dp = { 1, params.row_len - 1, 1, params.row_len - 1, params.interlines + 1, 0 };
	display(params.chunk[curr], &params, &dp, reduced_max_residuum, iter);

#ifdef SHOW_FIRST2_LINES
	MPI_Barrier(MPI_COMM_WORLD);

	fflush(stdout);
	fflush(stderr);
	MPI_Barrier(MPI_COMM_WORLD);
	usleep(200 + 100 * params.rank);

	if (1)
	{
		for (unsigned y = 0; y < 2; y ++)
		{
			fprintf(stdout, "%i: y: %u | ", params.rank, params.first_row + y);
			for (unsigned x = dp.x0; x < dp.x1; x += dp.advance)
			{
				fprintf(stdout, "%4.4f ", params.chunk[curr][y][x]);
			}
			fprintf(stdout, "\n");
		}
	}
	fflush(stdout);
	fflush(stderr);
#endif
	clean_up(&params);
	MPI_Finalize();
	return EXIT_SUCCESS;
}

