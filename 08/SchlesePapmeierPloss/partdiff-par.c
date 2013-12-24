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




typedef struct
{
	unsigned x0;
	unsigned x1;
	unsigned y0;
	unsigned y1;
	unsigned advance;
	int		 show_next_chunk_line;
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

double compute(double** const src, double** dest, const Params* params, unsigned first_row, unsigned num_rows)
{
	//#define LOG_COMP(...) fprintf(stderr, __VA_ARGS__);
	#define LOG_COMP(...)
	double max_residuum = 0;
	// TODO: move some the following Vars to Params.
	const double h = 1.0 / (double)params->row_len;
	const double pih = PI * h;
	const double fpisin = 0.25 * TWO_PI_SQUARE * h * h;//ist das *h*h Absicht?


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
			fpisin_i = fpisin * sin(pih * (double)world_y);
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
				fprintf(out, "%4.4f ", row[x]);
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

void print_params(const Params* params)
{
	printf("%i: num_tasks : %i\n", params->rank, params->num_tasks);
	printf("%i: num_chunks : %u\n", params->rank, params->num_chunks);
	printf("%i: use_stoerfunktion : %i\n", params->rank, params->use_stoerfunktion);
	printf("%i: target_residuum : %lf\n", params->rank, params->target_residuum);
	printf("%i: target_iteration : %lu\n", params->rank, params->target_iteration);
	printf("%i: interlines : %u\n", params->rank, params->interlines);
	printf("%i: row_len: %u\n", params->rank, params->row_len);
	printf("%i: first_row : %u\n", params->rank, params->first_row);
	printf("%i: num_rows : %u\n", params->rank, params->num_rows);
	printf("%i: chunks mem usage: : %6.3lf kb\n", params->rank, ((double)(params->num_chunks * params->num_rows * params->row_len * sizeof(double)) / (double)(1024)));
#ifdef _OPENMP
	printf("%i: omp_num_threads : %u\n", params->rank, params->omp_num_threads);
#endif

}

const char* usage = "usage: patdiff-par method interlines use_stoerfunc num_iterations target_residuum\n";

/**
 * parses input and stores it in params.
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
	if (argc < 4)
	{
		printf("arguments error\n%s", usage);
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	if ((sscanf(argv[1], "%u%n", &params->method, &pos) != 1) || pos != strlen(argv[1]) ||
		(params->method > 1))
	{
		printf("expecting 1nd argument of boolean type (0 or 1)\n");
		MPI_Abort(MPI_COMM_WORLD, 2);
	}

	if (sscanf(argv[2], "%u%n", &params->interlines, &pos) != 1 ||  pos != strlen(argv[2]))
	{
		printf("expecting 2st argument of unsigned type\n");
		MPI_Abort(MPI_COMM_WORLD, 3);
	}
	params->row_len = (params->interlines * 8) + 9;

	if ((sscanf(argv[3], "%u%n", &params->use_stoerfunktion, &pos) != 1) ||  pos != strlen(argv[3]) ||
		(params->use_stoerfunktion > 1))
	{
		printf("expecting 3rd argument of boolean type (0 or 1)\n");
		MPI_Abort(MPI_COMM_WORLD, 4);
	}

	if (sscanf(argv[4], "%lu%n", &params->target_iteration, &pos) != 1 ||  pos != strlen(argv[4]))
	{
		printf("expecting 4th argument of 64 bit unsigned type\n");
		MPI_Abort(MPI_COMM_WORLD, 5);
	}

	unsigned stop_after_precision_reached = (params->target_iteration == 0);
	if (stop_after_precision_reached)
	{
		if (argc < 5)
		{
			printf("expecting max residuum as a 5th argument\n");
			MPI_Abort(MPI_COMM_WORLD, 6);
		}

		if (sscanf(argv[5], "%lf%n", &params->target_residuum, &pos) != 1 ||  pos != strlen(argv[5]))
		{
			printf("max residuum should be a floating point number\n");
			MPI_Abort(MPI_COMM_WORLD, 7);
		}

		if (!isnormal(params->target_residuum) || params->target_residuum <= 0)
		{
			printf("max residuum should be a positive floating point number\n");
			MPI_Abort(MPI_COMM_WORLD, 7);
		}
	}

#ifdef _OPENMP
	params->omp_num_threads = omp_get_thread_num();
#else
	params->omp_num_threads = 0;
#endif
	params->num_chunks = 1 + (params->method == JACOBI);
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
	MPI_Barrier(MPI_COMM_WORLD);

	allocate_matrix_chunks(&params);
	for (unsigned i = 0; i < params.num_chunks; ++i)
	{
		init_chunk(params.chunk[i], &params);
	}

	Result result;
	struct timeval start_time, end_time;
	gettimeofday(&start_time, NULL);
	if (params.method == JACOBI)
	{
		do_jacobi(&params, &result);
	}
	else
	{
		do_gauss(&params, &result);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	gettimeofday(&end_time, NULL);

	fflush(stdout);
	fflush(stderr);
	usleep(500);

	// show full matrix
	//Display_Params dp = { 0, params.row_len, 0, params.row_len, 1, 1};

	Display_Params dp = { 1, params.row_len - 1, 1, params.row_len - 1, params.interlines + 1, 0 };
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

