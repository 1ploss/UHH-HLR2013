#pragma once
/*
 * common.h
 *
 *  Created on: Dec 20, 2013
 *      Author: jcd
 */

#include <stdint.h>

typedef struct
{
	enum
	{
		JACOBI = 0,
		GAUSS = 1
	} method;

	unsigned use_stoerfunktion;

	uint64_t target_iteration;
	double target_residuum;
	int rank;
	int prev_rank;
	int next_rank;
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

typedef struct
{
	double max_residuum;
	double** chunk;
	uint64_t num_iterations;
} Result;

double compute(double** const src, double** dest, const Params* params, unsigned first_row, unsigned num_rows);

static inline int is_first_rank(const Params* params)
{
	return params->rank == 0;
}

static inline int is_last_rank(const Params* params)
{
	return params->rank == params->num_tasks - 1;
}


#ifndef PI
#define PI           3.14159265358979323846
#endif
#define TWO_PI_SQUARE (2.0 * PI * PI)

#define ARRAY_SIZE(A) (sizeof(A) / sizeof(A[0]))
inline static double min(double a, double b) { return (a < b) ? a : b; }
inline static double max(double a, double b) { return (a > b) ? a : b; }

#define TAG_COMM_CMD 0
#define TAG_COMM_ROW_UP (1U << 8)
#define TAG_COMM_ROW_DOWN (2U << 8)

