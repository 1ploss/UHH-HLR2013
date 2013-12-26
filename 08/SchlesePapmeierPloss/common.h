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
		GAUSS = 1,
		JACOBI = 2
	} method; // calculating method

	unsigned use_stoerfunktion; // if non 0, 2 * pi^2 * sin(pi * x) * sin(pi * y) is used.

	uint64_t target_iteration; // in the iteration mode: the iteration when the programm will stop calculations
	double target_residuum; // if the mode is to run until precision is reached, then stop the program if the max_residuum
	                        // of current iteration was less than this value
	int rank; // this task mpi rank
	int prev_rank; // next mpi task rank
	int next_rank; // previous mpi task rank
	int num_tasks; // number of mpi tasks

	unsigned interlines; // number of rows between displayed rows
	unsigned row_len; // length of a row == interlines * 8 + 9
	unsigned first_row; // global row number, also sometimes called world y
	unsigned num_rows; // number of rows this task owns (and keeps in memory)

	unsigned num_threads;

	unsigned num_chunks; // ho much matrices (or pieces of those) we use
	double* mem_pool; // memory, that holds the chunks
	double*** chunk; // chunks array (addressed as chunk[matrix_nr][row][column])

	// we precomputed some values here
	double h;
	double fpisin;
	double pih;
} Params;

/**
 * Result data is accumulated in this structure.
 */
typedef struct
{
	double max_residuum;
	double** chunk;
	uint64_t num_iterations;
} Result;

/**
 * Computes a part of the new matrix.
 */
double compute(double** const src, double** dest, const Params* params, unsigned first_row, unsigned num_rows);

/**
 * Returns true if this is the first rank.
 */
static inline int is_first_rank(const Params* params)
{
	return params->rank == 0;
}

/**
 * Returns true if this is the last rank.
 */
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

