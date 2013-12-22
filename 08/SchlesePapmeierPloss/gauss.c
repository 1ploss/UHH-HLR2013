/*
 * gauss.c
 *
 *  Created on: Dec 20, 2013
 *      Author: jcd
 */

#include "gauss.h"
#include "common.h"
#include <mpi.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct
{
	MPI_Request max_res_request;
	MPI_Request max_res_post;
	MPI_Request zeroth_row_request;
	MPI_Request first_row_post;
	MPI_Request last_row_request;
	MPI_Request second2last_row_post;

	int max_res_requested;
	int max_res_posted;
	int zeroth_row_requested;
	int first_row_posted;
	int last_row_requested;
	int second2last_row_posted;

} Row_Comm;

void row_comm_init(Row_Comm* r)
{
	r->max_res_requested = 0;
	r->max_res_posted = 0;
	r->zeroth_row_requested = 0;
	r->first_row_posted = 0;
	r->last_row_requested = 0;
	r->second2last_row_posted = 0;
}

void row_comm_sync(Row_Comm* r)
{
	MPI_Status status;
	if (r->max_res_posted)
	{
		MPI_Wait(&r->max_res_post, &status);
	}
	if (r->max_res_request)
	{
		MPI_Wait(&r->max_res_request, &status);
	}
	if (r->zeroth_row_requested)
	{
		MPI_Wait(&r->zeroth_row_request, &status);
	}
	if (r->first_row_posted)
	{
		MPI_Wait(&r->first_row_post, &status);
	}
	if (r->last_row_requested)
	{
		MPI_Wait(&r->last_row_request, &status);
	}
	if (r->second2last_row_posted)
	{
		MPI_Wait(&r->second2last_row_post, &status);
	}
}

void row_comm_cancel(Row_Comm* r)
{
	if (r->max_res_posted)
	{
		MPI_Cancel(&r->max_res_post);
	}
	if (r->max_res_request)
	{
		MPI_Cancel(&r->max_res_request);
	}
	if (r->zeroth_row_requested)
	{
		MPI_Cancel(&r->zeroth_row_request);
	}
	if (r->first_row_posted)
	{
		MPI_Cancel(&r->first_row_post);
	}
	if (r->last_row_requested)
	{
		MPI_Cancel(&r->last_row_request);
	}
	if (r->second2last_row_posted)
	{
		MPI_Cancel(&r->second2last_row_post);
	}
}

#define LOG_REQ(...) fprintf(stderr, __VA_ARGS__);
/**
 * This function will post receive request for the 0th row
 */
static inline void gauss_post_recv_0th_row(double** chunk, const Params* params, Row_Comm* rc)
{
	LOG_REQ("%i: post_recv_0th_row to %i\n", params->rank, params->prev_rank);
	assert(rc->zeroth_row_requested == 0);
	MPI_Irecv(chunk[0], params->row_len, MPI_DOUBLE, params->prev_rank, TAG_COMM_ROW_DOWN, MPI_COMM_WORLD, &rc->zeroth_row_request);
	rc->zeroth_row_requested = 1;
}

static inline void gauss_sync_recv_0th_row(const Params* params, Row_Comm* rc)
{
	MPI_Status status;
	LOG_REQ("%i: sync_recv_0th_row\n", params->rank);
	assert(rc->zeroth_row_requested);
	MPI_Wait(&rc->zeroth_row_request, &status);
	rc->zeroth_row_requested = 0;
}

static inline void gauss_post_recv_last_row(double** chunk, const Params* params, Row_Comm* rc)
{
	LOG_REQ("%i: post_recv_last_row to %i\n", params->rank, params->next_rank);
	assert(rc->last_row_requested == 0);
	MPI_Irecv(chunk[params->num_rows - 1], params->row_len, MPI_DOUBLE, params->next_rank, TAG_COMM_ROW_UP, MPI_COMM_WORLD, &rc->last_row_request);
	rc->last_row_requested = 1;
}

static inline void gauss_sync_recv_last_row(const Params* params, Row_Comm* rc)
{
	MPI_Status status;
	LOG_REQ("%i: sync_recv_last_row\n", params->rank);
	assert(rc->last_row_requested);
	MPI_Wait(&rc->last_row_request, &status);
	rc->last_row_requested = 0;
}

static inline void gauss_post_send_1st_row(double** chunk, const Params* params, Row_Comm* rc)
{
	LOG_REQ("%i: post_send_1th to %i\n", params->rank, params->prev_rank);
	assert(rc->first_row_posted == 0);
	assert(params->num_rows >= 1);
	MPI_Isend(chunk[1], params->row_len, MPI_DOUBLE, params->prev_rank, TAG_COMM_ROW_UP, MPI_COMM_WORLD, &rc->first_row_post);
	rc->first_row_posted = 1;
}

static inline void gauss_sync_send_1st_row(const Params* params, Row_Comm* rc)
{
	MPI_Status status;
	LOG_REQ("%i: sync_send_1st_row\n", params->rank);
	assert(rc->first_row_posted);
	MPI_Wait(&rc->first_row_post, &status);
	rc->first_row_posted = 0;
}

static inline void gauss_post_send_2nd_to_last_row(double** chunk, const Params* params, Row_Comm* rc)
{
	LOG_REQ("%i: post_send_2nd_to_last to %i\n", params->rank, params->next_rank);
	assert(rc->second2last_row_posted == 0);
	assert(params->num_rows >= 2);
	MPI_Isend(chunk[params->num_rows - 2], params->row_len, MPI_DOUBLE, params->next_rank, TAG_COMM_ROW_DOWN, MPI_COMM_WORLD, &rc->second2last_row_post);
	rc->second2last_row_posted = 1;
}

static inline void gauss_sync_send_2nd_to_last_row(const Params* params, Row_Comm* rc)
{
	MPI_Status status;
	LOG_REQ("%i: sync_send_2nd_to_last_row\n", params->rank);
	assert(rc->second2last_row_posted);
	MPI_Wait(&rc->second2last_row_post, &status);
	rc->second2last_row_posted = 0;
}

static inline void gauss_post_send_last_residuum(double* max_residuum, const Params* params, Row_Comm* rc)
{
	LOG_REQ("%i: post_send_last_residuum to %i\n", params->rank, params->next_rank);
	assert(rc->max_res_posted == 0);
	MPI_Isend(max_residuum, 1, MPI_DOUBLE, params->next_rank, TAG_COMM_PASS_MAX_RESIDUUM, MPI_COMM_WORLD, &rc->max_res_post);
	rc->max_res_posted = 1;
}


static inline void gauss_sync_send_last_residuum(const Params* params, Row_Comm* rc)
{
	MPI_Status status;
	LOG_REQ("%i: sync_send_last_residuum\n", params->rank);
	assert(rc->max_res_posted);
	MPI_Wait(&rc->max_res_post, &status);
	rc->max_res_posted = 0;
}


static inline void gauss_post_request_last_residuum(double* max_residuum, const Params* params, Row_Comm* rc)
{
	LOG_REQ("%i: post_request_last_residuum to %i\n", params->rank, params->next_rank);
	assert(rc->max_res_requested == 0);
	MPI_Isend(max_residuum, 1, MPI_DOUBLE, params->next_rank, TAG_COMM_PASS_MAX_RESIDUUM, MPI_COMM_WORLD, &rc->max_res_request);
	rc->max_res_requested = 1;
}


static inline void gauss_sync_request_last_residuum(const Params* params, Row_Comm* rc)
{
	MPI_Status status;
	LOG_REQ("%i: sync_request_last_residuum\n", params->rank);
	assert(rc->max_res_requested);
	MPI_Wait(&rc->max_res_request, &status);
	rc->max_res_requested = 0;
}

double gauss_calc_comm(double** chunk, const Params* params, Row_Comm* rc)
{
	unsigned curr_row = 1;
	double max_residuum = 0;

	if (!is_first_rank(params))
	{
		gauss_sync_recv_0th_row(params, rc);
		gauss_sync_send_1st_row(params, rc);

		LOG_REQ("%i: calc bottom %u, +1 rows\n", params->rank, curr_row);
		max_residuum = max(compute(chunk, chunk, params, curr_row++, 1), max_residuum);

		gauss_post_send_1st_row(chunk, params, rc); // send out 1st row
		gauss_post_recv_0th_row(chunk, params, rc); // speculatively post receive next 0th row
	}

	unsigned num_middle_rows = params->num_rows - curr_row - (2 * !is_last_rank(params));
	if (num_middle_rows < params->num_rows)
	{
		LOG_REQ("%i: calc mid %u, +%u rows\n", params->rank, curr_row, num_middle_rows);
		max_residuum = max(compute(chunk, chunk, params, curr_row, num_middle_rows), max_residuum); // do the middle part
		curr_row += num_middle_rows;
	}

	// ensure we have the last row to be able to process second to last row
	if (!is_last_rank(params))
	{
		gauss_sync_recv_last_row(params, rc); // ensure the las post was received
		gauss_sync_send_2nd_to_last_row(params, rc); // ensure the last post succeded
		//LOG_REQ("%i: curr_row: %u, params->num_rows - 2: %u\n", params->rank, curr_row, params->num_rows - 2);
		assert(curr_row == params->num_rows - 2);

		LOG_REQ("%i: calc bottom %u, +1 rows\n", params->rank, curr_row);
		max_residuum = max(compute(chunk, chunk, params, curr_row++, 1),max_residuum); // do the second to last row

		gauss_post_recv_last_row(chunk, params, rc); // speculatively post receive next last row
		gauss_post_send_2nd_to_last_row(chunk, params, rc);	// send out recently computed second to last row
	}
	return max_residuum;
}

void do_gauss(const Params* params, Result* result)
{
#define LOG_GAUSS(...) fprintf(stderr, __VA_ARGS__);
//#define LOG_GAUSS(...)
	result->max_residuum = 0;
	unsigned stop_after_precision_reached = (params->target_iteration == 0);
	double** chunk = params->chunk[0];

	Row_Comm rc;
	row_comm_init(&rc);

	if (!is_first_rank(params)) /* last rank doesn't have any rank below */
	{
		gauss_post_recv_0th_row(chunk, params, &rc);
		gauss_post_send_1st_row(chunk, params, &rc);

	}
	if (!is_last_rank(params)) /* last rank doesn't have any rank below */
	{
		gauss_post_recv_last_row(chunk, params, &rc);
		gauss_post_send_2nd_to_last_row(chunk, params, &rc);
	}

	LOG_GAUSS("%i: main loop\n", params->rank)

	uint64_t target_iter = params->target_iteration;
	for (result->num_iterations = 0; stop_after_precision_reached || result->num_iterations < target_iter; result->num_iterations++)
	{
		LOG_GAUSS("%i:----------------[iter %lu]-------------------\n", params->rank, result->num_iterations);
		double recv_max_residuum = 0;
		unsigned is_not_rank0_in_0th_iter = !(is_first_rank(params) && result->num_iterations == 0);
		if (is_not_rank0_in_0th_iter)
		{
			gauss_post_request_last_residuum(&recv_max_residuum, params, &rc);
		}

		double max_residuum = gauss_calc_comm(chunk, params, &rc);

		if (is_not_rank0_in_0th_iter)
		{
			gauss_sync_request_last_residuum(params, &rc);
			max_residuum = max(recv_max_residuum, max_residuum);
		}

		if (rc.max_res_posted)
		{
			gauss_sync_send_last_residuum(params, &rc);
		}

		result->max_residuum = max(max_residuum, result->max_residuum);

		if (stop_after_precision_reached)
		{
			LOG_GAUSS("%i: max_res: recv: %lf, max_residuum: %lf, result->max_residuum: %lf \n",
							params->rank, recv_max_residuum, max_residuum, result->max_residuum);
			if (result->max_residuum <= params->target_residuum)
			{
				stop_after_precision_reached = 0;
				target_iter = result->num_iterations + (params->num_tasks - params->rank) ;
				LOG_GAUSS("%i: switch to iteration mode: num_iter: %lu, target: %lu\n",
								params->rank, result->num_iterations, target_iter);
			}
		}

		// if not the last iteration of the last rank
		if (!(is_last_rank(params) && (result->num_iterations + 1 == params->target_iteration)))
		{
			gauss_post_send_last_residuum(&result->max_residuum, params, &rc);
		}

	}

	LOG_GAUSS("%i: done in %lu iterations\n", params->rank, result->num_iterations);
	if (!is_first_rank(params) && rc.zeroth_row_requested)
	{
		LOG_GAUSS("%i: cancelling zeroth_row_requested\n", params->rank);
		MPI_Cancel(&rc.zeroth_row_request);
	}
	if (!is_last_rank(params) && rc.last_row_requested)
	{
		LOG_GAUSS("%i: cancelling last_row_request\n", params->rank);
		MPI_Cancel(&rc.last_row_request);
	}
	if (rc.max_res_posted)
	{
		LOG_GAUSS("%i: cancelling max_res_posted\n", params->rank);
		MPI_Cancel(&rc.max_res_post);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	LOG_GAUSS("%i: cancel done\n", params->rank);
	result->chunk = chunk;
}

