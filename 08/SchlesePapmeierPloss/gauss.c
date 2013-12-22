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

#define CMD_NONE 0
#define CMD_ASK_STOP 1
#define CMD_SWITCH_TO_ITER 2

typedef struct
{
	unsigned cmd;
	double max_residuum;
} Cmd;


typedef struct
{
	MPI_Request cmd_recv_post;
	MPI_Request cmd_send_post;
	MPI_Request zeroth_row_request;
	MPI_Request first_row_post;
	MPI_Request last_row_request;
	MPI_Request second2last_row_post;

	int cmd_recv_posted;
	int cmd_send_posted;
	int zeroth_row_requested;
	int first_row_posted;
	int last_row_requested;
	int second2last_row_posted;

} Row_Comm;

void row_comm_init(Row_Comm* r)
{
	r->cmd_recv_post = MPI_REQUEST_NULL;
	r->cmd_send_post = MPI_REQUEST_NULL;
	r->zeroth_row_request = MPI_REQUEST_NULL;
	r->first_row_post = MPI_REQUEST_NULL;
	r->last_row_request = MPI_REQUEST_NULL;
	r->second2last_row_post = MPI_REQUEST_NULL;

	r->cmd_recv_posted = 0;
	r->cmd_send_posted = 0;
	r->zeroth_row_requested = 0;
	r->first_row_posted = 0;
	r->last_row_requested = 0;
	r->second2last_row_posted = 0;
}

void row_comm_sync(Row_Comm* r)
{
	MPI_Status status;
	if (r->cmd_send_posted)
	{
		MPI_Wait(&r->cmd_send_post, &status);
	}
	if (r->cmd_recv_post)
	{
		MPI_Wait(&r->cmd_recv_post, &status);
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
	if (r->cmd_send_posted)
	{
		MPI_Cancel(&r->cmd_send_post);
	}
	if (r->cmd_recv_post)
	{
		MPI_Cancel(&r->cmd_recv_post);
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

//#define LOG_REQ(...) fprintf(stderr, __VA_ARGS__);
#define LOG_REQ(...)
/**
 * This function will post receive request for the 0th row
 */
static inline void gauss_post_recv_0th_row(double** chunk, const Params* params, Row_Comm* rc)
{
	LOG_REQ("%i:?: post_recv_0th_row to %i\n", params->rank, params->prev_rank);
	assert(rc->zeroth_row_requested == 0);
	MPI_Irecv(chunk[0], params->row_len, MPI_DOUBLE, params->prev_rank, TAG_COMM_ROW_DOWN, MPI_COMM_WORLD, &rc->zeroth_row_request);
	rc->zeroth_row_requested = 1;
}

static inline void gauss_sync_recv_0th_row(Row_Comm* rc)
{
	MPI_Status status;
	LOG_REQ("%i:?: sync_recv_0th_row\n", params->rank);
	assert(rc->zeroth_row_requested);
	MPI_Wait(&rc->zeroth_row_request, &status);
	rc->zeroth_row_requested = 0;
}

static inline void gauss_post_recv_last_row(double** chunk, const Params* params, Row_Comm* rc)
{
	LOG_REQ("%i:?: post_recv_last_row to %i\n", params->rank, params->next_rank);
	assert(rc->last_row_requested == 0);
	MPI_Irecv(chunk[params->num_rows - 1], params->row_len, MPI_DOUBLE, params->next_rank, TAG_COMM_ROW_UP, MPI_COMM_WORLD, &rc->last_row_request);
	rc->last_row_requested = 1;
}

static inline void gauss_sync_recv_last_row(Row_Comm* rc)
{
	MPI_Status status;
	LOG_REQ("%i:?: sync_recv_last_row\n", params->rank);
	assert(rc->last_row_requested);
	MPI_Wait(&rc->last_row_request, &status);
	rc->last_row_requested = 0;
}

static inline void gauss_post_send_1st_row(double** chunk, const Params* params, Row_Comm* rc)
{
	LOG_REQ("%i:?: post_send_1th to %i\n", params->rank, params->prev_rank);
	assert(rc->first_row_posted == 0);
	assert(params->num_rows >= 1);
	MPI_Isend(chunk[1], params->row_len, MPI_DOUBLE, params->prev_rank, TAG_COMM_ROW_UP, MPI_COMM_WORLD, &rc->first_row_post);
	rc->first_row_posted = 1;
}

static inline void gauss_sync_send_1st_row(Row_Comm* rc)
{
	MPI_Status status;
	LOG_REQ("%i:?: sync_send_1st_row\n", params->rank);
	assert(rc->first_row_posted);
	MPI_Wait(&rc->first_row_post, &status);
	rc->first_row_posted = 0;
}

static inline void gauss_post_send_2nd_to_last_row(double** chunk, const Params* params, Row_Comm* rc)
{
	LOG_REQ("%i:?: post_send_2nd_to_last to %i\n", params->rank, params->next_rank);
	assert(rc->second2last_row_posted == 0);
	assert(params->num_rows >= 2);
	MPI_Isend(chunk[params->num_rows - 2], params->row_len, MPI_DOUBLE, params->next_rank, TAG_COMM_ROW_DOWN, MPI_COMM_WORLD, &rc->second2last_row_post);
	rc->second2last_row_posted = 1;
}

static inline void gauss_sync_send_2nd_to_last_row(Row_Comm* rc)
{
	MPI_Status status;
	LOG_REQ("%i:?: sync_send_2nd_to_last_row\n", params->rank);
	assert(rc->second2last_row_posted);
	MPI_Wait(&rc->second2last_row_post, &status);
	rc->second2last_row_posted = 0;
}

#define LOG_CMD(...) fprintf(stderr, __VA_ARGS__);
static inline void gauss_post_send_cmd(Cmd* cmd, const Params* params, Row_Comm* rc)
{
	LOG_CMD("%i:?: post_send_cmd to %i\n", params->rank, params->next_rank);
	assert(rc->cmd_send_posted == 0);
	MPI_Isend(cmd, sizeof(Cmd), MPI_BYTE, params->next_rank, TAG_COMM_CMD, MPI_COMM_WORLD, &rc->cmd_send_post);
	rc->cmd_send_posted = 1;
}


static inline void gauss_sync_send_cmd(const Params* params, Row_Comm* rc)
{
	MPI_Status status;
	LOG_CMD("%i:?: sync_send_cmd\n", params->rank);
	assert(rc->cmd_send_posted);
	MPI_Wait(&rc->cmd_send_post, &status);
	rc->cmd_send_posted = 0;
}


static inline void gauss_post_recv_cmd(Cmd* cmd, const Params* params, Row_Comm* rc)
{
	LOG_CMD("%i:?: post_recv_cmd from %i\n", params->rank, params->prev_rank);
	assert(rc->cmd_recv_posted == 0);
	MPI_Irecv((uint8_t*)cmd, sizeof(Cmd), MPI_BYTE, params->prev_rank, TAG_COMM_CMD, MPI_COMM_WORLD, &rc->cmd_recv_post);
	rc->cmd_recv_posted = 1;
}


static inline void gauss_sync_recv_cmd(const Params* params, Row_Comm* rc)
{
	MPI_Status status;
	LOG_CMD("%i:?: sync_recv_cmd from %i\n", params->rank, params->prev_rank);
	assert(rc->cmd_recv_posted);
	MPI_Wait(&rc->cmd_recv_post, &status);
	rc->cmd_recv_posted = 0;
}

double gauss_calc_comm(double** chunk, const Params* params, Row_Comm* rc)
{
	unsigned curr_row = 1;
	double max_residuum = 0;

	if (!is_first_rank(params))
	{
		gauss_sync_recv_0th_row(rc);
		gauss_sync_send_1st_row(rc);

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
		gauss_sync_recv_last_row(rc); // ensure the las post was received
		gauss_sync_send_2nd_to_last_row(rc); // ensure the last post succeded
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
//#define LOG_MAX_RES(...) fprintf(stderr, __VA_ARGS__);
#define LOG_GAUSS(...)
//#define LOG_GAUSS(...) fprintf(stderr, __VA_ARGS__);

	result->max_residuum = 0.0;
	unsigned stop_after_precision_reached = (params->target_iteration == 0);
	double** chunk = params->chunk[0];

	Row_Comm rc;
	row_comm_init(&rc);

	Cmd recv_cmd = { CMD_NONE, 0 }, send_cmd = { CMD_NONE, 0 };

	if (is_last_rank(params))
	{
		LOG_GAUSS("%i:0: jumpstart cmd chain\n", params->rank);
		MPI_Isend(&send_cmd, sizeof(Cmd), MPI_BYTE, 0, TAG_COMM_CMD, MPI_COMM_WORLD, &rc.cmd_send_post);
		rc.cmd_send_posted = 1;
	}

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

	LOG_GAUSS("%i:0: main loop\n", params->rank);

	uint64_t target_iter = params->target_iteration, iter = 0;
	for (iter = 0; stop_after_precision_reached || iter < target_iter; iter++)
	{
		LOG_GAUSS("%i:%lu:-----------------------------------\n", params->rank, iter);

		gauss_post_recv_cmd(&recv_cmd, params, &rc);

		double max_residuum = gauss_calc_comm(chunk, params, &rc);
		gauss_sync_recv_cmd(params, &rc);
		max_residuum = max(max_residuum, recv_cmd.max_residuum);

		if (rc.cmd_send_posted)
		{
			gauss_sync_send_cmd(params, &rc);
		}

		send_cmd.cmd = recv_cmd.cmd;
		send_cmd.max_residuum = (is_last_rank(params) ? 0 : max_residuum);
		switch (recv_cmd.cmd) // check cmd
		{
			case CMD_SWITCH_TO_ITER:
				stop_after_precision_reached = 0;
				//target_iter = iter + (params->num_tasks - params->rank) ;
				//target_iter = iter + params->rank + 1;
				target_iter = iter + params->num_tasks;
				LOG_GAUSS("%i:%lu: switch to iteration mode new target_iter: %lu\n", params->rank, iter, target_iter);
				if (is_last_rank(params))
				{
					send_cmd.cmd = CMD_NONE;
				}
				break;

			case CMD_ASK_STOP:
				if (is_last_rank(params))
				{
					send_cmd.cmd = CMD_SWITCH_TO_ITER;
				}
				break;

			case CMD_NONE:
			default:
				if (stop_after_precision_reached)
				{
					if (max_residuum <= params->target_residuum)
					{
						send_cmd.cmd = is_last_rank(params) ? CMD_SWITCH_TO_ITER : CMD_ASK_STOP;
					}
				}
				break;
		}
		LOG_GAUSS("%i:%lu: cmd %u, cmd.max_residuum: %lf, our max res: %lf\n",
						params->rank, iter, send_cmd.cmd, send_cmd.max_residuum, max_residuum);

		// stopping cmd chain
		if (!(is_last_rank(params) && (iter + 1 == iter))) 		// if not the last iteration of the last rank
		{
			//LOG_MAX_RES("%i:%lu: post_send_max_residuum %lf to %i\n", params->rank, iter, result->max_residuum, params->next_rank);
			gauss_post_send_cmd(&send_cmd, params, &rc);
		}

	}

	result->num_iterations = iter;
	result->chunk = chunk;


	LOG_GAUSS("%i: done in %lu iterations, allreduce max_res\n", params->rank, result->num_iterations);


	MPI_Allreduce(&send_cmd.max_residuum, &result->max_residuum, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

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
	if (rc.cmd_send_posted)
	{
		LOG_GAUSS("%i: cancelling max_res_posted\n", params->rank);
		MPI_Cancel(&rc.cmd_send_post);
	}

	LOG_GAUSS("%i: cancel done\n", params->rank);
}

