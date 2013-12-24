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

/**
 * Default command, no special meaning, calculate as normal, if iteration mode, then abort after the target
 * iteration is reached.
 */
#define CMD_NONE 0

/**
 * This command causes the task to switch from max_residuum to iteration mode.
 */
#define CMD_SWITCH_TO_ITER 1


typedef struct
{
	unsigned	cmd;
	double		max_residuum;
} Cmd;

/**
 * A handy struct to keep all asynchronous requests in one place.
 */
typedef struct
{
	Cmd* send_cmd;
	Cmd* recv_cmd;
	MPI_Request* cmd_recv_post;
	MPI_Request* cmd_send_post;
	int*		cmd_recv_posted;
	int*		cmd_send_posted;
	MPI_Request zeroth_row_request;
	MPI_Request first_row_post;
	MPI_Request last_row_request;
	MPI_Request second2last_row_post;

	int zeroth_row_requested;
	int first_row_posted;
	int last_row_requested;
	int second2last_row_posted;

	unsigned num_cmd_buffers;
	unsigned curr_send_nr;
	unsigned curr_recv_nr;
} Row_Comm;

/**
 * Initializes Row_Comm.
 */
static void row_comm_init(const Params* params, Row_Comm* r)
{
	r->num_cmd_buffers = (unsigned)params->num_tasks;
	r->curr_send_nr = 0;
	r->curr_recv_nr = 0;
	r->send_cmd = malloc(r->num_cmd_buffers * sizeof(Cmd));
	r->recv_cmd = malloc(r->num_cmd_buffers * sizeof(Cmd));
	r->cmd_recv_post = malloc(r->num_cmd_buffers * sizeof(MPI_Request));
	r->cmd_send_post = malloc(r->num_cmd_buffers * sizeof(MPI_Request));
	r->cmd_recv_posted = malloc(r->num_cmd_buffers * sizeof(int));
	r->cmd_send_posted = malloc(r->num_cmd_buffers * sizeof(int));
	if (!(r->cmd_recv_post && r->cmd_recv_posted && r->cmd_send_post && r->cmd_send_posted &&
		  r->send_cmd && r->recv_cmd))
	{
		fprintf(stderr, "malloc failed to allocate memory");
		MPI_Abort(MPI_COMM_WORLD, 20);
	}

	for (unsigned i = 0; i < r->num_cmd_buffers; ++i)
	{
		r->recv_cmd[i].cmd = CMD_NONE;
		r->recv_cmd[i].max_residuum = 0;
		r->send_cmd[i].cmd = CMD_NONE;
		r->send_cmd[i].max_residuum = 0;
		r->cmd_recv_post[i] = MPI_REQUEST_NULL;
		r->cmd_recv_posted[i] = 0;
		r->cmd_send_post[i] = MPI_REQUEST_NULL;
		r->cmd_send_posted[i] = 0;
	}

	r->zeroth_row_request = MPI_REQUEST_NULL;
	r->first_row_post = MPI_REQUEST_NULL;
	r->last_row_request = MPI_REQUEST_NULL;
	r->second2last_row_post = MPI_REQUEST_NULL;

	r->zeroth_row_requested = 0;
	r->first_row_posted = 0;
	r->last_row_requested = 0;
	r->second2last_row_posted = 0;
}

static void row_comm_cleanup(Row_Comm* r)
{
	free(r->send_cmd);
	free(r->recv_cmd);
	free(r->cmd_recv_post);
	free(r->cmd_send_post);
	free(r->cmd_recv_posted);
	free(r->cmd_send_posted);
}

unsigned prev_send_nr(Row_Comm* r)
{
	return (r->curr_send_nr + r->num_cmd_buffers - 1) % r->num_cmd_buffers;
}

unsigned next_send_nr(Row_Comm* r)
{
	return (r->curr_send_nr + 1) % r->num_cmd_buffers;
}

unsigned prev_recv_nr(Row_Comm* r)
{
	return (r->curr_recv_nr + r->num_cmd_buffers - 1) % r->num_cmd_buffers;
}

unsigned next_recv_nr(Row_Comm* r)
{
	return (r->curr_recv_nr + 1) % r->num_cmd_buffers;
}


//#define LOG_REQ(...) fprintf(stderr, __VA_ARGS__);
#define LOG_REQ(...) do { (void)iter; } while (0)

/**
 * This function will post receive request for the 0th row
 */
static inline void gauss_post_recv_0th_row(double** chunk, uint64_t iter, const Params* params, Row_Comm* rc)
{
	LOG_REQ("%i:%lu: post_recv_0th_row to %i\n", params->rank, iter, params->prev_rank);
	assert(rc->zeroth_row_requested == 0);
	MPI_Irecv(chunk[0], params->row_len, MPI_DOUBLE, params->prev_rank, TAG_COMM_ROW_DOWN, MPI_COMM_WORLD, &rc->zeroth_row_request);
	rc->zeroth_row_requested = 1;
}

/**
 * This function will wait until the post from the previous function is complete.
 */
static inline void gauss_sync_recv_0th_row(Row_Comm* rc, uint64_t iter)
{
	MPI_Status status;
	LOG_REQ("%i:%lu: sync_recv_0th_row\n", params->rank, iter);
	assert(rc->zeroth_row_requested);
	MPI_Wait(&rc->zeroth_row_request, &status);
	rc->zeroth_row_requested = 0;
}

/**
 * This function will post receive request for the last row
 */
static inline void gauss_post_recv_last_row(double** chunk, uint64_t iter, const Params* params, Row_Comm* rc)
{
	LOG_REQ("%i:%lu: post_recv_last_row to %i\n", params->rank, iter, params->next_rank);
	assert(rc->last_row_requested == 0);
	MPI_Irecv(chunk[params->num_rows - 1], params->row_len, MPI_DOUBLE, params->next_rank, TAG_COMM_ROW_UP, MPI_COMM_WORLD, &rc->last_row_request);
	rc->last_row_requested = 1;
}

/**
 * This function will wait until the post from the previous function is complete.
 */
static inline void gauss_sync_recv_last_row(Row_Comm* rc, uint64_t iter)
{
	MPI_Status status;
	LOG_REQ("%i:%lu: sync_recv_last_row\n", params->rank, iter);
	assert(rc->last_row_requested);
	MPI_Wait(&rc->last_row_request, &status);
	rc->last_row_requested = 0;
}

/**
 * This function will post sending request for the 1st row
 */
static inline void gauss_post_send_1st_row(double** chunk, uint64_t iter, const Params* params, Row_Comm* rc)
{
	LOG_REQ("%i:%lu: post_send_1th to %i\n", params->rank, iter, params->prev_rank);
	assert(rc->first_row_posted == 0);
	assert(params->num_rows >= 1);
	MPI_Isend(chunk[1], params->row_len, MPI_DOUBLE, params->prev_rank, TAG_COMM_ROW_UP, MPI_COMM_WORLD, &rc->first_row_post);
	rc->first_row_posted = 1;
}

/**
 * This function will wait until the post from the previous function is complete.
 */
static inline void gauss_sync_send_1st_row(Row_Comm* rc, uint64_t iter)
{
	MPI_Status status;
	LOG_REQ("%i:%lu: sync_send_1st_row\n", params->rank, iter);
	assert(rc->first_row_posted);
	MPI_Wait(&rc->first_row_post, &status);
	rc->first_row_posted = 0;
}

/**
 * This function will post sending request for the 2nd to last row
 */
static inline void gauss_post_send_2nd_to_last_row(double** chunk, uint64_t iter, const Params* params, Row_Comm* rc)
{
	LOG_REQ("%i:%lu: post_send_2nd_to_last to %i\n", params->rank, iter, params->next_rank);
	assert(rc->second2last_row_posted == 0);
	assert(params->num_rows >= 2);
	MPI_Isend(chunk[params->num_rows - 2], params->row_len, MPI_DOUBLE, params->next_rank, TAG_COMM_ROW_DOWN, MPI_COMM_WORLD, &rc->second2last_row_post);
	rc->second2last_row_posted = 1;
}

/**
 * This function will wait until the post from the previous function is complete.
 */
static inline void gauss_sync_send_2nd_to_last_row(Row_Comm* rc, uint64_t iter)
{
	MPI_Status status;
	LOG_REQ("%i:%lu: sync_send_2nd_to_last_row\n", params->rank, iter);
	assert(rc->second2last_row_posted);
	MPI_Wait(&rc->second2last_row_post, &status);
	rc->second2last_row_posted = 0;
}

//#define LOG_CMD(...) fprintf(stderr, __VA_ARGS__);
#define LOG_CMD(...) do { (void)params; (void)iter; } while (0)

/**
 * This function will post sending request for a command
 */
static inline void gauss_post_send_cmd(const Params* params, Row_Comm* rc, uint64_t iter)
{
	LOG_CMD("%i:%lu: post_send_cmd to %i\n", params->rank, iter, params->next_rank);
	assert(rc->cmd_send_posted[rc->curr_send_nr] == 0);
	MPI_Isend(&rc->send_cmd[rc->curr_send_nr], sizeof(Cmd), MPI_BYTE, params->next_rank, TAG_COMM_CMD + rc->curr_send_nr, MPI_COMM_WORLD, &rc->cmd_send_post[rc->curr_send_nr]);
	rc->cmd_send_posted[rc->curr_send_nr] = 1;
}

/**
 * This function will wait until the post from the previous function is complete.
 */
static inline void gauss_sync_send_cmd(const Params* params, Row_Comm* rc, uint64_t iter)
{
	MPI_Status status;
	LOG_CMD("%i:%lu: sync_send_cmd\n", params->rank, iter);
	assert(rc->cmd_send_posted[rc->curr_send_nr]);
	MPI_Wait(&rc->cmd_send_post[rc->curr_send_nr], &status);
	rc->cmd_send_posted[rc->curr_send_nr] = 0;
}

/**
 * This function will post receive request for a command
 */
static inline void gauss_post_recv_cmd(const Params* params, Row_Comm* rc, uint64_t iter)
{
	LOG_CMD("%i:%lu: post_recv_cmd from %i\n", params->rank, iter, params->prev_rank);
	assert(rc->cmd_recv_posted[rc->curr_recv_nr] == 0);
	MPI_Irecv((uint8_t*)&rc->recv_cmd[rc->curr_recv_nr], sizeof(Cmd), MPI_BYTE, params->prev_rank, TAG_COMM_CMD  + rc->curr_recv_nr, MPI_COMM_WORLD, &rc->cmd_recv_post[rc->curr_recv_nr]);
	rc->cmd_recv_posted[rc->curr_recv_nr] = 1;
}

/**
 * This function will wait until the post from the previous function is complete.
 */
static inline void gauss_sync_recv_cmd(const Params* params, Row_Comm* rc, uint64_t iter)
{
	MPI_Status status;
	LOG_CMD("%i:%lu: sync_recv_cmd from %i\n", params->rank, iter, params->prev_rank);
	assert(rc->cmd_recv_posted[rc->curr_recv_nr]);
	MPI_Wait(&rc->cmd_recv_post[rc->curr_recv_nr], &status);
	rc->cmd_recv_posted[rc->curr_recv_nr] = 0;
}

/**
 * This function combines calculation and communication of the Gauss method.
 */
static double gauss_calc_comm(double** chunk, uint64_t iter, const Params* params, Row_Comm* rc)
{
//#define LOG_CALC(...) fprintf(stderr, __VA_ARGS__);
#define LOG_CALC(...)
	unsigned curr_row = 1;
	double max_residuum = 0;

	// first rank doesn't have to communicate its top rows.
	if (!is_first_rank(params))
	{
		// ensure last speculative receive and send posts are finished.
		gauss_sync_recv_0th_row(rc, iter);
		gauss_sync_send_1st_row(rc, iter);

		LOG_CALC("%i: calc top %u, +1 rows\n", params->rank, curr_row);
		max_residuum = max(compute(chunk, chunk, params, curr_row++, 1), max_residuum); // compute 1st row

		gauss_post_send_1st_row(chunk, iter, params, rc); // send out 1st row
		gauss_post_recv_0th_row(chunk, iter, params, rc); // speculatively post receive next 0th row
	}

	// the middle doesn't have to be send anywhere, just batch process it.
	unsigned num_middle_rows = params->num_rows - curr_row - 1 - !is_last_rank(params);
	if (num_middle_rows < params->num_rows)
	{
		LOG_CALC("%i: calc mid %u, +%u rows\n", params->rank, curr_row, num_middle_rows);
		max_residuum = max(compute(chunk, chunk, params, curr_row, num_middle_rows), max_residuum); // do the middle part
		curr_row += num_middle_rows;
	}

	// last rank doesn't have to communicate bottom rows.
	if (!is_last_rank(params))
	{
		// ensure last speculative receive and send posts are finished.
		gauss_sync_recv_last_row(rc, iter);
		gauss_sync_send_2nd_to_last_row(rc, iter);

		assert(curr_row == params->num_rows - 2 && "curr_row calculation went wrong somewhere before");

		LOG_CALC("%i: calc bottom %u, +1 rows\n", params->rank, curr_row);
		max_residuum = max(compute(chunk, chunk, params, curr_row++, 1),max_residuum); // compute 2nd to last row

		gauss_post_recv_last_row(chunk, iter, params, rc); // speculatively post receive next last row
		gauss_post_send_2nd_to_last_row(chunk, iter, params, rc);	// send out recently computed second to last row
	}
	return max_residuum;
}

//#define LOG_GAUSS(...) fprintf(stderr, __VA_ARGS__);
#define LOG_GAUSS(...)

/**
 * Initial step in the gauss implementation. Jumpstarts the command chain by sending default command to rank 0.
 * Also post send/receive message requests, which will be needed later in gaus_calc_comm()
 */
static void gauss_initial_post(const Params* params, Row_Comm* rc)
{
	double** chunk = params->chunk[0];

	/**
	 * Initiating command chain. Start num_cmd_buffers semi-parallel
	 */
	if (is_last_rank(params))
	{
		for (unsigned i = 0; i < rc->num_cmd_buffers; i++)
		{
			gauss_post_send_cmd(params, rc, (uint64_t)-1);
			rc->curr_send_nr = next_send_nr(rc);
		}
	}

	for (unsigned i = 0; i < rc->num_cmd_buffers; i++)
	{
		gauss_post_recv_cmd(params, rc, (uint64_t)-1);
		rc->curr_recv_nr = next_recv_nr(rc);
	}

	if (!is_first_rank(params)) /* last rank doesn't have any rank below */
	{
		gauss_post_recv_0th_row(chunk, (uint64_t)-1, params, rc);
		gauss_post_send_1st_row(chunk, (uint64_t)-1, params, rc);

	}
	if (!is_last_rank(params)) /* last rank doesn't have any rank below */
	{
		gauss_post_recv_last_row(chunk, (uint64_t)-1, params, rc);
		gauss_post_send_2nd_to_last_row(chunk, (uint64_t)-1, params, rc);
	}
}

/**
 * This is the last step of the Gauss implementation. After the last execution of gaus_calc_comm, there will be
 * a number of speculative requests impossible to complete. So we will cancel them here.
 */
static void gauss_cancel_speculative_posts(const Params* params, Row_Comm* rc)
{
	if (!is_first_rank(params) && rc->zeroth_row_requested)
	{
		LOG_GAUSS("%i: cancelling zeroth_row_requested\n", params->rank);
		MPI_Cancel(&rc->zeroth_row_request);
	}
	if (!is_last_rank(params) && rc->last_row_requested)
	{
		LOG_GAUSS("%i: cancelling last_row_request\n", params->rank);
		MPI_Cancel(&rc->last_row_request);
	}

	for (unsigned i = 0; i < rc->num_cmd_buffers; i++)
	{
		unsigned send_idx = (rc->curr_send_nr + i) % rc-> num_cmd_buffers;
		if (rc->cmd_send_posted[send_idx])
		{
			LOG_GAUSS("%i: cancelling max_res_posted\n", params->rank);
			MPI_Cancel(&rc->cmd_send_post[send_idx]);
		}

		unsigned recv_idx = (rc->curr_recv_nr + i) % rc-> num_cmd_buffers;
		if (rc->cmd_recv_posted[recv_idx])
		{
			LOG_GAUSS("%i: cancelling max_res_posted\n", params->rank);
			MPI_Cancel(&rc->cmd_recv_post[recv_idx]);
		}
	}

	LOG_GAUSS("%i: cancel done\n", params->rank);
}

/**
 * Gaus implementation top level.
 */
void do_gauss(const Params* params, Result* result)
{
	unsigned stop_after_precision_reached = (params->target_iteration == 0);
	double** chunk = params->chunk[0];
	Row_Comm rc;
	row_comm_init(params, &rc);

	gauss_initial_post(params, &rc);

	LOG_GAUSS("%i:0: main loop\n", params->rank);

	double max_residuum = 0;
	int switch_to_iter_was_sent = 0;
	uint64_t target_iter = params->target_iteration, iter = 0;
	for (iter = 0; stop_after_precision_reached || iter < target_iter; iter++)
	{
		//LOG_GAUSS("%i:%lu:-----------------------------------\n", params->rank, iter);
		if (rc.cmd_send_posted[rc.curr_send_nr])
		{
			gauss_sync_send_cmd(params, &rc, iter);
		}

		gauss_sync_recv_cmd(params, &rc, iter);

		Cmd* recv_cmd = &rc.recv_cmd[rc.curr_recv_nr];
		Cmd* send_cmd = &rc.send_cmd[rc.curr_send_nr];

		max_residuum = gauss_calc_comm(chunk, iter, params, &rc);
		max_residuum = max(max_residuum, recv_cmd->max_residuum);

		send_cmd->cmd = recv_cmd->cmd;
		send_cmd->max_residuum = max_residuum;

		if (recv_cmd->cmd == CMD_SWITCH_TO_ITER)
		{
			stop_after_precision_reached = 0;
			// num_cmd_buffers receive latency
			//target_iter = iter + rc.num_cmd_buffers * (params->num_tasks - params->rank);
			target_iter = iter + (params->num_tasks - params->rank);
			LOG_GAUSS("%i:%lu: switch to iteration mode new target_iter: %lu\n", params->rank, iter, target_iter);
		}
		if (is_last_rank(params))
		{
			send_cmd->max_residuum = 0;
			switch(recv_cmd->cmd)
			{
				default:
				case CMD_NONE:
					if (stop_after_precision_reached && max_residuum < params->target_residuum && switch_to_iter_was_sent == 0)
					{
						send_cmd->cmd = CMD_SWITCH_TO_ITER;
						switch_to_iter_was_sent = 1;
					}
					break;

				case CMD_SWITCH_TO_ITER:
					send_cmd->cmd = CMD_NONE;
					break;
			}
		}

		//LOG_GAUSS("%i:%lu: cmd %u->%u, max_residuum: %lf\n", params->rank, iter, recv_cmd->cmd, send_cmd->cmd, max_residuum);

		gauss_post_send_cmd(params, &rc, iter);
		gauss_post_recv_cmd(params, &rc, iter);

		rc.curr_send_nr = next_send_nr(&rc);
		rc.curr_recv_nr = next_recv_nr(&rc);
	}

	result->max_residuum = max_residuum;
	result->num_iterations = iter;
	result->chunk = chunk;


	LOG_GAUSS("%i: done in %lu iterations, max_residuum: %lf, doing bcast\n", params->rank, iter, result->max_residuum);

	// distribute the latest max_residuum
	MPI_Bcast(&result->max_residuum, 1, MPI_DOUBLE, (params->num_tasks - 1), MPI_COMM_WORLD);

	gauss_cancel_speculative_posts(params, &rc);
	row_comm_cleanup(&rc);
}

