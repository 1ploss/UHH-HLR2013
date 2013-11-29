#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <limits.h>
#include <mpi.h>
#define DEBUG
#ifdef DEBUG
#define LOG(...) fprintf(stderr, __VA_ARGS__);
#else
#define LOG(...)
#endif


int* init(unsigned N)
{
	int *buf = malloc(sizeof(int) * N);

	srand(time(NULL));

	for (unsigned i = 0; i < N; i++)
	{
		buf[i] = rand() % 25; //do not modify %25
	}

	return buf;
}


#define TAG_INITIAL_COMM 1
#define CMD_DO_CIRCLE 2
#define CMD_DO_STOP 3

#define TAG_SEND_RECEIVE 4

#define SWAP(a, b) b = a ^ b; a = b ^ a; b = a ^ b;

/**
 * @return index of the last receiving buffer.
 */
unsigned circle(int* buffers[], unsigned buffsz)
{
	int rank, num_tasks, target;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
	int left_rank = (rank + num_tasks - 1) % num_tasks;
	int right_rank = (rank + 1) % num_tasks;
	int last_rank = num_tasks - 1;
	unsigned recv_buff_index = 0;
	unsigned send_buff_index = 1;
	MPI_Status status;

	/**
	 * Initial communication: rank[0] -> rank[N - 1] tell about the target condition
	 */
	if (rank == 0)
	{
		LOG("0: sending target: %i\n", *buffers[send_buff_index]);
		MPI_Send(buffers[send_buff_index], 1, MPI_INT, last_rank, TAG_INITIAL_COMM, MPI_COMM_WORLD);
	}
	else if (rank == last_rank)
	{
		MPI_Recv(&target, 1, MPI_INT, 0, TAG_INITIAL_COMM, MPI_COMM_WORLD, &status);
		LOG("%i: received target: %i\n", last_rank, target);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	//return;

	/**
	 * Main send/receive loop
	 */
	for (unsigned iteration = 0; iteration < 10; iteration++)
	{
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
				 left_rank,    					/* receive from rank 0 */
				 TAG_SEND_RECEIVE , // use MPI_ANY_TAG for any type of message
				 MPI_COMM_WORLD,    /* default communicator */
				 &recv_request);          /* info about the received message */

		MPI_Wait(&send_request, &status);
		MPI_Wait(&recv_request, &status);
		LOG("%i: iteration %u send recv complete\n", rank, iteration);

		/**
		 * Last rank broadcasts if the termination condition is reached.
		 */
		char dummy = CMD_DO_CIRCLE;
		if (rank == last_rank) // master
		{
			if (buffers[recv_buff_index][0] != target)
			{
				dummy = CMD_DO_STOP;
			}
		}

		MPI_Bcast(&dummy, 1, MPI_SIGNED_CHAR, last_rank, MPI_COMM_WORLD);
		LOG("%i: iteration %u: bcast complete, dummy is %u\n", rank, iteration, dummy);

		if (dummy == CMD_DO_STOP)
		{
			return recv_buff_index;
		}

		SWAP(recv_buff_index, send_buff_index)
	}
	return recv_buff_index;
}

int main(int argc, char** argv)
{
	int rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS)
	{
		fprintf (stderr, "MPI_Init failed!\n");
		exit (EXIT_FAILURE);
	}

	unsigned N;

	if (argc < 2)
	{
		printf("Arguments error\n");
		MPI_Abort(MPI_COMM_WORLD, 2);
	}

	if (sscanf(argv[1], "%u", &N) != 1)
	{
		printf("expecting 1 argument of unsigned type\n");
		MPI_Abort(MPI_COMM_WORLD, 3);
	}

	int rank, num_tasks;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);


	div_t d = div(N, num_tasks);
	unsigned chunk_size = d.quot + (d.rem != 0);
	unsigned offset = rank * chunk_size;
	if (offset < N)
	{
		if (offset + chunk_size > N)
		{
			chunk_size = N - offset;
		}

		if (rank < 0 || offset + chunk_size > (unsigned)INT_MAX)
		{
			MPI_Abort(MPI_COMM_WORLD, 4);
		}

		LOG("Rank: %i, N: %u, chunk_size %u, offset %u\n", rank, N, chunk_size, offset);

		int* buffers[2];
		buffers[0] = init(chunk_size);
		buffers[1] = init(chunk_size);

		printf("\nBEFORE\n");

		for (unsigned i = 0; i < chunk_size; i++)
		{
			printf("rank %d: %d\n", rank, buffers[1][i]);
		}

		unsigned recv_buff_index = circle(buffers, chunk_size);
		printf("\nAFTER\n");

		for (unsigned j = 0; j < chunk_size; j++)
		{
			printf("rank %d: %d\n", rank, buffers[recv_buff_index][j]);
		}

		free(buffers[0]);
		free(buffers[1]);
	}
	else
	{
		printf("rank %d: unused\n", rank);
	}

	MPI_Finalize();
	return EXIT_SUCCESS;
}
