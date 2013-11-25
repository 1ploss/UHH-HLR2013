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

#define TAG_SEND_RIGHT 1
int* circle(int* buff, unsigned buffsz, int left, int right)
{
	MPI_Send(buff,						/* message buffer */
			 buffsz,						/* number of data items */
			 MPI_INT,			/* data item is an integer */
			 right,							/* destination process rank */
			 TAG_SEND_RIGHT,	/* user chosen message tag */
			 MPI_COMM_WORLD);   		/* default communicator */
	MPI_Status status;
	MPI_Recv(buff,           /* message buffer */
			 buffsz,                 /* one data item */
			 MPI_INT,        /* of type double real */
			 left,    					/* receive from rank 0 */
			 TAG_SEND_RIGHT , // use MPI_ANY_TAG for any type of message
			 MPI_COMM_WORLD,    /* default communicator */
			 &status);          /* info about the received message */
	return buff;
}

int main(int argc, char** argv)
{
	int rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS)
	{
		fprintf (stderr, "MPI_Init failed!\n");
		MPI_Abort(MPI_COMM_WORLD, 1);
	}

	unsigned N;
	int* buf;

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

		buf = init(chunk_size);

		printf("\nBEFORE\n");

		for (unsigned i = 0; i < chunk_size; i++)
		{
			printf("rank %d: %d\n", rank, buf[i]);
		}

		int left_rank = (rank + num_tasks - 1) % num_tasks;
		int right_rank = (rank + 1) % num_tasks;

		MPI_Barrier(MPI_COMM_WORLD);
		circle(buf, chunk_size, left_rank, right_rank);
		printf("\nAFTER\n");

		for (unsigned j = 0; j < chunk_size; j++)
		{
			printf("rank %d: %d\n", rank, buf[j]);
		}

		free(buf);
	}
	else
	{
		printf("rank %d: unused\n", rank);
	}

	MPI_Finalize();
	return EXIT_SUCCESS;
}
