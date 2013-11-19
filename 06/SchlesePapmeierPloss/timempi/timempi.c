#include "timempi.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#define __USE_XOPEN2K
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#define COMM_BUFF_SZ 250
#define TAG_HOSTNAME_TIMESTAMP 1
#define TAG_DIE_DIE_DIE 2
#define SLAVE_LOG(...) fprintf(stdout, "slave: "__VA_ARGS__);
#define MASTER_LOG(...) fprintf(stdout, "master: "__VA_ARGS__);

#define SLAVE_WAIT_FOR_RELEASE

void print_info()
{
	int version, subversion, num_tasks, len;
	char hostname[MPI_MAX_PROCESSOR_NAME];

	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
	MPI_Get_processor_name(hostname, &len);

	MPI_Get_version (&version,&subversion);
	fprintf (stdout, "Using MPI v%i.%i, number of tasks = %d, running on \"%s\"\n",
					version, subversion, num_tasks, hostname);
}

void master(int rc)
{
	int num_tasks;
	char buff[COMM_BUFF_SZ];
	MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
	if (num_tasks == 1)
	{
		fprintf(stderr, "only 1 proces is active, exiting\n");
		return;
	}
	const int num_slave_tasks = num_tasks - 1;

	char* received[num_slave_tasks];
	memset(received, 0, sizeof(received));

	MPI_Status status;
	/* has
    int MPI_SOURCE;
    int MPI_TAG;
    int MPI_ERROR;
 */
	// expecting messages from rank 1+
	for (int num_msgs = 1;; num_msgs++)
	{
		MPI_Recv(&buff,           /* message buffer */
				 sizeof(buff),                 /* one data item */
				 MPI_SIGNED_CHAR,        /* of type double real */
				 MPI_ANY_SOURCE,    /* receive from any sender */
				 TAG_HOSTNAME_TIMESTAMP, // use MPI_ANY_TAG for any type of message
				 MPI_COMM_WORLD,    /* default communicator */
				 &status);          /* info about the received message */

		int id_offset = status.MPI_SOURCE - 1;

		MASTER_LOG("received message from process %i\n", status.MPI_SOURCE);

		int msg_size;
		MPI_Get_count(&status, MPI_SIGNED_CHAR, &msg_size);

		if (id_offset >= num_tasks)
		{
			fprintf(stderr, "task rank is out of range");
			MPI_Abort(MPI_COMM_WORLD, rc); // Bye bye cruel world
		}
		else if (received[id_offset] != 0)
		{
			fprintf(stderr, "2 messages from the same rank");
			MPI_Abort(MPI_COMM_WORLD, rc); // Bye bye cruel world
		}
		else
		{
			MASTER_LOG("received message of size %i\n", msg_size);

			received[id_offset] = malloc(msg_size + 1);
			memcpy(received[id_offset], buff, msg_size);
			*(received[id_offset] + msg_size) = 0;
		}

		if (num_msgs == num_slave_tasks)
		{
			MASTER_LOG("evaluating all messages\n");
			for (int i = 0; i < num_slave_tasks; ++i)
			{
				if (received[i] == 0)
				{
					fprintf(stderr, "haven't received from process %i\n", i + 1);
					MPI_Abort(MPI_COMM_WORLD, rc); // Bye bye cruel world
				}
			}

			MASTER_LOG("received all messages\n");
			/* received all messages, now printing them out: */
			for (int i = 0; i < num_slave_tasks; ++i)
			{
				fprintf(stdout, "%s \n", received[i]);
				free(received[i]);
			}

#ifdef SLAVE_WAIT_FOR_RELEASE
			for (int i = 0; i < num_slave_tasks; ++i)
			{
				MPI_Send(received,					/* message buffer */
						 0,					/* number of data items */
						 MPI_SIGNED_CHAR,	/* data item is an integer */
						 (int)i + 1,		/* destination process rank */
						 TAG_DIE_DIE_DIE,	/* user chosen message tag */
						 MPI_COMM_WORLD);   /* default communicator */
			}
#endif
			break;
		}
	}
}


int fill_timestamp(char* buff, unsigned buffsz)
{
	char            fmt[64];
	struct timeval  tv;
	struct tm       *tm;
	int result = 0;
	if (gettimeofday(&tv, NULL))
	{
		perror("gettimeofday");
	}
	else
	{
		if((tm = localtime(&tv.tv_sec)) != NULL)
		{
			strftime(fmt, sizeof fmt, "%Y-%m-%d %H:%M:%S.%%06u", tm);
			result = snprintf(buff, buffsz, fmt, tv.tv_usec);
		}
		else
		{
			perror("localtime");
		}
	}
    return result;
}

void slave(int rc)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	SLAVE_LOG("%i: starting\n", rank);

	char buff[COMM_BUFF_SZ];
	if (gethostname(buff, sizeof buff))
	{
		perror("gethostname");
		MPI_Abort(MPI_COMM_WORLD, rc); // Bye bye cruel world
	}
	buff[(sizeof buff) - 1] = 0; // just in case
	unsigned buff_len = strlen(buff);
	buff[buff_len++] = ':';
	buff[buff_len++] = ' ';
	int result = fill_timestamp(&buff[buff_len], COMM_BUFF_SZ - buff_len);
	if (result <= 0)
	{
		printf("fill_timestamp");
		MPI_Abort(MPI_COMM_WORLD, rc); // Bye bye cruel world
	}
	buff_len += result;

	SLAVE_LOG("%i: sending %s\n", rank, buff);
    MPI_Send(buff,						/* message buffer */
    				buff_len,						/* number of data items */
             MPI_SIGNED_CHAR,			/* data item is an integer */
             0,							/* destination process rank */
             TAG_HOSTNAME_TIMESTAMP,	/* user chosen message tag */
             MPI_COMM_WORLD);   		/* default communicator */

#ifdef SLAVE_WAIT_FOR_RELEASE
    MPI_Status status;
	MPI_Recv(&buff,           /* message buffer */
			 sizeof(buff),                 /* one data item */
			 MPI_SIGNED_CHAR,        /* of type double real */
			 0,    					/* receive from rank 0 */
			 TAG_DIE_DIE_DIE , // use MPI_ANY_TAG for any type of message
			 MPI_COMM_WORLD,    /* default communicator */
			 &status);          /* info about the received message */
#endif
}


int main(int argc, char* argv[])
{
	int rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS)
	{
		fprintf (stderr, "MPI_Init failed!\n");
		MPI_Abort(MPI_COMM_WORLD, rc); // Bye bye cruel world
	}

	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if (rank == 0)
	{
		print_info();
		master(rc);
	}
	else
	{
		slave(rc);
	}

	printf("Rang %i beendet jetzt!\n", rank);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	return 0;
}
