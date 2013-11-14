/*
 ============================================================================
 Name        : queue.c
 Author      : 
 Version     :
 Copyright   : Your copyright notice
 Description : Hello World in C, Ansi-style
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <semaphore.h>
#include "thread_pool.h"

// compiler barrier: http://gcc.gnu.org/ml/gcc/2003-04/msg01180.html
#define compiler_barrier __asm__ __volatile__ ("")
#define IS_DONE_COOKIE 0xDEADBEEFU

typedef struct
{
	thread_job_t			function;
	THREAD_ARGUMENT_TYPE	argument;
	THREAD_RESULT_TYPE		result;
	unsigned				stop;
	volatile uint32_t		is_done; // set by the thread, cleared by the pool
	uint32_t				is_result_picked_up; // used only by the pool itself. Set when the result is picked up.
	pthread_t				id;
	sem_t					start_sem;
	sem_t *					thread_done_sem;
} thread_t;

struct thread_pool_t
{
	thread_t*		threads;
	size_t			num_threads;
	sem_t			thread_done_sem;
};

static void* thread_runner(void* data)
{
	thread_t* thread = data;

	do
	{
		sem_wait(&thread->start_sem);
		if (thread->is_done == 0)
		{
			if (thread->function)
			{
				thread->result = thread->function(thread->argument);
			}

			compiler_barrier;
			/**
			 * no need to use __sync_bool_compare_and_swap because
			 * a partial write will not be valid.
			 */
			thread->is_done = IS_DONE_COOKIE;
			// sem_post is a memory barrier	http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap04.html#tag_04_11
			sem_post(thread->thread_done_sem);
		}
	}
	while (thread->stop == 0);
	return 0;
}

static void thread_init(struct thread_pool_t* pool, thread_t* thread)
{
    int result;
	pthread_attr_t attr;

	thread->function = 0;
	thread->argument = 0;
	thread->result = 0;
    thread->stop = 0;
    thread->is_done = 0;
    thread->is_result_picked_up = 1;

    thread->thread_done_sem = &pool->thread_done_sem;

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    result = pthread_create(&thread->id, &attr, thread_runner, thread);
    if (result)
    {
    	perror("pthread_create failed");
    	exit(EXIT_FAILURE);
    }
    pthread_attr_destroy(&attr);

    result = sem_init(&thread->start_sem, 0, 0);
    if (result)
	{
		perror("sem_init failed");
		exit(EXIT_FAILURE);
	}

}

static void thread_deinit(thread_t* thread)
{
	thread->stop = 1;
	sem_post(&thread->start_sem);
	pthread_join(thread->id, &thread->result);
	int result = sem_destroy(&thread->start_sem);
	if (result)
	{
		perror("thread_deinit: sem_destroy failed");
		exit(EXIT_FAILURE);
	}
}

static inline void* alloc(size_t size, const char* extra)
{
	void* mem = malloc(size);
	if (!mem)
	{
		printf("alloc: %s: failed to allocate %lu bytes of memory", extra, size);
		exit(EXIT_FAILURE);
	}
	return mem;
}

struct thread_pool_t* thread_pool_create(unsigned num_threads)
{
	struct thread_pool_t* pool = alloc(sizeof (struct thread_pool_t), "struct thread_pool_t");
	pool->threads = alloc(num_threads * sizeof (thread_t), "threads");
	pool->num_threads = num_threads;
    int result = sem_init(&pool->thread_done_sem, 0, num_threads);
    if (result)
	{
		perror("sem_init failed");
		exit(EXIT_FAILURE);
	}

	for (unsigned i = 0; i < num_threads; ++i)
	{
		thread_init(pool, &pool->threads[i]);
	}

	return pool;
}

void thread_pool_destroy(struct thread_pool_t* pool)
{
	for (unsigned i = 0; i < pool->num_threads; ++i)
	{
		thread_deinit(&pool->threads[i]);
	}
	free(pool->threads);
	free(pool);
}

// can only be run from a single thread
int thread_pool_try_submit_job(struct thread_pool_t* pool, thread_job_t function, THREAD_ARGUMENT_TYPE argument)
{
	for (unsigned i = 0; i < pool->num_threads; ++i)
	{
		thread_t* thread = &pool->threads[i];
		if (thread->is_result_picked_up == 1)
		{
			thread->function = function;
			thread->argument = argument;
			thread->is_result_picked_up = 0;

			compiler_barrier;
			thread->is_done = 0;
			// sem_post is a memory barrier
			sem_post(&thread->start_sem);
			return 0;
		}
	}
	return -1;
}

// can only be run from a single thread
THREAD_RESULT_TYPE thread_pool_retrieve_result(struct thread_pool_t* pool)
{
	while (1)
	{
		sem_wait(&pool->thread_done_sem); // wait until a thread is done
		for (unsigned i = 0; i < pool->num_threads; ++i)
		{
			thread_t* thread = &pool->threads[i];
			if (thread->is_done == IS_DONE_COOKIE)
			{
				compiler_barrier;
				if (thread->is_result_picked_up == 0)
				{
					THREAD_RESULT_TYPE result = thread->result;
					thread->is_result_picked_up = 1;
					compiler_barrier;
					return result;
				}
			}
		}
	}
	return 0;
}

// Uncomment this to test
//#define TEST_THREAD_POOL
#ifdef TEST_THREAD_POOL

void* test_func(void* arg)
{
	for (unsigned i = 0; i < 2; i++)
	{
		printf("ping from thread %lu\n", (long unsigned)arg);
		sleep(1);
	}
	printf("thread %lu done\n", (long unsigned)arg);
	return arg;
}

int main(void)
{
	const unsigned num_threads = 5;

	printf("creating pool for %u threads\n", num_threads);
	thread_pool_handle pool = thread_pool_create(num_threads);
	for (unsigned i = 0; i < num_threads; ++i)
	{
		if (thread_pool_try_submit_job(pool, test_func, (void*)(unsigned long)i))
		{
			printf("error: failed to submit job nr. %u\n", i);
			thread_pool_destroy(pool);
			exit(EXIT_FAILURE);
		}
	}

	if (thread_pool_try_submit_job(pool, test_func, (void*)(unsigned long)12345) != -1)
	{
		printf("error: extra job didn't fail\n");
		thread_pool_destroy(pool);
		exit(EXIT_FAILURE);
	}

	printf("ok, submitted jobs, retreiving results\n");

	unsigned found_results[num_threads];
	memset(found_results, 0, sizeof(found_results));
	for (unsigned i = 0; i < num_threads; ++i)
	{
		const long unsigned r = (const long unsigned)thread_pool_retrieve_result(pool);
		if (r < num_threads)
		{
			found_results[r] = 1;
		}
		else
		{
			printf("result of the %u retrieval (%lu) is out of range\n", i, r);
			thread_pool_destroy(pool);
			exit (EXIT_FAILURE);
		}
	}

	printf("ok, got all results\n");
	for (unsigned i = 0; i < num_threads; ++i)
	{
		if (found_results[i] != 1)
		{
			printf("haven't found %u in results\n", i);
			thread_pool_destroy(pool);
			exit (EXIT_FAILURE);
		}
	}

	printf("ok, destroying pool\n");
	thread_pool_destroy(pool);
	printf("done\n");
	return EXIT_SUCCESS;
}

#endif
