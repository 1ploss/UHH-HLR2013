#pragma once
/*
 * thread_pool.h
 *
 *  Created on: Nov 13, 2013
 *      Author: jcd
 */

/**
 * Signature of a thread job.
 */
#define THREAD_RESULT_TYPE double
#define THREAD_ARGUMENT_TYPE void*
typedef THREAD_RESULT_TYPE (*thread_job_t)(THREAD_ARGUMENT_TYPE);

/**
 * Handle to the thread pool
 */
typedef struct thread_pool_t* thread_pool_handle;

/**
 * Creates a thread pool.
 *
 * @param num_threads number of threads in the pool.
 */
thread_pool_handle thread_pool_create(unsigned num_threads);

/**
 * Destroys the pool.
 *
 *  @param pool the pool to use
 */
void thread_pool_destroy(thread_pool_handle pool);

/**
 *  Retrieve any result. This function will block if there no results and some threads are still running.
 * This function is not reentrant and can only be run from a single thread, the same thread_pool_submit_job is run from.
 *
 *  @param pool the pool to use
 *  @return result or 0 if there are no results.
 */
THREAD_RESULT_TYPE thread_pool_retrieve_result(thread_pool_handle pool);

/**
 * Submit a job to this thread pool. This function will not block because the intention is that the
 * thread_pool_retrieve_result would block and after it is done, thread_pool_submit_job will always succeed.
 * This function is not reentrant and can only be run from a single thread.
 *
 *  @param pool the pool to use
 *  @param function the job to run
 *  @param argument job argument
 *
 *  @return 0 if the submission was successful, else -1.
 *
 */
int thread_pool_try_submit_job(thread_pool_handle pool, thread_job_t function, void* argument);

