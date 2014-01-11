#!/usr/bin/python3
import os
import glob
import copy
import numpy as np
import matplotlib.pyplot as plt


class Test:
	def __init__(self, job_name):
		self.complete = False
		self.job_name = job_name
		tokens = self.job_name.split("_")
		self.test_type = tokens[0].lower()
		self.nodes = int(tokens[3])
		self.tasks = int(tokens[4])
		self.interlines = int(tokens[5])
		N = self.interlines * 8 + 9
		self.matrix_size = N * N
		
	def __str__(self):
		if self.complete:
			s = ("job_name: " + self.job_name + "\n"
				 "test_type: " + self.test_type + "\n"
				 "nodes: " + str(self.nodes) + "\n"
				 "tasks: " + str(self.tasks) + "\n"
				 "interlines: " + str(self.interlines) + "\n"
				 "matrix_size: " + str(self.matrix_size) + "\n"				 
			     "num_tasks: " + str(self.num_tasks) + "\n"
			     "method: " + self.method + "\n"
			     "max_residuum: " + str(self.max_residuum) + "\n"
			     "num_iterations: " + str(self.num_iterations) + "\n"
			     "time: " + str(self.time)
			     )
			return s
			
	def __lt__(self, other):
		""" less than, used for sorting """
		if self.complete == False:
			return self
		elif other.complete == False:
			return other
		else:
			return self.matrix_size < other.matrix_size

def gen_tests_from_files(results_dir):
	tests = []
	job_output_filenames = glob.glob(results_dir + "/*.out")
	for job_output_filename in job_output_filenames:	
		with open(job_output_filename) as job_file:		
			job_name = os.path.splitext(os.path.basename(job_output_filename))[0]
			curr_test = Test(job_name)		
			for line in job_file:
				#print("line: " + line);
				tokens = line.split()
				if tokens[1] == "num_tasks":
					curr_test.num_tasks = int(tokens[3])
				elif tokens[1] == "method":
					curr_test.method = tokens[3].lower()
				elif tokens[0] == "max":
					curr_test.max_residuum = float(tokens[3][:-1])
					curr_test.num_iterations = int(tokens[8])
				elif tokens[0] == "time":
					curr_test.time = float(tokens[2])
					curr_test.complete = True;
					tests.append(curr_test)
					curr_test = Test(job_name)
	return tests


def mean(tests):
	max_residuum = 0;
	time = 0.0
	for test in tests:
		max_residuum = max_residuum + test.max_residuum
		time = time + test.time
	#result_test = copy.copy(tests[0])
	result_test = copy.deepcopy(tests[0])
	result_test.max_residuum = max_residuum / len(tests)
	result_test.time = time / len(tests)
	return result_test

def smooth_tests(tests_in, tests_per_job):
	i = 0
	tests_out = []
	batch = []
	for test in tests:
		batch.append(test)
		i = i + 1
		if i == tests_per_job:
			i = 0			
			m = mean(batch)
			print("appending" + str(m))
			tests_out.append(m)
			batch = []
	return tests_out				

def plot_weak_tests_matrix_size(tests, name):
	if not tests:
		raise BaseException("no tests passed to plot_tests")
	norm_matrix_size = tests[0].matrix_size
	norm_time = tests[0].time
	x = []
	y = []
	for test in tests:
		#x.append(test.matrix_size / norm_matrix_size)
		x.append(test.matrix_size) # matrix size
		y.append(norm_time / test.time) # speedup
	for i in range(0, len(x)):
		print("(" + str(x[i]) + ", " + str(y[i]) + ")")
	plt.plot(x, y, label=name)
	#plt.show()
	

results_dir = "/home/jcd/uni/hlr/10-Job-Scripte/out"
outpu_dir = "/home/jcd/out"
tests = gen_tests_from_files(results_dir)
tests = smooth_tests(tests, 3)
print(str(len(tests)))
weak_scaling_ja = []
weak_scaling_ga = []
strong_scaling = []
comm = []
for test in tests:	
	#print(test)
	#print("------------------\n")
	if test.test_type == "weak":
		if test.method == "jacobi":
			weak_scaling_ja.append(test)
		else:
			weak_scaling_ga.append(test)
	elif test.test_type == "strong":
		strong_scaling.append(test)
	elif test.test_type == "communication":
		comm.append(test)
	else:
		raise BaseException("unknown test type: " + test.test_type)

weak_scaling_ja.sort()
weak_scaling_ga.sort()
strong_scaling.sort()
comm.sort()

plt.xlabel("matrix size (nr. of elements)")
plt.ylabel("speedup")
plot_weak_tests_matrix_size(weak_scaling_ja, "Weak Scaling with Jacobi")
plot_weak_tests_matrix_size(weak_scaling_ga, "Weak Scaling with Gauss")
plt.legend()
#plt.savefig("weak_matrix_size.png")
plt.show()
plt.clf()
#plot_tests(strong_scaling)
#plot_tests(comm)



