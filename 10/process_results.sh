#!/usr/bin/python3
results_dir = "/home/jcd/uni/hlr/10-Job-Scripte/out"
outpu_dir = "/home/jcd/out"

class Test:
	def __init__(self, test_filename):
		self.complete = False
		self.test_filename = test_filename
		tokens = test_filename.split("_")
		
	def __str__(self):
		if self.complete:
			s = ("test_filename: " + self.test_filename + "\n"
			     "num_tasks: " + str(self.num_tasks) + "\n"
			     "method: " + self.method + "\n"
			     "max_residuum: " + str(self.max_residuum) + "\n"
			     "num_iterations: " + str(self.num_iterations) + "\n"
			     "time: " + str(self.time)
			     )
			return s
		else:
			return "<incomplete test>"
		
		
tests = []

import glob
test_filenames = glob.glob(results_dir + "/*.out")
for test_filename in test_filenames:
	
	with open(test_filename) as test_file:
		
		curr_test = Test(test_filename)
		for line in test_file:
			#print("line: " + line);
			tokens = line.split()
			if tokens[1] == "num_tasks":
				curr_test.num_tasks = int(tokens[3])
			elif tokens[1] == "method":
				curr_test.method = tokens[3]
			elif tokens[0] == "max":
				curr_test.max_residuum = float(tokens[3][:-1])
				curr_test.num_iterations = int(tokens[8])
			elif tokens[0] == "time":
				curr_test.time = float(tokens[2])
				curr_test.complete = True;
				tests.append(curr_test)
				curr_test = Test(test_filename)


def mean(tests):
	max_residuum = 0;
	time = 0.0
	for test in tests:
		max_residuum = max_residuum + test.max_residuum
		time = time + test.time
	result_test = Test(tests[0].test_filename)
	result_test.max_residuum = max_residuum / len(tests)
	result_test.time = time / len(tests)
	result_test.num_tasks = tests[0].num_tasks
	result_test.method = tests[0].method
	result_test.num_iterations = tests[0].num_iterations
	result_test.complete = True
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
			tests_out.append(mean(batch))
			to_smooth = []
	return tests_out				

tests = smooth_tests(tests, 3)

for test in tests:	
	print(test)
	print("------------------\n")
	
	

