from numpy import *
import multiprocessing as mp
import time

use_numpy = True  # try running with True and False
                  # notice how using built-in NumPy speedups can negate speedups from parallelization

processes = 4     # try N_CPU_CORES or N_CPU_CORES-1 (for MacBook Pro: physical, not virtual, cores)
jobs = 250        # some large number
if use_numpy: array_len = 1000000
else:         array_len = 100000

random_arrays = [random.rand(array_len) for j in range(jobs)]

def operation_with_numpy(input_array):
    input_array_copy = input_array.copy()
    new_array = input_array_copy**4 + (input_array_copy+1)**4
    return sum(new_array)

def operation_without_numpy(input_array):
    input_array_copy = input_array.copy()
    sublist1 = [val**4 for val in input_array_copy]
    sublist2 = [(val+1)**4 for val in input_array_copy]
    new_list = [sublist1[i] + sublist2[i] for i in range(len(sublist1))]
    sum = 0
    for i in range(len(sublist1)): sum += new_list[i]
    return sum

print('jobs = {0}, array_len = {1}'.format(jobs,array_len))

start = time.time()
if use_numpy: results = [operation_with_numpy(ra) for ra in random_arrays]
else:         results = [operation_without_numpy(ra) for ra in random_arrays]
end = time.time()
print('SERIAL: processes = 1, elapsed = {0:.02f} s'.format(end-start))

start = time.time()
pool = mp.Pool(processes=processes)
if use_numpy: results_pool = [pool.apply_async(operation_with_numpy,args=(ra,)) for ra in random_arrays]
else:         results_pool = [pool.apply_async(operation_without_numpy,args=(ra,)) for ra in random_arrays]
results = [pool_member.get() for pool_member in results_pool]
end = time.time()
print('PARALLEL: processes = {0}, elapsed = {1:.02f} s'.format(processes,end-start))