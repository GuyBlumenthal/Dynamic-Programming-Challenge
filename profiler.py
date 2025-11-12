
from test import run_solution
from timer import time_reset, time_return

import numpy as np

run_times = []

iterations = 100

for i in range(iterations):
    time_reset()
    for t_nr in range(4):
        run_solution(t_nr)
    if i % 20 == 0:
        print(f"Done {i} iterations")
    run_times.append(time_return())

run_times = np.array(run_times)
print(run_times)

print("Mean: ", np.mean(run_times))
print("Var: ", np.var(run_times))
