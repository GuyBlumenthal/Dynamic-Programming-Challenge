
from test import run_solution
from timer import time_reset, time_return

import numpy as np

run_times = []

iterations = 200

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


from datetime import datetime
import git

with open("profiler.txt", "a") as log:
    log.write(f"@{git.Repo().head.object.hexsha[:6]}{'~D' if git.Repo().is_dirty() else '  '} \t[{datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}] M({np.mean(run_times):.8f}) V({np.var(run_times):.8f})\n")
