import timeit

import pickle

from Solver import solution
from test import apply_overrides_and_instantiate

import timeit
import pickle
import concurrent.futures
from tqdm import tqdm
from statistics import mean
import multiprocessing
import git
from datetime import datetime

def run_solution(test_nr: int):
    # Load constants overrides
    with open(f"tests/test{test_nr}.pkl", "rb") as f:
        overrides = pickle.load(f)
    C = apply_overrides_and_instantiate(overrides)

    _ = solution(C)

def load_all():
    overrides = []
    # Load constants overrides
    for test_nr in range(4):
        with open(f"tests/test{test_nr}.pkl", "rb") as f:
            override = pickle.load(f)
        overrides.append(override)
    return overrides

def run_all(overrides):
    for override in overrides:
        C = apply_overrides_and_instantiate(override)
        solution(C)

def timeit_worker(number_of_runs):
    """
    Executes timeit.repeat() for a specified number of runs.
    This function is executed by each process for each "task".
    """
    stmt = "run_all(overrides)"
    setup = "overrides = load_all()" # Simplified, globals provides the functions

    # Note: We do NOT set timeit.template here to avoid garbled tqdm output

    times = [min(timeit.repeat(
        stmt=stmt,
        setup=setup,
        number=number_of_runs,
        repeat=1,  # Repeat a few times per task for stability
        globals=globals() # CRITICAL: Pass globals to the timeit context
    ))]
    return times

def one_core_timer(iters):
    timeit.template = """
def inner(_it, _timer{init}):
    from tqdm import tqdm
    {setup}
    _t0 = _timer()
    for _i in tqdm(_it, total=_it.__length_hint__()):
        {stmt}
    _t1 = _timer()
    return _t1 - _t0
"""

    stmt = """
from profiler_timeit import run_all
run_all(overrides)
"""

    setup = """
from profiler_timeit import load_all
overrides = load_all()
"""

    t = timeit.Timer(stmt=stmt, setup=setup)
    dur = t.timeit(number=iters) / iters
    return dur

def one_core_main(iters):
    dur = one_core_timer(iters)

    import git
    from datetime import datetime

    with open("profiler.txt", "a") as log:
        s = f"@{git.Repo().head.object.hexsha[:6]}{'~D' if git.Repo().is_dirty() else '  '} \t[{datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}] M({dur:.8f}), {iters}\n"
        print(f"Mean: {dur:8f}")
        log.write(s)

def multi_core_main():
    # --- Setup Parallel Execution ---
    NUM_PROCESSES = multiprocessing.cpu_count()

    # --- Hybrid "Chunking" Strategy ---
    # We create more tasks than cores for a smoother progress bar,
    # but keep RUNS_PER_TASK large enough to minimize overhead.
    RUNS_PER_TASK = 50

    # Number of runs to do across all cores
    TOTAL_RUNS = RUNS_PER_TASK * 8 * 4# Your original 'iters'


    # Calculate total tasks, handling remainders
    TOTAL_TASKS = TOTAL_RUNS // RUNS_PER_TASK
    remainder = TOTAL_RUNS % RUNS_PER_TASK

    # Create the list of workloads (arguments for timeit_worker)
    workloads = [RUNS_PER_TASK] * TOTAL_TASKS
    if remainder > 0:
        workloads.append(remainder)
        TOTAL_TASKS += 1

    print(f"Starting parallel benchmark on {NUM_PROCESSES} cores...")
    print(f"Total Iterations: {TOTAL_RUNS}")
    print(f"Total Tasks (progress bar): {TOTAL_TASKS}")
    print(f"Runs per Task (approx): {RUNS_PER_TASK}")

    all_times = [] # To store all timing results

    # --- 3. Run using ProcessPoolExecutor ---
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_PROCESSES) as executor:

        # executor.map runs timeit_worker for each item in workloads
        # tqdm tracks the completion of all tasks
        results_from_map = list(tqdm(
            executor.map(timeit_worker, workloads),
            total=len(workloads),
            desc="Benchmarking Tasks"
        ))

    # --- 4. Process Results ---

    # results_from_map is a list of lists, e.g., [[t1, t2], [t3, t4], ...]
    # We need to get the average *per-iteration* time.
    avg_iter_times = []

    # 'workloads' has the 'number=' (RUNS_PER_TASK) for each result
    for (task_times, number_of_runs) in zip(results_from_map, workloads):
        # task_times is a list, e.g., [0.5, 0.49, 0.51] (from repeat=3)
        # Each 't' is the total time for 'number_of_runs' iterations.
        for t in task_times:
            avg_iter_times.append(t / number_of_runs)

    # 'dur' is the mean of all measured single-iteration averages
    iters = TOTAL_RUNS * NUM_PROCESSES # For the log file
    dur = mean(avg_iter_times)

    # --- 5. Display and Log ---
    print("\n--- Results Summary ---")
    print(f"Total logical cores used: {NUM_PROCESSES}")
    print(f"Total timeit repetitions collected: {len(avg_iter_times)}")
    print(f"Total rnu_all calls: {NUM_PROCESSES * sum(workloads)}")
    print(f"Average time per 'run_all' call: {dur:.8f} seconds")

    with open("profiler.txt", "a") as log:
        log.write(f"@{git.Repo().head.object.hexsha[:6]}{'~D' if git.Repo().is_dirty() else '  '} \t[{datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}] M({dur:.8f}), {iters}\n")

if __name__ == "__main__":
    one_core_main(iters=100)
