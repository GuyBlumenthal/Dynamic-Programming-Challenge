import time
from functools import wraps, partial
from typing import Callable
import numpy as np
timings = []


import line_profiler
def profile(A):
    # return line_profiler.profile(A)
    return A

def time_def(func: Callable = None) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        # storing time before function execution
        begin = time.time()
        r = func(*args, **kwargs) # exec the actual function
        end = time.time()

        elapsed = end - begin

        timings.append((func.__name__, elapsed))

        return r

    return wrapper

def time_evaluate():
    total = f"{sum([t[1] for t in timings]):.8f}"
    mean = f"{np.mean([t[1] for t in timings]):.8f}"

    print(f"Total TIME {total}")
    print(f"Averg TIME {mean}")

    for timing in timings:
        print(f"\tDEF {timing[0]} : TIME {timing[1]:.8f}")

    from datetime import datetime
    import git

    with open("log.txt", "a") as log:
        log.write(f"@{git.Repo().head.object.hexsha[:6]}{'~D' if git.Repo().is_dirty() else '  '} \t[{datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}] {total}\n")

def time_return():
    return sum([t[1] for t in timings])

def time_reset():
    global timings
    timings = list()