import time
from functools import wraps, partial
from typing import Callable

timings = []


def time_def(func: Callable = None) -> Callable:
    """
    This function is a decorator
    It will run the function the decorator is applied to, and return its result
    Printing the execution time

    eg.

    @time_it
    def time_max(A):
        return max(A)

    time_max([1,2,3,4,6]) # will return max value and print execution time

    """

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
    print(f"Total TIME {total}")

    for timing in timings:
        print(f"\tDEF {timing[0]} : TIME {timing[1]:.8f}")

    from datetime import datetime
    import git

    with open("log.txt", "a") as log:
        log.write(f"@{git.Repo().head.object.hexsha[:6]}{'~D' if git.Repo().is_dirty() else ''} \t[{datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}] {total}\n")