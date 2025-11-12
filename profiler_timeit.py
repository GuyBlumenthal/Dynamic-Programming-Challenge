import timeit

import pickle

from Solver import solution
from test import apply_overrides_and_instantiate

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


if __name__ == "__main__":
    iters = 500
    dur = t.timeit(number=iters) / iters
    print(dur)

    import git
    from datetime import datetime

    with open("profiler.txt", "a") as log:
        log.write(f"@{git.Repo().head.object.hexsha[:6]}{'~D' if git.Repo().is_dirty() else '  '} \t[{datetime.now().strftime('%d/%m/%Y, %H:%M:%S')}] M({dur:.8f}), {iters}\n")

