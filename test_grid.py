import pickle
from pprint import pp
from itertools import product
import os

import numpy as np

from test import apply_overrides_and_instantiate

from line_profiler import LineProfiler
from tqdm import tqdm

from typing import Generator, Dict, Any

from timeit import Timer

# All imported files for profiling
from Solver import SOLUTION_FUNCTIONS, solution
from ComputeExpectedStageCosts import compute_expected_stage_cost_solver
from ComputeTransitionProbabilities import compute_transition_probabilities_sparse
from utils import CustomStateSpace

from test import main as challenge_main

import json

TEST_PARAM_RANGES = {
    "x": [6, 12],

    "Y": [6, 12],

    "V_max": [1, 2, 3],

    "U_no_flap": [0],

    "U_weak": [1, 3],

    "U_strong": [1, 3],

    "V_dev": [1, 2, 3],

    "D_min": [3, 4, 5],

    "G": [1, 3],

    "g": [1, 2],

    "lam_weak": [0.3, 0.5],

    "lam_strong": [0.5, 0.8],
}

TEST_PARAM_RANGES_SHORT = {
    "x": [6, 12],

    "Y": [6, 12],

    "V_max": [1, 2, 3],

    "U_no_flap": [0],

    "U_weak": [1],

    "U_strong": [3],

    "V_dev": [1, 3],

    "D_min": [3, 5],

    "G": [1, 3],

    "g": [1],

    "lam_weak": [0.3],

    "lam_strong": [0.5, 0.8],
}

TEST_PARAM_RANGES_SUPER_SHORT = {
    "x": [12],

    "Y": [12],

    "V_max": [3],

    "U_no_flap": [0],

    "U_weak": [1],

    "U_strong": [3],

    "V_dev": [3],

    "D_min": [3],

    "G": [1],

    "g": [1],

    "lam_weak": [0.3],

    "lam_strong": [0.8],
}


def gen_tests():
    no_s_h = [
        # i for i in product(*TEST_PARAM_RANGES.values())
        i for i in product(*TEST_PARAM_RANGES_SHORT.values())
        # i for i in product(*TEST_PARAM_RANGES_super_SHORT.values())
    ]

    tests = []
    s_hs = []

    out_tests = []

    for test in no_s_h:
        Y = test[1]

        s_h = [
            [0, Y-1],
            [0, Y-2],
            [0, Y//2, Y],
        ]

        for opt in s_h:
            tests.append(test)
            s_hs.append(opt)


    for i, (test, s_h) in enumerate(zip(tests, s_hs)):
        test_def = {
            k: v for k, v in zip(TEST_PARAM_RANGES.keys(), test)
        }
        test_def["S_h"] = s_h

        out_tests.append(test_def)

    return out_tests

def main():
    tests = gen_tests()

    lp = LineProfiler()

    profiled_functions = [
        CustomStateSpace.build_d_recursive,
        CustomStateSpace.custom_state_space,
        compute_expected_stage_cost_solver,
        compute_transition_probabilities_sparse,
        *SOLUTION_FUNCTIONS.values(),
    ]

    for func in profiled_functions:
        lp.add_callable(func)


    setup="from main import main as challenge_main"
    stmt="challenge_main()"

    t = Timer(stmt=stmt, setup=setup)
    # dur = t.timeit(number=iters) / iters

    wrapper = lp(t.timeit)
    wrapper(100)


    # for test in tqdm(tests, desc="Test Progress"):
    #     C = apply_overrides_and_instantiate(test)
    #     try:
    #         wrapper = lp(solution)
    #         wrapper(C)
    #     except Exception as e:
    #         print(f"Failed for constant description of {test}")
    #         print(e)


    with open("tests/profile_output.txt", "w") as f:
        lp.print_stats(stream=f,stripzeros=True)

    lp.dump_stats("tests/profile_output.pkl")

    lp.get_stats()

    # Show detailed per-line information for each function.
    for (fn, lineno, name), timings in lp.get_stats().timings.items():
        if name == "solution_linear_prog_sparse":
            total_time = sum(t[2] for t in timings)
            func_hits = timings[0][1]
            unit = lp.get_stats().unit
            runtime = ((total_time * unit)/func_hits)
            json.dump([{
                "name": "runtime",
                "unit": "s",
                "value": runtime
            }], open("tests/profile_runtime.json", "w"))

if __name__ == "__main__":
    main()

