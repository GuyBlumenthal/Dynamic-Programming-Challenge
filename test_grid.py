from line_profiler import LineProfiler
import Solver

from test import apply_overrides_and_instantiate
from itertools import product

from tqdm import tqdm
import json
import time

import numpy as np
import pandas as pd

from copy import deepcopy

TEST_PARAM_RANGES = {
    "x": [6, 8, 12],

    "Y": [6, 8, 12],

    "V_max": [1, 2, 3],

    "U_no_flap": [0],

    "U_weak": [1, 3],

    "U_strong": [1, 3],

    "V_dev": [1, 2],

    "D_min": [4, 5],

    "G": [1],

    "g": [1],

    "lam_weak": [0.3, 0.5],

    "lam_strong": [0.5, 0.8],
}

def gen_tests():
    no_s_h = [
        i for i in product(*TEST_PARAM_RANGES.values())
    ]

    tests = []
    s_hs = []

    out_tests = []

    for test in no_s_h:
        Y = test[1]

        s_h = [
            [1, Y-1],
            [1, Y-2],
            [1, Y//2, Y],
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

def select_LP(K, L):
    return Solver.solver_LP

def select_PI(K, L):
    return Solver.solver_PI_No_M

def select_PI_M(K, L):
    return Solver.solver_PI_With_M

def compare_solutions(sol_A, sol_B):
    J_a, U_a = sol_A
    J_b, U_b = sol_B

    RTOL = 1e-4
    ATOL = 1e-7

    if not np.allclose(J_a, J_b, rtol=RTOL, atol=ATOL):
        print("Wrong optimal cost!")
        return False

    # if not np.array_equal(U_a, U_b):
    #     print("Policy differs from golden (may be OK if ties exist)")

    return True

def main():


    # Enable line by line profiling
    line_profile = False
    if line_profile:
        lp = LineProfiler()
        lp.add_module(Solver)
        solution = lp(Solver.solution)
    else:
        solution = Solver.solution

    # Enable logging (print statement) in the solver
    verbose_solver = False
    if not verbose_solver:
        Solver.log = lambda x: None

    # True -> Test over grid of solution parameters
    # False -> Test over test cases in test.py and main.py
    test_grid = False
    if test_grid:
        tests = gen_tests()
    else:
        import pickle
        tests = [{}]
        # tests = []
        for test_nr in range(4):
            with open(f"tests/test{test_nr}.pkl", "rb") as f:
                tests.append(pickle.load(f))

    # Define the set of selectors to run the problem on. The first two will have their solution compared
    selectors = [
        select_LP,              # Only run LP -> Use this one first to remove invalid tests quickly
        select_PI,              # Only run PI
        select_PI_M,              # Only run PI
        # Solver.select_solver,   # Default selector (Defined in Solver.py)
    ]

    # Run tests
    for test in tqdm(tests, desc="Test Progress"):

        try:
            results = []
            C = apply_overrides_and_instantiate(test)
            for selector in selectors:
                C_instance = deepcopy(C)
                Solver.select_solver = selector
                results.append(solution(C_instance))

            if not compare_solutions(results[0], results[1]):
                raise Exception(f"""
                ##################################################
                    Cost mismatch on problem {test}
                ##################################################""")
        except Exception as e:
            # This is an infinite problem or we had problem mismatch
            print(e)

    total_df = pd.DataFrame(Solver.timing_array, columns=[
        "prob_size",
        "solver",
        "total_t",
        "state_t",
        "prob_t",
        "cost_t",
        "solver_t",
    ]).sort_values(["prob_size", "solver"])

    save_output = False
    if save_output:
        total_df.to_csv("extended_testing/profiles/tests.csv")

    timing_cols = total_df.loc[:, "total_t":"solver_t"]
    mean_times_df = timing_cols.groupby(total_df.index % len(selectors)).mean()
    mean_times_df.index = [selector.__name__ for selector in selectors]

    print(total_df)
    # print(mean_times_df)

    if line_profile:
        with open(f"extended_testing/profiles/profile_{time.strftime('%H_%M_%S')}.txt", 'w') as f:
            lp.print_stats(f, stripzeros=True)

if __name__ == "__main__":
    main()