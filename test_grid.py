from line_profiler import LineProfiler
import Solver

from test import apply_overrides_and_instantiate
from itertools import product

from tqdm import tqdm
import json

import numpy as np
import pandas as pd

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

def compare_solutions(sol_A, sol_B):
    J_a, U_a = sol_A
    J_b, U_b = sol_B

    RTOL = 1e-4
    ATOL = 1e-7

    if not np.allclose(J_a, J_b, rtol=RTOL, atol=ATOL):
        print("Wrong optimal cost!")
        return False

    if not np.array_equal(U_a, U_b):
        print("Policy differs from golden (may be OK if ties exist)")

    return True

def main():
    tests = gen_tests()

    lp = LineProfiler()

    lp.add_module(Solver)

    # Select solution configuration
    line_profile = True
    solution = lp(Solver.solution) if line_profile else Solver.solution

    verbose_solver = False
    if not verbose_solver:
        Solver.log = lambda x: None

    # Run tests
    for test in tqdm(tests, desc="Test Progress"):

        try:
            # Run LP first as it will crash for impossible problems, PI will run for a long time
            C = apply_overrides_and_instantiate(test)
            Solver.select_solver = lambda K, L: Solver.solver_LP
            LP = solution(C)

            C = apply_overrides_and_instantiate(test)
            Solver.select_solver = lambda K, L: Solver.solver_PI
            PI = solution(C)

            if not compare_solutions(PI, LP):
                raise Exception(f"""
                ##################################################
                    Cost mismatch on problem {test}
                ##################################################""")
        except Exception as e:
            # This is an infinite problem or we had problem mismatch
            print(e)

    pd.DataFrame(Solver.timing_array, columns=[
        "prob_size",
        "solver",
        "total_t",
        "state_t",
        "prob_t",
        "cost_t",
        "solver_t",
    ]).to_csv("extended_testing/profiles/tests.csv")

if __name__ == "__main__":
    main()