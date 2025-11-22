"""main.py

Python script that calls all the functions for computing the optimal cost
and policy of the given problem.

Dynamic Programming and Optimal Control
Fall 2025
Programming Exercise

Contact: Antonio Terpin aterpin@ethz.ch
Authors: Marius Baumann, Antonio Terpin

--
ETH Zurich
Institute for Dynamic Systems and Control
--
"""

import os
import numpy as np
import argparse

from Const import Const
from ComputeTransitionProbabilities import compute_transition_probabilities
from ComputeExpectedStageCosts import compute_expected_stage_cost
from Solver import solution, generate_state_space
import simulation

from line_profiler import LineProfiler
from Solver import compute_transition_probabilities_vectorized
import time

def main(use_solution_if_exist=True) -> None:
    """Main function to compute the optimal policy and run a simulation.

    Args:
        use_solution_if_exist (bool): If True, tries to load an existing
            optimal policy from disk. If not found, computes it from scratch.
    """
    C = Const()
    u_opt = None

    ws_dir = "workspaces"
    u_path = os.path.join(ws_dir, "u_opt.npy")
    os.makedirs(ws_dir, exist_ok=True)

    if use_solution_if_exist and os.path.exists(u_path):
        u_opt = np.load("workspaces/u_opt.npy")
        if len(u_opt)!=C.K:
            u_opt = None

    profiler = True

    if u_opt == None:
        # Build P and Q
        # print("Computing transition probabilities P ...")
        # P = compute_transition_probabilities(C)
        # print(f"P shape: {P.shape}")

        # print("Computing expected stage costs Q ...")
        # Q = compute_expected_stage_cost(C)
        # print(f"Q shape: {Q.shape}")

        # Solve for optimal cost and policy
        if profiler:
            print("Solving for optimal policy ...")
            lp = LineProfiler()
            lp.add_callable(compute_transition_probabilities_vectorized)
            lp.add_callable(generate_state_space)
            wrapper = lp(solution)
            J_opt, u_opt = wrapper(C)
            # J_opt, u_opt = solution(C)
            with open(f"extended_testing/profiles/profile_{time.strftime('%H_%M_%S')}.txt", 'w') as f:
                lp.print_stats(f)
            print("Solution obtained.")
            print("J_opt (min/max):", float(np.min(J_opt)), float(np.max(J_opt)))
        else:
            J_opt, u_opt = solution(C)
            print("Solution obtained.")
            print("J_opt (min/max):", float(np.min(J_opt)), float(np.max(J_opt)))

    # Run simulation
    # simulation.run_simulation(C, policy=u_opt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-solution", action="store_true")
    args = parser.parse_args()
    main(use_solution_if_exist=args.use_solution)
