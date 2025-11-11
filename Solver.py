"""Solver.py

Template to solve the stochastic shortest path problem.

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

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix, vstack, eye
from Const import Const

from ComputeExpectedStageCosts import compute_expected_stage_cost
from ComputeTransitionProbabilities import compute_transition_probabilities

from timer import time_def

def solution_template(C: Const) -> tuple[np.array, np.array]:
    """Computes the optimal cost and the optimal control policy.

    You can solve the SSP by any method:
    - Value Iteration
    - Policy Iteration
    - Linear Programming
    - A combination of the above
    - Others?

    Args:
        C (Const): The constants describing the problem instance.

    Returns:
        np.array: The optimal cost to go for the stochastic SPP,
            of shape (C.K,), where C.K is the number of states.
        np.array: The optimal control policy for the stochastic SPP,
            of shape (C.K,), where each entry is in {0,...,C.L-1}.
    """
    J_opt = np.zeros(C.K)
    u_opt = np.zeros(C.K)

    # You're free to use the functions below, implemented in the previous
    # tasks, or come up with something else.
    # If you use them, you need to add the corresponding imports
    # at the top of this file.
    # P = compute_transition_probabilities(C)
    # Q = compute_expected_stage_cost(C)

    # TODO: implement Value Iteration, Policy Iteration, Linear Programming
    # or a combination of these

    return J_opt, u_opt

def solution_linear_prog(C: Const) -> tuple[np.array, np.array]:
    """Computes the optimal cost and the optimal control policy.

    You can solve the SSP by any method:
    - Value Iteration
    - Policy Iteration
    - Linear Programming
    - A combination of the above
    - Others?

    Args:
        C (Const): The constants describing the problem instance.

    Returns:
        np.array: The optimal cost to go for the stochastic SPP,
            of shape (C.K,), where C.K is the number of states.
        np.array: The optimal control policy for the stochastic SPP,
            of shape (C.K,), where each entry is in {0,...,C.L-1}.
    """
    J_opt = np.zeros(C.K)
    u_opt = np.zeros(C.K)

    P = compute_transition_probabilities(C)
    Q = compute_expected_stage_cost(C)

    c = -1 * np.ones(C.K)

    A = np.empty((C.K * C.L, C.K))
    I = np.eye(C.K)
    A[0:C.K, :] = I - P[:, :, 0]
    A[C.K:2*C.K, :] = I - P[:, :, 1]
    A[2*C.K:3*C.K, :] = I - P[:, :, 2]

    b = Q.flatten(order='F')

    J_opt = linprog(c, A_ub=A, b_ub=b, bounds=[None, 0]).x

    expected_values = Q + np.tensordot(P, J_opt, axes=([1], [0]))
    optimal_indices = np.argmin(expected_values, axis=1)
    u_opt = np.array(C.input_space)[optimal_indices]

    return J_opt, u_opt

def solution_linear_prog_sparse(C: Const) -> tuple[np.array, np.array]:
    """Computes the optimal cost and the optimal control policy.
    (Optimized with sparse LP)
    """
    J_opt = np.zeros(C.K)
    u_opt = np.zeros(C.K)

    P = compute_transition_probabilities(C)
    Q = compute_expected_stage_cost(C)

    c = -1 * np.ones(C.K)

    # --- OPTIMIZATION: Build A as a Sparse Matrix ---

    # 1. Create a sparse identity matrix (CSC format is good for column math)
    I_sparse = eye(C.K, format='csc')

    # 2. Create a list to hold the sparse blocks (I - P_l)
    A_blocks = []

    # 3. Loop over all actions (assuming C.L is the number of actions)
    for l in range(C.L):
        # Convert the dense P slice to a sparse matrix
        P_l_sparse = csc_matrix(P[:, :, l])

        # Add the sparse (I - P_l) block to our list
        A_blocks.append(I_sparse - P_l_sparse)

    # 4. Stack all blocks vertically into one tall sparse matrix
    # This is the sparse equivalent of np.concatenate or pre-allocation
    A = vstack(A_blocks, format='csc')

    # --- END OPTIMIZATION ---

    b = Q.flatten(order='F')

    # --- OPTIMIZATION: Specify a fast solver method ---
    # 'highs-ipm' (Interior-Point Method) is excellent for
    # large, sparse problems like this one.
    res = linprog(c, A_ub=A, b_ub=b, bounds=[None, 0], method='highs-ipm')

    if not res.success:
        print("Warning: Linear program did not solve successfully.")
        # Fallback to the default solver if IPM fails
        res = linprog(c, A_ub=A, b_ub=b, bounds=[None, 0], method='highs')

    J_opt = res.x
    # --- END OPTIMIZATION ---

    # This part is already fast and vectorized
    expected_values = Q + np.tensordot(P, J_opt, axes=([1], [0]))
    optimal_indices = np.argmin(expected_values, axis=1)
    u_opt = np.array(C.input_space)[optimal_indices]

    return J_opt, u_opt

solution = time_def(solution_linear_prog_sparse)
