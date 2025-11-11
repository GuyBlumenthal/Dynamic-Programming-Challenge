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

def solution_value_iteration(C: Const) -> tuple[np.array, np.array]:
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

    next_states = {i: {} for i in range(C.K)}
    for state in C.state_space:
        state_index = C.state_to_index(state)
        for action in range(3):
            next_states[state_index][action] = np.where(P[state_index, :, action] > 0)[0]

    # Stages of value iteration
    # 1) Check if completed
    # 2) Pick the next cell
    # 3) For the cell, update the value using the iteration equation

    epsilon = 1e-8
    gamma = 0.9999

    def exp_value(state, action):
        # TODO: Reduce complexity to only need the possible next states instead of all of them
        return P[state, :, action].T @ (Q[state, action] + gamma * J_opt[:])

    def bellman(state):
        return min([exp_value(state, action) for action in range(3)])

    delta = float("inf")

    while delta > epsilon:
        delta = 0

        for state in C.state_space:
            state_index = C.state_to_index(state)

            prev = J_opt[state_index]
            J_opt[state_index] = bellman(state_index)
            diff = abs(prev - J_opt[state_index])

            if diff > delta:  # Store the max diff of the iteration
                delta = diff

            while diff > epsilon:
                # While we are on this cell, continue iterating until we reach a temporary minimum
                prev = J_opt[state_index]
                J_opt[state_index] = bellman(state_index)
                diff = abs(prev - J_opt[state_index])

    # Optimal action
    for state in C.state_space:
        state_index = C.state_to_index(state)
        u_opt[state_index] = sorted(
            range(3), key=lambda a: exp_value(state_index, a)
        )[0]

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
    A = np.concatenate([
        np.eye(C.K) - P[:, :, 0],
        np.eye(C.K) - P[:, :, 1],
        np.eye(C.K) - P[:, :, 2],
    ])

    b = []
    for u in range(3):
        for state in C.state_space:
            i = C.state_to_index(state)
            b.append(Q[i, u])
    b = np.array(b)


    J_opt = linprog(c, A_ub=A, b_ub=b, bounds=[None, 0]).x

    expected_values = Q + np.tensordot(P, J_opt, axes=([1], [0]))
    optimal_indices = np.argmin(expected_values, axis=1)
    u_opt = np.array(C.input_space)[optimal_indices]

    return J_opt, u_opt

solution = time_def(solution_linear_prog)
