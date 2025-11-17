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

from utils import CustomStateSpace

from ComputeExpectedStageCosts import compute_expected_stage_cost_solver as compute_expected_stage_cost
from ComputeExpectedStageCosts import compute_expected_stage_cost as compute_expected_stage_cost_old
from ComputeTransitionProbabilities import compute_transition_probabilities_sparse as compute_transition_probabilities

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
    ss = CustomStateSpace()
    K, state_array, valid_states = ss.custom_state_space(C)

    J_opt = np.zeros(K)
    u_opt = np.zeros(K)

    P = compute_transition_probabilities(C, state_array, K, valid_states)
    Q, b = compute_expected_stage_cost(C, K)

    c = np.full(K, -1, np.int64)

    # 1. Create a sparse identity matrix
    I_sparse = eye(K, format='csc')

    # 2. Create a list to hold the sparse blocks (I - P_l)
    A_blocks = []

    # 3. Loop over all actions
    for l in range(C.L):
        # Add the sparse (I - P_l) block to our list
        A_blocks.append(I_sparse - P[l])

    # 4. Stack all blocks vertically into one sparse matrix
    A = vstack(A_blocks, format='csc')

    # 'highs' is the best for sparse problems
    # TODO: Need to deal with situation there is no termination EVER! we need optimal actions?
    res = linprog(c, A_ub=A, b_ub=b, bounds=[None, 0], method='highs')

    J_opt = res.x

    # Create a list of weighted_J vectors, one for each action l
    weighted_J_cols = []
    for l in range(C.L):
        # P[l] is (K, K) sparse, J_opt is (K,) dense
        # The @ operator performs efficient sparse-dot-dense
        weighted_J_l = P[l] @ J_opt  # Result is a (K,) dense vector
        weighted_J_cols.append(weighted_J_l)

    # Stack the (K,) vectors as columns into a (K, L) dense array
    weighted_J_all = np.stack(weighted_J_cols, axis=1)
    expected_values = Q + weighted_J_all

    optimal_indices = np.argmin(expected_values, axis=1)
    u_opt = np.array(C.input_space)[optimal_indices]

    return J_opt, u_opt

def solution_value_iteration(C: Const, epsilon=1e-8, max_iter=10000) -> tuple[np.array, np.array]:
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

    ss = CustomStateSpace()
    K, state_array, valid_states = ss.custom_state_space(C)
    P = compute_transition_probabilities(C, state_array, K, valid_states)
    Q, _ = compute_expected_stage_cost(C, K)

    # 2. Initialize J (Value function)
    J = np.zeros(K)

    # Pre-allocate arrays
    J_current = np.zeros(K)
    J_next = np.empty(K)

    val_l = np.empty(K)

    for i in range(max_iter):
        # 1. Initialize J_next with the value of the first action (l=0)
        # Q[:, 0] is (K,), P[0] @ J_old is (K,)
        np.add(Q[:, 0], (P[0] @ J_current), out=J_next)

        # 2. Loop from the second action (l=1)
        for l in range(1, C.L):
            # Calculate the value for this action: Q_l + P_l * J_current
            np.add(Q[:, l], (P[l] @ J_current), out=val_l)

            # Update J_next to be the element-wise minimum
            np.minimum(J_next, val_l, out=J_next)

        if np.max(np.abs(J_next - J_current)) < epsilon:
            J_current = J_next  # Converged. J_next is the final value.
            break

        J_current, J_next = J_next, J_current

    # 4. Recover the optimal policy (u_opt)
    expected_values_cols = []
    for l in range(C.L):
        # Use the final converged J_opt
        val_l = Q[:, l] + (P[l] @ J_current)
        expected_values_cols.append(val_l)

    expected_values = np.stack(expected_values_cols, axis=1)

    optimal_indices = np.argmin(expected_values, axis=1)
    u_opt = np.array(C.input_space)[optimal_indices]

    return J_current, u_opt

SOL_LP = 0
SOL_VI = 1

SOLUTION_FUNCTIONS = {
    SOL_LP: solution_linear_prog_sparse,
    SOL_VI: solution_value_iteration,
}

def solution_picker(C: Const) -> tuple[np.array, np.array]:
    # TODO: Minor, but this is technically overhead

    selected_solution = SOLUTION_FUNCTIONS[SOL_LP]

    return selected_solution(C)

solution = solution_picker
