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
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from Const import Const
from ComputeTransitionProbabilities import compute_transition_probabilities
from ComputeExpectedStageCosts import compute_expected_stage_cost

from utils import compute_transition_probabilities_fast, compute_expected_stage_cost_fast, build_A_fast_setup, build_A_fast, make_preconditioner, compute_transition_probabilities_vectorized

import time

def solution(C: Const) -> tuple[np.ndarray, np.ndarray]:
    # 1. Setup Phase
    # -----------------------------------------------------
    Total_start = time.perf_counter()
    print("Computing Transition Probabilities... (Optimized Hashing)")
    
    T_start = time.perf_counter()
    
    # 1. Compute P matrices (Fastest method: Coordinate Hashing)
    P_list = compute_transition_probabilities_vectorized(C)
    
    # 2. Stack them vertically: [P_action0; P_action1; ...]
    # Shape becomes (L * K, K)
    # This allows us to extract the specific row for (state k, action u) 
    # using a single efficient slice operation later.
    P_stack = sp.vstack(P_list).tocsr()
    
    T_end = time.perf_counter()

    print("Computing Stage Costs...")
    Q = compute_expected_stage_cost_fast(C, C.K)
    
    # 2. Solver Parameters
    # -----------------------------------------------------
    K, L = C.K, C.L
    gamma = 1.0
    dtype = np.float64
    
    # Initialization
    J = np.zeros(K, dtype=dtype)
    policy = np.zeros(K, dtype=int)
    
    # Pre-allocate indices for slicing
    range_k = np.arange(K, dtype=int)
    
    # Pre-build Identity matrix
    I_mat = sp.eye(K, format='csr', dtype=dtype)

    # Tuning for GMRES speed
    max_outer_iters = 200
    outer_tol = 1e-7
    gmres_tol_max = 1e-4
    gmres_tol_min = 1e-9
    gmres_restart = 60
    max_inner_iters = 30  # Keep high to minimize restarts

    delta_J_prev = 1.0
    
    solve_start = time.perf_counter()

    # 3. Policy Iteration Loop
    # -----------------------------------------------------
    for outer_iter in range(max_outer_iters):
        J_prev = J.copy()

        # --- A. Optimized Matrix Construction (Slicing) ---
        # Goal: Construct A = I - gamma * P_pi
        # Method: Row Slicing (Fancy Indexing) on P_stack
        #
        # Logic: P_stack has structure:
        # Rows 0 to K-1: Action 0
        # Rows K to 2K-1: Action 1
        # ...
        # Therefore, the row for state k taking action policy[k] is at:
        # index = policy[k] * K + k
        
        selection_indices = policy * K + range_k
        
        # This slice is heavily optimized in Scipy (C-level copy of rows)
        P_pi = P_stack[selection_indices, :]
        
        # A = I - gamma * P_pi
        A_sparse = I_mat - gamma * P_pi

        # --- B. Preconditioner ---
        # A simple preconditioner speeds up GMRES significantly
        M_precond = make_preconditioner(A_sparse, omega=0.8, inner_iters=3, dtype=dtype)

        # --- C. Right-Hand Side ---
        b = Q[range_k, policy].astype(dtype)

        # --- D. Adaptive Tolerance ---
        # Relax tolerance when error is high to save time
        if outer_iter > 0:
            tol_k = 0.5 * delta_J_prev 
        else:
            tol_k = gmres_tol_max
        tol_k = min(max(tol_k, gmres_tol_min), gmres_tol_max)

        # --- E. Linear Solve (GMRES) ---
        try:
            J_eval, info = spla.gmres(
                A_sparse, b, x0=J, 
                tol=tol_k, 
                restart=gmres_restart, 
                maxiter=max_inner_iters, 
                M=M_precond
            )
            if info != 0:
                J_eval = spla.spsolve(A_sparse, b)
        except Exception:
            J_eval = spla.spsolve(A_sparse, b)

        # --- F. Policy Improvement ---
        # Compute Expected Future Costs for ALL actions simultaneously
        # P_stack.dot(J) calculates [P_0*J, P_1*J] in one go.
        P_J_all = P_stack.dot(J_eval)
        
        # Reshape to (L, K) -> Transpose to (K, L) to get Q values
        future_costs = P_J_all.reshape((L, K)).T
        
        Q_J = Q + gamma * future_costs
        
        # Greedy Update
        new_policy = np.argmin(Q_J, axis=1)

        # --- G. Convergence Check ---
        policy_changes = np.sum(new_policy != policy)
        delta_J = np.max(np.abs(J_eval - J_prev))

        delta_J_prev = delta_J
        J = J_eval
        policy = new_policy

        if policy_changes == 0 and delta_J < outer_tol:
            print(f"Converged in {outer_iter+1} iterations.")
            break
        
        if outer_iter == max_outer_iters - 1:
            print("Warning: Max outer iterations reached without convergence.")

    # 4. Final Timing Stats
    # -----------------------------------------------------
    Total_time = time.perf_counter() - Total_start
    T_setup = T_end - T_start
    T_solve = time.perf_counter() - solve_start
    
    print("\n--- Timing Summary (Fixed Slicing) ---")
    print(f"Transition Setup:  {T_setup:.6f}s")
    print(f"Solver Loop:       {T_solve:.6f}s")
    print(f"Total Runtime:     {Total_time:.6f}s")

    return J, policy