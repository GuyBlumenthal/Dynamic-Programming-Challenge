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
    
    # 1. Compute P matrices
    P_list = compute_transition_probabilities_vectorized(C)
    
    # 2. Stack P: Shape (L * K, K)
    P_stack = sp.vstack(P_list).tocsr()
    
    T_end = time.perf_counter()

    print("Computing Stage Costs...")
    Q = compute_expected_stage_cost_fast(C, C.K)
    
    # 2. Solver Parameters & Pre-calculation
    # -----------------------------------------------------
    K, L = C.K, C.L
    gamma = 1.0
    dtype = np.float64
    
    print("Pre-calculating System Matrices (A_all)...")
    # --- OPTIMIZATION: Build A_all once ---
    A_all = build_A_fast_setup(K, L, P_stack, gamma, dtype)
    
    # Initialization
    J = np.zeros(K, dtype=dtype)
    policy = np.zeros(K, dtype=int)
    
    # Pre-allocate indices
    range_k = np.arange(K, dtype=int)
    
    # Tuning
    max_outer_iters = 200
    outer_tol = 1e-7
    gmres_tol_max = 1e-4
    gmres_tol_min = 1e-9
    gmres_restart = 60
    max_inner_iters = 30
    
    delta_J_prev = 1.0
    solve_start = time.perf_counter()

    # 3. Policy Iteration Loop
    # -----------------------------------------------------
    for outer_iter in range(max_outer_iters):
        J_prev = J.copy()

        # --- A. Ultra-Fast Matrix Slicing ---
        # No arithmetic here, just picking rows from A_all
        A_sparse = build_A_fast(A_all, K, policy, range_k)

        # --- B. Custom Preconditioner ---
        # Uses the specific logic provided (omega relaxation)
        M_precond = make_preconditioner(A_sparse, omega=0.8, inner_iters=3, dtype=dtype)

        # --- C. Right-Hand Side ---
        b = Q[range_k, policy].astype(dtype)

        # --- D. Adaptive Tolerance ---
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
        # P_stack.dot(J) calculates costs for ALL actions in one go
        P_J_all = P_stack.dot(J_eval)
        
        # Reshape to (L, K) -> Transpose to (K, L)
        future_costs = P_J_all.reshape((L, K)).T
        Q_J = Q + gamma * future_costs
        
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
    
    print("\n--- Timing Summary (Pre-calc A_all + Custom Precond) ---")
    print(f"Transition Setup:  {T_setup:.6f}s")
    print(f"Solver Loop:       {T_solve:.6f}s")
    print(f"Total Runtime:     {Total_time:.6f}s")

    return J, policy