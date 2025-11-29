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

from utils import compute_transition_probabilities_fast, build_A, make_preconditioner, compute_transition_probabilities_vectorized

import time

def solution(C: Const) -> tuple[np.ndarray, np.ndarray]: 
    Total_start = time.perf_counter()

    print("Computing Transition Probabilities... (fast)")
    T_start_time = time.perf_counter()
    P_list = compute_transition_probabilities_vectorized(C)#compute_transition_probabilities_fast(C)
    
    # --- OPTIMIZATION: Pre-Stack P Matrices ---
    P_stack = sp.vstack(P_list).tocsr()
    
    del P_list 
    
    T_end_time = time.perf_counter()

    print("Computing Stage Costs...")
    SC_start_time = time.perf_counter()
    Q = compute_expected_stage_cost(C)
    SC_end_time = time.perf_counter()

    # --- iGMRES & Solver Parameters ---
    K, L = C.K, C.L
    gamma = 1.0
    dtype = np.float64

    J = np.zeros(K, dtype=dtype)
    policy = np.zeros(K, dtype=int)  

    # Tolerances
    gmres_tol_max = 1e-5
    gmres_tol_min = 1e-9
    eta = 0.5
    outer_tol = 1e-7

    # Iteration limits
    max_outer_iters = 200
    gmres_restart = 60
    max_inner_iters = 15
    
    delta_J_prev = 1.0
    range_k = np.arange(K, dtype=int) # Pre-compute for slicing

    start_time = time.perf_counter()

    # Begin iterations
    for outer_iter in range(max_outer_iters):
        J_prev = J.copy()

        # --- 1. Build A (Optimized) ---
        A_sparse = build_A(K, P_stack, policy, range_k, gamma, dtype)

        # --- 2. Preconditioner ---
        M = make_preconditioner(A_sparse, omega=0.8, inner_iters=3, dtype=dtype)

        # Right-hand side
        b = Q[range_k, policy].astype(dtype)

        # Adaptive GMRES Tolerance
        # If delta_J is large, we don't need a perfect linear solve yet.
        if outer_iter > 0:
            tol_k = eta * delta_J_prev
        else:
            tol_k = gmres_tol_max
            
        tol_k = min(max(tol_k, gmres_tol_min), gmres_tol_max)

        # --- 3. Solve Linear System ---
        try:
            J_eval, info = spla.gmres(
                A_sparse,
                b,
                x0=J,
                tol=tol_k,
                restart=gmres_restart,
                maxiter=max_inner_iters,
                M=M
            )
            if info != 0:
                J_eval = spla.spsolve(A_sparse, b)
        except Exception:
            J_eval = spla.spsolve(A_sparse, b)

        # --- 4. Policy Improvement (Optimized) ---
        P_J_all = P_stack.dot(J_eval)
        
        expected_future_costs = P_J_all.reshape((L, K)).T
        
        # Bellman update
        Q_J = Q + gamma * expected_future_costs

        # Greedy improvement
        new_policy = np.argmin(Q_J, axis=1)

        # Stats
        policy_changes = np.sum(new_policy != policy)
        delta_J = np.max(np.abs(J_eval - J_prev))

        delta_J_prev = delta_J
        J = J_eval
        policy = new_policy

        if policy_changes == 0 and delta_J < outer_tol:
            print(f"Converged in {outer_iter+1} iterations.")
            break
    else:
        print("Warning: Max outer iterations reached without convergence.")

    end_time = time.perf_counter()

    T_TP = T_end_time - T_start_time
    T_SC = SC_end_time - SC_start_time
    T_J = end_time - start_time
    Total_time = time.perf_counter() - Total_start

    print("\n--- Timing Summary (Optimized with Slicing) ---")
    print(f"Transition Probabilities: {T_TP:.6f}s")
    print(f"Stage Costs:              {T_SC:.6f}s")
    print(f"Policy Iteration:         {T_J:.6f}s")
    print(f"Total Runtime:            {Total_time:.6f}s")

    return J, policy