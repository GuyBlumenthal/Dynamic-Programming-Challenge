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

#from utils import compute_transition_probabilities_fast, compute_expected_stage_cost_fast, build_A_fast_setup, build_A_fast, make_preconditioner, compute_transition_probabilities_vectorized

import time

def make_preconditioner(A_csr, omega=0.8, inner_iters=3, dtype=np.float64):
    """
    Same as before (kept).
    """
    K = A_csr.shape[0]
    diag = A_csr.diagonal().astype(dtype)
    diag_nonzero = diag.copy()
    diag_nonzero[diag_nonzero == 0.0] = 1.0
    Dinv = 1.0 / diag_nonzero

    def apply_inner_iterations(r):
        z = np.zeros_like(r, dtype=dtype)
        for _ in range(inner_iters):
            Az = A_csr.dot(z)
            z += omega * (Dinv * (r - Az))
        return z

    return spla.LinearOperator((K, K), matvec=apply_inner_iterations, dtype=dtype)

def build_A_fast_setup(K, L, P_stack, gamma, dtype=np.float64):
    I = sp.vstack(sp.eye(K, format='csr', dtype=dtype) for _ in range(L))
    A_all = I - gamma * P_stack
    return A_all

def build_A_fast(A_all, K, policy, range_k):
    # --- VECTORIZED SLICING ---
    # We want the row from P corresponding to action policy[i] for state i.
    # Since P_stack is stacked vertically, the row for state i and action l
    # is located at index: (l * K) + i
    selector_indices = policy * K + range_k
    return A_all[selector_indices, :]

def compute_transition_probabilities_vectorized(C):
    """
    Optimized transition probability calculation using Coordinate Hashing.

    Improvements:
    1. Coordinate Hashing: Maps (y, v, d, h) -> Integer Index in O(1).
       Replaces np.searchsorted (O(N log N)).
    2. Vectorized Physics: Strictly follows PDF dynamics (Collision, Spawning, Motion).
    """

    # 1. Parse State Space & Setup Hashing
    # ---------------------------------------------------------
    S_arr = np.array(C.state_space, dtype=np.int32)
    N = S_arr.shape[0]
    num_cols = S_arr.shape[1]

    # Calculate ranges and strides for Perfect Hashing
    # We map every state to a unique integer: Hash = Sum( (val - min) * stride )
    mins = S_arr.min(axis=0)
    maxs = S_arr.max(axis=0)
    ranges = maxs - mins + 1

    # Compute strides (column-major-like logic, but order doesn't matter as long as consistent)
    strides = np.zeros(num_cols, dtype=np.int64)
    current_stride = 1
    for i in range(num_cols):
        strides[i] = current_stride
        current_stride *= ranges[i]

    # Total size of the dense lookup table
    table_size = current_stride

    # Safety Check: If table is > 100MB (approx 25M entries), warn or fallback.
    # For Flappy Bird, table_size is usually < 5,000,000 (20MB), which is fine.

    # Create the lookup table
    # lookup_table[hash] = index_in_state_space
    lookup_table = np.full(table_size, -1, dtype=np.int32)

    # Compute hashes for all existing valid states
    # Formula: sum((S_arr[:, i] - mins[i]) * strides[i])
    state_hashes = np.dot((S_arr - mins), strides)
    lookup_table[state_hashes] = np.arange(N, dtype=np.int32)

    # Helper for O(1) Lookup
    def get_state_indices(next_states):
        # 1. Check bounds (vectorized)
        # Any state component outside [min, max] is invalid
        valid_bounds = np.all((next_states >= mins) & (next_states <= maxs), axis=1)

        indices = np.full(next_states.shape[0], -1, dtype=np.int32)

        if np.any(valid_bounds):
            # 2. Compute Hashes
            # Subset only valid bounds to avoid overflow/segfaults on hash calculation
            valid_states = next_states[valid_bounds]
            hashes = np.dot((valid_states - mins), strides)

            # 3. Lookup
            # Since we checked bounds, hashes are guaranteed < table_size
            found_indices = lookup_table[hashes]
            indices[valid_bounds] = found_indices

        # Return indices and a boolean mask of which ones were found
        mask_found = (indices != -1)
        return indices[mask_found], mask_found

    # 2. Pre-process State Columns
    # ---------------------------------------------------------
    Y = S_arr[:, 0]
    V = S_arr[:, 1]
    D = S_arr[:, 2 : 2 + C.M]
    H = S_arr[:, 2 + C.M : 2 + 2 * C.M]

    # Collision Logic (PDF: "transition to a cost-free termination state")
    # We treat these as absorbing (rows of 0 in P), effectively removing them from the game flow.
    if C.M > 0:
        gap_tol = (C.G - 1) // 2
        # Collision if inside pipe horizontally (D=0) AND outside gap vertically
        is_collided = (D[:, 0] == 0) & (np.abs(Y - H[:, 0]) > gap_tol)
    else:
        is_collided = np.zeros(N, dtype=bool)

    # 3. Deterministic Pipe Dynamics (Pre-calculated)
    # ---------------------------------------------------------
    Hat_D = D.copy()
    Hat_H = H.copy()

    if C.M > 0:
        # Mask: Pipes that are currently at x=0 (Passing/Recycling)
        mask_passing = (D[:, 0] == 0)

        # Shift pipes left
        if C.M > 1:
            Hat_D[mask_passing, :-1] = D[mask_passing, 1:]
            Hat_H[mask_passing, :-1] = H[mask_passing, 1:]

        # Reset last pipe (will be filled by spawn logic if applicable)
        Hat_D[mask_passing, C.M-1] = 0
        if len(C.S_h) > 0:
            Hat_H[mask_passing, C.M-1] = C.S_h[0] # Default height

        # Decrement horizontal distance (Drift)
        # Only decrement if not just reset (checked by > 0)
        mask_dec = (Hat_D[:, 0] > 0)
        Hat_D[mask_dec, 0] -= 1

    # Spawn Probability Logic (Linear Ramp)
    sum_hat_D = np.sum(Hat_D, axis=1)
    s_values = (C.X - 1) - sum_hat_D # Free space

    # p = (s - (Dmin-1)) / (X - Dmin)
    numerator = s_values - (C.D_min - 1)
    denominator = float(C.X - C.D_min)
    p_spawn_vec = np.clip(numerator / denominator, 0.0, 1.0)

    # Identify which pipe index 'k' to spawn into (first available slot)
    k_spawn_indices = np.full(N, C.M - 1, dtype=int)
    if C.M > 1:
        # Find first column where D=0
        is_zero = (Hat_D[:, 1:] == 0)
        any_zero = np.any(is_zero, axis=1)
        first_zero = np.argmax(is_zero, axis=1)
        k_spawn_indices[any_zero] = first_zero[any_zero] + 1

    # 4. Input Loop & Matrix Construction
    # ---------------------------------------------------------
    prob_h_new = 1.0 / len(C.S_h) if len(C.S_h) > 0 else 0.0
    U_array = np.array(C.input_space)

    P_sparse_list = []

    for u in U_array:
        # Initialize Coordinate Lists for Sparse Matrix
        rows_list = []
        cols_list = []
        data_list = []

        # Resolve Flap/Wind randomness
        if u == C.U_strong:
            W_flap = np.arange(-C.V_dev, C.V_dev + 1)
        else:
            W_flap = np.array([0])

        prob_flap = 1.0 / len(W_flap)
        n_w = len(W_flap)

        # Vectorized Next State Calculation
        # v_{k+1} = v_k + u + w - g
        # Broadcast: (N, 1) + scalar + (1, n_w) -> (N, n_w)
        V_next_matrix = V[:, None] + u + W_flap[None, :] - C.g
        V_next_flat = np.clip(V_next_matrix, -C.V_max, C.V_max).flatten()

        # y_{k+1} = y_k + v_k (Note: Uses CURRENT v_k)
        Y_repeated = np.repeat(Y, n_w)
        V_curr_repeated = np.repeat(V, n_w)
        Y_next_flat = np.clip(Y_repeated + V_curr_repeated, 0, C.Y - 1).astype(np.int32)

        # Filter Collided Sources (They don't transition)
        source_idxs_base = np.repeat(np.arange(N), n_w)
        valid_src_mask = ~np.repeat(is_collided, n_w)

        # Apply mask
        V_next = V_next_flat[valid_src_mask]
        Y_next = Y_next_flat[valid_src_mask]
        src_idxs = source_idxs_base[valid_src_mask]

        # Get pipe state for these sources
        Hat_D_sub = Hat_D[src_idxs]
        Hat_H_sub = Hat_H[src_idxs]
        p_spawn_sub = p_spawn_vec[src_idxs]

        # --- PATH A: NO SPAWN ---
        # Prob = prob_flap * (1 - p_spawn)
        probs_ns = prob_flap * (1.0 - p_spawn_sub)
        mask_ns = probs_ns > 0

        if np.any(mask_ns):
            # Construct Next States Matrix
            NS_states = np.column_stack((
                Y_next[mask_ns],
                V_next[mask_ns],
                Hat_D_sub[mask_ns],
                Hat_H_sub[mask_ns]
            ))

            # FAST LOOKUP via Hashing
            dest_idxs, found = get_state_indices(NS_states)

            if len(dest_idxs) > 0:
                rows_list.append(src_idxs[mask_ns][found])
                cols_list.append(dest_idxs)
                data_list.append(probs_ns[mask_ns][found])

        # --- PATH B: SPAWN ---
        mask_s = (p_spawn_sub > 0)
        if np.any(mask_s) and len(C.S_h) > 0:
            # Subset for spawn calculations
            S_Y = Y_next[mask_s]
            S_V = V_next[mask_s]
            S_src = src_idxs[mask_s]
            S_D = Hat_D_sub[mask_s]
            S_H = Hat_H_sub[mask_s]
            S_k = k_spawn_indices[src_idxs][mask_s]

            # Base probability for this branch
            S_base_prob = prob_flap * p_spawn_sub[mask_s]

            # Calculate 's' distance to fill
            # s = X - 1 - sum(d)
            S_s_fill = np.clip((C.X - 1) - np.sum(S_D, axis=1), C.D_min, C.X - 1)

            # Iterate over possible new heights (Uniform probability)
            for h_new in C.S_h:
                # Construct new pipe arrays
                D_new = S_D.copy()
                H_new = S_H.copy()

                # Update the specific pipe k
                row_indices = np.arange(len(S_Y))
                D_new[row_indices, S_k] = S_s_fill
                H_new[row_indices, S_k] = h_new

                # Construct State Matrix
                S_states = np.column_stack((S_Y, S_V, D_new, H_new))

                # FAST LOOKUP
                dest_idxs, found = get_state_indices(S_states)

                if len(dest_idxs) > 0:
                    rows_list.append(S_src[found])
                    cols_list.append(dest_idxs)
                    # p = base_prob * (1/|Sh|)
                    data_list.append(S_base_prob[found] * prob_h_new)

        # Build CSR Matrix for this input u
        if len(data_list) > 0:
            data = np.concatenate(data_list)
            rows = np.concatenate(rows_list)
            cols = np.concatenate(cols_list)
            P_mat = sp.csr_matrix((data, (rows, cols)), shape=(C.K, C.K))
        else:
            P_mat = sp.csr_matrix((C.K, C.K))

        P_sparse_list.append(P_mat)

    return P_sparse_list

# Keeping this purely for reference, though logic is now inlined for performance
def spawn_probability(C, s):
    if s <= C.D_min - 1:
        return 0.0
    elif s >= C.X:
        return 1.0
    else:
        return (s - (C.D_min - 1)) / (C.X - C.D_min)

def compute_expected_stage_cost_fast(C: Const, K: int):
    return np.tile(np.array([
        -1,                # Cost for action 0 (None)
        C.lam_weak - 1,    # Cost for action 1 (Weak)
        C.lam_strong - 1   # Cost for action 2 (Strong)
    ]), (K, 1))


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

    # Tolerances
    gmres_tol = 1e-5
    outer_tol = 1e-7

    # Iteration limits
    max_outer_iters = 200
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

        # --- 2. Preconditioner ---
        M = make_preconditioner(A_sparse, omega=0.8, inner_iters=5, dtype=dtype)

        # --- C. Right-Hand Side ---
        b = Q[range_k, policy].astype(dtype)

        # --- 3. Solve Linear System ---
        try:
            J_eval, info = spla.gmres(
                A_sparse,
                b,
                x0=J,
                tol=gmres_tol,
                restart=gmres_restart,
                maxiter=max_inner_iters,
                M=M
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