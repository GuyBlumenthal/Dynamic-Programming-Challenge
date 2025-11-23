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

from scipy.optimize import linprog

from Const import Const
from ComputeTransitionProbabilities import compute_transition_probabilities
from ComputeExpectedStageCosts import compute_expected_stage_cost

#from utils import compute_transition_probabilities_fast, compute_expected_stage_cost_fast, build_A_fast_setup, build_A_fast, make_preconditioner, compute_transition_probabilities_vectorized

import time

from itertools import product

SOLVER_DEV_MODE = True
timing_array = []

log = print if SOLVER_DEV_MODE else lambda x: None
record_time = time.perf_counter if SOLVER_DEV_MODE else lambda: 0

class CustomStateSpace:
    # ================== D-Vector Recursive Builder ==================
    def build_d_recursive(self, y, v, current_d_list, current_d_sum, d_index, spot0):
        """
        Recursively builds the D-vector (d1, ..., dM) for a given
        (y, v) prefix.

        Pruning:
        - sum(d) > X-1
        - Trailing zeros (if d_i=0, all d_j for j>i must be 0)
        - d2=0 if d1=0
        """
        # --- Base Case: D-vector is complete ---
        if d_index == self.M:
            # D-vector is built, now start building the H-vector
            h_iterable = self.possible_h_iterables[spot0]

            prefix = (y, v) + tuple(current_d_list)

            # 2. Loop over the product of these allowed H-options
            for h_tuple in h_iterable:
                state = prefix + h_tuple
                self.valid_states_with_indices[self.current_index, :] = state
                self.current_index += 1

            return        # --- Recursive Step: Add d_i ---
        if d_index == 0:
            d_options = self.S_d1
        elif spot0 > 0:
            d_options = self.S_d0
        else:
            d_options = self.S_d

        for d in d_options:
            # 1. Sum constraint
            if current_d_sum + d > self.X_limit:
                continue

            # 3. d1/d2 constraint (d1=0 -> d2>0)
            # d_index == 1 is d2
            if d_index == 1:
                d1 = current_d_list[0]
                d2 = d
                if d1 <= 0 and d2 == 0:
                    continue

            next_spot0 = spot0
            if spot0 == 0 and d_index > 0 and d == 0:
                next_spot0 = d_index  # This is the first zero

            # Recurse with the added d
            current_d_list[d_index] = d
            self.build_d_recursive(
                y, v,
                current_d_list,
                current_d_sum + d,
                d_index + 1,
                # Update zero_seen flag:
                # (zero_seen is True if it was already True, OR
                # if we are adding a zero *after* d1)
                next_spot0
            )

    def custom_state_space(self, C: Const):
        """
        Computes the state space and returns a state -> index dictionary
        using a recursive, pruning-based generation method.

        This function maintains the strict lexicographical ordering
        from the problem statement, ensuring the state-to-index
        mapping is identical to the original 'itertools.product' method,
        but is significantly faster by pruning invalid branches early.
        """


        self.current_index = 0

        # --- Cache constants from C for minor speedup ---
        self.S_y, self.S_v = C.S_y, C.S_v
        self.S_d, self.S_d1 = C.S_d, C.S_d1
        self.S_h, self.S_h_default = C.S_h, C.S_h[0]
        self.M, self.X_limit = C.M, C.X - 1

        # --- Create mappings for non-contiguous state variables ---
        self.v_offset = C.V_max

        # --- Initialize state_to_index_array ---
        dims = [C.Y, 2 * C.V_max + 1]
        for _ in range(self.M):
            dims.append(C.X) # d values are 0..X-1
        for _ in range(self.M):
            dims.append(C.Y) # h values are 0..Y-1

        self.state_to_index_array = np.full(dims, -1, dtype=np.int32)

        d = (np.prod(dims), 2 + 2 * self.M)
        self.valid_states_with_indices = np.zeros(d, dtype=np.int32)


        # Pre-build the two possible lists for H-options
        self.h_options_all = self.S_h
        self.h_options_default = [self.S_h_default]

        # Pre build an empty D-options list
        self.S_d0 = [0]

        possible_h_iterables = [[self.h_options_all] + [self.h_options_all for i in range(1, self.M)]]
        for spot0 in range(1, self.M):
            possible_h_iterables.append([self.h_options_all] + [
                self.h_options_default if i >= spot0 else self.h_options_all
                for i in range(1, self.M)
            ])
        self.possible_h_iterables = [list(product(*h_iter)) for h_iter in possible_h_iterables]

        # The outer loops *must* be y, then v, to maintain order
        for y in self.S_y:
            for v in self.S_v:
                current_d_list_for_v = [0] * self.M
                # Start the recursion for the D-vector
                self.build_d_recursive(y, v, current_d_list_for_v, 0, 0, 0)

        return self.current_index, self.valid_states_with_indices[:self.current_index, :]

def generate_state_space(C: Const):
    if hasattr(C, '_state_space'):
        K, S_arr = C.K, np.array(C.state_space, dtype=np.int32)
    else:
        css = CustomStateSpace()
        K, S_arr = css.custom_state_space(C)

    return K, S_arr

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

def compute_transition_probabilities_vectorized(C, K, S_arr):
    """
    Optimized transition probability calculation using Coordinate Hashing.

    Improvements:
    1. Coordinate Hashing: Maps (y, v, d, h) -> Integer Index in O(1).
       Replaces np.searchsorted (O(N log N)).
    2. Vectorized Physics: Strictly follows PDF dynamics (Collision, Spawning, Motion).
    """

    # 1. Parse State Space & Setup Hashing
    # ---------------------------------------------------------
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
            P_mat = sp.csr_matrix((data, (rows, cols)), shape=(K, K))
        else:
            P_mat = sp.csr_matrix((K, K))

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


def solver_PI(C, K, L, P_list, P_stack, Q):
    gamma = 1.0
    dtype = np.float64

    log("Pre-calculating System Matrices (A_all)...")
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
        # P.dot(J) calculates costs for ALL actions in one go
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
            log(f"Converged in {outer_iter+1} iterations.")
            break

        if outer_iter == max_outer_iters - 1:
            log("Warning: Max outer iterations reached without convergence.")

    return J, policy

def solver_LP(C, K, L, P_list, P_stack, Q):
    b = Q.flatten(order='F')

    c = np.full(K, -1, np.int64)

    # 1. Create a sparse identity matrix
    I_sparse = sp.eye(K, format='csc')

    # 2. Create a list to hold the sparse blocks (I - P_l)
    A_blocks = []

    # 3. Loop over all actions
    for l in range(L):
        # Add the sparse (I - P_l) block to our list
        A_blocks.append(I_sparse - P_list[l])

    # 4. Stack all blocks vertically into one sparse matrix
    A = sp.vstack(A_blocks, format='csc')

    res = linprog(c, A_ub=A, b_ub=b, bounds=[None, 0], method='highs')

    J_opt = res.x

    # Create a list of weighted_J vectors, one for each action l
    weighted_J_cols = []
    for l in range(L):
        # P[l] is (K, K) sparse, J_opt is (K,) dense
        # The @ operator performs efficient sparse-dot-dense
        weighted_J_l = P_list[l] @ J_opt  # Result is a (K,) dense vector
        weighted_J_cols.append(weighted_J_l)

    # Stack the (K,) vectors as columns into a (K, L) dense array
    weighted_J_all = np.stack(weighted_J_cols, axis=1)
    expected_values = Q + weighted_J_all

    optimal_indices = np.argmin(expected_values, axis=1)
    u_opt = np.array(C.input_space)[optimal_indices]

    return J_opt, u_opt

def select_solver(K, L):
    if K > 1000:
        log("PI solver selected")
        return solver_PI
    else:
        log("LP solver selected")
        return solver_LP

def solution(C: Const) -> tuple[np.ndarray, np.ndarray]:
    T_start = record_time()

    log("Generating state space")
    T_state_start = record_time()
    K, S_arr = generate_state_space(C)
    L = C.L
    T_state_end = record_time()


    log("Computing Transition Probabilities...")
    T_prob_start = record_time()
    P_list = compute_transition_probabilities_vectorized(C, K, S_arr)
    P_stack = sp.vstack(P_list).tocsr()
    T_prob_end = record_time()


    log("Computing Stage Costs...")
    T_cost_start = record_time()
    Q = compute_expected_stage_cost_fast(C, K)
    T_cost_end = record_time()


    log("Running solver")
    T_solver_start = record_time()
    solver = select_solver(K, L)
    J, policy = solver(C, K, L, P_list, P_stack, Q)
    T_solver_end = record_time()

    # Final Timing Stats
    # -----------------------------------------------------
    if SOLVER_DEV_MODE:
        T_total_ms  = (record_time() - T_start)   * 1e3
        T_state_ms  = (T_state_end - T_state_start)     * 1e3
        T_prob_ms   = (T_prob_end - T_prob_start)       * 1e3
        T_cost_ms   = (T_cost_end - T_cost_start)       * 1e3
        T_solver_ms = (T_solver_end - T_solver_start)   * 1e3

        global timing_array
        timing_array.append((
            K, solver.__name__, (
                T_total_ms,
                T_state_ms,
                T_prob_ms,
                T_cost_ms,
                T_solver_ms,
            )
        ))

        log("\n--- Timing Summary (Pre-calc A_all + Custom Precond) ---")
        log(f"State generation:   {T_state_ms:.3f}ms")
        log(f"Transition Setup:   {T_prob_ms:.3f}ms")
        log(f"Stage Costs:        {T_cost_ms:.3f}ms")
        log(f"Solver:             {T_solver_ms:.3f}ms")
        log(f"Total Runtime:      {T_total_ms:.3f}ms")

    return J, policy