"""Solver.py

Template to solve the stochastic shortest path problem.

Dynamic Programming and Optimal Control
Fall 2025
Programming Exercise

Contact: Antonio Terpin aterpin@ethz.ch

Authors: Marius Baumann, Antonio Terpin

Problem Solved by: Guy Blumenthal, Tobias Tichy

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
import time

from itertools import product

SOLVER_DEV_MODE = True
timing_array = []

log = print if SOLVER_DEV_MODE else lambda x: None
record_time = time.perf_counter if SOLVER_DEV_MODE else lambda: 0

class CustomStateSpace:
    def custom_state_space(self, C):
        self.S_y, self.S_v = C.S_y, C.S_v
        self.S_d, self.S_d1 = sorted(C.S_d), sorted(C.S_d1)
        self.S_h, self.S_h_default = C.S_h, C.S_h[0]
        self.M, self.X_limit = C.M, C.X - 1

        # Precompute H blocks
        h_options_all = self.S_h
        h_options_default = [self.S_h_default]

        # Dictionary: spot0 -> 2D Array of H combinations
        self.h_blocks = {}

        # Logic to build h_iterables mapping spot0 to H-combinations
        # Case 0: All options allowed
        h_iter_0 = [h_options_all] * self.M
        self.h_blocks[0] = np.array(list(product(*h_iter_0)), dtype=np.int32)

        # Case 1..M:
        for spot0 in range(1, self.M):
            h_iter = [h_options_all] + [
                h_options_default if i >= spot0 else h_options_all
                for i in range(1, self.M)
            ]
            self.h_blocks[spot0] = np.array(list(product(*h_iter)), dtype=np.int32)

        # Generate core DH Matirx
        # We build the combinations of D and H once.
        # This list will hold arrays of shape (N, M + M) -> [d1...dM, h1...hM]
        self.dh_rows_list = []

        self.S_d0 = [0]
        d_buffer = [0] * self.M

        # Start recursion
        self.build_dh_core(d_buffer, 0, 0, 0)

        # Core Matrix: Columns [d0...dM, h0...hM]
        self.core_dh_matrix = np.vstack(self.dh_rows_list)

        # EXPAND WITH Y AND V (Vectorized)
        n_dh = self.core_dh_matrix.shape[0]
        n_y = len(self.S_y)
        n_v = len(self.S_v)
        total_rows = n_y * n_v * n_dh

        cols = 2 + 2 * self.M
        self.valid_states = np.zeros((total_rows, cols), dtype=np.int32)

        current_idx = 0

        block_size = n_v * n_dh
        v_block = np.repeat(self.S_v, n_dh)                 # Shape: (block_size,)
        dh_block = np.tile(self.core_dh_matrix, (n_v, 1))   # Shape: (block_size, cols-2)

        # Loop only over Y
        for y in self.S_y:
            end_idx = current_idx + block_size

            # Fill Y
            self.valid_states[current_idx:end_idx, 0] = y

            # Fill V using Block Copy
            self.valid_states[current_idx:end_idx, 1] = v_block

            # Fill D and H using the pre-computed core
            self.valid_states[current_idx:end_idx, 2:] = dh_block

            current_idx = end_idx

        return total_rows, self.valid_states


    def build_dh_core(self, current_d_list, current_d_sum, d_index, spot0):
        # Base Case: D-vector is complete
        if d_index == self.M:
            # Retrieve the pre-computed H-block for this spot0
            h_block = self.h_blocks[spot0] # Shape (N_h, M)
            n_h_rows = h_block.shape[0]

            # Create the D block (repeat current_d_list n_h_rows times)
            d_row = np.array(current_d_list, dtype=np.int32)
            d_block = np.tile(d_row, (n_h_rows, 1)) # Shape (N_h, M)

            # Concatenate D and H side-by-side
            # Result shape: (N_h, 2*M)
            dh_chunk = np.hstack((d_block, h_block))

            # Store this chunk
            self.dh_rows_list.append(dh_chunk)
            return

        # Recursive Step
        if d_index == 0:
            d_options = self.S_d1
        elif spot0 > 0:
            d_options = self.S_d0
        else:
            d_options = self.S_d

        for d in d_options:
            if current_d_sum + d > self.X_limit:
                break

            if d_index == 1:
                d1 = current_d_list[0]
                if d1 <= 0 and d == 0:
                    continue

            next_spot0 = spot0
            if spot0 == 0 and d_index > 0 and d == 0:
                next_spot0 = d_index

            current_d_list[d_index] = d

            self.build_dh_core(
                current_d_list,
                current_d_sum + d,
                d_index + 1,
                next_spot0
            )

def generate_state_space(C: Const):
    if hasattr(C, '_state_space'):
        K, S_arr = C.K, np.array(C.state_space, dtype=np.int32)
    else:
        css = CustomStateSpace()
        K, S_arr = css.custom_state_space(C)

    return K, S_arr

def make_preconditioner(A_csr, omega=0.8, inner_iters=5, dtype=np.float64):
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
    selector_indices = policy * K + range_k
    return A_all[selector_indices, :]

def compute_transition_probabilities_lookup(C: Const, K, S_arr):
    IDX_DTYPE = np.int32
    HASH_DTYPE = np.int64

    N_full = S_arr.shape[0]
    num_cols = S_arr.shape[1]

    # Setup Strides
    # Using int64 to prevent overflow
    mins = np.array([0, -C.V_max, *[0]*C.M, *[C.S_h[0]]*C.M], dtype=HASH_DTYPE)
    maxs = np.array([C.Y, C.V_max, *[C.X]*C.M, *[C.S_h[-1]]*C.M], dtype=HASH_DTYPE)
    ranges = maxs - mins + 1

    strides = np.zeros(num_cols, dtype=HASH_DTYPE)
    current_stride = 1
    for i in range(num_cols):
        strides[i] = current_stride
        current_stride *= ranges[i]

    table_size = current_stride
    stride_y = strides[0]
    stride_v = strides[1]
    strides_d = strides[2 : 2+C.M]
    strides_h = strides[2+C.M : 2+2*C.M]

    # Create Lookup Table
    lookup_table = np.full(table_size, -1, dtype=IDX_DTYPE)

    # Compute current hashes
    # We use a raw matrix multiply for speed, ensuring int64
    current_hashes = np.dot((S_arr.astype(HASH_DTYPE) - mins), strides)
    lookup_table[current_hashes] = np.arange(N_full, dtype=IDX_DTYPE)

    # Determine validity once and slice arrays.

    Y_full = S_arr[:, 0]
    V_full = S_arr[:, 1]
    D_full = S_arr[:, 2 : 2 + C.M]
    H_full = S_arr[:, 2 + C.M : 2 + 2 * C.M]

    if C.M > 0:
        gap_tol = (C.G - 1) // 2
        is_collided = (D_full[:, 0] == 0) & (np.abs(Y_full - H_full[:, 0]) > gap_tol)
        valid_mask = ~is_collided
    else:
        valid_mask = np.ones(N_full, dtype=bool)

    # Convert to indices for COO matrix construction later
    valid_src_indices = np.where(valid_mask)[0].astype(IDX_DTYPE)
    N = len(valid_src_indices) # This is the reduced state space size

    if N == 0:
        return [sp.csr_matrix((K, K)) for _ in C.input_space]

    # Work only with valid subsets from here on
    Y = Y_full[valid_mask]
    V = V_full[valid_mask]
    D = D_full[valid_mask]
    H = H_full[valid_mask]

    # Pre-compute Deterministic Dynamics (Invariant to Action)

    Hat_D = D.copy()
    Hat_H = H.copy()

    if C.M > 0:
        mask_passing = (D[:, 0] == 0)
        if C.M > 1:
            Hat_D[mask_passing, :-1] = D[mask_passing, 1:]
            Hat_H[mask_passing, :-1] = H[mask_passing, 1:]

        Hat_D[mask_passing, C.M-1] = 0
        if len(C.S_h) > 0:
            Hat_H[mask_passing, C.M-1] = C.S_h[0] # Default height

        mask_dec = (Hat_D[:, 0] > 0)
        Hat_D[mask_dec, 0] -= 1

    # Base Pipe Hash (Next state of pipes excluding spawn variation)
    d_part = np.dot(Hat_D.astype(HASH_DTYPE) - mins[2:2+C.M], strides_d)
    h_part = np.dot(Hat_H.astype(HASH_DTYPE) - mins[2+C.M:], strides_h)
    base_pipe_hash = d_part + h_part # Shape (N,)

    # Y Dynamics (Y_next = Y + V)
    # Since Y_next is independent of 'u' (action) and 'w' (noise),
    # we precompute its contribution to the hash.
    Y_next = Y + V
    np.clip(Y_next, 0, C.Y - 1, out=Y_next)

    # Pre-multiply by stride to get the Y-component of the hash
    hash_y_part = (Y_next.astype(HASH_DTYPE) - mins[0]) * stride_y # Shape (N,)

    # Spawn Logic Pre-calc
    # Calculate spawn probability
    sum_hat_D = np.sum(Hat_D, axis=1)
    numerator = (C.X - 1) - sum_hat_D - (C.D_min - 1)
    denom = max(1.0, float(C.X - C.D_min))
    p_spawn = np.clip(numerator / denom, 0.0, 1.0) # Shape (N,)

    # Identify spawning states
    has_spawn_chance = (p_spawn > 0) & (C.M > 0)

    # Prepare Spawn Offsets
    # If a spawn happens, we add offsets to the hash relative to base_pipe_hash
    k_spawn_indices = np.full(N, C.M - 1, dtype=int)
    if C.M > 1:
        is_zero = (Hat_D[:, 1:] == 0)
        any_zero = np.any(is_zero, axis=1)
        # Get first 0
        k_spawn_indices[any_zero] = np.argmax(is_zero[any_zero], axis=1) + 1

    s_fill = np.clip((C.X - 1) - sum_hat_D, C.D_min, C.X - 1).astype(HASH_DTYPE)

    # Hash offsets for D and H
    # Offset D: s_fill * stride_d[k]
    spawn_offset_D = s_fill * strides_d[k_spawn_indices]

    # Offset H: We need offsets for EACH possible height in C.S_h
    # Result: (N_spawn_candidates, num_heights)
    if len(C.S_h) > 0:
        h_vals = np.array(C.S_h, dtype=HASH_DTYPE)
        h_diffs = h_vals - C.S_h[0] # Relative to the default we put in Hat_H
        # Broadcast multiply: (N,) * (H,) -> (N, H)
        spawn_stride_h = strides_h[k_spawn_indices]
        spawn_offset_H = np.outer(spawn_stride_h, h_diffs) # (N, n_heights)
    else:
        spawn_offset_H = np.zeros((N, 1), dtype=HASH_DTYPE)

    # Action Loop
    P_sparse_list = []
    min_v, max_v = mins[1], maxs[1]

    # Probabilities
    prob_h_new = 1.0 / len(C.S_h) if len(C.S_h) > 0 else 0.0

    for u in C.input_space:

        # Noise
        if u == C.U_strong:
            W_flap = np.arange(-C.V_dev, C.V_dev + 1, dtype=np.int32)
        else:
            W_flap = np.array([0], dtype=np.int32)

        prob_flap = 1.0 / len(W_flap)

        # V_next: Shape (N, n_w)
        V_next = V[:, None] + u + W_flap[None, :] - C.g
        np.clip(V_next, -C.V_max, C.V_max, out=V_next)

        # Hash Part V: (N, n_w)
        hash_v_part = (V_next.astype(HASH_DTYPE) - min_v) * stride_v

        # Combine invariants: Y_part + Pipe_part (Broadcasting N,1) + V_part (N,W)
        # base_hash shape: (N, n_w)
        base_hash = (hash_y_part[:, None] + base_pipe_hash[:, None]) + hash_v_part

        # No Spawn Scenario (Happens for all states)
        # Prob = prob_flap * (1 - p_spawn)
        probs_ns = prob_flap * (1.0 - p_spawn) # (N,)

        # Filter where prob > 0 to save lookup time
        mask_ns = (probs_ns > 0)

        if np.any(mask_ns):
            # Subset
            h_ns = base_hash[mask_ns].ravel() # (N_subset * W)
            src_ns = valid_src_indices[mask_ns]
            p_ns = probs_ns[mask_ns]

            # Lookup
            dest_ns = lookup_table[h_ns]

            # Source must repeat W times
            rows_ns = np.repeat(src_ns, len(W_flap))
            cols_ns = dest_ns
            data_ns = np.repeat(p_ns, len(W_flap)) # Prob is constant across W for this part

            # Filter invalid destinations (-1)
            valid_t = (cols_ns != -1)
            rows_ns = rows_ns[valid_t]
            cols_ns = cols_ns[valid_t]
            data_ns = data_ns[valid_t]
        else:
            rows_ns, cols_ns, data_ns = [], [], []

        # Spawn Scenario
        if np.any(has_spawn_chance) and len(C.S_h) > 0:
            # Subset to only spawning candidates
            # Base Hash for these candidates: (N_spawn, W)
            h_base_s = base_hash[has_spawn_chance]
            src_s = valid_src_indices[has_spawn_chance]
            p_base_s = p_spawn[has_spawn_chance] * prob_flap * prob_h_new # Scalar per state

            # Offsets (N_spawn, n_heights)
            off_d = spawn_offset_D[has_spawn_chance] # (N_spawn,)
            off_h = spawn_offset_H[has_spawn_chance] # (N_spawn, H)

            # We need to broadcast (N, W) with (N, H)
            # Result: (N, W, H)
            # Total Hash = Base(N,W,1) + OffD(N,1,1) + OffH(N,1,H)
            total_hash_s = (h_base_s[:, :, None] + off_d[:, None, None] + off_h[:, None, :])

            # Flatten
            h_flat_s = total_hash_s.ravel()
            dest_s = lookup_table[h_flat_s]

            # Create COO data
            n_w = len(W_flap)
            n_h = len(C.S_h)

            # Rows: repeat src N times (W*H)
            rows_s = np.repeat(src_s, n_w * n_h)
            cols_s = dest_s
            data_s = np.repeat(p_base_s, n_w * n_h)

            valid_t = (cols_s != -1)
            rows_s = rows_s[valid_t]
            cols_s = cols_s[valid_t]
            data_s = data_s[valid_t]
        else:
            rows_s, cols_s, data_s = [], [], []

        # Concatenate lists is faster than appending numpy arrays incrementally
        all_rows = np.concatenate([rows_ns, rows_s]) if len(rows_s) > 0 else rows_ns
        all_cols = np.concatenate([cols_ns, cols_s]) if len(rows_s) > 0 else cols_ns
        all_data = np.concatenate([data_ns, data_s]) if len(rows_s) > 0 else data_ns

        # Construct CSR directly from COO data
        # Sum duplicates (implicitly handled by coo -> csr conversion
        P = sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(K, K)).tocsr()
        P_sparse_list.append(P)

    return P_sparse_list

def compute_transition_probabilities_memsafe(C: Const, K, S_arr):
    # Convert State Space to Matrix
    N = K

    # Columns: Y, V, D[0]...D[M-1], H[0]...H[M-1]
    Y = S_arr[:, 0]
    V = S_arr[:, 1]
    D = S_arr[:, 2 : 2 + C.M]
    H = S_arr[:, 2 + C.M : 2 + 2 * C.M]

    dtype_view = np.dtype((np.void, S_arr.dtype.itemsize * S_arr.shape[1]))
    S_void = np.ascontiguousarray(S_arr).view(dtype_view).ravel()
    sort_order = np.argsort(S_void)
    S_void_sorted = S_void[sort_order]

    def lookup_state_indices(next_states_matrix):
        next_void = np.ascontiguousarray(next_states_matrix.astype(np.int32)).view(dtype_view).ravel()
        search_indices = np.searchsorted(S_void_sorted, next_void)
        search_indices = np.clip(search_indices, 0, N - 1)
        found_void = S_void_sorted[search_indices]
        valid_mask = (found_void == next_void)
        return sort_order[search_indices], valid_mask

    if C.M > 0:
        gap_tol = (C.G - 1) // 2
        is_collided = (D[:, 0] == 0) & (np.abs(Y - H[:, 0]) > gap_tol)
    else:
        is_collided = np.zeros(N, dtype=bool)

    Hat_D = D.copy()
    Hat_H = H.copy()

    if C.M > 0:
        mask_passing = (D[:, 0] == 0)

        if C.M > 1:
            Hat_D[mask_passing, :-1] = D[mask_passing, 1:]
            Hat_H[mask_passing, :-1] = H[mask_passing, 1:]

        # Set last element to 0 / default height
        Hat_D[mask_passing, C.M-1] = 0
        if len(C.S_h) > 0:
            Hat_H[mask_passing, C.M-1] = C.S_h[0]

        mask_dec = (Hat_D[:, 0] > 0)
        Hat_D[mask_dec, 0] -= 1

    # Spawn Parameter Calculation
    sum_hat_D = np.sum(Hat_D, axis=1)
    s_values = (C.X - 1) - sum_hat_D

    # PDF Formula for p_spawn(s) implemented vectorized
    # s <= Dmin - 1: 0
    # Dmin <= s <= X - 1: linear ramp
    # s >= X: 1
    numerator = s_values - (C.D_min - 1)
    denominator = float(C.X - C.D_min)
    p_linear = numerator / denominator
    p_spawn_vec = np.clip(p_linear, 0.0, 1.0)

    # Find m_min: "smallest index with no assigned obstacle"
    # Python indices 1..M-If none, default to M-1.
    k_spawn_indices = np.full(N, C.M - 1, dtype=int)

    if C.M > 1:
        search_view = Hat_D[:, 1:]
        is_zero_view = (search_view == 0)
        first_zero_rel = np.argmax(is_zero_view, axis=1)
        any_zero_found = np.any(is_zero_view, axis=1)
        k_spawn_indices[any_zero_found] = first_zero_rel[any_zero_found] + 1

    # Iterate Inputs & Build Matrix
    prob_h_new = 1.0 / len(C.S_h) if len(C.S_h) > 0 else 0.0
    U_array = np.array(C.input_space)

    P_data = [[] for _ in range(C.L)]
    P_rows = [[] for _ in range(C.L)]
    P_cols = [[] for _ in range(C.L)]

    for l_idx, u in enumerate(U_array):
        # Probabilistic Velocities
        if u == C.U_strong:
            W_flap = np.arange(-C.V_dev, C.V_dev + 1)
        else:
            W_flap = np.array([0])
        prob_flap = 1.0 / len(W_flap)
        n_w = len(W_flap)

        # Broadcast V + u + W (Calculates v_{k+1})
        V_next_matrix = V[:, None] + u + W_flap[None, :] - C.g
        V_next_flat = np.clip(V_next_matrix, -C.V_max, C.V_max).flatten()

        # Calculate y_{k+1} based on CURRENT velocity v_k (PDF Page 5 formula)
        # y_{k+1} = min(max(y_k + v_k, 0), Y-1)
        # We repeat Y and V (current) to match the shape of the expanded arrays
        Y_repeated = np.repeat(Y, n_w)
        V_current_repeated = np.repeat(V, n_w)
        Y_next_flat = np.clip(Y_repeated + V_current_repeated, 0, C.Y - 1).astype(np.int32)

        # Source filtering (Collided states are dead ends)
        source_idxs = np.repeat(np.arange(N), n_w)
        valid_src = ~np.repeat(is_collided, n_w)

        V_next = V_next_flat[valid_src]
        Y_next = Y_next_flat[valid_src]
        source_idxs = source_idxs[valid_src]

        # Get pre-computed pipe params for valid rows
        Hat_D_sub = Hat_D[source_idxs]
        Hat_H_sub = Hat_H[source_idxs]
        p_spawn_sub = p_spawn_vec[source_idxs]
        k_spawn_sub = k_spawn_indices[source_idxs]

        # Path 1: No Spawn
        probs_ns = prob_flap * (1.0 - p_spawn_sub)
        mask_ns = probs_ns > 0

        if np.any(mask_ns):
            NS_states = np.column_stack((Y_next[mask_ns], V_next[mask_ns], Hat_D_sub[mask_ns], Hat_H_sub[mask_ns]))
            dest_idx, valid = lookup_state_indices(NS_states)

            keep = valid
            P_rows[l_idx].append(source_idxs[mask_ns][keep])
            P_cols[l_idx].append(dest_idx[keep])
            P_data[l_idx].append(probs_ns[mask_ns][keep])

        # Path 2: Spawn
        mask_s = (p_spawn_sub > 0)

        if np.any(mask_s) and len(C.S_h) > 0:
            S_Y = Y_next[mask_s]
            S_V = V_next[mask_s]
            S_D = Hat_D_sub[mask_s]
            S_H = Hat_H_sub[mask_s]
            S_k = k_spawn_sub[mask_s]
            S_base_prob = prob_flap * p_spawn_sub[mask_s]

            # Calculate s to fill (re-computed for flattened subset)
            S_sum_d = np.sum(S_D, axis=1)
            s_fill = np.clip((C.X - 1) - S_sum_d, C.D_min, C.X - 1)

            # Iterate over all possible new heights (uniform prob)
            for h_new in C.S_h:
                D_new = S_D.copy()
                H_new = S_H.copy()
                rows = np.arange(len(S_Y))

                # Assign new pipe to the identified slot
                D_new[rows, S_k] = s_fill
                H_new[rows, S_k] = h_new

                S_states = np.column_stack((S_Y, S_V, D_new, H_new))
                dest_idx, valid = lookup_state_indices(S_states)

                keep = valid
                P_rows[l_idx].append(source_idxs[mask_s][keep])
                P_cols[l_idx].append(dest_idx[keep])
                P_data[l_idx].append(S_base_prob[keep] * prob_h_new)

    # Construct Sparse Matrices
    P_sparse = []
    for l in range(C.L):
        if len(P_data[l]) > 0:
            data = np.concatenate(P_data[l])
            rows = np.concatenate(P_rows[l])
            cols = np.concatenate(P_cols[l])
            P_l = sp.csr_matrix((data, (rows, cols)), shape=(C.K, C.K))
            P_sparse.append(P_l)
        else:
            P_sparse.append(sp.csr_matrix((C.K, C.K)))

    return P_sparse

def compute_transition_probabilities(C: Const, K, S_arr):
    try:
        return compute_transition_probabilities_lookup(C, K, S_arr)
    except:
        log("Failed to allocate lookup table, trying memsafe transition probabilities")
        return compute_transition_probabilities_memsafe(C, K, S_arr)

def compute_expected_stage_cost_fast(C: Const, K: int):
    return np.tile(np.array([
        -1,                # Cost for action 0 (None)
        C.lam_weak - 1,    # Cost for action 1 (Weak)
        C.lam_strong - 1   # Cost for action 2 (Strong)
    ]), (K, 1))


def solver_PI_With_M(C, K, L, P_list, P_stack, Q):
    gamma = 1.0
    dtype = np.float64

    log("Pre-calculating System Matrices (A_all)...")
    A_all = build_A_fast_setup(K, L, P_stack, gamma, dtype)

    # Initialization
    J = np.zeros(K, dtype=dtype)
    policy = np.zeros(K, dtype=int)

    # Pre-allocate indices
    range_k = np.arange(K, dtype=int)

    # Tuning

    # Tolerances
    gmres_tol = 1e-5

    # Iteration limits
    max_outer_iters = 200
    gmres_restart = 60
    max_inner_iters = 30

    # Policy Iteration Loop
    for outer_iter in range(max_outer_iters):
        J_prev = J.copy()

        A_sparse = build_A_fast(A_all, K, policy, range_k)

        # Preconditioner
        M = make_preconditioner(A_sparse, omega=0.8, inner_iters=5, dtype=dtype)

        b = Q[range_k, policy].astype(dtype)

        # Solve Linear System
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

        P_J_all = P_stack.dot(J_eval)

        # Reshape to (L, K) -> Transpose to (K, L)
        future_costs = P_J_all.reshape((L, K)).T
        Q_J = Q + gamma * future_costs

        new_policy = np.argmin(Q_J, axis=1)

        # Check for convergence
        policy_changes = np.sum(new_policy != policy)

        J = J_eval
        policy = new_policy

        if policy_changes == 0 and np.allclose(J_eval, J_prev, atol=1e-4, rtol=1e-7):
            log(f"Converged in {outer_iter+1} iterations.")
            break

        if outer_iter == max_outer_iters - 1:
            log("Warning: Max outer iterations reached without convergence.")

    return J, policy

def solver_PI_No_M(C, K, L, P_list, P_stack, Q):
    gamma = 1.0
    dtype = np.float64

    log("Pre-calculating System Matrices (A_all)...")
    A_all = build_A_fast_setup(K, L, P_stack, gamma, dtype)

    # Initialization
    J = np.zeros(K, dtype=dtype)
    policy = np.zeros(K, dtype=int)

    # Pre-allocate indices
    range_k = np.arange(K, dtype=int)

    # Tuning

    # Tolerances
    gmres_tol = 1e-5

    # Iteration limits
    max_outer_iters = 200
    gmres_restart = 60
    max_inner_iters = 30

    # Policy Iteration Loop
    for outer_iter in range(max_outer_iters):
        J_prev = J.copy()

        # No arithmetic here, just picking rows from A_all
        A_sparse = build_A_fast(A_all, K, policy, range_k)

        # C. Right-Hand Side
        b = Q[range_k, policy].astype(dtype)

        # Solve Linear System
        try:
            J_eval, info = spla.gmres(
                A_sparse,
                b,
                x0=J,
                tol=gmres_tol,
                restart=gmres_restart,
                maxiter=max_inner_iters,
            )
            if info != 0:
                J_eval = spla.spsolve(A_sparse, b)
        except Exception:
            J_eval = spla.spsolve(A_sparse, b)

        P_J_all = P_stack.dot(J_eval)

        # Reshape to (L, K) -> Transpose to (K, L)
        future_costs = P_J_all.reshape((L, K)).T
        Q_J = Q + gamma * future_costs

        new_policy = np.argmin(Q_J, axis=1)

        policy_changes = np.sum(new_policy != policy)

        J = J_eval
        policy = new_policy

        if policy_changes == 0 and np.allclose(J_eval, J_prev, atol=1e-4, rtol=1e-7):
            log(f"Converged in {outer_iter+1} iterations.")
            break

        if outer_iter == max_outer_iters - 1:
            log("Warning: Max outer iterations reached without convergence.")

    return J, policy

def solver_LP(C, K, L, P_list, P_stack, Q):
    b = Q.flatten(order='F')

    c = np.full(K, -1, np.int64)

    # Create a sparse identity matrix
    I_sparse = sp.eye(K, format='csc')

    # Create a list to hold the sparse blocks (I - P_l)
    A_blocks = []

    # Loop over all actions
    for l in range(L):
        # Add the sparse (I - P_l) block to our list
        A_blocks.append(I_sparse - P_list[l])

    # Stack all blocks vertically into one sparse matrix
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

def select_solver(K, L, cutoff=7000):
    if K > cutoff:
        return solver_PI_With_M
    else:
        return solver_PI_No_M

def solution(C: Const) -> tuple[np.ndarray, np.ndarray]:
    T_start = record_time()

    log("Generating state space")
    T_state_start = record_time()
    K, S_arr = generate_state_space(C)
    L = C.L
    T_state_end = record_time()


    log("Computing Transition Probabilities...")
    T_prob_start = record_time()
    P_list = compute_transition_probabilities(C, K, S_arr)
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
    if SOLVER_DEV_MODE:
        T_total_ms  = (record_time() - T_start)   * 1e3
        T_state_ms  = (T_state_end - T_state_start)     * 1e3
        T_prob_ms   = (T_prob_end - T_prob_start)       * 1e3
        T_cost_ms   = (T_cost_end - T_cost_start)       * 1e3
        T_solver_ms = (T_solver_end - T_solver_start)   * 1e3

        global timing_array
        timing_array.append((
            K, solver.__name__,
                T_total_ms,
                T_state_ms,
                T_prob_ms,
                T_cost_ms,
                T_solver_ms,
        ))

        log("\n--- Timing Summary (Pre-calc A_all + Custom Precond)")
        log(f"State generation:   {T_state_ms:.3f}ms")
        log(f"Transition Setup:   {T_prob_ms:.3f}ms")
        log(f"Stage Costs:        {T_cost_ms:.3f}ms")
        log(f"Solver:             {T_solver_ms:.3f}ms")
        log(f"Total Runtime:      {T_total_ms:.3f}ms")

    return J, policy