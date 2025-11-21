"""utils.py

Python script containg utility functions. Modify if needed,
but be careful as these functions are used, e.g., in simulation.py.

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

from multiprocessing import Pool, cpu_count
from collections import defaultdict

from Const import Const
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import numpy as np
from tqdm import tqdm

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

def spawn_probability(C: Const, s: int) -> float:
    """Distance-dependent spawn probability p_spawn(s).

    Args:
        C (Const): The constants describing the problem instance.
        s (int): Free distance, as defined in the assignment.

    Returns:
        float: The spawn probability p_spawn(s).
    """
    return max(min((s - C.D_min + 1) / (C.X - C.D_min), 1.0), 0.0)

def is_in_gap(C: Const, y: int, h1: int) -> bool:
    """Returns true if bird in gap.

    Args:
        C (Const): The constants describing the problem instance.
        y (int): Vertical position of the bird.
        h1 (int): Center of the gap of the first obstacle.

    Returns:
        bool: True if bird is in the gap, False otherwise.
    """
    half = (C.G - 1) // 2
    return abs(y - h1) <= half

def is_passing(C: Const, y: int, d1: int, h1: int) -> bool:
    """Return true if bird is currently passing the gap without colliding.

    Args:
        C (Const): The constants describing the problem instance.
        y (int): Vertical position of the bird.
        d1 (int): Distance to the first obstacle.
        h1 (int): Center of the gap of the first obstacle.

    Returns:
        bool: True if bird is passing the gap, False otherwise.
    """
    return (d1 == 0) and is_in_gap(C, y, h1)

def is_collision(C: Const, y: int, d1: int, h1: int) -> bool:
    """Return true if bird is colliding with obstacle.

    Args:
        C (Const): The constants describing the problem instance.
        y (int): Vertical position of the bird.
        d1 (int): Distance to the first obstacle.
        h1 (int): Center of the gap of the first obstacle.

    Returns:
        bool: True if bird is colliding with obstacle, False otherwise.
    """
    return (d1 == 0) and not is_in_gap(C, y, h1)

def precompute_velocity_transitions(C: Const):
    VELOCITY_TRANSITIONS = {}
    for v in range(-C.V_max, C.V_max + 1):
        for l, u in enumerate(C.input_space):

            if u == C.U_strong:
                W_flap = range(-C.V_dev, C.V_dev + 1)
                prob_flap = 1.0 / len(W_flap)
            else:
                W_flap = [0]
                prob_flap = 1.0

            v_agg = defaultdict(float)
            for w_flap in W_flap:
                v_next = int(np.clip(v + u + w_flap - C.g, -C.V_max, C.V_max))
                v_agg[v_next] += prob_flap

            VELOCITY_TRANSITIONS[(l, v)] = list(v_agg.items())

    return VELOCITY_TRANSITIONS

def compute_transition_probabilities_fast(C: Const):
    P_data = [[] for _ in range(C.L)]
    P_rows = [[] for _ in range(C.L)]
    P_cols = [[] for _ in range(C.L)]

    STATE_MAP = {s: i for i, s in enumerate(C.state_space)}

    VELOCITY_TRANSITIONS = precompute_velocity_transitions(C)

    prob_h_new = 1 / len(C.S_h) if len(C.S_h) > 0 else 0

    for state_tuple, i in STATE_MAP.items():

        y, v = state_tuple[0], state_tuple[1]
        D = list(state_tuple[2:2+C.M])
        H = list(state_tuple[2+C.M:])

        if D[0] == 0 and abs(y - H[0]) > (C.G - 1) // 2:
            continue

        y_next_base = int(np.clip(y + v, 0, C.Y-1))

        hat_D = D.copy()
        hat_H = H.copy()

        if C.M > 0 and D[0] == 0:
            for k in range(C.M - 1):
                hat_D[k] = D[k+1]
                hat_H[k] = H[k+1]
            hat_D[C.M-1] = 0
            hat_H[C.M-1] = C.S_h[0]

        if C.M > 0 and hat_D[0] > 0:
            hat_D[0] = hat_D[0] - 1

        s = (C.X - 1) - sum(hat_D)
        p_spawn = spawn_probability(C, s)

        k_spawn = -1
        if p_spawn > 0:
            for k in range(1, C.M):
                if hat_D[k] == 0:
                    k_spawn = k
                    break

        if p_spawn > 0 and k_spawn != -1:
            D_template_list = list(hat_D)
            D_template_list[k_spawn] = np.clip(s, C.D_min, C.X-1)
            D_template = tuple(D_template_list)

            H_prefix = hat_H[:k_spawn]
            H_suffix = hat_H[k_spawn+1:]

        for l in range(C.L):
            v_transition_list = VELOCITY_TRANSITIONS.get((l, v), [])

            for v_next, prob_flap_agg in v_transition_list:

                YV_template = (y_next_base, v_next)

                if p_spawn < 1:
                    next_state_no_spawn = (*YV_template, *hat_D, *hat_H)
                    try:
                        j = STATE_MAP[next_state_no_spawn]
                        prob_to_add = prob_flap_agg * (1 - p_spawn)

                        P_data[l].append(prob_to_add)
                        P_rows[l].append(i)
                        P_cols[l].append(j)
                    except (KeyError, TypeError):
                        pass

                if p_spawn > 0 and k_spawn != -1:
                    for h_new in C.S_h:

                        H_vector_tuple = (*H_prefix, h_new, *H_suffix)
                        next_state_spawn = (*YV_template, *D_template, *H_vector_tuple)

                        try:
                            j = STATE_MAP[next_state_spawn]
                            prob_to_add = prob_flap_agg * p_spawn * prob_h_new

                            P_data[l].append(prob_to_add)
                            P_rows[l].append(i)
                            P_cols[l].append(j)
                        except(KeyError, TypeError):
                            pass

    P_sparse = []
    for l in range(C.L):
        P_l = sp.csr_matrix((P_data[l], (P_rows[l], P_cols[l])), shape=(C.K, C.K))
        P_l.sum_duplicates()
        P_sparse.append(P_l)

    return P_sparse

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
