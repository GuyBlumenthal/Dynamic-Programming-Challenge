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

def build_A(K, P_stack, policy, range_k, gamma, dtype=np.float64):
    """
    Constructs A = I - gamma * P_pi using vectorized slicing on a stacked P matrix.

    Args:
        K: State space size
        P_stack: Vertically stacked P matrices (shape L*K, K)
        policy: Array of shape (K,) containing actions
        range_k: Pre-computed np.arange(K) for speed
        gamma: Discount factor
    """
    # --- VECTORIZED SLICING ---
    # We want the row from P corresponding to action policy[i] for state i.
    # Since P_stack is stacked vertically, the row for state i and action l
    # is located at index: (l * K) + i
    selector_indices = policy * K + range_k

    # Scipy CSR slicing is highly optimized in C.
    # It extracts the specific rows into a new CSR matrix much faster
    # than adding masked matrices in Python.
    P_pi = P_stack[selector_indices, :]

    I = sp.eye(K, format='csr', dtype=dtype)
    A = I - gamma * P_pi
    return A

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
    Fully vectorized calculation of transition probabilities.
    Strictly adheres to PE_instructions.pdf.

    Corrections Implemented:
    1. Vertical Motion: y_{k+1} = y_k + v_k (Uses CURRENT velocity, not next).
    2. Pipe Logic: Shifts happen first, decrement checks NEW state.
    3. Spawn Probability: Exact linear ramp formula from PDF.
    4. m_min Logic: Defaults to M-1 (last pipe) if buffer is full.
    """

    # 1. Convert State Space to Matrix
    # ---------------------------------------------------------
    S_arr = np.array(C.state_space, dtype=np.int32)
    N = S_arr.shape[0]

    # Columns: Y, V, D[0]...D[M-1], H[0]...H[M-1]
    Y = S_arr[:, 0]
    V = S_arr[:, 1]
    D = S_arr[:, 2 : 2 + C.M]
    H = S_arr[:, 2 + C.M : 2 + 2 * C.M]

    # 2. Pre-process State Lookup (SearchSorted Trick)
    # ------------------------------------------------------
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

    # 3. Deterministic Pipe Dynamics (Per PDF "Dynamics" section)
    # -----------------------------------------------------------

    # A. Collision check
    # "On collision... transition to a cost-free termination state"
    # We identify these source states and ensure they generate NO transitions in P
    # (effectively making them absorbing or exiting the set).
    if C.M > 0:
        gap_tol = (C.G - 1) // 2
        is_collided = (D[:, 0] == 0) & (np.abs(Y - H[:, 0]) > gap_tol)
    else:
        is_collided = np.zeros(N, dtype=bool)

    Hat_D = D.copy()
    Hat_H = H.copy()

    if C.M > 0:
        # B. Intermediate Quantities
        # Case 1: Passing (d[1]=0, no collision). Logic: Shift indices left.
        mask_passing = (D[:, 0] == 0)

        if C.M > 1:
            # Shift indices 2..M to 1..M-1 (Python indices 1..M-1 to 0..M-2)
            Hat_D[mask_passing, :-1] = D[mask_passing, 1:]
            Hat_H[mask_passing, :-1] = H[mask_passing, 1:]

        # Set last element to 0 / default height
        Hat_D[mask_passing, C.M-1] = 0
        if len(C.S_h) > 0:
            Hat_H[mask_passing, C.M-1] = C.S_h[0]

        # C. Drift Decrement
        # For "Normal Drift" (d[1] > 0), hat_d = d - 1.
        # For "Passing", hat_d[1] = d[2] - 1.
        # Since we already shifted d[2] into position 0 for passing states,
        # we simply decrement Hat_D[0] wherever it is > 0.
        mask_dec = (Hat_D[:, 0] > 0)
        Hat_D[mask_dec, 0] -= 1

    # 4. Spawn Parameter Calculation
    # -----------------------------------------------
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
    # Python indices 1..M-1. If none, default to M-1.
    k_spawn_indices = np.full(N, C.M - 1, dtype=int)

    if C.M > 1:
        search_view = Hat_D[:, 1:]
        is_zero_view = (search_view == 0)
        first_zero_rel = np.argmax(is_zero_view, axis=1)
        any_zero_found = np.any(is_zero_view, axis=1)
        k_spawn_indices[any_zero_found] = first_zero_rel[any_zero_found] + 1

    # 5. Iterate Inputs & Build Matrix
    # --------------------------------
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

        # --- Path 1: No Spawn ---
        probs_ns = prob_flap * (1.0 - p_spawn_sub)
        mask_ns = probs_ns > 0

        if np.any(mask_ns):
            NS_states = np.column_stack((Y_next[mask_ns], V_next[mask_ns], Hat_D_sub[mask_ns], Hat_H_sub[mask_ns]))
            dest_idx, valid = lookup_state_indices(NS_states)

            keep = valid
            P_rows[l_idx].append(source_idxs[mask_ns][keep])
            P_cols[l_idx].append(dest_idx[keep])
            P_data[l_idx].append(probs_ns[mask_ns][keep])

        # --- Path 2: Spawn ---
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

    # 6. Construct Sparse Matrices
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

# Keeping this purely for reference, though logic is now inlined for performance
def spawn_probability(C, s):
    if s <= C.D_min - 1:
        return 0.0
    elif s >= C.X:
        return 1.0
    else:
        return (s - (C.D_min - 1)) / (C.X - C.D_min)