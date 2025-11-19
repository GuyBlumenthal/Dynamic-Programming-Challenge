"""ComputeTransitionProbabilities.py

Template to compute the transition probability matrix.

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
from Const import Const

from functools import lru_cache

# TODO: Try different versions (csc, csr, bsr, coo)
from scipy.sparse import csc_matrix, coo_matrix

from utils import CustomStateSpace

def compute_transition_probabilities_sparse(C:Const, state_to_index_array, K, valid_states_with_indices) -> list:
    """Computes the transition probability matrix P as a list of sparse matrices."""

    # Each action in C.input_space will have its own sparse probability matrix
    num_inputs = len(C.input_space)

    # 1. Estimate Max NNZ per input layer to pre-allocate memory.
    # Max transitions: A state splits into multiple outcomes via wind (W_v) and spawn heights (S_h).
    # Formula: K states * Max_Wind_Variations * Max_Spawn_Variations
    max_branches = len(C.W_v) * len(C.S_h) + 2 # +2 for safety buffer
    est_nnz = K * max_branches

    # 2. Pre-allocate arrays. Use int32 for indices to save memory/bandwidth.
    # We create a list of arrays, one set for each input action.
    data_buf = [
        np.zeros(est_nnz, dtype=np.float64)
        for _ in range(num_inputs)
    ]

    rows_buf = [
        np.zeros(est_nnz, dtype=np.int32)
        for _ in range(num_inputs)
    ]


    cols_buf = [
        np.zeros(est_nnz, dtype=np.int32)
        for _ in range(num_inputs)
    ]

    bufs = [
        (
            data_buf[input_index],
            rows_buf[input_index],
            cols_buf[input_index],
        ) for input_index in range(num_inputs)
    ]

    bufs_iters = [
        zip(
            np.nditer(data_buf[input_index], op_flags=['readwrite']),
            np.nditer(rows_buf[input_index], op_flags=['readwrite']),
            np.nditer(cols_buf[input_index], op_flags=['readwrite'])
        ) for input_index in range(num_inputs)
    ]

    # Store variables once instead of recalculating
    Y_limit = C.Y - 1
    G_limit = (C.G - 1) / 2
    X, D_min, D_min_1, X_D_min = C.X, C.D_min, C.D_min - 1, C.X - C.D_min
    V_max, g = C.V_max, C.g
    S_h, S_h_0 = C.S_h, C.S_h[0]
    p_height = 1 / len(S_h)
    U_strong_prob = 1 / (2 * C.V_dev + 1)
    W_v = C.W_v

    U = [
        [0, C.U_no_flap, 1, [0]],
        [1, C.U_weak, 1, [0]],
        [2, C.U_strong, U_strong_prob, W_v]
    ]

    max_v = V_max + max(C.input_space) + max(W_v) - g
    min_v = -V_max + min(C.input_space) + min(W_v) - g

    Y_LOOKUP = {y_v: min(Y_limit, max(0, y_v)) for y_v in range(0-V_max, Y_limit+V_max+1)}
    V_LOOKUP = {(v+g): min(V_max, max(-V_max, v)) for v in range(min_v, max_v+1)}

    for y_i, v_i, d_i, h_i, state_index, m_min in valid_states_with_indices:
        if d_i[0] == 0: # Passing
            if abs(y_i - h_i[0]) > G_limit:
                continue # Crash!
            dhat_j = (d_i[1] - 1, *d_i[2:], 0)
            hhat_j = (*h_i[1:], S_h_0)

            # We need to update m_min!
            m_min = dhat_j[1:].index(0) + 1
        else:
            dhat_j = (d_i[0] - 1, *d_i[1:])
            hhat_j = h_i

        s = X - 1 - sum(dhat_j)

        if s < D_min:
            p_spawn = 0
            p_no_spawn = 1
        elif s >= X - 1:
            p_spawn = 1
            p_no_spawn = 0
        else:
            p_spawn = (s - D_min_1) / X_D_min
            p_no_spawn = 1 - p_spawn

        if p_spawn > 0:
            spawn_array = state_to_index_array[Y_LOOKUP[y_i + v_i], :, *dhat_j[:m_min], s, *dhat_j[m_min+1:], *hhat_j[:m_min], :, *hhat_j[m_min+1:]]

        if p_no_spawn > 0:
            no_spawn_array = state_to_index_array[Y_LOOKUP[y_i + v_i], :, *dhat_j, *hhat_j]

        # Each input will push to a seperate index in coo_*
        # Important note: Duplicate entries will be SUMMED!
        # This means that we do not have to worry about two inputs resulting in the same next_state (See += in compute_transition_probabilities)
        for input_index, u_k, p_flap, W_v_list in U:
            buf_iter = bufs_iters[input_index]

            p_a = p_flap * p_no_spawn
            p_b = p_flap * p_spawn * p_height
            for w_v in W_v_list:
                v_j = V_LOOKUP[v_i + u_k + w_v]

                # Case 1: No spawn
                if p_no_spawn > 0:
                    j_index = no_spawn_array[v_j]

                    # DIRECT ARRAY FILL - NO APPEND
                    data, row, col = next(buf_iter)
                    data[...], row[...], col[...] = p_a, state_index, j_index

                # Case 2: Spawn
                if p_spawn > 0:
                    for height in S_h:
                        j_index = spawn_array[v_j, height]
                        data, row, col = next(buf_iter)
                        data[...], row[...], col[...] = p_b, state_index, j_index

    # Construct the sparse matrices
    P_sparse_list = []
    for l in range(num_inputs):
        data, rows, cols = bufs[l]

        # FAST: Creating coo_matrix from existing numpy arrays is O(1) pointer copy
        P_l_coo = coo_matrix(
            (data, (rows, cols)),
            shape=(K, K)
        )

        # FAST: Summing duplicates happens here in C++
        P_sparse_list.append(P_l_coo.tocsc())

    return P_sparse_list

def compute_transition_probabilities(C:Const) -> np.array:
    """Computes the transition probability matrix P.

    Args:
        C (Const): The constants describing the problem instance.

    Returns:
        np.array: Transition probability matrix of shape (K,K,L), where:
            - K is the size of the state space;
            - L is the size of the input space.
            - P[i,j,l] corresponds to the probability of transitioning
              from the state i to the state j when input l is applied.
    """

    state_space = CustomStateSpace()

    K, state_to_index_dict, valid_state_with_indices = state_space.custom_state_space(C)
    P_sparse_list = compute_transition_probabilities_sparse(C, state_to_index_dict, K, valid_state_with_indices)

    P = np.empty((K, K, C.L), dtype=P_sparse_list[0].dtype)

    # Fill P
    for l in range(C.L):
        P[:, :, l] = P_sparse_list[l].toarray()

    return P