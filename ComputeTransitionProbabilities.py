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

    # A calculated probability P[curr_state, next_state, action] is stored as
    #   coo_data[action] = P[curr_state, next_state, action]
    #   coo_cols[action] = curr_state
    #   coo_rows[action] = next_state
    coo_data = [[] for input_i in range(num_inputs)]
    coo_cols = [[] for input_i in range(num_inputs)]
    coo_rows = [[] for input_i in range(num_inputs)]

    # Helper functions for populating coo table
    append_data = [l.append for l in coo_data]
    append_rows = [l.append for l in coo_rows]
    append_cols = [l.append for l in coo_cols]

    # Store variables once instead of recalculating
    Y_limit = C.Y - 1
    G_limit = (C.G - 1) / 2
    M = C.M
    X, D_min = C.X, C.D_min
    V_max, g = C.V_max, C.g
    S_h, S_h_0 = C.S_h, C.S_h[0]
    p_height = 1 / len(S_h)
    U_strong_prob = 1 / (2 * C.V_dev + 1)
    W_v = C.W_v

    # --- Create mappings for non-contiguous state variables ---
    v_offset = C.V_max

    StateVar_Y = 0
    StateVar_V = 1
    StateVar_D_1 = 2
    StateVar_D_2 = StateVar_D_1 + 1
    StateVar_D_M = StateVar_D_1 + M - 1
    StateVar_H_1 = StateVar_D_M + 1
    StateVar_H_2 = StateVar_H_1 + 1
    StateVar_H_M = StateVar_H_1 + M - 1

    U = [
        [0, C.U_no_flap, 1, [0]],
        [1, C.U_weak, 1, [0]],
        [2, C.U_strong, U_strong_prob, W_v]
    ]

    # Cache all possible Vs
    v_space = C.S_v
    u_space = C.input_space

    max_v = V_max + max(C.input_space) + max(W_v) - g
    min_v = -V_max + min(C.input_space) + min(W_v) - g

    Y_LOOKUP_OFFSET = abs(0-V_max)
    Y_LOOKUP = [min(Y_limit, max(0, y_v)) for y_v in range(0-V_max, Y_limit+V_max+1)]

    V_LOOKUP_OFFSET = abs(min_v)
    V_LOOKUP = [min(V_max, max(-V_max, v)) for v in range(min_v, max_v+1)]

    for state_i, state_index in valid_states_with_indices:
        y_j = Y_LOOKUP[Y_LOOKUP_OFFSET + state_i[StateVar_Y] + state_i[StateVar_V]]
        if state_i[StateVar_D_1] == 0: # Passing
            if abs(state_i[StateVar_Y] - state_i[StateVar_H_1]) > G_limit:
                continue # Crash!
            dhat_j = (state_i[StateVar_D_2] - 1, *state_i[StateVar_D_2+1:StateVar_D_M+1], 0)
            hhat_j = (*state_i[StateVar_H_2:StateVar_H_M + 1], S_h_0)
        else:
            dhat_j = (state_i[StateVar_D_1] - 1, *state_i[StateVar_D_2:StateVar_D_M + 1])
            hhat_j = (*state_i[StateVar_H_1:StateVar_H_M + 1], )
        s = X - 1 - sum(dhat_j)
        p_spawn = (s - (D_min - 1)) / (X - D_min)
        # TODO: Can p_spawn ever be 1 or greater?
        if p_spawn < 0:
            p_spawn = 0
        elif p_spawn > 1:
            p_spawn = 1
        # p_spawn = min(1, max(0, p_spawn))
        p_no_spawn = 1 - p_spawn
        try:
            # m_min will be the Python index from the start of the list
            m_min = dhat_j[1:].index(0) + 1
        except ValueError:
            # No zero was found in the slice, so m_min defaults to M-1
            m_min = M - 1
        dspawn_j = None
        if p_spawn > 0:
            dspawn_j = tuple(dhat_j[:m_min] + (s, ) + dhat_j[m_min+1:])
            h_prefix = tuple(hhat_j[:m_min])
            h_suffix = tuple(hhat_j[m_min+1:])

        # Each input will push to a seperate index in coo_*
        # Important note: Duplicate entries will be SUMMED!
        # This means that we do not have to worry about two inputs resulting in the same next_state (See += in compute_transition_probabilities)
        for input_index, u_k, p_flap, W_v_list in U:
            for w_v in W_v_list:
                v_j = V_LOOKUP[V_LOOKUP_OFFSET + state_i[StateVar_V] + u_k + w_v - g]

                # Case 1: No spawn
                if p_no_spawn > 0:
                    index_tuple = tuple([y_j, v_j + v_offset] + list(dhat_j) + list(hhat_j))
                    j_index = state_to_index_array[index_tuple]

                    append_data[input_index](p_flap * p_no_spawn)
                    append_rows[input_index](state_index)
                    append_cols[input_index](j_index)

                # Case 2: Spawn
                if p_spawn > 0:
                    p_combined = p_flap * p_spawn * p_height
                    for height in S_h:
                        h_spawn = h_prefix + (height,) + h_suffix
                        
                        index_tuple = tuple([y_j, v_j + v_offset] + list(dspawn_j) + list(h_spawn))
                        j_index = state_to_index_array[index_tuple]

                        append_data[input_index](p_combined)
                        append_rows[input_index](state_index)
                        append_cols[input_index](j_index)

    # Construct the sparse matrices
    P_sparse_list = []
    for l in range(C.L):
        # Build as COO first (fastest for this input)
        P_l_coo = coo_matrix(
            (coo_data[l], (coo_rows[l], coo_cols[l])),
            shape=(K, K)
        )
        # Convert to CSC (fast, sums duplicates)
        P_l = P_l_coo.tocsc()
        P_sparse_list.append(P_l)

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