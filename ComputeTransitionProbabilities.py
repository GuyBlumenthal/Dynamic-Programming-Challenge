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

from timer import profile

@profile
def compute_transition_probabilities_sparse(C:Const, state_to_index_dict, K) -> list:
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

    for state_i, state_index in state_to_index_dict.items():
        y_j = Y_LOOKUP[Y_LOOKUP_OFFSET + state_i[StateVar_Y] + state_i[StateVar_V]]
        if state_i[StateVar_D_1] == 0:
            if abs(state_i[StateVar_Y] - state_i[StateVar_H_1]) > G_limit:
                continue
            dhat_j = [state_i[StateVar_D_2] - 1, *state_i[StateVar_D_2+1:StateVar_D_M+1], 0]
            hhat_j = [*state_i[StateVar_H_2:StateVar_H_M + 1], S_h_0]
        else:
            dhat_j = [state_i[StateVar_D_1] - 1, *state_i[StateVar_D_2:StateVar_D_M + 1]]
            hhat_j = [*state_i[StateVar_H_1:StateVar_H_M + 1]]
        s = X - 1 - sum(dhat_j)
        p_spawn = (s - (D_min - 1)) / (X - D_min)
        # TODO: Can p_spawn ever be 1 or greater?
        if p_spawn < 0:
            p_spawn = 0
        elif p_spawn > 1:
            p_spawn = 1
        # p_spawn = min(1, max(0, p_spawn))
        p_no_spawn = 1 - p_spawn
        m_min = M - 1
        for m in range(1, M):
            if dhat_j[m] == 0:
                m_min = m
                break
        dspawn_j = None
        if p_spawn > 0:
            dspawn_j = dhat_j[:]
            dspawn_j[m_min] = s
            dspawn_j = tuple(dspawn_j)
            hspawn_j = hhat_j[:]
            h_prefix = tuple(hhat_j[:m_min])
            h_suffix = tuple(hhat_j[m_min+1:])

        dhat_tuple = tuple(dhat_j)
        hhat_tuple = tuple(hhat_j)

        # Each input will push to a seperate index in coo_*
        # Important note: Duplicate entries will be SUMMED!
        # This means that we do not have to worry about two inputs resulting in the same next_state (See += in compute_transition_probabilities)
        for input_index, u_k, p_flap, W_v_list in U:
            for w_v in W_v_list:
                v_j = V_LOOKUP[V_LOOKUP_OFFSET + state_i[StateVar_V] + u_k + w_v - g]

                # Case 1: No spawn
                if p_no_spawn > 0:
                    next_state = (y_j, v_j) + dhat_tuple + hhat_tuple
                    j_index = state_to_index_dict[next_state]

                    append_data[input_index](p_flap * p_no_spawn)
                    append_rows[input_index](state_index)
                    append_cols[input_index](j_index)

                # Case 2: Spawn
                if p_spawn > 0:
                    spawn_prefix = (y_j, v_j) + dspawn_j
                    p_combined = p_flap * p_spawn * p_height
                    for height in S_h:
                        next_state = spawn_prefix + h_prefix + (height,) + h_suffix
                        j_index = state_to_index_dict[next_state]

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
    P = np.zeros((C.K, C.K, C.L))

    class StateVar:
        Y = 0
        V = 1
        D_1 = 2
        D_2 = D_1 + 1
        D_M = D_1 + C.M - 1
        H_1 = D_M + 1
        H_2 = H_1 + 1
        H_M = H_1 + C.M - 1

    state_to_index = lru_cache(maxsize=None)(C.state_to_index)

    # Avoids repeated attribute lookups (e.g., 'C.Y') inside the loop.
    Y_limit = C.Y - 1
    G_limit = (C.G - 1) / 2
    M = C.M
    X, D_min = C.X, C.D_min
    V_max, g = C.V_max, C.g
    S_h, S_h_0 = C.S_h, C.S_h[0]
    p_height = 1 / len(S_h)
    U_strong_prob = 1 / (2 * C.V_dev + 1)
    W_v = C.W_v

    # Loop through each state and possible actions, set the associated probabilities in P, and the rest remain 0
    for state_i in C.state_space:
        # Handle non input related state dynamics
        state_index = state_to_index(state_i)

        # Next height is known
        y_j = min(Y_limit, max(0, state_i[StateVar.Y] + state_i[StateVar.V]))

        # Are we passing or drifting
        if state_i[StateVar.D_1] == 0:
            # Handle passing
            if abs(state_i[StateVar.Y] - state_i[StateVar.H_1]) > G_limit:
                # Collision, no transition in P
                continue
            dhat_j = [state_i[StateVar.D_2] - 1, *state_i[StateVar.D_2:StateVar.D_M], 0]
            hhat_j = [*state_i[StateVar.H_2:StateVar.H_M + 1], S_h_0]
        else:
            # Handle drifting -> This is deterministic
            dhat_j = [state_i[StateVar.D_1] - 1, *state_i[StateVar.D_2:StateVar.D_M + 1]]
            hhat_j = [*state_i[StateVar.H_1:StateVar.H_M + 1]]

        s = X - 1 - sum(dhat_j)

        p_spawn = (s - (D_min - 1)) / (X - D_min)
        p_spawn = min(1, max(0, p_spawn))
        p_no_spawn = 1 - p_spawn

        # Find first empty d
        m_min = M - 1
        for m in range(1, M):
            if dhat_j[m] == 0:
                m_min = m
                break

        dspawn_j = None
        hspawn_j = None
        if p_spawn > 0:
            dspawn_j = dhat_j[:]
            dspawn_j[m_min] = s
            hspawn_j = hhat_j[:]

        U = [ # input_index, u_k, p_flap, W_v
            [0, C.U_no_flap, 1, [0]],
            [1, C.U_weak, 1, [0]],
            [2, C.U_strong, U_strong_prob, W_v]
        ]

        for input_index, u_k, p_flap, W_v in U:
            # Select flap disturbance
            for w_v in W_v:

                v_j = min(V_max, max(-V_max, state_i[StateVar.V] + u_k + w_v - g))

                # Case 1: No spawn
                if p_no_spawn > 0:
                    next_state = (
                        y_j,
                        v_j,
                        *dhat_j,
                        *hhat_j
                    )

                    P[state_index, state_to_index(next_state), input_index] += p_flap * p_no_spawn

                # Case 2: Spawn
                if p_spawn > 0:
                    p_combined = p_flap * p_spawn * p_height
                    for height in S_h:
                        hspawn_j[m_min] = height

                        next_state = (
                            y_j,
                            v_j,
                            *dspawn_j,
                            *hspawn_j
                        )

                        P[state_index, state_to_index(next_state), input_index] += p_combined

    return P