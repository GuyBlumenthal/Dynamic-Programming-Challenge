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
from copy import copy

from functools import lru_cache

mdp_dict = None

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

    # Print Hello


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
            dspawn_j = copy(dhat_j)
            dspawn_j[m_min] = s
            hspawn_j = copy(hhat_j)

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