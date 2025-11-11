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

    # Loop through each state and possible actions, set the associated probabilities in P, and the rest remain 0
    for state_i in C.state_space:
        # Handle non input related state dynamics
        state_index = state_to_index(state_i)

        # Next height is known
        y_j = min(C.Y - 1, max(0, state_i[StateVar.Y] + state_i[StateVar.V]))

        # Are we passing or drifting
        if state_i[StateVar.D_1] == 0:
            # Handle passing
            if abs(state_i[StateVar.Y] - state_i[StateVar.H_1]) > (C.G - 1) / 2:
                # Collision, no transition in P
                continue
            dhat_j = [state_i[StateVar.D_2] - 1, *state_i[StateVar.D_2:StateVar.D_M], 0]
            hhat_j = [*state_i[StateVar.H_2:StateVar.H_M + 1], Const.S_h[0]]
        else:
            # Handle drifting -> This is deterministic
            dhat_j = [state_i[StateVar.D_1] - 1, *state_i[StateVar.D_2:StateVar.D_M + 1]]
            hhat_j = [*state_i[StateVar.H_1:StateVar.H_M + 1]]

        s = C.X - 1 - sum(dhat_j)

        p_spawn = (s - (C.D_min - 1)) / (C.X - C.D_min)
        p_spawn = min(1, max(0, p_spawn))

        p_height = 1 / len(C.S_h)

        # Find first empty d
        m_min = C.M - 1
        for m in range(1, C.M):
            if dhat_j[m] == 0:
                m_min = m
                break

        U = [ # input_index, u_k, p_flap, W_v
            [0, C.U_no_flap, 1, [0]],
            [1, C.U_weak, 1, [0]],
            [2, C.U_strong, 1 / (2 * C.V_dev + 1), C.W_v]
        ]

        for input_index, u_k, p_flap, W_v in U:
            # Select flap disturbance
            for w_v in W_v:

                v_j = min(C.V_max, max(-C.V_max, state_i[StateVar.V] + u_k + w_v - C.g))

                # Case 1: No spawn
                if p_spawn < 1:
                    next_state = (
                        y_j,
                        v_j,
                        *dhat_j,
                        *hhat_j
                    )

                    P[state_index, state_to_index(next_state), input_index] += p_flap * (1 - p_spawn)

                # Case 2: Spawn
                dspawn_j = copy(dhat_j)
                hspawn_j = copy(hhat_j)
                if p_spawn > 0:
                    dspawn_j[m_min] = s
                    p_combined = p_flap * p_spawn * p_height
                    for height in C.S_h:
                        hspawn_j[m_min] = height

                        next_state = (
                            y_j,
                            v_j,
                            *dspawn_j,
                            *hspawn_j
                        )

                        P[state_index, state_to_index(next_state), input_index] += p_combined

    return P