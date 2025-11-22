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

