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

from Const import Const

from typing import Tuple, Dict

from itertools import product

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


from line_profiler import profile

@profile
def custom_state_space(C: Const) -> Tuple[int, Dict[Tuple[int, ...], int]]:
    """
    Computes the state space and returns a state -> index dictionary
    using a recursive, pruning-based generation method.

    This function maintains the strict lexicographical ordering
    from the problem statement, ensuring the state-to-index
    mapping is identical to the original 'itertools.product' method,
    but is significantly faster by pruning invalid branches early.
    """

    state_to_index_dict = {}
    current_index = 0

    # --- Cache constants from C for minor speedup ---
    S_y, S_v = C.S_y, C.S_v
    S_d, S_d1 = C.S_d, C.S_d1
    S_h, S_h_default = C.S_h, C.S_h[0]
    M, X_limit = C.M, C.X - 1

    # Pre-build the two possible lists for H-options
    h_options_all = S_h
    h_options_default = [S_h_default]

    # Pre build an empty D-options list
    S_d0 = [0]

    # ================== D-Vector Recursive Builder ==================
    @profile
    def _build_d_recursive(y, v, current_d_list, current_d_sum, d_index, zero_seen):
        """
        Recursively builds the D-vector (d1, ..., dM) for a given
        (y, v) prefix.

        Pruning:
        - sum(d) > X-1
        - Trailing zeros (if d_i=0, all d_j for j>i must be 0)
        - d2=0 if d1=0
        """
        # --- Base Case: D-vector is complete ---
        if d_index == M:
            # D-vector is built, now start building the H-vector
            nonlocal current_index

            # 1. Build the list of allowed H-options for this d_tuple
            # h1 always has all options
            h_iterables = [h_options_all] + [
                h_options_default if current_d_list[i] == 0 else h_options_all
                for i in range(1, M)
            ]

            prefix = (y, v) + tuple(current_d_list)

            # 2. Loop over the product of these allowed H-options
            for h_tuple in product(*h_iterables):
                state = prefix + h_tuple
                state_to_index_dict[state] = current_index
                current_index += 1

            return

        # --- Recursive Step: Add d_i ---
        if d_index == 0:
            d_options = S_d1
        elif zero_seen:
            d_options = S_d0
        else:
            d_options = S_d

        for d in d_options:
            # 1. Sum constraint
            if current_d_sum + d > X_limit:
                continue

            # 3. d1/d2 constraint (d1=0 -> d2>0)
            # d_index == 1 is d2
            if d_index == 1:
                d1 = current_d_list[0]
                d2 = d
                if d1 <= 0 and d2 == 0:
                    continue

            # Recurse with the added d
            current_d_list[d_index] = d
            _build_d_recursive(
                y, v,
                current_d_list,
                current_d_sum + d,
                d_index + 1,
                # Update zero_seen flag:
                # (zero_seen is True if it was already True, OR
                # if we are adding a zero *after* d1)
                zero_seen or (d_index > 0 and d == 0)
            )

    # The outer loops *must* be y, then v, to maintain order
    for y in S_y:
        for v in S_v:
            current_d_list_for_v = [0] * M
            # Start the recursion for the D-vector
            _build_d_recursive(y, v, current_d_list_for_v, 0, 0, False)

    return len(state_to_index_dict), state_to_index_dict
