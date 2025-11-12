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

# def custom_state_space(C: Const) -> Tuple[int, Dict[Tuple[int, ...], int]]:
#     """Returns the full state space as a list of tuples.

#     Returns:
#         List[Tuple[int, ...]]: list of admissible states
#     """
#     iterables = (
#         [C.S_y, C.S_v, C.S_d1]
#         + [C.S_d] * (C.M - 1)
#         + [C.S_h] * C.M
#     )

#     counter = 0

#     state_dict = {}

#     for x in product(*iterables):
#         if C.is_valid_state(x):
#             state_dict[tuple(x)] = counter
#             counter = counter + 1
#     return counter, state_dict

# def custom_state_space(C: Const) -> Tuple[int, Dict[Tuple[int, ...], int]]:
#     return C.K, {state: i for i, state in enumerate(C.state_space)}

def custom_state_space(C: Const) -> Tuple[int, Dict[Tuple[int, ...], int]]:
    """
    Computes the state space and returns a state -> index dictionary
    using a recursive, pruning-based generation method.

    This function maintains the strict lexicographical ordering
    from the problem statement, ensuring the state-to-index
    mapping is identical to the original 'itertools.product' method,
    but is significantly faster by pruning invalid branches early.

    Args:
        C (Const): The constants describing the problem instance.

    Returns:
        Dict[Tuple[int, ...], int]:
            A dictionary mapping each valid state tuple to its
            unique integer index (from 0 to K-1).
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

    # ================== H-Vector Recursive Builder ==================
    def _build_h_recursive(y, v, d_tuple, current_h_list, h_index):
        """
        Recursively builds the H-vector (h1, ..., hM) for a given
        (y, v, d_tuple) prefix.

        Pruning:
        - If d_i = 0 (for i>=2), only the default h_i is allowed.
        """
        nonlocal current_index

        # --- Base Case: H-vector is complete ---
        if h_index == M:
            state = (y, v, *d_tuple, *current_h_list)
            state_to_index_dict[state] = current_index
            current_index += 1
            return

        # --- Recursive Step: Add h_i ---

        # Pruning check: if i>=2 (h_index > 0) and d_i = 0...
        if h_index > 0 and d_tuple[h_index] == 0:
            # ...then h_i *must* be the default value. Prune all other options.
            current_h_list.append(S_h_default)
            _build_h_recursive(y, v, d_tuple, current_h_list, h_index + 1)
            current_h_list.pop() # Backtrack
        else:
            # d_i > 0 or i=1 (h1). All h_options are valid.
            for h in S_h:
                current_h_list.append(h)
                _build_h_recursive(y, v, d_tuple, current_h_list, h_index + 1)
                current_h_list.pop() # Backtrack

    # ================== D-Vector Recursive Builder ==================
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
            _build_h_recursive(y, v, tuple(current_d_list), [], 0)
            return

            # nonlocal current_index

            # d_tuple = tuple(current_d_list)

            # # --- Your H-vector logic starts here ---
            # # 1. Build the list of allowed H-options for this d_tuple
            # h_iterables = [h_options_all] # h1 always has all options

            # for i in range(1, M): # For d2...dM
            #     if d_tuple[i] == 0:
            #         h_iterables.append(h_options_default)
            #     else:
            #         h_iterables.append(h_options_all)

            # # 2. Loop over the product of these allowed H-options
            # for h_tuple in product(*h_iterables):
            #     state = (y, v, *d_tuple, *h_tuple)
            #     state_to_index_dict[state] = current_index
            #     current_index += 1
            # # --- Your H-vector logic ends here ---

            # return # End this recursive branch

        # --- Recursive Step: Add d_i ---
        if d_index == 0:
            d_options = S_d1
        elif zero_seen:
            d_options = [0]
        else:
            d_options = S_d

        for d in d_options:
            # --- PRUNING CHECKS ---

            # 1. Sum constraint
            if current_d_sum + d > X_limit:
                continue

            # 3. d1/d2 constraint (d1=0 -> d2>0)
            #    (Here, d_index=1 means we are *adding* d2)
            if d_index == 1:
                d1 = current_d_list[0]
                d2 = d
                if d1 <= 0 and d2 == 0:
                    continue

            # --- RECURSE ---
            current_d_list.append(d)
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
            current_d_list.pop() # Backtrack

    # ================== Start Generation ==================
    # The outer loops *must* be y, then v, to maintain
    # lexicographical order.

    for y in S_y:
        for v in S_v:
            # Start the recursion for the D-vector
            _build_d_recursive(y, v, [], 0, 0, False)

    return len(state_to_index_dict), state_to_index_dict
