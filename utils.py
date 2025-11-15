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

import numpy as np
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


class CustomStateSpace:
    # ================== D-Vector Recursive Builder ==================
    def build_d_recursive(self, y, v, current_d_list, current_d_sum, d_index, spot0):
        """
        Recursively builds the D-vector (d1, ..., dM) for a given
        (y, v) prefix.

        Pruning:
        - sum(d) > X-1
        - Trailing zeros (if d_i=0, all d_j for j>i must be 0)
        - d2=0 if d1=0
        """
        # --- Base Case: D-vector is complete ---
        if d_index == self.M:
            # D-vector is built, now start building the H-vector
            h_iterable = self.possible_h_iterables[spot0]

            # 2. Loop over the product of these allowed H-options
            prefix = (y, v) + tuple(current_d_list)
            for h_tuple in h_iterable:
                self.state_to_index_array[*prefix, *h_tuple] = self.current_index
                self.valid_states_with_indices.append((y, v, current_d_list[:], h_tuple, self.current_index, spot0))
                self.current_index += 1

            return        # --- Recursive Step: Add d_i ---
        if d_index == 0:
            d_options = self.S_d1
        elif spot0 > 0:
            d_options = self.S_d0
        else:
            d_options = self.S_d

        for d in d_options:
            # 1. Sum constraint
            if current_d_sum + d > self.X_limit:
                break # These are stored in increasing order

            # 3. d1/d2 constraint (d1=0 -> d2>0)
            # d_index == 1 is d2
            if d_index == 1:
                d1 = current_d_list[0]
                d2 = d
                if d1 <= 0 and d2 == 0:
                    continue

            next_spot0 = spot0
            if spot0 == 0 and d_index > 0 and d == 0:
                next_spot0 = d_index  # This is the first zero

            # Recurse with the added d
            current_d_list[d_index] = d
            self.build_d_recursive(
                y, v,
                current_d_list,
                current_d_sum + d,
                d_index + 1,
                next_spot0
            )

    def custom_state_space(self, C: Const) -> Tuple[int, np.ndarray, list]:
        """
        Computes the state space and returns a state -> index dictionary
        using a recursive, pruning-based generation method.

        This function maintains the strict lexicographical ordering
        from the problem statement, ensuring the state-to-index
        mapping is identical to the original 'itertools.product' method,
        but is significantly faster by pruning invalid branches early.
        """


        self.current_index = 0
        self.valid_states_with_indices = []

        # --- Cache constants from C for minor speedup ---
        self.S_y, self.S_v = C.S_y, C.S_v
        self.S_d, self.S_d1 = C.S_d, C.S_d1
        self.S_h, self.S_h_default = C.S_h, C.S_h[0]
        self.M, self.X_limit = C.M, C.X - 1

        # --- Create mappings for non-contiguous state variables ---
        self.v_offset = C.V_max

        # --- Initialize state_to_index_array ---
        dims = [C.Y, self.v_offset + 2 * C.V_max + 1]
        for _ in range(self.M):
            dims.append(C.X) # d values are 0..X-1
        for _ in range(self.M):
            dims.append(C.Y) # h values are 0..Y-1

        self.state_to_index_array = np.full(dims, -1, dtype=np.int32)


        # Pre-build the two possible lists for H-options
        self.h_options_all = self.S_h
        self.h_options_default = [self.S_h_default]

        # Pre build an empty D-options list
        self.S_d0 = [0]

        possible_h_iterables = [[self.h_options_all] + [self.h_options_all for i in range(1, self.M)]]
        for spot0 in range(1, self.M):
            possible_h_iterables.append([self.h_options_all] + [
                self.h_options_default if i >= spot0 else self.h_options_all
                for i in range(1, self.M)
            ])
        self.possible_h_iterables = [list(product(*h_iter)) for h_iter in possible_h_iterables]

        # The outer loops *must* be y, then v, to maintain order
        for y in self.S_y:
            for v in self.S_v:
                current_d_list_for_v = [0] * self.M
                # Start the recursion for the D-vector
                self.build_d_recursive(y, v, current_d_list_for_v, 0, 0, 0)

        return self.current_index, self.state_to_index_array, self.valid_states_with_indices
