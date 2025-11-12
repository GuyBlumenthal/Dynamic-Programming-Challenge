"""ComputeExpectedStageCosts.py

Template to compute the expected stage cost.

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

from timer import time_def

def compute_expected_stage_cost_solver(C: Const) -> np.array:
    """Computes the expected stage cost for the given problem.

    Args:
        C (Const): The constants describing the problem instance.

    Returns:
        np.array: Expected stage cost Q of shape (K,L), where
            - K is the size of the state space;
            - L is the size of the input space; and
            - Q[i,l] corresponds to the expected stage cost incurred when using input l at state i.
    """
    return np.tile(np.array([
        -1,                # Cost for action 0 (None)
        C.lam_weak - 1,    # Cost for action 1 (Weak)
        C.lam_strong - 1   # Cost for action 2 (Strong)
    ]), (C.K, 1))

@time_def
def compute_expected_stage_cost(C: Const) -> np.array:
    """Computes the expected stage cost for the given problem.

    Args:
        C (Const): The constants describing the problem instance.

    Returns:
        np.array: Expected stage cost Q of shape (K,L), where
            - K is the size of the state space;
            - L is the size of the input space; and
            - Q[i,l] corresponds to the expected stage cost incurred when using input l at state i.
    """
    return np.tile(np.array([
        -1,                # Cost for action 0 (None)
        C.lam_weak - 1,    # Cost for action 1 (Weak)
        C.lam_strong - 1   # Cost for action 2 (Strong)
    ]), (C.K, 1))