import traceback
from math import ceil
from itertools import product
from typing import List, Tuple
from Const import ConstMod

from SolverMod import solution_mod


TEST_CASES = [
    # Case 1: The PDF Example
    # Exact parameters from Page 6 of the instructions.
    {
        "name": "PDF Example (Standard)",
        "X": 9, "Y": 8, "V_max": 1,
        "U_no_flap": 0, "U_weak": 1, "U_strong": 3, # U_weak not specified, assuming 1
        "V_dev": 2, "D_min": 3, "G": 3, "S_h": [0, 2, 5],
        "g": 1, "lam_weak": 0.5, "lam_strong": 0.7
    },
    
    # Case 2: Tiny Grid (Min Dimensions)
    # Tests boundaries and small state spaces.
    {
        "name": "Tiny Grid",
        "X": 4, "Y": 6, "V_max": 1,
        "U_no_flap": 0, "U_weak": 1, "U_strong": 2,
        "V_dev": 0, "D_min": 3, "G": 3, "S_h": [2, 3],
        "g": 1, "lam_weak": 0.1, "lam_strong": 0.2
    },

    # Case 3: The Long Corridor
    # Large X, small Y. Many obstacles (High M).
    {
        "name": "Long Corridor (High M)",
        "X": 9, "Y": 7, "V_max": 1,
        "U_no_flap": 0, "U_weak": 1, "U_strong": 2,
        "V_dev": 1, "D_min": 3, "G": 3, "S_h": [1, 3, 5],
        "g": 1, "lam_weak": 0.5, "lam_strong": 0.8
    },

    # Case 4: High Velocity & Gravity
    # Tests physics limits and clipping.
    {
        "name": "High Physics Dynamics",
        "X": 8, "Y": 15, "V_max": 3,
        "U_no_flap": 0, "U_weak": 2, "U_strong": 5,
        "V_dev": 1, "D_min": 4, "G": 3, "S_h": [5, 10],
        "g": 2, "lam_weak": 0.3, "lam_strong": 0.6
    },

    # Case 5: The "Impossible" Gap
    # Gap size 1 (G=1) requires perfect precision.
    {
        "name": "Tight Gap (G=1)",
        "X": 7, "Y": 8, "V_max": 1,
        "U_no_flap": 0, "U_weak": 1, "U_strong": 2,
        "V_dev": 1, "D_min": 3, "G": 1, "S_h": [2, 4, 6],
        "g": 1, "lam_weak": 0.5, "lam_strong": 0.9
    },

    # Case 6: Deterministic Mode
    # V_dev = 0. The system behaves deterministically (except spawn).
    {
        "name": "Deterministic Physics",
        "X": 8, "Y": 10, "V_max": 2,
        "U_no_flap": 0, "U_weak": 2, "U_strong": 3,
        "V_dev": 0, "D_min": 4, "G": 3, "S_h": [4, 6],
        "g": 1, "lam_weak": 0.4, "lam_strong": 0.8
    },

    # Case 7: High Uncertainty
    # V_dev is large relative to Y. Hard to control.
    {
        "name": "High Uncertainty",
        "X": 8, "Y": 12, "V_max": 2,
        "U_no_flap": 0, "U_weak": 2, "U_strong": 3,
        "V_dev": 2, "D_min": 4, "G": 5, "S_h": [4, 8],
        "g": 1, "lam_weak": 0.5, "lam_strong": 0.7
    },

    # Case 8: Expensive Inputs
    # High Lambda costs. Solver should prefer risking collisions over flapping?
    {
        "name": "Expensive Inputs",
        "X": 6, "Y": 8, "V_max": 1,
        "U_no_flap": 0, "U_weak": 1, "U_strong": 2,
        "V_dev": 1, "D_min": 3, "G": 3, "S_h": [3],
        "g": 1, "lam_weak": 0.9, "lam_strong": 0.99
    },

    # Case 9: Large Gaps (Easy Mode)
    # G=5, easier to pass.
    {
        "name": "Large Gaps (Easy)",
        "X": 8, "Y": 14, "V_max": 2,
        "U_no_flap": 0, "U_weak": 2, "U_strong": 3,
        "V_dev": 1, "D_min": 4, "G": 5, "S_h": [5, 9],
        "g": 1, "lam_weak": 0.5, "lam_strong": 0.7
    },

    # Case 10: Edge D_min
    # D_min is large (close to X). M should be small (likely 2).
    {
        "name": "Large Spacing (Low M)",
        "X": 10, "Y": 10, "V_max": 2,
        "U_no_flap": 0, "U_weak": 2, "U_strong": 3,
        "V_dev": 1, "D_min": 6, "G": 3, "S_h": [3, 7],
        "g": 1, "lam_weak": 0.5, "lam_strong": 0.7
    }
]

def main():
    print(f"--- Starting Batch Run of {len(TEST_CASES)} Predefined Cases ---")
    
    for i, params_dict in enumerate(TEST_CASES):
        case_name = params_dict.pop("name", f"Case {i+1}")
        print(f"\nInstance {i+1}/{len(TEST_CASES)}: {case_name}")
        print(f"   Params: X={params_dict['X']}, Y={params_dict['Y']}, "
              f"D_min={params_dict['D_min']}, G={params_dict['G']}, V_dev={params_dict['V_dev']}")
        
        try:
            # 1. Instantiate the Object
            const_instance = ConstMod(**params_dict)
            
            # Check calculated properties
            # print(f"   -> Calculated M: {const_instance.M}")
            
            # 2. Call the Solver
            V_opt, Policy_opt = solution_mod(const_instance)
            
            #print("   -> Success.")
            
        except Exception as e:
            print(f"   -> FAILED: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()