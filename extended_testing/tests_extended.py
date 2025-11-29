import numpy as np
from Const import Const
from ComputeTransitionProbabilities import compute_transition_probabilities
from ComputeExpectedStageCosts import compute_expected_stage_cost
from Solver import solution

def run_edge_case_test(test_name, params):
    """
    Runs a test for an edge case configuration.
    It prints the shapes and summary statistics of the results.
    """
    print(f"--- Running edge case test: {test_name} ---")
    print(f"Parameters: {params}")
    
    try:
        C = Const()
        for key, value in params.items():
            setattr(C, key, value)

        # Ensure M is re-calculated if X or D_min change
        if 'X' in params or 'D_min' in params:
            C.M = np.ceil(C.X / C.D_min)

        print(f"State space size (K): {C.K}")
        
        P = compute_transition_probabilities(C)
        Q = compute_expected_stage_cost(C)
        J, u = solution(C)

        print(f"  P matrix shape: {P.shape}")
        print(f"  Q matrix shape: {Q.shape}")
        print(f"  J vector shape: {J.shape}")
        print(f"  u vector shape: {u.shape}")
        
        if C.K > 0:
            print(f"  J stats: min={np.min(J):.2f}, max={np.max(J):.2f}, mean={np.mean(J):.2f}")
            print(f"  u stats: unique values={np.unique(u)}")
        
        print(f"[SUCCESS] {test_name} ran without errors.")

    except Exception as e:
        print(f"[ERROR] An error occurred in {test_name}: {e}")
    
    print(f"--- Finished edge case test: {test_name} ---")
    print()

def run_all_extended_tests():
    """Runs all extended test cases."""
    print("\n--- Running All Extended Tests ---")
    # Edge Case 1: Very small grid
    small_grid_params = {'X': 3, 'Y': 3, 'D_min': 2, 'G': 1, 'S_h': [1]}
    run_edge_case_test("Small Grid", small_grid_params)

    # Edge Case 2: No vertical velocity
    no_velocity_params = {'V_max': 0}
    run_edge_case_test("No Velocity", no_velocity_params)

    # Edge Case 3: Gap size almost as large as grid height
    large_gap_params = {'Y': 15, 'G': 13, 'S_h': [7]}
    run_edge_case_test("Large Gap", large_gap_params)
    
    # Edge Case 4: High gravity
    high_gravity_params = {'g': 5}
    run_edge_case_test("High Gravity", high_gravity_params)
    
    # Edge Case 5: All inputs are the same
    same_inputs_params = {'U_no_flap': 1, 'U_weak': 1, 'U_strong': 1}
    run_edge_case_test("Same Inputs", same_inputs_params)
    print("--- Finished All Extended Tests ---")
    print()


if __name__ == "__main__":
    run_all_extended_tests()
