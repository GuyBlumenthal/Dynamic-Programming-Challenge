
import pickle
import numpy as np
from Const import Const
from ComputeTransitionProbabilities import compute_transition_probabilities
from ComputeExpectedStageCosts import compute_expected_stage_cost
from Solver import solution # Import the solution function from Solver.py
import tests_extended # Import the extended tests module

def run_test(test_name):
    """
    Runs a test case by loading data, executing functions, and comparing results.
    """
    print (f'-- Running test: {test_name} --')
    try:
        # Load test data
        with open(f'../tests/{test_name}.pkl', 'rb') as f:
            const_data = pickle.load(f)
        
        test_data = np.load(f'../tests/{test_name}.npz', allow_pickle=True)
        P_expected = test_data['P']
        Q_expected = test_data['Q']
        J_expected = test_data['J'] # Load expected J
        u_expected = test_data['u'] # Load expected u

        # Create and configure Const object
        C = Const()
        for key, value in const_data.items():
            setattr(C, key, value)

        # Compute actual values
        P_actual = compute_transition_probabilities(C)
        Q_actual = compute_expected_stage_cost(C)
        J_actual, u_actual = solution(C) # Compute J and u using the solution function

        RTOL = 1e-4
        ATOL = 1e-7

        # Compare results
        np.testing.assert_allclose(P_actual, P_expected, rtol=RTOL, atol=ATOL)
        print(f"[SUCCESS] P matrix for {test_name} is correct.")
        np.testing.assert_allclose(Q_actual, Q_expected, rtol=RTOL, atol=ATOL)
        print(f"[SUCCESS] Q matrix for {test_name} is correct.")
        np.testing.assert_allclose(J_actual, J_expected, rtol=RTOL, atol=ATOL)
        print(f"[SUCCESS] J matrix for {test_name} is correct.")
        #np.testing.assert_allclose(u_actual, u_expected, rtol=RTOL, atol=ATOL)
        #
        # print(f"[SUCCESS] u (policy) for {test_name} is correct.")

    except AssertionError as e:
        print(f"[FAILURE] {test_name}: {e}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred in {test_name}: {e}")
    print(f"-- Finished test: {test_name} --")


if __name__ == "__main__":
    run_test('test0')
    run_test('test1')
    run_test('test2')
    run_test('test3')
    
    # Run the extended tests
    tests_extended.run_all_extended_tests()
