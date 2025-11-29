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
import scipy.sparse as sp


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
    
    def spawn_probability(C, s):
        if s <= C.D_min - 1:
            return 0.0
        elif s >= C.X:
            return 1.0
        else:
            return (s - (C.D_min - 1)) / (C.X - C.D_min)
    
    P_data = [[] for _ in range(C.L)]
    P_rows = [[] for _ in range(C.L)]
    P_cols = [[] for _ in range(C.L)]

    for i, state_tuple in enumerate(C.state_space):
        
        y, v = state_tuple[0], state_tuple[1] #position, velocity
        D = list(state_tuple[2:2+C.M]) # from the second index to max possible number of obstacles
        H = list(state_tuple[2+C.M:]) 
        
        if D[0] == 0 and abs(y - H[0]) > (C.G - 1) // 2:
            continue
        
        for l, u in enumerate(C.input_space):
            if u == C.U_strong:
                W_flap = range(-C.V_dev, C.V_dev+1)
                prob_flap = 1 / len(W_flap)
            else:
                W_flap = [0]
                prob_flap = 1.0
            
            for w_flap in W_flap:
                v_next = int(np.clip(v + u + w_flap - C.g, -C.V_max, C.V_max))
                y_next = int(np.clip(y + v, 0, C.Y-1))
                        
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

                if p_spawn < 1:
                    next_state_no_spawn = (y_next, v_next, *hat_D, *hat_H)
                    try:
                        j = C.state_to_index(next_state_no_spawn)
                        P_data[l].append(prob_flap * (1 - p_spawn))
                        P_rows[l].append(i)
                        P_cols[l].append(j)
                    except (KeyError, TypeError):
                        pass
                    
                if p_spawn > 0:
                    k_spawn = -1  
                    for k in range(1, C.M): 
                        if hat_D[k] == 0:
                            k_spawn = k
                            break
                    
                    if k_spawn != -1:
                        for h_new in C.S_h:
                            hat_D_spawn = hat_D.copy()
                            hat_H_spawn = hat_H.copy()
                            hat_D_spawn[k_spawn] = np.clip(s, C.D_min, C.X-1)
                            hat_H_spawn[k_spawn] = h_new
                            next_state_spawn = (y_next, v_next, *hat_D_spawn, *hat_H_spawn)
                            try:
                                j = C.state_to_index(next_state_spawn)
                                P_data[l].append(prob_flap * p_spawn * (1 / len(C.S_h)))
                                P_rows[l].append(i)
                                P_cols[l].append(j)
                            except(KeyError, TypeError):
                                pass
                            
    P = np.zeros((C.K, C.K, C.L))
    
    for l in range(C.L):
        np.add.at(P[:, :, l], (P_rows[l], P_cols[l]), P_data[l])

    print("shape", np.shape(P))
    return P

