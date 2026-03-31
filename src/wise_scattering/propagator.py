import numpy as np
from numba import njit

@njit(cache=True)
def renormalized_numerov(W_r: np.ndarray, step: float, direction: int, initial_ratio: complex) -> np.ndarray:
    """
    1D Renormalized Numerov propagator solving y'' = W(r)y.
    
    If direction == 1 (outwards): Computes Q_n = y_{n-1} / y_n
    If direction == -1 (inwards): Computes Q_n = y_{n+1} / y_n
    """
    steps = len(W_r)
    Q = np.zeros(steps, dtype=np.complex128)
    
    # Standard Numerov F_n factor
    # Since y'' = W_r * y, the coupling matrix equivalent is -W_r
    # Therefore: F_n = 1 + (h^2 / 12) * (-W_r)
    F = 1.0 - (step**2 / 12.0) * W_r
    
    if direction == 1:
        # --- Outwards Propagation ---
        Q[0] = initial_ratio
        
        current_A = -F[0]
        current_B = 12.0 - 10.0 * F[0]
        current_C = F[1]
        
        Q[1] = current_C / (current_A * Q[0] + current_B)
        
        for n in range(1, steps - 1):
            current_A = (current_B - 12.0) / 10.0
            current_B = 12.0 - 10.0 * current_C
            current_C = F[n+1]
            Q[n+1] = current_C / (current_A * Q[n] + current_B)
            
    else:
        # --- Inwards Propagation ---
        Q[-1] = initial_ratio
        
        current_A = -F[-1] # this is an approximation, valid for small step sizes/large r
        current_B = 12.0 - 10.0 * F[-1]
        current_C = F[-2]
        
        Q[-2] = current_C / (current_A * Q[-1] + current_B)
        
        for n in range(steps - 2, 0, -1):
            current_A = (current_B - 12.0) / 10.0
            current_B = 12.0 - 10.0 * current_C
            current_C = F[n-1]
            Q[n-1] = current_C / (current_A * Q[n] + current_B)
            
    return Q