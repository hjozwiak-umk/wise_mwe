import numpy as np
from numba import njit

@njit(cache=True)
def renormalized_numerov(W_r: np.ndarray, step: float, direction: int, initial_ratio: complex) -> np.ndarray:
    """
    1D Renormalized Numerov propagator for solving the single-channel radial Schrödinger equation.

    Parameters
    ----------
    W_r : np.ndarray
        The effective potential computed for all the N grid points:
        W(r) = 2μ*V(r) - k^2 + l(l+1)/r^2. Shape: (N,).
    step : float
        The spatial grid step size (h).
    direction : int
        Propagation direction flag. 
        `1` for outward propagation (computes Q^x_n = x_{n-1} / x_n).
        `-1` for inward propagation (computes Q^y_n = y_{n+1} / y_n).
    initial_ratio : complex
        The boundary condition for the ratio at the start of the propagation.
        For outward: exactly 0.0 at the origin.
        For inward: the analytical ratio y(r_max) / y(r_max - h).

    Returns
    -------
    np.ndarray
        An array of physical wavefunction ratios Q of shape (N,).
    """
    steps = len(W_r)
    Q = np.zeros(steps, dtype=np.complex128)
    
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