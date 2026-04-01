import numpy as np
from scipy.special import spherical_jn, spherical_yn, kv, iv

def riccati_bessel(k, l, r):
    """
    Returns the Riccati-Bessel functions of the first and second kind. Based on scipy's spherical Bessel functions.

    Parameters
    ----------
    k : float
        The asymptotic wavevector.
    l : int
        The end-over-end rotational angular momentum quantum number.
    r : float
        The radial distance.

    Returns
    -------
    tuple[float, float]
        A tuple containing (j_l(kr), y_l(kr)), scaled by kr.
    """
    kr = k * r
    return kr * spherical_jn(l, kr), kr * spherical_yn(l, kr)

def modified_bessel_kv(kappa, l, r):
    """
    Computes the scaled modified spherical Bessel function of the second kind.
    Used for asymptotically closed channels. 
    Based on scipy's modified Bessel function of the second kind, with a sqrt(r) scaling to match the asymptotic form of the radial wavefunction.

    Parameters
    ----------
    kappa : float
        The decay parameter (sqrt(|k^2|)).
    l : int
        The end-over-end rotational angular momentum quantum number.
    r : float
        The radial distance.

    Returns
    -------
    float
        The modified Bessel function value at kappa * r.
    """
    return np.sqrt(r) * kv(l + 0.5, kappa * r)

def modified_bessel_iv(kappa, l, r):
    """
    Computes the scaled modified spherical Bessel function of the first kind.
    Used for matching closed-channel Green's functions.
    Based on scipy's modified Bessel function of the first kind, with a sqrt(r) scaling to match the asymptotic form of the radial wavefunction.

    Parameters
    ----------
    kappa : float
        The decay parameter (sqrt(|k^2|)).
    l : int
        The end-over-end rotational angular momentum quantum number.
    r : float
        The radial distance.

    Returns
    -------
    float
        The modified Bessel function value at kappa * r.
    """
    return np.sqrt(r) * iv(l + 0.5, kappa * r)

def get_inward_initial_ratio(k_sq, l, r_max, step):
    """
    Calculates the boundary condition ratio for the irregular solution.
    Computes Q_in = y(r_max) / y(r_max - h) stepping off the grid.
    See Eqs. (7) and (12) in the manuscript.

    Parameters
    ----------
    k_sq : float
        The squared asymptotic wavevector (k^2 = 2μ(E - E_asymp)).
    l : int
        The end-over-end rotational angular momentum quantum number.
    r_max : float
        The final grid point distance.
    step : float
        The spatial grid step size (h).

    Returns
    -------
    complex
        The exact analytical inward boundary ratio.
    """
    r_N = r_max + step
    r_Nm1 = r_max

    if k_sq > 0:
        k = np.sqrt(k_sq)
        j_N, y_N = riccati_bessel(k, l, r_N)
        j_Nm1, y_Nm1 = riccati_bessel(k, l, r_Nm1)
        return (j_N + 1j * y_N) / (j_Nm1 + 1j * y_Nm1)
    else:
        kappa = np.sqrt(np.abs(k_sq))
        return modified_bessel_kv(kappa, l, r_N) / modified_bessel_kv(kappa, l, r_Nm1)

def process_asymptotics_and_greens(k_sq, l, grid, Q_out, Q_in):
    """
    Extracts asymptotic physical observables and backpropagates the diagonal Green's function.

    Computes phase shifts and reference wavefunctions directly from the regular and
    irregular Numerov ratios, ensuring mathematical stability in closed channels.

    Parameters
    ----------
    k_sq : float
        The squared asymptotic wavevector.
    l : int
        The end-over-end rotational angular momentum quantum number.
    grid : np.ndarray
        The radial spatial grid.
    Q_out : np.ndarray
        The regular outward solution ratios Q^x_n = x_{n-1} / x_n.
    Q_in : np.ndarray
        The irregular inward solution ratios Q^y_n = y_{n+1} / y_n.

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        - delta (float): The asymptotic phase shift (set to 0.0 for closed channels).
        - u_reg (np.ndarray): The reference regular wavefunction across the grid.
        - G_ii (np.ndarray): The diagonal Green's function across the grid. See Eqs. (13) and (14) in the manuscript.
    """
    n_points = len(grid)
    G_ii = np.zeros(n_points, dtype=np.complex128)
    u_reg = np.zeros(n_points, dtype=np.complex128)
    delta = 0.0

    r1 = grid[-2]
    r2 = grid[-1]
    Q_end = Q_out[-1] # Q_out[-1] is y(r1) / y(r2)

    if k_sq > 0:
        k = np.sqrt(k_sq)
        j1, y1 = riccati_bessel(k, l, r1)
        j2, y2 = riccati_bessel(k, l, r2)

        # 1. Phase Shift
        tan_delta = (Q_end * j2 - j1) / (Q_end * y2 - y1)
        delta = np.arctan(tan_delta.real)

        # 2. G_ii at r_max
        u_reg_r2 = np.cos(delta) * j2 - np.sin(delta) * y2
        u_irreg_r2 = -1j * np.exp(1j * delta) * (j2 + 1j * y2) / k
        G_ii[-1] = u_reg_r2 * u_irreg_r2
        u_reg[-1] = u_reg_r2

        # 3. Backpropagate u_reg (Only needed for open channels!)
        for i in range(n_points - 1, 0, -1):
            u_reg[i-1] = Q_out[i] * u_reg[i]

    else:
        kappa = np.sqrt(np.abs(k_sq))
        i1 = modified_bessel_iv(kappa, l, r1)
        i2 = modified_bessel_iv(kappa, l, r2)
        k1 = modified_bessel_kv(kappa, l, r1)
        k2 = modified_bessel_kv(kappa, l, r2)

        iv_ratio = i1 / i2
        kv_ratio = k1 / k2

        # 1. G_ii at r_max
        numerator = Q_end - iv_ratio
        denominator = Q_end - kv_ratio
        G_ii[-1] = - (1.0 - numerator / denominator) * i2 * k2

    # Backpropagate G_ii directly from the ratios for BOTH open and closed channels.
    for i in range(n_points - 1, 0, -1):
        G_ii[i-1] = G_ii[i] * Q_out[i] / Q_in[i-1]
    return delta, u_reg, G_ii