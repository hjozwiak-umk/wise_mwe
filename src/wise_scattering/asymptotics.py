import numpy as np
from scipy.special import spherical_jn, spherical_yn, kv, iv

def riccati_bessel(k, l, r):
    kr = k * r
    return kr * spherical_jn(l, kr), kr * spherical_yn(l, kr)

def modified_bessel_kv(kappa, l, r):
    return np.sqrt(r) * kv(l + 0.5, kappa * r)

def modified_bessel_iv(kappa, l, r):
    return np.sqrt(r) * iv(l + 0.5, kappa * r)

def get_inward_initial_ratio(k_sq, l, r_max, step):
    """Calculates Q_in[-1] = y(r_max + h) / y(r_max) for the irregular solution."""
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
    Computes phase shift, open-channel wavefunctions, and the full G_ii diagonal,
    strictly avoiding the reconstruction of closed-channel irregular wavefunctions!
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