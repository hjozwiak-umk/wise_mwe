"""
WISE Scattering Minimal Working Example (MWE)
---------------------------------------------
This script is the main entry point for demonstrating the matrix-free 
Weinberg-regularized Iterative Series Expansion (WISE) framework for the example
Atom-Diatom scattering (CO+He) system.
"""
import numpy as np
import time
from scipy.sparse.linalg import eigs, LinearOperator

# Imports from our standalone MWE modules
from wise_scattering.asymptotics import get_inward_initial_ratio, process_asymptotics_and_greens
from wise_scattering.physics_utilities import (generate_space_fixed_channels, precompute_potential_sparsity, 
                           precompute_centrifugal_sparsity, load_radial_potential, init_wigner_symbols, compute_diagonal_potential_jit)
from wise_scattering.propagator import renormalized_numerov
from wise_scattering.wise_core import apply_K_matvec, apply_KH_matvec, apply_K_P

def main():
    """
    Executes the full pipeline.
    
    Steps:
    1. System and Grid Initialization.
    2. Basis Set Generation (Space-Fixed representation).
    3. Precomputation of the potential centrifugal terms in the sparse matrix format.
    4. Generation of Green's function from solutions of the single-channel problems (Renormalized Numerov).
    5. Construction of the matrix-free operators.
    6. Determination of the largest Weinberg eigenvalues of the scattering kernel.
    7. Regularized Born Series.
    8. S-Matrix column evaluation and convergence checking.
    """
    print("Starting WISE scattering MWE for CO+He...")

    # Initialize Wigner symbols
    init_wigner_symbols()

    # Mathematical and physical constants
    ATOMIC_MASS_UNIT_TO_ELECTRON_MASS = 1822.8884862
    HARTREE_TO_INVERSE_CM = 2.1947463136314e5 # NIST: https://physics.nist.gov/cgi-bin/cuu/Value?hrminv|search_for=hartree

    # CO+He physical parameters
    MASS_CO = 28.0101 * ATOMIC_MASS_UNIT_TO_ELECTRON_MASS # a.m.u. converted to a.u. of mass (electron masses)
    MASS_HE = 4.002602 * ATOMIC_MASS_UNIT_TO_ELECTRON_MASS  # a.m.u. converted to a.u. of mass (electron masses)
    REDUCED_MASS = (MASS_CO * MASS_HE) / (MASS_CO + MASS_HE)
    B_ROT = 1.922521 / HARTREE_TO_INVERSE_CM          # cm-1 converted to Hartree

    PREFACTOR = 2.0 * REDUCED_MASS

    # Target collision & algorithm setup
    E_col_cm1 = 8.84506
    E_total = E_col_cm1 / HARTREE_TO_INVERSE_CM
    J_tot = 0
    parity = 1
    j_max = 6
    lambda_max = 20

    # Target incoming channel
    incoming_j = 0
    incoming_l = 0

    # Solver parameters
    n_eigs = 20
    conv_radius = 0.95
    max_iter = 1000
    conv_threshold = 1e-6

    # Grid setup
    r_min, r_max, step = 3.0, 20.0, 0.01
    grid = np.arange(r_min, r_max + step, step)
    n_points = len(grid)
    
    # Integration weights (Trapezoidal)
    weights = np.ones_like(grid) * step
    weights[0] = weights[-1] = 0.5 * step
    sqrt_w = np.sqrt(weights)

    # Basis Set & Potential Precomputation
    print(f"Setting up CO+He scattering (E={E_col_cm1} cm-1, J={J_tot}, j_max={j_max})...")
    channel_j, channel_l, channel_E = generate_space_fixed_channels(J_tot, parity, j_max, B_ROT)
    n_channels = len(channel_j)
    print(f"Total channels (open + closed): {n_channels}")
    
    # Dynamically find the incoming channel index
    try:
        incoming_idx = next(i for i, (j, l) in enumerate(zip(channel_j, channel_l)) if j == incoming_j and l == incoming_l)
        print(f"Target incoming channel (j={incoming_j}, l={incoming_l}) mapped to index: {incoming_idx}")
    except StopIteration:
        raise ValueError(f"Incoming channel (j={incoming_j}, l={incoming_l}) not found in the generated basis set.")

    # Load raw potential and compress into sparse coupling data
    radial_pot = load_radial_potential('CO-He-coupling-terms.dat', grid, lambda_max)    
    pot_data = precompute_potential_sparsity(channel_j, channel_l, J_tot, lambda_max, offdiag_only=True)
    cent_data = precompute_centrifugal_sparsity(channel_j, channel_l, offdiag_only=True)
    
    print("Computing reference Green's functions...")
    G_diag = np.zeros((n_channels, n_points), dtype=np.complex128)
    R_ratio = np.zeros((n_channels, n_points), dtype=np.complex128)
    u_reg_all = np.zeros((n_channels, n_points), dtype=np.complex128)
    phase_shifts = np.zeros(n_channels)
    k_sq_all = np.zeros(n_channels)

    for c in range(n_channels):
        j_c = channel_j[c]
        l_c = channel_l[c]
        E_asymp = channel_E[c]
        k_sq = PREFACTOR * (E_total - E_asymp)
        k_sq_all[c] = k_sq

        # Compute effective potential for the renormalized Numerov propagator
        V_ii = compute_diagonal_potential_jit(J_tot, j_c, l_c, lambda_max, radial_pot)
        W_r = PREFACTOR * V_ii - k_sq + (l_c * (l_c + 1)) / (grid**2)
        
        # Outward Propagation (Regular ratio)
        Q_out = renormalized_numerov(W_r, step, direction=1, initial_ratio=0)

        # Calculate Inward Initial Condition (y_N / y_N-1)
        ratio_inward_start = get_inward_initial_ratio(k_sq, l_c, grid[-1], step)

        # Inward Propagation (Irregular ratio)
        Q_in = renormalized_numerov(W_r, step, direction=-1, initial_ratio=ratio_inward_start)
        
        # Extract Physics directly from ratios
        delta, u_reg_all[c, :], G_diag[c, :] = process_asymptotics_and_greens(k_sq, l_c, grid, Q_out, Q_in)
        phase_shifts[c] = delta
        R_ratio[c, :] = Q_out + 0.0j

    # Define the source term (initial guess): pure incoming wave in the chosen channel
    u_0 = np.zeros((n_channels, n_points), dtype=np.complex128)
    u_0[incoming_idx, :] = u_reg_all[incoming_idx, :]
    
    # Symmetrize source term for the iterative loop
    u_source_symm = (u_0.T.flatten() * np.repeat(sqrt_w, n_channels))

    # Matrix-Free Operator Setup
    size = n_channels * n_points

    def matvec(v):
        return apply_K_matvec(v, grid, sqrt_w, PREFACTOR, G_diag, R_ratio, pot_data, radial_pot, cent_data)

    def rmatvec(v):
        return apply_KH_matvec(v, grid, sqrt_w, PREFACTOR, G_diag, R_ratio, pot_data, radial_pot, cent_data)

    K_op = LinearOperator((size, size), matvec=matvec, rmatvec=rmatvec, dtype=np.complex128)

    # Arnoldi Eigensolver
    print(f"Searching for {n_eigs} largest eigenvalues using ARPACK...")

    t0 = time.perf_counter()
    eigvals, right_vecs = eigs(K_op, k=n_eigs, which='LM')
    left_eigvals_conj, left_vecs_conj = eigs(K_op.H, k=n_eigs, which='LM')
    t1 = time.perf_counter()
    print(f"Eigenspace computed in {t1 - t0:.2f} seconds.")

    for idx, val in enumerate(eigvals):
        print(f"  Eigenvalue {idx}: |eta| = {np.abs(val):.6f}, eta = {val:.6e}")

    # Filter for divergent eigenvalues based on the adjustable convergence radius
    bad_indices = np.where(np.abs(eigvals) >= conv_radius)[0]
    n_D = len(bad_indices)
    print(f"Found {n_D} divergent Weinberg eigenvalues with |eta| >= {conv_radius}.")

    if n_D > 0:
        # Match left and right eigenspaces and compute biorthogonal overlap
        matched_eigvals = eigvals[bad_indices]
        u_T = right_vecs[:, bad_indices].T
        
        # Find matching left eigenvectors by minimum distance to complex conjugate
        match_idx = [np.argmin(np.abs(left_eigvals_conj - np.conj(ev))) for ev in matched_eigvals]
        v_vecs = np.conj(left_vecs_conj[:, match_idx])
        
        # Enforce biorthogonality: M_ij = <v_i | u_j>
        overlap_matrix = np.matmul(v_vecs.T.conj(), right_vecs[:, bad_indices])
        inverse_overlap = np.linalg.inv(overlap_matrix)
        v_T = np.ascontiguousarray((v_vecs @ inverse_overlap.conj().T).T)
    else:
        matched_eigvals, u_T, v_T = [], [], []

    # Iterative Born Series with S-Matrix Convergence
    from wise_scattering.wise_core import compute_U_psi_jit
    
    print("Launching regularized iterative series with physical S-matrix convergence...")
    
    u_R = u_source_symm.copy()
    converged = False
    
    S_old = None
    log_S = []

    t0 = time.perf_counter()
    for it in range(max_iter):
        u_R_old = u_R.copy()
        
        # 1. Apply kernel K
        K_y = matvec(u_R_old)
        
        # 2. Apply projection P (if divergent subspace exists)
        if n_D > 0:
            K_D_y = apply_K_P(u_R_old, matched_eigvals, u_T, v_T)
            u_R = u_source_symm + K_y - K_D_y
        else:
            u_R = u_source_symm + K_y
            
        # 3. Reconstruct the full physical wavefunction for this iteration
        psi_current_symm = u_R.copy()
        if n_D > 0:
            for alpha in range(n_D):
                overlap = np.vdot(v_T[alpha], np.ascontiguousarray(u_R))
                c_alpha = overlap / (1.0 - matched_eigvals[alpha])
                psi_current_symm += c_alpha * matched_eigvals[alpha] * u_T[alpha]

        # De-symmetrize and reshape back to physical channels (C, N)
        psi_phys_flat = psi_current_symm / np.repeat(sqrt_w, n_channels)
        psi_phys = psi_phys_flat.reshape((n_points, n_channels)).T
        
        # 4. Compute S-Matrix column
        U_psi = compute_U_psi_jit(psi_phys, grid, PREFACTOR, pot_data, radial_pot, cent_data)
        S_current = np.zeros(n_channels, dtype=np.complex128)
        
        for j in range(n_channels):
            if k_sq_all[j] <= 0:
                continue  # Closed channels do not contribute to the S-matrix
                
            integrand = u_reg_all[j, :] * U_psi[j, :]
            integral = np.trapz(integrand, grid)
            
            delta_ij = 1.0 if j == incoming_idx else 0.0
            
            # Exact flux normalization matching the original pyscatter reference
            norm = (np.sqrt(k_sq_all[incoming_idx]) * np.sqrt(k_sq_all[j]))**(-0.5)
            phase_factor = np.exp(1j * (phase_shifts[incoming_idx] + phase_shifts[j]))
            
            S_current[j] = phase_factor * (delta_ij - 2j * norm * integral)
            
        log_S.append(S_current)
        
        # 5. Check Convergence
        if S_old is not None:
            # Only compare open channel elements
            delta_S = np.max(np.abs(S_current[k_sq_all > 0] - S_old[k_sq_all > 0]))
            # print(f"Iter {it+1:^3d} | Delta S = {delta_S:.2e}")
            
            if delta_S < conv_threshold:
                converged = True
                print(f"S-matrix converged within {conv_threshold:.2e} in {it + 1} iterations.")
                break
                
        S_old = S_current

    t1 = time.perf_counter()
    if not converged:
        print(f"Warning: Reached max_iter ({max_iter}) without full convergence.")

    # Final Output
    print("\n--- Schmidt Process Completed ---")
    print(f"Total solver time: {t1 - t0:.2f} s")
    print("\nFinal S-matrix probabilities (S_ij):")
    for j in range(n_channels):
        if k_sq_all[j] > 0:
            val = S_old[j]
            prob = np.abs(val)**2
            print(f"  S({incoming_idx},{j}) = {val.real: .6f} + i{val.imag: .6f}  |  P = {prob:.6f}")

if __name__ == "__main__":
    main()