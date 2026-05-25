"""
WISE Contour Integration Minimal Working Example (Decoupled Contours)
---------------------------------------------
"""
import numpy as np
import time
from scipy.sparse.linalg import eigs, LinearOperator

from wise_scattering.asymptotics import get_inward_initial_ratio, process_asymptotics_and_greens
from wise_scattering.physics_utilities import (generate_space_fixed_channels, precompute_potential_sparsity, 
                           precompute_centrifugal_sparsity, load_radial_potential, init_wigner_symbols, compute_diagonal_potential_jit)
from wise_scattering.propagator import renormalized_numerov
from wise_scattering.wise_core import apply_K_matvec, apply_KH_matvec, compute_U_psi_jit, apply_contour_correction, apply_contour_projector

def main():
    print("Starting WISE Contour MWE...")
    init_wigner_symbols()

    ATOMIC_MASS_UNIT_TO_ELECTRON_MASS = 1822.8884862
    HARTREE_TO_INVERSE_CM = 2.1947463136314e5 
    MASS_CO = 28.0101 * ATOMIC_MASS_UNIT_TO_ELECTRON_MASS 
    MASS_HE = 4.002602 * ATOMIC_MASS_UNIT_TO_ELECTRON_MASS  
    REDUCED_MASS = (MASS_CO * MASS_HE) / (MASS_CO + MASS_HE)
    B_ROT = 1.922521 / HARTREE_TO_INVERSE_CM          
    PREFACTOR = 2.0 * REDUCED_MASS

    # Target collision & algorithm setup
    E_col_cm1 = 8.84506
    E_total = E_col_cm1 / HARTREE_TO_INVERSE_CM
    J_tot = 0
    parity = 1
    j_max = 2
    lambda_max = 20

    # Target incoming channel
    incoming_j = 0
    incoming_l = 0

    max_iter = 1000
    conv_threshold = 5e-4 # Convergence threshold for u_R
    solver_tol = 1e-5 # Tolerance for linear solves in contour integration
    outer_delta = 0.2 # Extra buffer for R_out beyond eta_max to ensure all eigenvalues are enclosed

    # CONTOUR SETTINGS
    # Stage 1: should enclose all diverging eigenvalues, does not worry about z=1
    N_q_1 = 32 # Number of quadrature points for Stage 1
    R_in_1 = 1.002 # Inner radius for Stage 1 contour
    
    # Stage 2: must avoid z=1 but catch divergent states.
    N_q_2 = 23200  
    R_in_2 = 1.002
    # ----------------------------------

    grid = np.arange(3.0, 20.0 + 0.01, 0.01)
    n_points = len(grid)
    weights = np.ones_like(grid) * 0.01
    weights[0] = weights[-1] = 0.005
    sqrt_w = np.sqrt(weights)

    print(f"Setting up CO+He scattering (E={E_col_cm1} cm-1, J={J_tot}, j_max={j_max})...")
    channel_j, channel_l, channel_E = generate_space_fixed_channels(J_tot, parity, j_max, B_ROT)
    n_channels = len(channel_j)
    print(f"Total channels (open + closed): {n_channels}")

    try:
        incoming_idx = next(i for i, (j, l) in enumerate(zip(channel_j, channel_l)) if j == incoming_j and l == incoming_l)
        print(f"Target incoming channel (j={incoming_j}, l={incoming_l}) mapped to index: {incoming_idx}")
    except StopIteration:
        raise ValueError(f"Incoming channel (j={incoming_j}, l={incoming_l}) not found in the generated basis set.")

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
        k_sq = PREFACTOR * (E_total - channel_E[c])
        k_sq_all[c] = k_sq
        V_ii = compute_diagonal_potential_jit(J_tot, channel_j[c], channel_l[c], lambda_max, radial_pot)
        W_r = PREFACTOR * V_ii - k_sq + (channel_l[c] * (channel_l[c] + 1)) / (grid**2)
        
        Q_out = renormalized_numerov(W_r, 0.01, direction=1, initial_ratio=0)
        ratio_inward_start = get_inward_initial_ratio(k_sq, channel_l[c], grid[-1], 0.01)
        Q_in = renormalized_numerov(W_r, 0.01, direction=-1, initial_ratio=ratio_inward_start)
        
        delta, u_reg_all[c, :], G_diag[c, :] = process_asymptotics_and_greens(k_sq, channel_l[c], grid, Q_out, Q_in)
        phase_shifts[c] = delta
        R_ratio[c, :] = Q_out + 0.0j

    u_0 = np.zeros((n_channels, n_points), dtype=np.complex128)
    u_0[incoming_idx, :] = u_reg_all[incoming_idx, :]
    u_source_symm = (u_0.T.flatten() * np.repeat(sqrt_w, n_channels))

    def matvec(v): return apply_K_matvec(v, grid, sqrt_w, PREFACTOR, G_diag, R_ratio, pot_data, radial_pot, cent_data)
    def rmatvec(v): return apply_KH_matvec(v, grid, sqrt_w, PREFACTOR, G_diag, R_ratio, pot_data, radial_pot, cent_data)
    K_op = LinearOperator((n_channels * n_points, n_channels * n_points), matvec=matvec, rmatvec=rmatvec, dtype=np.complex128)

    print("\n--- Stage 0: Contour Geometry ---")
    t_eig_start = time.perf_counter()
    val_max, _ = eigs(K_op, k=1, which='LM')
    eta_max = np.abs(val_max[0])
    R_out = eta_max + outer_delta
    print(f"Found eta_max = {eta_max:.4f}")
    print(f"Stage 1 Bounds: R_in = {R_in_1:.3f}, R_out = {R_out:.3f} (N_q = {N_q_1})")
    print(f"Stage 2 Bounds: R_in = {R_in_2:.3f}, R_out = {R_out:.3f} (N_q = {N_q_2})")

    print("\n--- Stage 1: Regularized Born Series ---")
    u_R = u_source_symm.copy()
    converged = False
    total_inner_iters = 0
    
    t_born_start = time.perf_counter()
    for it in range(max_iter):
        u_R_old = u_R.copy()
        
        P_D_y, inner_iters = apply_contour_projector(u_R_old, K_op, R_out, R_in_1, N_q_1, tol=solver_tol)
        total_inner_iters += inner_iters

        u_diff = u_R_old - P_D_y
        K_R_y = matvec(u_diff)
        u_R = u_source_symm + K_R_y
            
        delta_u = np.max(np.abs(u_R - u_R_old))
        print(f"Iter {it+1:3d} | Delta u_R = {delta_u:.2e} | BiCGSTAB iters this step: {inner_iters}")

        if delta_u < conv_threshold:
            converged = True
            print(f"Born Series converged in {it + 1} iterations.")
            break

    print("\n--- Stage 2: Contour Wavefunction Correction ---")
    t_corr_start = time.perf_counter()
    
    full_correction, stage2_iters = apply_contour_correction(u_R, K_op, R_out, R_in_2, N_q_2, tol=solver_tol)
    psi_final_symm = u_R + full_correction
    print(f"Correction computed in {time.perf_counter() - t_corr_start:.2f} s | BiCGSTAB iters: {stage2_iters}")

    psi_phys_flat = psi_final_symm / np.repeat(sqrt_w, n_channels)
    psi_phys = psi_phys_flat.reshape((n_points, n_channels)).T
    
    print("\n--- Stage 3: S-Matrix Extraction ---")
    U_psi = compute_U_psi_jit(psi_phys, grid, PREFACTOR, pot_data, radial_pot, cent_data)
    S_final = np.zeros(n_channels, dtype=np.complex128)
    
    for j in range(n_channels):
        if k_sq_all[j] <= 0: continue
        integrand = u_reg_all[j, :] * U_psi[j, :]
        integral = np.trapz(integrand, grid)
        delta_ij = 1.0 if j == incoming_idx else 0.0
        norm = (np.sqrt(k_sq_all[incoming_idx]) * np.sqrt(k_sq_all[j]))**(-0.5)
        phase_factor = np.exp(1j * (phase_shifts[incoming_idx] + phase_shifts[j]))
        S_final[j] = phase_factor * (delta_ij - 2j * norm * integral)

    t_total_end = time.perf_counter()
    print("\n--- Process Completed ---")
    print(f"Total solver time: {t_total_end - t_eig_start:.2f} s")
    print(f"Total BiCGSTAB linear solver iterations: {total_inner_iters}")
    print("\nFinal S-matrix probabilities (S_ij):")
    for j in range(n_channels):
        if k_sq_all[j] > 0:
            val = S_final[j]
            print(f"  S({incoming_idx},{j}) = {val.real: .6f} + i{val.imag: .6f}  |  P = {np.abs(val)**2:.6f}")

if __name__ == "__main__":
    main()