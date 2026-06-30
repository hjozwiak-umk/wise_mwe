"""
WISE Contour Integration Minimal Working Example (MWE)
-------------------------------------------------------------------
This script is the entry point for demonstrating the matrix-free 
Weinberg-regularized Iterative Series Expansion (WISE) framework via 
complex contour integration for the Atom-Diatom scattering (CO+He) system.
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
    """
    Executes the full contour-projection WISE pipeline.
    
    Steps:
    1. System and Grid Initialization.
    2. Basis Set Generation & Matrix-Free Operator Construction.
    3. Stage 0: Evaluate the outer contour boundary (largest Weinberg eigenvalue).
    4. Stage 1: Born Series using an origin-centered inner contour.
    5. Stage 2: Divergent Wavefunction Correction (optionally: using a shifted inner contour 
       to safely bypass the singularity at z=1).
    6. Stage 3: S-Matrix column evaluation and flux normalization.
    """
    print("Starting WISE Contour MWE...")
    
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

    max_iter = 1000
    conv_threshold = 1e-6 # Convergence threshold for the regularized wavefunction u_R
    solver_tol = 1e-5     # Tolerance for BiCGSTAB linear solves at each quadrature point
    outer_delta = 0.2     # Safety buffer added to |eta_max| to strictly enclose all eigenvalues
    
    # --- CONTOUR GEOMETRY SETTINGS ---
    # Common Outer Ring: Encloses the entire divergent spectrum.
    N_q_outer = 30 # the outer ring can be treated with a limited number of points
    
    # Stage 1 (Born Series):
    # Extracts the regularized source term. The inner ring is centered at the 
    # origin (z=0) with R < 1 to cleanly enclose the divergent eigenvalues.
    N_q_1 = 50
    R_in_1 = 0.95
    
    # Stage 2 (Divergent Correction):
    # Evaluates the z/(1-z) weighted integral. To avoid the severe singularity 
    # exactly at z=1, the center of the inner ring can be optionally shifted 
    # along the positive real axis (x_c_2)
    N_q_2 = 50  
    R_in_2 = 1.0
    x_c_2 = 0.1
    # ----------------------------------

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
        Q_out = renormalized_numerov(W_r, 0.01, direction=1, initial_ratio=0)

        # Calculate Inward Initial Condition (y_N / y_N-1)
        ratio_inward_start = get_inward_initial_ratio(k_sq, channel_l[c], grid[-1], 0.01)
        
        # Inward Propagation (Irregular ratio)
        Q_in = renormalized_numerov(W_r, 0.01, direction=-1, initial_ratio=ratio_inward_start)
        
        # Extract phase shift, wavefunction, and the diagonal Green's function
        delta, u_reg_all[c, :], G_diag[c, :] = process_asymptotics_and_greens(k_sq, channel_l[c], grid, Q_out, Q_in)
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
    
    K_op_shape = (size, size)
    K_op = LinearOperator((size, size), matvec=matvec, rmatvec=rmatvec, dtype=np.complex128)

    print("\n--- Stage 0: Contour Geometry ---")
    # Dynamically determine the outer boundary by finding the largest Weinberg eigenvalue.
    t_eig_start = time.perf_counter()
    val_max, _ = eigs(K_op, k=1, which='LM')
    eta_max = np.abs(val_max[0])
    R_out = eta_max + outer_delta

    print(f"Found eta_max = {eta_max:.4f}, N_q_outer = {N_q_outer}")
    print(f"Stage 1 Bounds: R_in = {R_in_1:.3f}, R_out = {R_out:.3f} (N_q = {N_q_1})")
    print(f"Stage 2 Bounds: R_in = {R_in_2:.3f}, R_out = {R_out:.3f} (N_q = {N_q_2}, x_c_in = {x_c_2})")

    print("\n--- Stage 1: Regularized Born Series ---")
    u_R = u_source_symm.copy()
    converged = False
    total_inner_iters = 0
    
    t_born_start = time.perf_counter()
    for it in range(max_iter):
        u_R_old = u_R.copy()
        
        # 1. Evaluate the contour integral for the projector P_D acting on u_R_old
        P_D_y, inner_iters = apply_contour_projector(
            u_R_old, K_op_shape, R_out, R_in_1, N_q_outer, N_q_1, 
            grid, sqrt_w, PREFACTOR, G_diag, R_ratio, pot_data, radial_pot, cent_data, 
            tol=solver_tol
        )
        total_inner_iters += inner_iters

        # 2. Define the regularized kernel: K_R = K - K * P_D
        u_diff = u_R_old - P_D_y
        K_R_y = matvec(u_diff)

        # 3. Update the regularized wavefunction
        u_R = u_source_symm + K_R_y
            
        delta_u = np.max(np.abs(u_R - u_R_old))
        print(f"Iter {it+1:3d} | Delta u_R = {delta_u:.2e} | BiCGSTAB iters this step: {inner_iters}")

        if delta_u < conv_threshold:
            converged = True
            print(f"Born Series converged in {it + 1} iterations.")
            break

    print("\n--- Stage 2: Contour Wavefunction Correction ---")
    t_corr_start = time.perf_counter()
    
    full_correction, stage2_iters = apply_contour_correction(
        u_R, K_op_shape, R_out, R_in_2, N_q_outer, N_q_2, 
        grid, sqrt_w, PREFACTOR, G_diag, R_ratio, pot_data, radial_pot, cent_data, 
        tol=solver_tol, x_c_in=x_c_2
    )

    # Construct the final wavefunction
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