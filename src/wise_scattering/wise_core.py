from numba import njit, prange
import numpy as np
from wise_scattering.physics_utilities import flat_to_matrix_coords

@njit(cache=True)
def construct_coupling_matrix_jit(r_val, prefactor, n_channels,
                                  pot_flat_pos, pot_terms_per_elem, pot_lambda_idx, pot_coeffs, 
                                  radial_pot_row,
                                  cent_flat_pos, cent_coeffs):
    """
    Constructs the total V(r) coupling matrix at a specific grid point r.
    This exactly replicates: V = -prefactor * V_pot - V_cent / r^2
    """
    # Allocate as float64, just like the original functions
    V = np.zeros((n_channels, n_channels), dtype=np.float64)
    
    # --- 1. Potential Contribution ---
    innermost_index = 0
    for idx in range(len(pot_flat_pos)):
        position = pot_flat_pos[idx]
        i, j = flat_to_matrix_coords(position)
        
        value = 0.0
        n_terms = pot_terms_per_elem[idx]
        for n_lambda in range(n_terms):
            value += pot_coeffs[innermost_index] * radial_pot_row[pot_lambda_idx[innermost_index]]
            innermost_index += 1
            
        # Apply the -prefactor exactly as the original formulation
        scaled_value = -prefactor * value
        V[i, j] = scaled_value
        V[j, i] = scaled_value

    # --- 2. Centrifugal Contribution ---
    for idx in range(len(cent_flat_pos)):
        position = cent_flat_pos[idx]
        i, j = flat_to_matrix_coords(position)
        
        # Apply the -1/r^2 scaling exactly as the original formulation
        cent_val = -cent_coeffs[idx] / (r_val**2)
        
        # Add to existing potential (don't overwrite!)
        V[i, j] += cent_val
        if i != j:
            V[j, i] += cent_val
        
    return V

@njit(fastmath=True, parallel=True)
def compute_U_psi_jit(psi_phys, grid, prefactor, pot_data, radial_pot, cent_data):
    """Computes U @ psi on the physical wavefunctions for the S-matrix integral."""
    C, N = psi_phys.shape
    U_psi = np.zeros_like(psi_phys)
    for j in prange(N):
        # We use the exact same negative constructor to get U
        V_j = -construct_coupling_matrix_jit(grid[j], prefactor, C, *pot_data, radial_pot[j], *cent_data)
        U_psi[:, j] = V_j.astype(np.complex128) @ psi_phys[:, j]
    return U_psi

@njit(fastmath=True, parallel=True)
def apply_K_matvec(psi_in, grid, sqrt_w, prefactor, G_diag, R_ratio, 
                   pot_data, radial_pot, cent_data):
    """
    Computes K @ psi using the matrix-free sweep.
    """
    C, N = G_diag.shape[0], len(grid)
    
    # Symmetrize
    psi_symm = np.empty_like(psi_in)
    for i in prange(N):
        for k in range(C):
            psi_symm[i*C + k] = psi_in[i*C + k] * sqrt_w[i]

    # V @ psi Stage
    V_psi = np.zeros((N, C), dtype=np.complex128)
    for j in prange(N):
        V_j = - construct_coupling_matrix_jit(grid[j], prefactor, C, *pot_data, radial_pot[j], *cent_data)
        V_psi[j, :] = V_j.astype(np.complex128) @ psi_symm[j*C : (j+1)*C]

    # Green's Function Sweep Stage
    psi_out_symm = np.zeros_like(psi_in)
    for c in prange(C):
        # Forward sweep
        prev_L = G_diag[c, 0] * V_psi[0, c]
        psi_out_symm[0*C + c] = prev_L
        for i in range(1, N):
            factor = (G_diag[c, i] / G_diag[c, i-1]) * R_ratio[c, i]
            curr_L = factor * prev_L + G_diag[c, i] * V_psi[i, c]
            psi_out_symm[i*C + c] = curr_L
            prev_L = curr_L
        # Backward sweep
        prev_U = 0.0j
        for i in range(N-2, -1, -1):
            curr_U = R_ratio[c, i+1] * (G_diag[c, i+1] * V_psi[i+1, c] + prev_U)
            psi_out_symm[i*C + c] += curr_U
            prev_U = curr_U

    # De-symmetrize
    psi_out = np.empty_like(psi_in)
    for i in prange(N):
        for k in range(C):
            psi_out[i*C + k] = psi_out_symm[i*C + k] * sqrt_w[i]
    return psi_out

@njit(fastmath=True, parallel=True)
def apply_KH_matvec(psi_in, grid, sqrt_w, prefactor, G_diag, R_ratio, 
                    pot_data, radial_pot, cent_data):
    """
    Computes psi_out = K.H @ psi_in.
    Math: K^H = V^H @ G^H
    """
    C, N = G_diag.shape[0], len(grid)

    # Symmetrize input
    psi_symm = np.empty_like(psi_in)
    for i in prange(N):
        for k in range(C):
            psi_symm[i*C + k] = psi_in[i*C + k] * sqrt_w[i]

    G_psi_H = np.zeros_like(psi_in)

    # --- STAGE 1: Green's Function Sweeps (Apply G^H) ---
    for c in prange(C):
        G_diag_conj = np.conj(G_diag[c, :])
        R_ratio_conj = np.conj(R_ratio[c, :])
        
        # Forward Sweep
        prev_S = G_diag_conj[0] * psi_symm[0*C + c]
        G_psi_H[0*C + c] = prev_S
        for i in range(1, N):
            factor = (G_diag_conj[i] / G_diag_conj[i-1]) * R_ratio_conj[i]
            current_S = factor * prev_S + G_diag_conj[i] * psi_symm[i*C + c]
            G_psi_H[i*C + c] = current_S
            prev_S = current_S

        # Backward Sweep
        prev_U = 0.0j
        for i in range(N-2, -1, -1):
            current_U = R_ratio_conj[i+1] * (G_diag_conj[i+1] * psi_symm[(i+1)*C + c] + prev_U)
            G_psi_H[i*C + c] += current_U
            prev_U = current_U

    # --- STAGE 2: Apply Potential V^H ---
    psi_out_symm = np.zeros_like(psi_in)
    for j in prange(N):
        # The * unpacking works identically to the forward K matvec
        V_j = - construct_coupling_matrix_jit(grid[j], prefactor, C, *pot_data, radial_pot[j], *cent_data)
        
        # Complex conjugate transpose of V
        V_j_H = V_j.T.astype(np.complex128)
        
        # Multiply: V^H @ (G^H psi)
        psi_out_symm[j*C : (j+1)*C] = V_j_H @ G_psi_H[j*C : (j+1)*C]

    # --- STAGE 3: De-symmetrize (using the safe explicit loop!) ---
    psi_out = np.empty_like(psi_in)
    for i in prange(N):
        for k in range(C):
            psi_out[i*C + k] = psi_out_symm[i*C + k] * sqrt_w[i]
            
    return psi_out

@njit
def apply_K_P(psi_vec, eigvals, u_T, v_T):
    """Action of the projection operator for divergent eigenvalues."""
    res = np.zeros_like(psi_vec)
    psi_contig = np.ascontiguousarray(psi_vec)
    for i in range(len(eigvals)):
        coeff = np.vdot(v_T[i], psi_contig)
        res += eigvals[i] * u_T[i] * coeff
    return res