from numba import njit, prange
import numpy as np
from wise_scattering.physics_utilities import flat_to_matrix_coords
from scipy.sparse.linalg import bicgstab, LinearOperator, gmres

@njit(cache=True)
def apply_V_action_jit(r_val, prefactor, n_channels,
                       pot_flat_pos, pot_terms_per_elem, pot_lambda_idx, pot_coeffs, 
                       radial_pot_row,
                       cent_flat_pos, cent_coeffs, psi_in, out_buf):
    """
    Computes the action of the scaled interaction potential matrix directly on a vector:
    (-V) |psi> = (2μ * V_pot(r) + l(l+1)/r^2) |psi>
    
    This replaces the O(N^2) dense matrix allocation inside the tight loops.
    """
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
            
        scaled_value = prefactor * value # sign flipped to apply -V
        out_buf[i] += scaled_value * psi_in[j]
        out_buf[j] += scaled_value * psi_in[i]

    # --- 2. Centrifugal Contribution ---
    for idx in range(len(cent_flat_pos)):
        position = cent_flat_pos[idx]
        i, j = flat_to_matrix_coords(position)
        
        cent_val = cent_coeffs[idx] / (r_val**2) # sign flipped
        out_buf[i] += cent_val * psi_in[j]
        out_buf[j] += cent_val * psi_in[i]

@njit(fastmath=True, parallel=True)
def compute_U_psi_jit(psi_phys, grid, prefactor, pot_data, radial_pot, cent_data):
    """
    Computes the action of the physical potential operator U on the full wavefunction.
    
    Used for computing S-matrix elements via Eq. (11) in the manuscript.
    
    Parameters
    ----------
    psi_phys : np.ndarray
        The un-symmetrized physical wavefunction matrix of shape (C, N).
    grid : np.ndarray
        The radial spatial grid.
    prefactor : float
        The mass prefactor (2μ).
    pot_data, cent_data : tuple
        Unpacked tuples containing the sparse potential and centrifugal mappings.
    radial_pot : np.ndarray
        The fully interpolated dense 2D radial potential matrix.

    Returns
    -------
    np.ndarray
        A 2D complex128 array of shape (C, N) representing the scaled potential action.
    """
    C, N = psi_phys.shape
    U_psi = np.zeros_like(psi_phys)
    for j in prange(N):
        v_j = np.zeros(C, dtype=np.complex128)
        slice_in = np.empty(C, dtype=np.complex128)
        for c in range(C):
            slice_in[c] = psi_phys[c, j]
        apply_V_action_jit(grid[j], prefactor, C, *pot_data, radial_pot[j], *cent_data, slice_in, v_j)
        for c in range(C):
            U_psi[c, j] = v_j[c]
    return U_psi

@njit(fastmath=True, parallel=True)
def apply_K_matvec(psi_in, grid, sqrt_w, prefactor, G_diag, R_ratio, forward_ratios,
                   pot_data, radial_pot, cent_data):
    """
    Matrix-free evaluator passed to the Arnoldi eigensolver: K|ψ⟩ = G * U * |ψ⟩.

    Parameters
    ----------
    psi_in : np.ndarray
        A flattened 1D array representing the state vector.
    grid : np.ndarray
        The radial spatial grid.
    sqrt_w : np.ndarray
        The square roots of the integration weights (for symmetrization).
    prefactor : float
        The reduced mass prefactor.
    G_diag : np.ndarray
        The diagonal elements of the reference Green's functions.
    R_ratio : np.ndarray
        The regular solution outward Numerov ratios.
    forward_ratios : np.ndarray
        Precomputed ratio for avoiding complex divisions.
    pot_data, cent_data : tuple
        Unpacked tuples containing sparse interaction matrix data.
    radial_pot : np.ndarray
        Dense radial potential matrix.

    Returns
    -------
    np.ndarray
        The resulting flattened 1D array after applying K.
    """
    C, N = G_diag.shape[0], len(grid)
    
    # V @ psi Stage
    V_psi_cn = np.zeros((C, N), dtype=np.complex128)
    for j in prange(N):
        v_j = np.zeros(C, dtype=np.complex128)
        psi_slice = np.empty(C, dtype=np.complex128)
        for c in range(C):
            psi_slice[c] = psi_in[j*C + c] * sqrt_w[j]
        
        apply_V_action_jit(grid[j], prefactor, C, *pot_data, radial_pot[j], *cent_data, psi_slice, v_j)
        for c in range(C):
            V_psi_cn[c, j] = v_j[c]

    # Green's Function Sweep Stage
    psi_out_symm_cn = np.zeros((C, N), dtype=np.complex128)
    for c in prange(C):
        # Forward sweep
        prev_L = G_diag[c, 0] * V_psi_cn[c, 0]
        psi_out_symm_cn[c, 0] = prev_L
        for i in range(1, N):
            curr_L = forward_ratios[c, i] * prev_L + G_diag[c, i] * V_psi_cn[c, i]
            psi_out_symm_cn[c, i] = curr_L
            prev_L = curr_L
            
        # Backward sweep
        prev_U = 0.0j
        for i in range(N-2, -1, -1):
            curr_U = R_ratio[c, i+1] * (G_diag[c, i+1] * V_psi_cn[c, i+1] + prev_U)
            psi_out_symm_cn[c, i] += curr_U
            prev_U = curr_U

    # De-symmetrize
    psi_out = np.empty_like(psi_in)
    for i in prange(N):
        for k in range(C):
            psi_out[i*C + k] = psi_out_symm_cn[k, i] * sqrt_w[i]
    return psi_out

@njit(fastmath=True, parallel=True)
def apply_KH_matvec(psi_in, grid, sqrt_w, prefactor, G_diag, R_ratio, forward_ratios,
                    pot_data, radial_pot, cent_data):
    """
    Computes the adjoint operator K^H |ψ⟩ = U^H * G^H * |ψ⟩.

    Required by the Arnoldi eigensolver for left-eigenspace calculations to identify biorthogonal 
    eigenvectors of the non-Hermitian operator K.

    Parameters
    ----------
    (Matches apply_K_matvec exactly)

    Returns
    -------
    np.ndarray
        The resulting flattened 1D array after applying K^H.
    """
    C, N = G_diag.shape[0], len(grid)

    # Symmetrize input
    psi_symm_cn = np.empty((C, N), dtype=np.complex128)
    for i in prange(N):
        for k in range(C):
            psi_symm_cn[k, i] = psi_in[i*C + k] * sqrt_w[i]

    G_psi_H_cn = np.zeros((C, N), dtype=np.complex128)

    # --- STAGE 1: Green's Function Sweeps (Apply G^H) ---
    for c in prange(C):
        G_diag_conj = np.conj(G_diag[c, :])
        R_ratio_conj = np.conj(R_ratio[c, :])
        forward_ratios_conj = np.conj(forward_ratios[c, :])
        
        # Forward Sweep
        prev_S = G_diag_conj[0] * psi_symm_cn[c, 0]
        G_psi_H_cn[c, 0] = prev_S
        for i in range(1, N):
            current_S = forward_ratios_conj[i] * prev_S + G_diag_conj[i] * psi_symm_cn[c, i]
            G_psi_H_cn[c, i] = current_S
            prev_S = current_S

        # Backward Sweep
        prev_U = 0.0j
        for i in range(N-2, -1, -1):
            current_U = R_ratio_conj[i+1] * (G_diag_conj[i+1] * psi_symm_cn[c, i+1] + prev_U)
            G_psi_H_cn[c, i] += current_U
            prev_U = current_U

    # --- STAGE 2: Apply Potential V^H ---
    psi_out = np.empty_like(psi_in)
    for j in prange(N):
        slice_in = np.empty(C, dtype=np.complex128)
        for c in range(C):
            slice_in[c] = G_psi_H_cn[c, j]
            
        v_j = np.zeros(C, dtype=np.complex128)
        # Note: V is real-symmetric, so V^H = V
        apply_V_action_jit(grid[j], prefactor, C, *pot_data, radial_pot[j], *cent_data, slice_in, v_j)
        
        for c in range(C):
            psi_out[j*C + c] = v_j[c] * sqrt_w[j]
            
    return psi_out

@njit
def apply_K_P(psi_vec, eigvals, u_T, v_T):
    """
    Computes the action of the divergent part of the kernel, K_D, on a state vector (Eq. (16) in the manuscript).

    K_D|ψ⟩ = sum(eta_i * |u_i⟩ * ⟨v_i|ψ⟩) for all divergent eigenvalues.
    
    Parameters
    ----------
    psi_vec : np.ndarray
        The state vector |ψ⟩ to project.
    eigvals : np.ndarray
        The divergent Weinberg eigenvalues (eta).
    u_T : np.ndarray
        Transposed right-eigenvectors (the |u_i⟩ states).
    v_T : np.ndarray
        Transposed, biorthogonalized left-eigenvectors (the ⟨v_i| states).

    Returns
    -------
    np.ndarray
        The resulting regularized state vector correction.
    """
    psi_c = np.ascontiguousarray(psi_vec)
    coeffs = (np.conj(v_T) @ psi_c) * eigvals
    return np.ascontiguousarray(u_T.T) @ coeffs

class SolverCounter:
    """
    A simple callback class to tally the total number of iterative solver 
    iterations (e.g., Bi-CGSTAB) across multiple linear system evaluations.
    
    Attributes
    ----------
    n_iters : int
        The running total of iterations expended by the solver.
    """
    def __init__(self):
        self.n_iters = 0
    def __call__(self, xk=None):
        self.n_iters += 1

@njit
def apply_shifted_K_matvec(v, z, grid, sqrt_w, PREFACTOR, G_diag, R_ratio, forward_ratios, pot_data, radial_pot, cent_data):
    """
    Computes the action of the shifted kernel operator (z * I - K) on a state vector.
    
    Parameters
    ----------
    v : np.ndarray
        A flattened 1D array representing the state vector.
    z : complex
        The complex shift, corresponding to a quadrature point on the contour.
    grid : np.ndarray
        The radial spatial grid.
    sqrt_w : np.ndarray
        The square roots of the integration weights.
    PREFACTOR : float
        The reduced mass prefactor.
    G_diag : np.ndarray
        The diagonal elements of the reference Green's functions.
    R_ratio : np.ndarray
        The regular solution outward Numerov ratios.
    forward_ratios : np.ndarray
        Precomputed ratio for avoiding complex divisions.
    pot_data, cent_data : tuple
        Unpacked tuples containing sparse interaction matrix data.
    radial_pot : np.ndarray
        Dense radial potential matrix.

    Returns
    -------
    np.ndarray
        The resulting 1D complex vector after applying the shifted operator.
    """
    # 1. Compute the standard K * v
    K_v = apply_K_matvec(v, grid, sqrt_w, PREFACTOR, G_diag, R_ratio, forward_ratios, pot_data, radial_pot, cent_data)
    
    # 2. Apply the shift and subtract
    result = np.empty_like(v)
    for i in range(len(v)):
        result[i] = z * v[i] - K_v[i]
        
    return result

def apply_contour_projector(v, K_op_shape, R_out, R_in, N_q_out, N_q_in, grid, sqrt_w, PREFACTOR, G_diag, R_ratio, forward_ratios, pot_data, radial_pot, cent_data, tol=1e-5, x_c_in=0.0):
    """
    Evaluates the contour integral for the divergent subspace projector acting on a vector.
    
    Parameters
    ----------
    v : np.ndarray
        A flattened 1D array representing the state vector.
    K_op_shape : tuple
        The shape of the linear operator.
    R_out : float
        The radius of the outer integration ring (enclosing all divergent eigenvalues).
    R_in : float
        The radius of the inner integration ring.
    N_q_out : int
        Number of trapezoidal quadrature points on the outer ring.
    N_q_in : int
        Number of trapezoidal quadrature points on the inner ring.
    grid, sqrt_w, PREFACTOR, G_diag, R_ratio, forward_ratios, pot_data, radial_pot, cent_data : 
        Physical system and potential parameters required for the K operator.
    tol : float, optional
        Convergence tolerance for the internal Bi-CGSTAB linear solver (default is 1e-5).
    x_c_in : float, optional
        The center coordinate of the inner ring on the real axis (default is 0.0).
        
    Returns
    -------
    tuple
        - P_v (np.ndarray): The projected vector containing only the divergent components.
        - n_iters (int): The total number of Bi-CGSTAB iterations expended across all nodes.
    """
    theta_out = np.linspace(0, 2 * np.pi, N_q_out, endpoint=False)
    d_theta = (2 * np.pi) / N_q_in
    theta_in = np.linspace(0, 2 * np.pi, N_q_in, endpoint=False) + (d_theta / 2.0) # phase shift to increase the distance to the eigenvalues
    
    z_out = R_out * np.exp(1j * theta_out)
    z_in = x_c_in + R_in * np.exp(1j * theta_in) # Applies the shift (0.0 by default)
    
    P_v = np.zeros_like(v, dtype=np.complex128)
    counter = SolverCounter()
    
    for k in range(N_q_out):
        z_curr = z_out[k]
        def matvec_wrapper(x, z=z_curr):
            return apply_shifted_K_matvec(x, z, grid, sqrt_w, PREFACTOR, G_diag, R_ratio, forward_ratios, pot_data, radial_pot, cent_data)
        A = LinearOperator(K_op_shape, matvec=matvec_wrapper, dtype=np.complex128)
        y, _ = bicgstab(A, v, tol=tol, callback=counter)
        P_v += (z_curr / N_q_out) * y

    for k in range(N_q_in):
        z_curr = z_in[k]
        def matvec_wrapper(x, z=z_curr):
            return apply_shifted_K_matvec(x, z, grid, sqrt_w, PREFACTOR, G_diag, R_ratio, forward_ratios, pot_data, radial_pot, cent_data)
        A = LinearOperator(K_op_shape, matvec=matvec_wrapper, dtype=np.complex128)
        y, _ = bicgstab(A, v, tol=tol, callback=counter)
        P_v -= ((z_curr - x_c_in) / N_q_in) * y
        
    return P_v, counter.n_iters

def apply_contour_correction(v, K_op_shape, R_out, R_in, N_q_out, N_q_in, grid, sqrt_w, PREFACTOR, G_diag, R_ratio, forward_ratios, pot_data, radial_pot, cent_data, tol=1e-5, x_c_in=0.0):
    """
    Evaluates the contour integral for the divergent wavefunction correction (Stage 2).
    
    Parameters
    ----------
    v : np.ndarray
        The regularized state vector to be corrected.
    K_op_shape : tuple
        The shape of the linear operator.
    R_out : float
        The radius of the outer integration ring.
    R_in : float
        The radius of the inner integration ring.
    N_q_out : int
        Number of trapezoidal quadrature points on the outer ring.
    N_q_in : int
        Number of trapezoidal quadrature points on the inner ring.
    grid, sqrt_w, PREFACTOR, G_diag, R_ratio, forward_ratios, pot_data, radial_pot, cent_data : 
        Physical system and potential parameters required for the K operator.
    tol : float, optional
        Convergence tolerance for the internal Bi-CGSTAB linear solver (default is 1e-5).
    x_c_in : float, optional
        The center coordinate of the inner ring on the real axis.
        
    Returns
    -------
    tuple
        - correction (np.ndarray): The integrated wavefunction correction vector.
        - n_iters (int): The total number of Bi-CGSTAB iterations expended across all nodes.
    """
    theta_out = np.linspace(0, 2 * np.pi, N_q_out, endpoint=False)
    d_theta = (2 * np.pi) / N_q_in
    theta_in = np.linspace(0, 2 * np.pi, N_q_in, endpoint=False) + (d_theta / 2.0) # phase shift to increase the distance to the eigenvalues
    
    z_out = R_out * np.exp(1j * theta_out)
    z_in = x_c_in + R_in * np.exp(1j * theta_in) # Apply the shift
    
    correction = np.zeros_like(v, dtype=np.complex128)
    counter = SolverCounter()
    
    for k in range(N_q_out):
        z_curr = z_out[k]
        def matvec_wrapper(x, z=z_curr):
            return apply_shifted_K_matvec(x, z, grid, sqrt_w, PREFACTOR, G_diag, R_ratio, forward_ratios, pot_data, radial_pot, cent_data)
        A = LinearOperator(K_op_shape, matvec=matvec_wrapper, dtype=np.complex128)
        y, _ = bicgstab(A, v, tol=tol, callback=counter)
        weight = (z_curr / N_q_out) * (z_curr / (1.0 - z_curr))
        correction += weight * y

    for k in range(N_q_in):
        z_curr = z_in[k]
        def matvec_wrapper(x, z=z_curr):
            return apply_shifted_K_matvec(x, z, grid, sqrt_w, PREFACTOR, G_diag, R_ratio, forward_ratios, pot_data, radial_pot, cent_data)
        A = LinearOperator(K_op_shape, matvec=matvec_wrapper, dtype=np.complex128)
        y, _ = bicgstab(A, v, tol=tol, callback=counter)
        
        weight = ((z_curr - x_c_in) / N_q_in) * (z_curr / (1.0 - z_curr))
        correction -= weight * y
        
    return correction, counter.n_iters