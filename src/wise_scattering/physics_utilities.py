import numpy as np
from numba import njit
from numba.core.typing import cffi_utils as cffi_support
import pywigxjpf as wig
import pywigxjpf_ffi
from scipy.interpolate import CubicSpline

# Register the ultra-fast Wigner symbol C-library
cffi_support.register_module(pywigxjpf_ffi)
nb_wig3jj = pywigxjpf_ffi.lib.wig3jj
nb_wig6jj = pywigxjpf_ffi.lib.wig6jj

def init_wigner_symbols(size=2000, max_wig=9):
    """
    Allocates memory for the pywigxjpf C-backend to compute Wigner symbols.
    
    Parameters
    ----------
    size : int, optional
        Size of the precomputed factorial tables.
    max_wig : int, optional
        Maximum angular momentum value expected in the calculation.
    """
    wig.wig_table_init(size, max_wig)
    wig.wig_temp_init(size)

# --- 1. Utility Functions ---
@njit(cache=True)
def matrix_to_flat(i, j):
    """
    Maps 2D symmetric matrix indices to a 1D flattened lower-triangle array index.

    Parameters
    ----------
    i : int
        Row index.
    j : int
        Column index.

    Returns
    -------
    int
        The 1D flattened index corresponding to the lower triangle position.
    """
    if i < j:
        i, j = j, i
    return (i * (i + 1)) // 2 + j

@njit(cache=True)
def flat_to_matrix_coords(pos):
    """
    Inverse of matrix_to_flat. Recovers the 2D matrix indices from a 1D flattened index.

    Parameters
    ----------
    pos : int
        The 1D flattened index.

    Returns
    -------
    tuple[int, int]
        The (row, column) indices of the symmetric matrix.
    """
    row = int(np.floor((np.sqrt(8 * pos + 1) - 1) / 2))
    col = pos - (row * (row + 1)) // 2
    return row, col

# --- 2. Channel Generation ---
def generate_space_fixed_channels(total_J: int, parity: int, j_max: int, B_rot: float):
    """
    Generates scattering channels for an Atom + Rigid Rotor system (space-fixed representation).

    Builds the basis set $|j, l\\rangle$ constrained by angular momentum coupling 
    and strict parity conservation.

    Parameters
    ----------
    total_J : int
        The total angular momentum of the collision complex.
    parity : int
        The total spatial parity of the system (+1 or -1).
    j_max : int
        The maximum rigid rotor angular momentum state to include in the basis.
    B_rot : float
        The rotational constant of the rigid rotor (in cm^-1 or target energy units).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Arrays containing the rotor angular momentum (j), orbital angular 
        momentum (l), and asymptotic channel energies (E) respectively.
    """
    channels = []
    
    for j in range(j_max + 1):
        # Triangle inequality for angular momentum: |J - j| <= l <= J + j
        for l in range(abs(j - total_J), j + total_J + 1):
            
            # Check if this combination matches the requested spatial parity
            if ((-1)**(j + l)) == parity:
                # Asymptotic energy of the rigid rotor: E = B * j * (j + 1)
                E_asymp = B_rot * j * (j + 1)
                channels.append({'j': j, 'l': l, 'E': E_asymp})
                
    # Sort by asymptotic energy
    channels.sort(key=lambda x: x['E'])
    
    # Return as fast NumPy arrays for Numba
    channel_j = np.array([ch['j'] for ch in channels], dtype=np.int64)
    channel_l = np.array([ch['l'] for ch in channels], dtype=np.int64)
    channel_E = np.array([ch['E'] for ch in channels], dtype=np.float64)
    
    return channel_j, channel_l, channel_E

# --- 3. Potential Matrix Precomputation ---
@njit(cache=True)
def get_allowed_lambdas_jit(j_bra, l_bra, j_ket, l_ket, lambda_max):
    """
    Determines which Legendre multipole terms (lambda) non-trivially couple two scattering channels.
    
    Enforces angular momentum triangle rules and parity selection rules for the 
    intermolecular potential expansion.

    Parameters
    ----------
    j_bra, l_bra : int
        Angular momentum quantum numbers of the bra state.
    j_ket, l_ket : int
        Angular momentum quantum numbers of the ket state.
    lambda_max : int
        The truncation limit of the potential multipole expansion.

    Returns
    -------
    np.ndarray
        An array of allowed lambda integers.
    """
    low_j  = abs(j_bra - j_ket)
    high_j = j_bra + j_ket
    low_l  = abs(l_bra - l_ket)
    high_l = l_bra + l_ket

    low  = max(low_j, low_l)
    high = min(high_j, high_l, lambda_max)

    c = 0
    for lam in range(low, high + 1):
        if ((j_bra + j_ket + lam) % 2 == 0) and ((l_bra + l_ket + lam) % 2 == 0):
            c += 1

    out = np.empty(c, dtype=np.int64)
    i = 0
    for lam in range(low, high + 1):
        if ((j_bra + j_ket + lam) % 2 == 0) and ((l_bra + l_ket + lam) % 2 == 0):
            out[i] = lam
            i += 1
    return out

@njit(fastmath=True)
def compute_coefficients_jit(J, j_bra, l_bra, j_ket, l_ket, lambdas):
    """
    Evaluates the exact angular coupling coefficients using Wigner 3-j and 6-j symbols.

    Parameters
    ----------
    J : int
        Total angular momentum.
    j_bra, l_bra : int
        Quantum numbers of the bra state.
    j_ket, l_ket : int
        Quantum numbers of the ket state.
    lambdas : np.ndarray
        Array of allowed multipole expansion indices.

    Returns
    -------
    np.ndarray
        The angular coupling coefficients corresponding to each allowed lambda.
    """
    n = lambdas.shape[0]
    coefficients = np.empty(n, dtype=np.float64)

    # double values expected by wigxjpf
    d_j_bra = 2 * j_bra
    d_j_ket = 2 * j_ket
    d_l_bra = 2 * l_bra
    d_l_ket = 2 * l_ket
    d_J     = 2 * J
    
    phase_exp = J + j_bra + j_ket
    phase = -1.0 if (phase_exp & 1) else 1.0
    
    prod = float(d_j_bra + 1) * float(d_j_ket + 1) * float(d_l_bra + 1) * float(d_l_ket + 1)
    sqrt_term = np.sqrt(prod)

    for i in range(n):
        lam = lambdas[i]
        d_lam = 2 * lam
        
        # Calculate Wigner symbols
        threej_j = nb_wig3jj(d_j_bra, d_lam, d_j_ket, 0, 0, 0)
        threej_l = nb_wig3jj(d_l_bra, d_lam, d_l_ket, 0, 0, 0)
        sixj_val = nb_wig6jj(d_j_bra, d_l_bra, d_J, d_l_ket, d_j_ket, d_lam)
        
        coefficients[i] = phase * sqrt_term * threej_j * threej_l * sixj_val
    
    return coefficients

@njit(fastmath=True)
def precompute_potential_sparsity(channel_j, channel_l, J, lambda_max, offdiag_only=True):
    """
    Scans the basis set and precomputes the sparse memory mapping for the potential matrix.

    Parameters
    ----------
    channel_j, channel_l : np.ndarray
        The basis set quantum numbers.
    J : int
        Total angular momentum.
    lambda_max : int
        Maximum multipole expansion term.
    offdiag_only : bool, optional
        If True, only maps the off-diagonal elements (used for matrix-free sweeps).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - flat_positions: 1D matrix coordinates for the non-zero couplings.
        - terms_per_element: Number of lambda terms contributing to each coordinate.
        - lambda_indices: The specific lambda values for each term.
        - coefficients: The computed angular coefficients.
    """
    num_channels = channel_j.shape[0]

    # PASS 1: Count elements to allocate arrays
    nz_elements = 0
    total_lambda_terms = 0

    for bra in range(num_channels):
        limit = bra if offdiag_only else (bra + 1)
        for ket in range(limit):
            allowed_lambdas = get_allowed_lambdas_jit(channel_j[bra], channel_l[bra], channel_j[ket], channel_l[ket], lambda_max)
            if allowed_lambdas.shape[0] > 0:
                nz_elements += 1
                total_lambda_terms += allowed_lambdas.shape[0]

    # Allocate outputs
    flat_positions = np.empty(nz_elements, dtype=np.int32)
    terms_per_element = np.empty(nz_elements, dtype=np.int32)
    lambda_indices = np.empty(total_lambda_terms, dtype=np.int32)
    coefficients = np.empty(total_lambda_terms, dtype=np.float64)

    # PASS 2: Fill arrays
    elem_idx = 0
    term_idx = 0

    for bra in range(num_channels):
        limit = bra if offdiag_only else (bra + 1)
        for ket in range(limit):
            lambdas = get_allowed_lambdas_jit(channel_j[bra], channel_l[bra], channel_j[ket], channel_l[ket], lambda_max)
            
            n_lambdas = lambdas.shape[0]
            if n_lambdas == 0:
                continue

            coeff_vec = compute_coefficients_jit(J, channel_j[bra], channel_l[bra], channel_j[ket], channel_l[ket], lambdas)

            flat_positions[elem_idx] = np.int32(matrix_to_flat(bra, ket))
            terms_per_element[elem_idx] = np.int32(n_lambdas)

            for q in range(n_lambdas):
                lambda_indices[term_idx + q] = np.int32(lambdas[q]) # The index maps directly to the lambda value!
                coefficients[term_idx + q] = coeff_vec[q]

            elem_idx += 1
            term_idx += n_lambdas

    return flat_positions, terms_per_element, lambda_indices, coefficients

# --- 4. Centrifugal Matrix Precomputation ---
@njit(cache=True, fastmath=True)
def precompute_centrifugal_sparsity(channel_j, channel_l, offdiag_only=True):
    """
    Precomputes the sparse memory mapping for the purely diagonal centrifugal potential.

    Parameters
    ----------
    channel_j, channel_l : np.ndarray
        The basis set quantum numbers.
    offdiag_only : bool, optional
        If True, only maps off-diagonal elements (which are always zero for centrifugal terms).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - flat_positions: 1D matrix coordinates.
        - coefficients: The evaluated l(l+1) terms.
    """
    num_channels = channel_j.shape[0]
    
    nz_count = 0
    for bra in range(num_channels):
        limit = bra if offdiag_only else (bra + 1)
        for ket in range(limit):
            if (channel_j[bra] == channel_j[ket] and channel_l[bra] == channel_l[ket]):
                nz_count += 1

    flat_positions = np.empty(nz_count, dtype=np.int32)
    coefficients = np.empty(nz_count, dtype=np.float64)

    write_idx = 0
    for bra in range(num_channels):
         limit = bra if offdiag_only else (bra + 1)
         for ket in range(limit):
            if (channel_j[bra] == channel_j[ket] and channel_l[bra] == channel_l[ket]):
                l_val = channel_l[bra]
                flat_positions[write_idx] = matrix_to_flat(bra, ket)
                coefficients[write_idx] = l_val * (l_val + 1)
                write_idx += 1
                
    return flat_positions, coefficients

@njit(fastmath=True)
def compute_diagonal_potential_jit(J, j_channel, l_channel, lambda_max, radial_pot_matrix):
    """
    Computes the diagonal interaction potential V_ii(r) for a single channel across the grid.

    Parameters
    ----------
    J : int
        Total angular momentum.
    j_channel, l_channel : int
        Quantum numbers defining the target channel.
    lambda_max : int
        Maximum multipole expansion term.
    radial_pot_matrix : np.ndarray
        The interpolated 2D dense radial potential matrix.

    Returns
    -------
    np.ndarray
        The exact diagonal interaction potential trace evaluated across the radial grid.
    """
    n_points = radial_pot_matrix.shape[0]
    V_ii = np.zeros(n_points, dtype=np.float64)
    
    # Get allowed lambdas for the diagonal (bra == ket)
    lambdas = get_allowed_lambdas_jit(j_channel, l_channel, j_channel, l_channel, lambda_max)
    
    if lambdas.shape[0] == 0:
        return V_ii
        
    # Get the angular coefficients
    coeffs = compute_coefficients_jit(J, j_channel, l_channel, j_channel, l_channel, lambdas)
    
    # Sum the potential terms: V_ii(r) = sum( c_lambda * V_lambda(r) )
    for i in range(lambdas.shape[0]):
        lam = lambdas[i]
        c = coeffs[i]
        for r_idx in range(n_points):
            V_ii[r_idx] += c * radial_pot_matrix[r_idx, lam]
            
    return V_ii

def load_radial_potential(filepath: str, r_grid: np.ndarray, lambda_max: int, 
                          header_lines: int = 16, points_per_lambda: int = 9941):
    """
    Reads CO-He radial potential terms from a file and interpolates them onto the solver's grid.
    The potential terms were computed from the He+CO PES of Peterson and McBane 
    [J. Chem. Phys. 123 (8) (2005), http://dx.doi.org/10.1063/1.1947194.]
    as explained in the Methods section.

    Parameters
    ----------
    filepath : str
        Path to the multi-column interaction potential .dat file.
    r_grid : np.ndarray
        The 1D NumPy array of radial grid points used by the solver (in Bohr).
    lambda_max : int
        The maximum lambda (Legendre term) required by the chosen basis set.
    header_lines : int, optional
        Number of header lines to skip at the top of the file.
    points_per_lambda : int, optional
        The number of data points provided in the file per lambda term block.
        
    Returns
    -------
    np.ndarray
        A 2D dense array of shape (len(r_grid), lambda_max + 1) representing 
        the radial potential V(r) evaluated in atomic units (Hartree).
    """
    HARTREE_TO_INVERSE_CM = 2.1947463136314e5
    
    # Initialize the output matrix with zeros (handles any missing lambdas safely)
    n_points = len(r_grid)
    V_matrix = np.zeros((n_points, lambda_max + 1), dtype=np.float64)
    
    with open(filepath, 'r') as f:
        # Skip the header
        for _ in range(header_lines):
            next(f)
            
        while True:
            try:
                line = next(f).strip()
                if not line:
                    continue
                
                # Read the lambda value (e.g., "0", "1", "2")
                lam = int(line)
                
                # Pre-allocate arrays for the raw file data
                r_raw = np.zeros(points_per_lambda)
                v_raw = np.zeros(points_per_lambda)
                
                # Read the grid points and potential values
                for i in range(points_per_lambda):
                    parts = next(f).split()
                    r_raw[i] = float(parts[0])
                    v_raw[i] = float(parts[1]) / HARTREE_TO_INVERSE_CM  # Convert cm-1 to Hartree
                    
                # If this lambda term is needed for our basis set, interpolate it!
                if lam <= lambda_max:
                    spline = CubicSpline(r_raw, v_raw)
                    # Evaluate the spline exactly at the solver's grid points
                    V_matrix[:, lam] = spline(r_grid)
                    
            except StopIteration:
                break # End of file
                
    return V_matrix