[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19395019.svg)](https://doi.org/10.5281/zenodo.19395019)

# WISE Scattering: Minimal Working Example

This repository contains standalone Minimal Working Examples (MWEs) of the **WISE (Weinberg-regularized Iterative Series Expansion)** algorithm for quantum scattering, as described in [the WISE manuscript](https://arxiv.org/abs/2601.01159). 
This repository implements the algorithm for a rigid-rotor + atom collision system (CO + He).

**Note:** These are pedagogical, stripped-down implementations designed specifically to reproduce the S-matrix column calculations and algorithmic scaling discussed in the manuscript. The complete, generalized, multi-system orchestration framework used for production calculations will be published in a separate software release.

## Implementations
This repository provides two mathematical formulations of the WISE solver:

1. **Arnoldi Subspace Projection (`run_wise_arnoldi`):** Computes the divergent Weinberg eigenvalues using an iterative Arnoldi solver (ARPACK) and projects them out via a biorthogonal Schmidt process [Eqs. (16)-(21) in the manuscript].
2. **Matrix-Free Contour Integration (`run_wise_contour`):** Evaluates the regularized Born series and divergent wavefunction correction via complex contour integration [Eqs. (26)-(28) in the manuscript].

## Repository structure
* `src/wise_scattering/run_wise_arnoldi.py`: Main execution script for the Arnoldi-based projection.
* `src/wise_scattering/run_wise_contour.py`: Main execution script for the contour integration method.
* `src/wise_scattering/wise_core.py`: The JIT-compiled matrix-vector operations (**K**, **K†**, and projection/contour operators).
* `src/wise_scattering/propagator.py`: A fast 1D Renormalized Numerov propagator for reference Green's function generation.
* `src/wise_scattering/asymptotics.py`: Boundary condition matching and Green's function normalization.
* `src/wise_scattering/physics_utilities.py`: Channel generation and angular momentum coupling (Wigner 3-j and 6-j symbols).
* `CO-He-coupling-terms.dat`: Tabulated Legendre expansion coefficients for the CO-He interaction potential.

## Installation

This package requires Python 3.10+ and relies strictly on core scientific libraries (`numpy`, `scipy`, `numba`, and `pywigxjpf`). 

It is highly recommended to install the package in an isolated virtual environment to ensure exact dependency matching for the Numba JIT compiler.

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 2. Install the package and its exact dependencies
pip install -e .
```

## Usage

Once installed, the package registers two console scripts that automatically execute the CO+He scattering benchmarks. Ensure your terminal is in the root directory (where `CO-He-coupling-terms.dat` is located).

1. **Arnoldi projection**
Run the Arnoldi-based solver:


```bash
run_wise_arnoldi
```

## Expected output


```plaintext
Starting WISE scattering MWE for CO+He...
Setting up CO+He scattering (E=8.84506 cm-1, J=0, j_max=6)...
Total channels (open + closed): 7
Target incoming channel (j=0, l=0) mapped to index: 0
Computing reference Green's functions...
Searching for 20 largest eigenvalues using ARPACK...
Eigenspace computed in 2.62 seconds.
  Eigenvalue 0: |eta| = 2.864501, eta = -2.819178e+00-5.075458e-01j
  ...
  Eigenvalue 19: |eta| = 0.811541, eta = -8.115318e-01-3.847148e-03j
Found 17 divergent Weinberg eigenvalues with |eta| >= 0.95.
Launching regularized iterative series with physical S-matrix convergence...
S-matrix converged within 1.00e-06 in 89 iterations.

--- Schmidt Process Completed ---
Total solver time: 0.86 s

Final S-matrix probabilities (S_ij):
  S(0,0) =  0.020858 + i 0.209322  |  P = 0.044251
  S(0,1) = -0.569089 + i 0.794947  |  P = 0.955803
```

2. **Contour integration**
Run the contour integration pipeline:

```bash
run_wise_contour
```

## Expected output

```plaintext
Starting WISE Contour MWE...
Setting up CO+He scattering (E=8.84506 cm-1, J=0, j_max=6)...
Total channels (open + closed): 7
Target incoming channel (j=0, l=0) mapped to index: 0
Computing reference Green's functions...

--- Stage 0: Contour Geometry ---
Found eta_max = 2.8645, N_q_outer = 30
Stage 1 Bounds: R_in = 0.950, R_out = 3.065 (N_q = 50)
Stage 2 Bounds: R_in = 1.000, R_out = 3.065 (N_q = 50, x_c_in = 0.1)

--- Stage 1: Regularized Born Series ---
Iter   1 | Delta u_R = 2.56e-02 | BiCGSTAB iters this step: 1109
...
Born Series converged in 40 iterations.

--- Stage 2: Contour Wavefunction Correction ---
Correction computed in 1.41 s | BiCGSTAB iters: 1047

--- Stage 3: S-Matrix Extraction ---

--- Process Completed ---
Total solver time: 64.37 s
Total BiCGSTAB linear solver iterations: 44781

Final S-matrix probabilities (S_ij):
  S(0,0) =  0.023277 + i 0.222407  |  P = 0.050007
  S(0,1) = -0.560634 + i 0.796150  |  P = 0.948166
```

## Configuration & Tunability

**Note on Default Parameters:** The default parameters provided in the `run_wise_arnoldi` script correspond exactly to those used to generate **Figure 4** in the manuscript. Specifically, the total collision energy of 8.84506 cm⁻¹ corresponds to exactly 5 cm⁻¹ of *kinetic* energy with respect to the *j*=1 rotational threshold. The radial grid setup (*r* = 3.0-20.0 Bohr, step = 0.01 Bohr) also identically matches the production calculations. The default parameters in the `run_wise_contour` script allow obtaining results for the same test calculations with the contour integration method.

Both scripts are designed designed to be easily modified. You can open either file and adjust the parameters under the **Target Collision & Algorithm Setup** section to explore different physical regimes and solver behaviors:

* **Collision Physics:**
  * `E_col_cm1`: The total collision energy.
  * `J_tot` and `parity`: The total angular momentum and parity of the collision complex.
  * `incoming_j` and `incoming_l`: The specific scattering channels to compute the single-column S-matrix from.
* **Basis Set:**
  * `j_max`: The maximum rotational state of the CO molecule.
* **Arnoldi-Specific Settings:**
  * `conv_radius`: Controls the eigenvalue cutoff for the Schmidt projection (default is `0.95`). Values < 1.0 ensure unconditional convergence.
  * `n_eigs`: The number of eigenvalues ARPACK searches for. If the solver diverges, increase this number to capture more of the divergent subspace.
* **Contour-Specific Settings**
  * `R_in_1`, `R_in_2`, `x_c_2`: Control the geometry and position of the inner integration rings for Born series [referred to as Stage 1 in the script; see the discussion below Eq. (27) in the manuscript] and the divergent subspace correction [referred to as Stage 2 in the script; see Eq. (28) in the manuscript].
  * `N_q_1`, `N_q_2`, `N_q_outer`: Determine the density of the trapezoidal quadrature grids.
  * `solver_tol`: Sets the convergence threshold for the internal BiCGSTAB iterative linear solver.

## Funding

This work was supported by the National Science Centre in Poland through Project No. 2024/53/N/ST2/02090 and by the NSF CAREER award No. PHY-2045681.
