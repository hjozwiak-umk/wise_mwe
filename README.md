# WISE Scattering: Minimal Working Example

This repository contains a standalone Minimal Working Example (MWE) of the **WISE (Weinberg-regularized Iterative Series Expansion)** algorithm for quantum scattering, as described in [the WISE manuscript](https://arxiv.org/abs/2601.01159).
This repository implements the algorithm for a rigid-rotor + atom collision system (CO + He).

**Note:** This is a pedagogical, stripped-down implementation designed specifically to reproduce the S-matrix column calculations and algorithmic scaling discussed in the manuscript. The complete, generalized, multi-system orchestration framework used for production calculations will be published in a separate software release.

## Repository Structure
* `src/wise_scattering/run_wise_scattering.py`: The main execution script. Orchestrates the physical setup, the matrix-free Arnoldi eigensolver, and the regularized Born series.
* `src/wise_scattering/wise_core.py`: The JIT-compiled matrix-vector operations (**K** and **K†** and projection operators).
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

Once installed, the package registers a console script that automatically executes the CO+He scattering benchmark.

Ensure your terminal is in the root directory (where `CO-He-coupling-terms.dat` is located) and run:

```bash
run_wise
```

## Expected Output

The script will output the channel setup, compute the selected eigenvalues of the scattering kernel, and execute the regularized Born series. A successful run will look like this:

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

## Configuration & Tunability

**Note on Default Parameters:** The default parameters provided in this script correspond exactly to those used to generate **Figure 4** in the manuscript. Specifically, the total collision energy of 8.84506 cm⁻¹ corresponds to exactly 5 cm⁻¹ of *kinetic* energy with respect to the *j*=1 rotational threshold. The radial grid setup (*r* = 3.0-20.0 Bohr, step = 0.01 Bohr) also identically matches the production calculations.

The `run_wise_scattering.py` script is designed to be easily modified. You can open the file and adjust the parameters under the **Target Collision & Algorithm Setup** section to explore different physical regimes and solver behaviors:

* **Collision Physics:**
  * `E_col_cm1`: The total collision energy.
  * `J_tot` and `parity`: The total angular momentum and parity of the collision complex.
  * `incoming_j` and `incoming_l`: The specific scattering channels to compute the single-column S-matrix from.
* **Basis Set:**
  * `j_max`: The maximum rotational state of the CO molecule.
* **Solver Settings:**
  * `conv_radius`: Controls the eigenvalue cutoff for the Schmidt projection (default is `0.95`). Values < 1.0 ensure unconditional convergence.
  * `n_eigs`: The number of eigenvalues ARPACK searches for. If the solver diverges, increase this number to capture more of the divergent subspace.

## Funding

This work was supported by the National Science Centre in Poland through Project No. 2024/53/N/ST2/02090 and by the NSF CAREER award No. PHY-2045681.