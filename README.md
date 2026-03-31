# WISE Scattering: Minimal Working Example

This repository contains a standalone Minimal Working Example (MWE) of the **WISE (Weinberg-regularized Iterative Series
Expansion)** algorithm for quantum scattering, as described in [the WISE manuscript](https://arxiv.org/abs/2601.01159).
This repository implements the algorithm for a rigid-rotor + atom collision system (CO + He). 

**Note:** This is a pedagogical, stripped-down implementation designed specifically to reproduce the S-matrix column calculations and algorithmic scaling discussed in the manuscript. The complete, generalized, multi-system orchestration framework used for production calculations will be published in a separate software release.

## Repository Structure
* `src/wise_scattering/run_wise_scattering.py`: The main execution script. Orchestrates the physical setup, the matrix-free Arnoldi eigensolver, and the regularized Born series.
* `src/wise_scattering/wise_core.py`: The JIT-compiled matrix-vector operations ($\mathbf{K}$ and $\mathbf{K}^{\dagger}$) and projection operators.
* `src/wise_scattering/propagator.py`: A fast 1D Renormalized Numerov propagator for reference Green's function generation.
* `src/wise_scattering/asymptotics.py`: Boundary condition matching and Green's function normalization.
* `src/wise_scattering/physics_utilities.py`: Channel generation and angular momentum coupling (Wigner 3-$j$ and 6-$j$ symbols).
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

The script will output the channel setup, the progress of the matrix-free Arnoldi eigensolver (identifying the divergent Weinberg eigenvalues), the convergence of the regularized iterative series, and the final extracted scattering information.
