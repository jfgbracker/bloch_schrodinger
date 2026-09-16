# Bloch-Schrödinger solver 

This package provides the tools to solve the Bloch-Schrödinger equation in 1D, 2D and 3D. It contains potential construction tools, a finite-difference solver (`FDSolver`, including support for coupled multi-field equations such as TE/TM polarization splitting), a plane-wave solver (`PWSolver`) for smooth potentials, a `Wannier` class for computing maximally localized Wannier functions, and a `hubbard` subpackage that computes the **exact** Hubbard on-site interaction `U` for quasi-low-dimensional optical lattices.

## Exact Hubbard parameters

`bloch_schrodinger.hubbard` is a validated port of the Mathematica package `cold-atom-hubbard-parameters` (H. S. Adlong *et al.*). Rather than estimating `U` as the single-band Wannier overlap `g ∫|w|⁴`, it *defines* `U` by matching the Hubbard two-body scattering amplitude to the exact lattice amplitude as the relative quasi-momentum goes to zero — so all bands and all transverse harmonic modes are resummed. The Wannier overlap is its leading Born term; the two agree only at weak coupling.

```python
from bloch_schrodinger.hubbard import hubbard_u_quasi1d

res = hubbard_u_quasi1d(depth=12,              # in units of the lattice recoil
                        omega_perp=8.1,        # hbar*omega, caller energy units
                        lattice_constant=np.pi,
                        alpha=1.0)             # H = alpha k^2 + V
res(-10.0)   # U at 1/a_1D = -10, in the caller's energy units
```

The published method factorises the pair problem per lattice direction, so it covers a quasi-1D lattice and a **separable square** quasi-2D lattice — not a honeycomb or triangular one. For non-separable geometries use `matched_square_depth_from_quartic` to build a reference lattice with the same on-site Wannier overlap and treat the result as an estimate; see the module docstring.

Units are handled by `UnitConverter`, which maps between this package's `H = alpha k^2 + V` and the solver's `m = hbar = d = 1`. The conversion is pinned by `tests/test_hubbard.py`, whose load-bearing test states one physical problem in four different unit conventions and requires the same `U/E_rec` from all of them.


## Installation

First download the repository and extract it where you want. Then, run in your python environment 

`bash`
pip install path\\to\\package\\bloch_schrodinger

or 

`bash`
pip install -e path\\to\\package\\bloch_schrodinger

if you want to be able to modify it in place.

## Getting started

Once you have installed the package, open in a jupyter viewer the [Getting Started](docs/GettingStarted.ipynb) notebook. For the full list of tutorials and a suggested reading order, see [docs/README.md](docs/README.md).



