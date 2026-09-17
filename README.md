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


## This fork

This is a fork of [GuillotMartin/bloch_schrodinger](https://github.com/GuillotMartin/bloch_schrodinger)
carrying three things that are not (yet) upstream:

1. **`bloch_schrodinger.hubbard`** — the exact Hubbard `U`, described above.
2. **`Potential.find_minima()`, `Potential.smooth()`, `Potential.plot_3d()`**, and
   `Potential.plot(show_minima=…)` — helpers for locating and displaying the sites of a
   lattice. `find_minima()` returns one array per cartesian coordinate, then the potential
   value, then a `Dataset` of lattice coordinates, so the 2D call reads
   `x, y, v, coords = pot.find_minima()`.
3. **A Python 3.11 floor** instead of 3.12+ — see below.
4. **Lazy `plotly` / `scikit-image` imports.** Both are needed only by `plot_isosurface`,
   but importing them at module scope made `bloch_schrodinger.plotting` unimportable
   without them — a real cost on a compute node where neither is installed. They are now
   imported inside the functions that use them; `plot_isosurface` raises a plain
   `ImportError` if they are missing, and everything else in the module works.

Upstream is tracked as the `upstream` remote; `git pull --rebase upstream main` brings in
new work from there.

## Python version

`REQUIRES_PYTHON = '>=3.11'`, and every source file in the package parses under 3.11.

Upstream declares 3.12+, but nothing in the package needs it: the only 3.12-only construct
was a handful of [PEP 695](https://peps.python.org/pep-0695/) type-alias statements,

```python
type paramType = int | float | xr.DataArray     # SyntaxError on 3.11
```

which are used purely as annotations. Dropping the `type` keyword leaves a plain
assignment that behaves identically here — PEP 695 aliases are lazily evaluated and plain
assignments are eager, but no right-hand side forward-references a name defined later, so
eager evaluation is safe. There are no 3.12-only stdlib APIs anywhere in the package.

This matters because compute clusters usually lag several Python releases behind. Keeping
the floor at 3.11 means the same commit runs on the cluster and on a current desktop, with
no patch step in between.

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



