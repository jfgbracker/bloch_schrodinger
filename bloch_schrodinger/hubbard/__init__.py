"""Exact Hubbard parameters for atoms in quasi-low-dimensional optical lattices.

A validated port of the Mathematica package ``cold-atom-hubbard-parameters``
(H. S. Adlong et al.), wrapped in this package's unit conventions.

``U`` is fixed by matching the Hubbard two-body scattering amplitude to the
exact lattice amplitude as the relative quasi-momentum goes to zero, so it
includes all bands and all transverse harmonic modes -- unlike the single-band
Wannier overlap ``g int |w|^4``, which is its leading Born term.

Quick start
-----------
>>> from bloch_schrodinger.hubbard import hubbard_u_quasi1d
>>> res = hubbard_u_quasi1d(depth=12, omega_perp=40.0,
...                         lattice_constant=np.pi, alpha=1.0)
>>> res(-10.0)          # U at 1/a_1D = -10, in the caller's energy units

Geometry restriction
--------------------
The published method factorises the pair problem into independent lattice
directions: it covers a quasi-1D lattice and a **separable square** quasi-2D
lattice, not a honeycomb or triangular one.  See :mod:`.api` for what to do
instead in the non-separable case.

Layout
------
``quadrature``, ``special``, ``bands``, ``hint``, ``tmatrix``, ``quasi1d``,
``quasi2d``, ``wannier1d``
    The solver, in its own ``m = hbar = d = 1`` units -- a direct port, kept
    close to the Mathematica source so the two can be compared line by line.
``units``
    The conversion layer to this package's ``H = alpha k^2 + V`` conventions.
``api``
    The public, unit-aware front end re-exported here.
"""

from .api import (V_REC, ExactHubbardResult, a1d_inv_from_a3d, hopping,
                  hubbard_u_quasi1d, hubbard_u_quasi2d,
                  hubbard_u_wannier_1d, hubbard_u_wannier_2d_square,
                  log_a2d_inv_from_a3d, matched_square_depth,
                  matched_square_depth_from_quartic,
                  site_trap_frequency, wannier_profile_1d,
                  wannier_quartic_1d, wannier_quartic_2d_square)
from .units import UnitConverter

__all__ = [
    "V_REC",
    "ExactHubbardResult",
    "UnitConverter",
    "hubbard_u_quasi1d",
    "hubbard_u_quasi2d",
    "hubbard_u_wannier_1d",
    "hubbard_u_wannier_2d_square",
    "a1d_inv_from_a3d",
    "log_a2d_inv_from_a3d",
    "wannier_profile_1d",
    "hopping",
    "site_trap_frequency",
    "matched_square_depth",
    "matched_square_depth_from_quartic",
    "wannier_quartic_1d",
    "wannier_quartic_2d_square",
]
