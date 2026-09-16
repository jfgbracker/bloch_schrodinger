"""Wannier functions of the lowest band - the *standard* Hubbard-U benchmark.

This module is not part of the Mathematica package.  It supplies the textbook
estimate that the exact calculation is meant to improve on:

    U_standard = g * Integral |w(x)|^4 dx        (1D)

with ``w`` the maximally-localised Wannier function of the lowest band.  It
therefore serves two purposes here: an independent check that the exact U has
the right weak-coupling limit, and the reference curve for the benchmark plots.

Conventions
-----------
Bloch states ``phi_k(x) = e^{i k x} Sum_n c_n(k) e^{i 2 pi n x}`` with
``Sum_n |c_n|^2 = 1``, and

    w(x) = (1/2 pi) Integral[phi_k(x), {k, -pi, pi}] ,

which satisfies ``Integral |w|^2 dx = 1``.  The gauge is fixed by demanding
``phi_k(0) > 0``, the standard choice that makes ``w`` real, even and
maximally localised for a symmetric 1D potential.
"""

from __future__ import annotations

import numpy as np

from .bands import eig_h0_sigma_ground

__all__ = ["wannier_function", "wannier_u_integral"]


def wannier_function(v, x, nmax=40, nk=512):
    """Lowest-band Wannier function ``w(x)`` centred on the site at ``x = 0``.

    Parameters
    ----------
    v : float
        Lattice depth (units of ``1/(m d^2)``).
    x : array_like
        Positions in units of the lattice spacing.
    nmax, nk : int
        Plane-wave cutoff and number of Brillouin-zone points.  A uniform
        BZ grid is used: it is the exact Wannier construction on a periodic
        ``nk``-site lattice, and converges exponentially in ``nk``.
    """
    x = np.asarray(x, dtype=float)
    n = np.arange(-nmax, nmax + 1)
    # uniform BZ grid, endpoint excluded (periodic)
    k = -np.pi + 2.0 * np.pi * np.arange(nk) / nk

    coeffs = np.empty((nk, 2 * nmax + 1))
    for ik, ki in enumerate(k):
        _, c = eig_h0_sigma_ground(v, nmax, ki)
        if c.sum() < 0:          # gauge: phi_k(0) > 0
            c = -c
        coeffs[ik] = c

    lattice = np.exp(2j * np.pi * np.outer(x, n))        # (nx, 2nmax+1)
    u = lattice @ coeffs.T                                # (nx, nk)
    w = (u * np.exp(1j * np.outer(x, k))).sum(axis=1) / nk
    return np.real(w)


def wannier_u_integral(v, nmax=40, nk=512, n_sites=10, n_per_site=400):
    """``Integral |w(x)|^4 dx`` (and the norm check ``Integral |w|^2 dx``).

    Returns
    -------
    quartic : float
        ``Integral |w|^4 dx`` - the geometric factor in ``U = g * quartic``.
    norm : float
        ``Integral |w|^2 dx``; should equal 1 to the accuracy of the grid.
    """
    x = np.linspace(-n_sites, n_sites, 2 * n_sites * n_per_site + 1)
    w = wannier_function(v, x, nmax=nmax, nk=nk)
    dx = x[1] - x[0]
    quartic = np.trapezoid(w ** 4, dx=dx)
    norm = np.trapezoid(w ** 2, dx=dx)
    return float(quartic), float(norm)
