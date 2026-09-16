"""Single-particle band structure of the 1D optical lattice.

Units: ``m = hbar = d = 1``; the recoil energy is ``V_REC = pi^2 / 2``.

The lattice potential is ``V_sigma(x) = v_sigma sin^2(pi x)``.  In the plane-wave
basis ``|k + 2 pi n>`` (``n = -nmax .. nmax``) it gives the symmetric tridiagonal
matrix used by ``EigSystH0sigma`` / ``Ek0sigma`` / ``EigSysH0sigmaGround``::

    diagonal      (k + 2 pi n)^2 / 2 + v/2
    off-diagonal  -v/4

``EigSystHc`` is the same construction for the *pair* (centre-of-mass) problem:
mass 2 (hence ``/4`` on the kinetic term) in the potential ``V_up + V_down``.

Every Hamiltonian here is tridiagonal, so ``scipy.linalg.eigh_tridiagonal`` is
used instead of a dense solver - the single biggest speed-up over a literal
transcription, and it changes no results.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal

__all__ = [
    "V_REC",
    "h0_sigma_tridiag",
    "hc_tridiag",
    "eig_h0_sigma",
    "e_k0_sigma",
    "eig_h0_sigma_ground",
    "eig_hc",
    "hopping_t",
]

V_REC = np.pi ** 2 / 2.0


def h0_sigma_tridiag(v, nmax, k):
    """Diagonal and off-diagonal of the single-atom Hamiltonian ``H0sigma``."""
    n = np.arange(-nmax, nmax + 1, dtype=float)
    d = (k + 2.0 * np.pi * n) ** 2 / 2.0 + v / 2.0
    e = np.full(2 * nmax, -v / 4.0)
    return d, e


def hc_tridiag(vup, vdown, nmax, k=0.0):
    """Diagonal and off-diagonal of the pair centre-of-mass Hamiltonian ``Hc``."""
    n = np.arange(-nmax, nmax + 1, dtype=float)
    d = (k + 2.0 * np.pi * n) ** 2 / 4.0 + (vup + vdown) / 2.0
    e = np.full(2 * nmax, -(vup + vdown) / 4.0)
    return d, e


def eig_h0_sigma(v, nmax, k):
    """``EigSystH0sigma``: all eigenvalues (ascending) and eigenvectors.

    Returns ``(evals, evecs)`` with ``evecs[nu, n]`` - eigenvectors are **rows**,
    as in Mathematica's ``Eigensystem``.
    """
    d, e = h0_sigma_tridiag(v, nmax, k)
    w, V = eigh_tridiagonal(d, e)
    return w, V.T


def e_k0_sigma(v, nmax, k):
    """``Ek0sigma``: lowest-band energy at quasi-momentum ``k``."""
    d, e = h0_sigma_tridiag(v, nmax, k)
    return float(eigh_tridiagonal(d, e, select="i", select_range=(0, 0),
                                  eigvals_only=True)[0])


def eig_h0_sigma_ground(v, nmax, k):
    """``EigSysH0sigmaGround``: lowest eigenvalue and its eigenvector."""
    d, e = h0_sigma_tridiag(v, nmax, k)
    w, V = eigh_tridiagonal(d, e, select="i", select_range=(0, 0))
    return float(w[0]), V[:, 0]


def eig_hc(vup, vdown, nmax, k=0.0):
    """``EigSystHc``: pair centre-of-mass eigensystem (eigenvectors as rows)."""
    d, e = hc_tridiag(vup, vdown, nmax, k)
    w, V = eigh_tridiagonal(d, e)
    return w, V.T


def hopping_t(v, nmax, nk=200):
    """``t = -(1/2 pi) Integral[E0(k) Cos[k], {k, -pi, pi}]``.

    The first Fourier coefficient of the lowest band, i.e. the Wannier-basis
    nearest-neighbour hopping, exactly as computed in both ``.wl`` packages.
    """
    from .quadrature import gaussian_quadrature_weights
    k, dk = gaussian_quadrature_weights(nk, -np.pi, np.pi)
    e = np.array([e_k0_sigma(v, nmax, ki) for ki in k])
    return float(-np.sum(dk * e * np.cos(k)) / (2.0 * np.pi))
