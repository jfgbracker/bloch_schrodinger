"""Quadrature grids, root finding and numerical differentiation.

Python ports of the Mathematica helpers used by ``1DHubbardParameters.wl`` and
``2DHubbardParameters.wl``:

===========================================  ================================
Mathematica                                   Python
===========================================  ================================
``GaussianQuadratureWeights[n, a, b]``        :func:`gaussian_quadrature_weights`
``setupKIntReg[nk]``                          :func:`setup_k_int_reg`
``setupKIntPV[nk, poles, k0, k1]``            :func:`setup_k_int_pv`
``FindAllCrossings[f, {k, a, b}]``            :func:`find_all_crossings`
``ND[f, k, k0]``                              :func:`nd`
===========================================  ================================
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "gaussian_quadrature_weights",
    "setup_k_int_reg",
    "setup_k_int_pv",
    "find_all_crossings",
    "nd",
    "round_to_multiple",
]


def gaussian_quadrature_weights(n: int, a: float = 0.0, b: float = 1.0):
    """Gauss-Legendre nodes/weights on ``[a, b]``, nodes in ascending order.

    Mirrors ``NumericalDifferentialEquationAnalysis`GaussianQuadratureWeights``.
    """
    x, w = np.polynomial.legendre.leggauss(int(n))
    half = 0.5 * (b - a)
    return half * x + 0.5 * (a + b), half * w


def setup_k_int_reg(nk: int):
    """``setupKIntReg``: plain Gauss-Legendre grid on the Brillouin zone."""
    return gaussian_quadrature_weights(nk, -np.pi, np.pi)


def round_to_multiple(x: float, unit: float) -> float:
    """``Round[x, unit]`` - round to the nearest multiple of ``unit``.

    Uses banker's rounding (round-half-to-even), matching Mathematica.
    """
    q = x / unit
    # np.round / Python round both implement round-half-to-even.
    return float(np.round(q)) * unit


def setup_k_int_pv(nk: int, poles, k0: float, k1: float):
    """``setupKIntPV``: principal-value grid on ``[k0, k1]``.

    The interval is split into sub-domains whose end points are placed
    symmetrically about every pole (``pole - s``, ``pole``, ``pole + s``), so
    that the Gauss nodes on the two sub-intervals flanking a pole are exact
    mirror images.  The ``1/(k - pole)`` parts of the integrand then cancel
    identically between the two sides, which is what realises the Cauchy
    principal value.

    Parameters
    ----------
    nk : int
        Number of Gauss-Legendre nodes *per sub-domain*.
    poles : array_like
        Pole positions, sorted ascending, strictly inside ``(k0, k1)``.
    """
    poles = np.atleast_1d(np.asarray(poles, dtype=float))
    if poles.size == 0:
        # Not reachable from the Mathematica code (it would raise a Part error);
        # fall back to a plain grid so callers degrade gracefully.
        return gaussian_quadrature_weights(nk, k0, k1)

    min_pole_to_endpoint = np.min(np.abs([k0 - poles[0], k1 - poles[-1]]))
    if poles.size == 1:
        spacing = min_pole_to_endpoint
    else:
        min_pole_spacing = np.min(np.abs(np.diff(poles)))
        if min_pole_spacing > min_pole_to_endpoint:
            spacing = min_pole_spacing / np.ceil(min_pole_spacing / min_pole_to_endpoint)
        else:
            spacing = min_pole_spacing

    edges = np.concatenate(([k0], poles - spacing, poles + spacing, poles, [k1]))
    edges = np.round(edges / 1e-9) * 1e-9          # Round[..., 10^-9]
    edges = np.unique(edges)                        # DeleteDuplicates + Sort

    sub_domains = np.stack([edges[:-1], edges[1:]], axis=1)

    t, dt = gaussian_quadrature_weights(nk, 0.0, 1.0)
    widths = sub_domains[:, 1] - sub_domains[:, 0]
    k = (np.outer(widths, t) + sub_domains[:, 0][:, None]).ravel()
    dk = np.outer(widths, dt).ravel()
    return k, dk


def find_all_crossings(f, a: float, b: float, n_sample: int = 1025,
                       xtol: float = 1e-14):
    """``FindAllCrossings``: every sign change of ``f`` on ``[a, b]``.

    ``f`` is sampled on a uniform grid, brackets are formed around sign
    changes, and each bracket is refined with Brent's method.  Like the
    Mathematica original this detects *sign changes* only - tangential
    (double) roots are invisible to both implementations.
    """
    xs = np.linspace(a, b, int(n_sample))
    ys = np.array([f(x) for x in xs], dtype=float)

    roots = list(xs[ys == 0.0])
    sign = np.sign(ys)
    idx = np.nonzero(sign[:-1] * sign[1:] < 0)[0]
    for i in idx:
        roots.append(brentq(f, xs[i], xs[i + 1], xtol=xtol, rtol=8.9e-16))

    if not roots:
        return np.array([])
    roots = np.unique(np.asarray(roots, dtype=float))
    return roots[(roots >= a) & (roots <= b)]


def nd(f, x0: float, h0: float = 0.1, terms: int = 7) -> float:
    """``ND[f, x, x0]``: Richardson-extrapolated first derivative.

    Central differences on a halving sequence of step sizes are combined with
    Neville/Richardson extrapolation, giving ~1e-11 accuracy on the smooth
    band-structure functions this is applied to.
    """
    h = float(h0)
    table = []
    for i in range(terms):
        table.append((f(x0 + h) - f(x0 - h)) / (2.0 * h))
        for j in range(i - 1, -1, -1):
            fac = 4.0 ** (i - j)
            table[j] = (fac * table[j + 1] - table[j]) / (fac - 1.0)
        h *= 0.5
    return table[0]
