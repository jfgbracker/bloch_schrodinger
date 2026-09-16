"""Closed forms for the transverse-confinement mode sums.

Both packages reduce the sum over harmonic-oscillator states of the tight
transverse direction(s) to a closed form:

* quasi-1D (2D transverse harmonic trap, ``omegaperp``)::

      quasi1DSum[a, b, smin, smax] == Sum[1/(a + 2 b s), {s, smin, smax}]

* quasi-2D (1D transverse harmonic trap, ``omegaz``)::

      quasi2DSum0toInf[var, w] == Sum[(2s)!/(2^s s!)^2 / (var + 2 w s), {s, 0, Infinity}]
      quasi2DSum1toInf[var, w] == same sum starting at s == 1

``quasi2DSum0toInfApprox`` is the large-argument asymptotic form the Mathematica
package switches to in order to dodge ``Gamma`` overflow; it is reproduced here
so the port can match Mathematica element for element (``mode="faithful"``).
The ``gammaln``-based evaluation used here does not overflow, so ``mode="exact"``
is also offered and is strictly more accurate.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma, gammaln, psi

__all__ = [
    "quasi1d_sum",
    "quasi2d_sum_0_inf",
    "quasi2d_sum_1_inf",
    "quasi2d_sum_0_inf_approx",
    "quasi2d_sum_1_inf_approx",
]

_HALF_LOG_PI = 0.5 * np.log(np.pi)


def quasi1d_sum(a, b, smin, smax):
    """``quasi1DSum[a, b, smin, smax]`` = ``Sum[1/(a + 2 b s), {s, smin, smax}]``."""
    x = np.asarray(a, dtype=float) / (2.0 * b)
    return (psi(1.0 + x + smax) - psi(x + smin)) / (2.0 * b)


def _sqrtpi_gamma_ratio(x):
    """``Sqrt[Pi] Gamma[1 + x] / Gamma[x + 1/2]``, overflow-free for large x."""
    x = np.asarray(x, dtype=float)
    out = np.empty(x.shape, dtype=float)
    safe = x > -0.5
    if safe.all():
        return np.exp(_HALF_LOG_PI + gammaln(1.0 + x) - gammaln(x + 0.5))
    xs = x[safe]
    out[safe] = np.exp(_HALF_LOG_PI + gammaln(1.0 + xs) - gammaln(xs + 0.5))
    xu = x[~safe]
    out[~safe] = np.sqrt(np.pi) * gamma(1.0 + xu) / gamma(xu + 0.5)
    return out


def _sqrtpi_gamma_ratio_m1(x):
    """``Sqrt[Pi] Gamma[1 + x] / Gamma[x + 1/2] - 1`` without cancellation at x -> 0."""
    x = np.asarray(x, dtype=float)
    out = np.empty(x.shape, dtype=float)
    safe = x > -0.5
    if safe.all():
        return np.expm1(_HALF_LOG_PI + gammaln(1.0 + x) - gammaln(x + 0.5))
    xs = x[safe]
    out[safe] = np.expm1(_HALF_LOG_PI + gammaln(1.0 + xs) - gammaln(xs + 0.5))
    out[~safe] = _sqrtpi_gamma_ratio(xu := x[~safe]) - 1.0
    return out


def quasi2d_sum_0_inf(var, omega):
    """``quasi2DSum0toInf[var, omega]``."""
    var = np.asarray(var, dtype=float)
    return _sqrtpi_gamma_ratio(var / (2.0 * omega)) / var


def quasi2d_sum_1_inf(var, omega):
    """``quasi2DSum1toInf[var, omega]``."""
    var = np.asarray(var, dtype=float)
    return _sqrtpi_gamma_ratio_m1(var / (2.0 * omega)) / var


def quasi2d_sum_0_inf_approx(var, omega):
    """``quasi2DSum0toInfApprox[var, omega]``."""
    var = np.asarray(var, dtype=float)
    x = var / (2.0 * omega)
    return np.sqrt(np.pi) / var * (x * x + 0.5 * x + 0.125) ** 0.25


def quasi2d_sum_1_inf_approx(var, omega):
    """``quasi2DSum1toInfApprox[var, omega]``."""
    var = np.asarray(var, dtype=float)
    return quasi2d_sum_0_inf_approx(var, omega) - 1.0 / var
