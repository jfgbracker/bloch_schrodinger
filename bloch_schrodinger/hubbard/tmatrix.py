"""Fast evaluation of the on-shell T matrix as the coupling is scanned.

Both packages evaluate

    T(c) = v . Inverse[c I - Pi] . v

for many values of the single scalar ``c`` (set by the scattering length) while
``Pi`` and ``v`` stay fixed.  A literal implementation re-solves an ``n x n``
system per point.  Since only a multiple of the identity changes, one complex
Schur decomposition ``Pi = Q S Q^H`` (``S`` upper triangular, ``Q`` unitary)
turns every subsequent evaluation into two triangular solves:

    T(c) = (Q^T v)^T . Inverse[c I - S] . (Q^H v) .

This is the same linear algebra, just reordered - results agree with the direct
solve to round-off - and it makes scanning thousands of scattering lengths free.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import schur, solve_triangular

__all__ = ["TMatrixScan"]


class TMatrixScan:
    """``c -> v . Inverse[c I - Pi] . v`` evaluated over an array of ``c``."""

    def __init__(self, pi_matrix, vec):
        pi = np.asarray(pi_matrix)
        self.n = pi.shape[0]
        self.pi_matrix = pi
        self.vec = np.asarray(vec, dtype=float)
        s, q = schur(pi.astype(complex), output="complex")
        self._s = s
        self._p = q.conj().T @ self.vec          # Q^H v
        self._q = q.T @ self.vec                 # Q^T v

    def __call__(self, c):
        """``c`` may be a scalar or an array; ``inf`` gives ``T = 0``."""
        arr = np.asarray(c, dtype=float)
        out = np.empty(arr.shape, dtype=complex)
        eye = np.eye(self.n)
        for idx in np.ndindex(arr.shape):
            ci = arr[idx]
            if not np.isfinite(ci):
                out[idx] = 0.0
                continue
            m = ci * eye - self._s
            out[idx] = self._q @ solve_triangular(m, self._p)
        return out if out.shape else complex(out)

    def direct(self, c):
        """Reference implementation: a full dense solve (for verification)."""
        arr = np.asarray(c, dtype=float)
        out = np.empty(arr.shape, dtype=complex)
        eye = np.eye(self.n)
        for idx in np.ndindex(arr.shape):
            ci = arr[idx]
            if not np.isfinite(ci):
                out[idx] = 0.0
                continue
            out[idx] = self.vec @ np.linalg.solve(ci * eye - self.pi_matrix,
                                                  self.vec)
        return out if out.shape else complex(out)
