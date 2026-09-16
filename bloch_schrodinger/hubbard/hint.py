"""Interaction (contact) matrix elements in the pair eigenbasis.

Port of ``buildHintTensor`` and ``buildHintTensornu1nu2eq00``.

Physics
-------
A contact interaction only connects states at coincident positions, so its
matrix element between the pair centre-of-mass state ``eta_nu`` (at zero CM
quasi-momentum) and two atoms in bands ``nu1`` (spin up, quasi-momentum ``-k``)
and ``nu2`` (spin down, quasi-momentum ``+k``) is the *real-space* overlap

    Hint[nu, nu1, nu2] = Integral dx eta_nu(x) phi_up,nu1(x) phi_down,nu2(x).

The relative quasi-momenta cancel, so in the plane-wave basis this becomes a
convolution over the reciprocal-lattice indices,

    Hint[nu, nu1, nu2] = Sum_N eta[nu, N] Sum_{n_up + n_down = N}
                             phi_up[nu1, n_up] phi_down[nu2, n_down],

with the centre-of-mass index ``N`` truncated at ``ncm_max`` and the relative
index truncated at ``nrel_max``.  Those two truncations are precisely the
``Reverse``-and-slice construction in the Mathematica source; the slice is
asymmetric for odd ``N`` because the relative index is then half-integer.

Sign conventions
----------------
Eigenvector signs are arbitrary and differ between linear-algebra backends.
They cancel identically here: ``Hint`` only ever enters the T matrix through
``Hint x Hint`` with *every* index either contracted between the two factors
(``nu1``, ``nu2``) or carried by both (``nu``), so the final T matrix - and
hence U - is independent of the convention.
"""

from __future__ import annotations

import numpy as np

from .bands import eig_h0_sigma, eig_h0_sigma_ground, eig_hc

__all__ = [
    "low_energy_grid_size",
    "pair_window",
    "build_hint_tensor",
    "build_hint_tensor_nu00",
]


def low_energy_grid_size(ncm_max: int, nrel_max: int) -> int:
    """``lowEnergyGridSize = Ceiling[nCMmax/2] + nRelmax + 1``."""
    return -(-int(ncm_max) // 2) + int(nrel_max) + 1


def pair_window(N: int, shift: int, nrel_max: int):
    """0-based ``[start, stop)`` slice of the reciprocal-lattice index.

    Mathematica (1-based) uses
    ``shift + N/2 + (1/2 if N odd else 0) - nrel  ;;  shift + N/2 - (1/2 if N odd else 0) + nrel``
    which contains ``2 nrel + 1`` indices for even ``N`` and ``2 nrel`` for odd ``N``.
    """
    if N % 2:                       # odd N (Mod[N, 2] == 1 for negative N too)
        start = shift + (N + 1) // 2 - nrel_max
        stop = shift + (N - 1) // 2 + nrel_max
    else:
        start = shift + N // 2 - nrel_max
        stop = shift + N // 2 + nrel_max
    return start - 1, stop          # -> python slice


def build_hint_tensor(vup, vdown, kgrid, ncm_max, nrel_max, eta_vecs=None):
    """``buildHintTensor``.

    Returns
    -------
    hint : (nk, 2 ncm_max + 1, nst, nst) ndarray
        ``Hint[k, nu, nu1, nu2]``, ``nst = 2 lowEnergyGridSize + 1``.
    eig_up, eig_down : (nk, nst) ndarray
        Single-atom band energies at ``-k`` and ``+k``, stored for reuse in the
        Green's-function kernels (as the Mathematica routine also does).
    """
    kgrid = np.atleast_1d(np.asarray(kgrid, dtype=float))
    L = low_energy_grid_size(ncm_max, nrel_max)
    shift = L + 1
    n_cm = 2 * ncm_max + 1
    n_st = 2 * L + 1

    if eta_vecs is None:
        _, eta_vecs = eig_hc(vup, vdown, ncm_max, 0.0)

    windows = [pair_window(N, shift, nrel_max) for N in range(-ncm_max, ncm_max + 1)]

    hint = np.empty((kgrid.size, n_cm, n_st, n_st))
    eig_up = np.empty((kgrid.size, n_st))
    eig_down = np.empty((kgrid.size, n_st))
    phi = np.empty((n_cm, n_st, n_st))

    for ik, k in enumerate(kgrid):
        eu, Vu = eig_h0_sigma(vup, L, -k)
        ed, Vd = eig_h0_sigma(vdown, L, k)
        eig_up[ik] = eu
        eig_down[ik] = ed
        for iN, (a, b) in enumerate(windows):
            phi[iN] = Vu[:, a:b] @ Vd[:, a:b][:, ::-1].T
        hint[ik] = np.tensordot(eta_vecs, phi, axes=([1], [0]))

    return hint, eig_up, eig_down


def build_hint_tensor_nu00(vup, vdown, ks, ncm_max, nrel_max, eta_vecs):
    """``buildHintTensornu1nu2eq00``: the ``nu1 = nu2 = 0`` (lowest-band) column.

    ``ks`` may be a scalar or an array; the return has shape
    ``(2 ncm_max + 1,)`` or ``(len(ks), 2 ncm_max + 1)`` accordingly.
    """
    scalar = np.isscalar(ks) or np.asarray(ks).ndim == 0
    ks = np.atleast_1d(np.asarray(ks, dtype=float))
    L = low_energy_grid_size(ncm_max, nrel_max)
    shift = L + 1
    windows = [pair_window(N, shift, nrel_max) for N in range(-ncm_max, ncm_max + 1)]

    phi = np.empty((ks.size, 2 * ncm_max + 1))
    for ik, k in enumerate(ks):
        _, vu = eig_h0_sigma_ground(vup, L, -k)
        _, vd = eig_h0_sigma_ground(vdown, L, k)
        for iN, (a, b) in enumerate(windows):
            phi[ik, iN] = vu[a:b] @ vd[a:b][::-1]

    out = phi @ eta_vecs.T                      # Sum_N eta[nu, N] phi[N]
    return out[0] if scalar else out
