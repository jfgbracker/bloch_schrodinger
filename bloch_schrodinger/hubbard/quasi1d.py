"""Hubbard U for a quasi-1D optical lattice.

Port of ``1DHubbardParameters.wl``.

Method
------
``U`` is fixed by demanding that the *Hubbard* two-body scattering amplitude
reproduces the *exact* lattice + confinement amplitude as the relative
quasi-momentum goes to zero.  With

    T_exact(1/a1D)^-1 = 1/g_1D - Pi ,        1/g_1D = -a1D/2
    T_Hubbard^-1      = 1/U     - Pi_Hubbard

the imaginary parts (fixed by the on-shell density of states) are matched by a
single factor ``effMassHubbard``, and ``U`` follows from the real parts:

    1/U = effMassHubbard * Re[T_exact^-1]        (Re[Pi_Hubbard] = 0 in 1D).

``Pi`` is the pair propagator in the basis of centre-of-mass lattice states.
It is split into

* ``s = 0``  - transverse ground state: principal value plus on-shell pole,
* ``s >= 1`` - transverse excited states (``quasi1DSum``), plus the counter
  term that renormalises the contact interaction,
* high-energy tails, where relative motion above the ``nrel_max`` cutoff is
  treated as free.
"""

from __future__ import annotations

import numpy as np

from .bands import V_REC, e_k0_sigma, eig_hc
from .hint import build_hint_tensor, build_hint_tensor_nu00, low_energy_grid_size
from .quadrature import (find_all_crossings, gaussian_quadrature_weights, nd,
                         setup_k_int_pv, setup_k_int_reg)
from .special import quasi1d_sum
from .tmatrix import TMatrixScan

__all__ = [
    "V_REC",
    "hubbard_inv_free_green",
    "hubbard_t_matrix_integral",
    "pi_pole_component",
    "setup_t_matrix_on_shell_quasi1d",
    "setup_hubbard_u_quasi1d_all_params",
    "setup_hubbard_u_quasi1d",
    "a1d_inv_from_a3d",
    "Quasi1DResult",
]

ZETA_HALF = -1.4603545088095868  # Zeta[1/2]


# --------------------------------------------------------------------------
# Hubbard-model side
# --------------------------------------------------------------------------
def hubbard_inv_free_green(tup, tdown, p, k):
    """``HubbardInvFreeGreenFunc`` = ``E0 - E(k)`` for the 1D Hubbard model."""
    return 2 * tup * (np.cos(-k) - np.cos(-p)) + 2 * tdown * (np.cos(k) - np.cos(p))


def hubbard_t_matrix_integral(tup, tdown, p, n_sample=1025):
    """``HubbardTMatrixIntegral``: ``(i/2) Sum_poles 1/|dGinv/dk|``.

    The principal-value part vanishes by symmetry in 1D, so the result is
    purely imaginary.  Analytically it equals ``i / (2 (tup + tdown) |sin p|)``.
    """
    f = lambda k: hubbard_inv_free_green(tup, tdown, p, k)
    poles = find_all_crossings(f, -np.pi, np.pi, n_sample=n_sample)
    return 0.5j * sum(1.0 / abs(nd(f, kp)) for kp in poles)


# --------------------------------------------------------------------------
# Pi matrix
# --------------------------------------------------------------------------
def pi_pole_component(vup, vdown, p_on_shell, nk, ncm_max, nrel_max,
                      n_sample=1025):
    """``PiPoleComponent``: the ``nu1 = nu2 = 0`` on-shell channel.

    Using ``1/(x + i0) = P(1/x) - i pi delta(x)``: a principal-value integral
    on a pole-symmetric grid, plus ``-i/2 Sum_poles 1/|dGinv/dk|``.
    """
    L = low_energy_grid_size(ncm_max, nrel_max)
    e0 = e_k0_sigma(vup, L, -p_on_shell) + e_k0_sigma(vdown, L, p_on_shell)

    def ginv(k):
        return e0 - (e_k0_sigma(vup, L, -k) + e_k0_sigma(vdown, L, k))

    _, eta = eig_hc(vup, vdown, ncm_max, 0.0)
    poles = find_all_crossings(ginv, -np.pi, np.pi, n_sample=n_sample)

    k_pv, dk_pv = setup_k_int_pv(nk, poles, -np.pi, np.pi)
    h_pv = build_hint_tensor_nu00(vup, vdown, k_pv, ncm_max, nrel_max, eta)
    w_pv = np.array([1.0 / ginv(k) for k in k_pv]) * dk_pv / (2 * np.pi)
    re_part = np.einsum("ia,ib->ab", h_pv * w_pv[:, None], h_pv)

    h_p = build_hint_tensor_nu00(vup, vdown, poles, ncm_max, nrel_max, eta)
    w_p = np.array([1.0 / abs(nd(ginv, kp)) for kp in poles])
    im_part = -0.5j * np.einsum("ia,ib->ab", h_p * w_p[:, None], h_p)

    return re_part + im_part


def _high_energy_s0(e0, ncm_max, nrel_max):
    """``highEnergyIntegrals_N``: free relative motion above the cutoff, s = 0.

    Analytically ``(1/pi) Integral[1/(A^2/4 - k^2), {k, Lambda_N, Infinity}]``
    with ``A = Sqrt[4 E0 - (2 pi N)^2]``.  The two ``10^-6`` offsets are kept
    exactly as in the Mathematica source.
    """
    N = np.arange(-ncm_max, ncm_max + 1)
    lam = np.where(N % 2 == 0, np.pi * (2 * nrel_max + 1), 2 * np.pi * nrel_max)
    A = np.sqrt((4.0 * e0 - (2 * np.pi * N) ** 2).astype(complex))
    return np.real(1.0 / (np.pi * A + 1e-6)
                   * np.log(-((A - 2 * lam) / (A + 2 * lam)) + 1e-6))


def _high_energy_s_ge_1(e0, ncm_max, nrel_max, omega_perp, smax):
    """``highEnergyIntegralsQuasi_N``, evaluated in closed form.

    ``(1/pi) Integral[Sum_s 1/(C_N - k^2 - 2 w s), {k, Lambda_N, Infinity}]``
    term by term, using
    ``Integral[1/(c - k^2), {k, L, Infinity}] = -ArcTanh[Sqrt[c]/L]/Sqrt[c]``
    which is valid for either sign of ``c`` on the complex branch.
    """
    N = np.arange(-ncm_max, ncm_max + 1)
    lam = np.where(N % 2 == 0, np.pi * (2 * nrel_max + 1), 2 * np.pi * nrel_max)
    s = np.arange(1, smax + 1)
    c = (e0 - (2 * np.pi * N[:, None]) ** 2 / 4.0
         - 2.0 * omega_perp * s[None, :]).astype(complex)
    sc = np.sqrt(c)
    return np.real(-np.arctanh(sc / lam[:, None]) / sc).sum(axis=1) / np.pi


def _counter_term(omega_perp, smax):
    """``counterTerm`` = ``(1/2pi) Integral[Sum_s 1/(k^2 + 2 w s), {k,-Inf,Inf}]``.

    Term by term ``Integral[1/(k^2 + c)] = pi/Sqrt[c]``, hence
    ``(1/2) Sum_{s=1}^{smax} 1/Sqrt[2 w s]``.
    """
    s = np.arange(1, smax + 1, dtype=float)
    return 0.5 * np.sum(1.0 / np.sqrt(2.0 * omega_perp * s))


def setup_t_matrix_on_shell_quasi1d(vup, vdown, ncm_max, nrel_max, nk_reg,
                                    nk_pole, p_on_shell, omega_perp, smax,
                                    n_sample=1025):
    """``setupTMatrixOnShellQuasi1D``: returns ``T(1/a1D)`` as a callable."""
    L = low_energy_grid_size(ncm_max, nrel_max)
    k_reg, dk_reg = setup_k_int_reg(nk_reg)
    e0 = e_k0_sigma(vup, L, -p_on_shell) + e_k0_sigma(vdown, L, p_on_shell)

    _, eta = eig_hc(vup, vdown, ncm_max, 0.0)
    n_cm = 2 * ncm_max + 1

    hei_s0 = _high_energy_s0(e0, ncm_max, nrel_max)
    pi_he_s0 = np.einsum("an,bn->ab", eta, eta * hei_s0)

    hei_q = _high_energy_s_ge_1(e0, ncm_max, nrel_max, omega_perp, smax)
    pi_he_q = np.einsum("an,bn->ab", eta, eta * hei_q)

    counter = _counter_term(omega_perp, smax)

    hint, eig_up, eig_down = build_hint_tensor(vup, vdown, k_reg, ncm_max,
                                               nrel_max, eta)
    n_st = hint.shape[-1]

    # --- s >= 1 -----------------------------------------------------------
    pi_ge1 = np.zeros((n_cm, n_cm))
    for ik in range(k_reg.size):
        var = e0 - eig_up[ik][:, None] - eig_down[ik][None, :]
        g = quasi1d_sum(var, -omega_perp, 1, smax) * (dk_reg[ik] / (2 * np.pi))
        a = hint[ik].reshape(n_cm, n_st * n_st)
        b = (hint[ik] * g).reshape(n_cm, n_st * n_st)
        pi_ge1 += a @ b.T
    pi_ge1 += counter * np.eye(n_cm) + pi_he_q

    # --- s = 0 (the on-shell pole channel is removed and handled separately)
    hint[:, :, 0, 0] = 0.0
    pi_s0 = np.zeros((n_cm, n_cm))
    for ik in range(k_reg.size):
        var = e0 - eig_up[ik][:, None] - eig_down[ik][None, :]
        g = (1.0 / var) * (dk_reg[ik] / (2 * np.pi))
        a = hint[ik].reshape(n_cm, n_st * n_st)
        b = (hint[ik] * g).reshape(n_cm, n_st * n_st)
        pi_s0 += a @ b.T
    pi_s0 += pi_he_s0

    pi_pole = pi_pole_component(vup, vdown, p_on_shell, nk_pole, ncm_max,
                                nrel_max, n_sample=n_sample)

    pi_matrix = pi_s0 + pi_ge1 + pi_pole
    hint_p = build_hint_tensor_nu00(vup, vdown, p_on_shell, ncm_max, nrel_max,
                                    eta)
    scan = TMatrixScan(pi_matrix, hint_p)

    def t_matrix(a1d_inv):
        # 1/g_1D = -a1D/2 = 1/(-2 (1/a1D)); 1/a1D = 0 gives c = inf, T = 0
        with np.errstate(divide="ignore"):
            return scan(1.0 / (-2.0 * np.asarray(a1d_inv, dtype=float)))

    t_matrix.pi_matrix = pi_matrix
    t_matrix.e0 = e0
    t_matrix.hint_p = hint_p
    t_matrix.scan = scan
    return t_matrix


class Quasi1DResult:
    """Callable ``U(1/a1D)`` that also carries the intermediate quantities."""

    def __init__(self, u_func, **kw):
        self._u = u_func
        self.__dict__.update(kw)

    def __call__(self, a1d_inv):
        return self._u(a1d_inv)


def setup_hubbard_u_quasi1d_all_params(vup, vdown, ncm_max, nrel_max, nk_reg,
                                       nk_pole, p_on_shell, omega_perp, smax,
                                       nk_hopping=200, im_ref_a1d_inv=-20.0,
                                       n_sample=1025):
    """``setupHubbardUQuasi1DAllParams``."""
    L = low_energy_grid_size(ncm_max, nrel_max)
    t_exact = setup_t_matrix_on_shell_quasi1d(vup, vdown, ncm_max, nrel_max,
                                              nk_reg, nk_pole, p_on_shell,
                                              omega_perp, smax,
                                              n_sample=n_sample)

    kk, dkk = gaussian_quadrature_weights(nk_hopping, -np.pi, np.pi)
    tup = -np.sum(dkk * np.array([e_k0_sigma(vup, L, k) for k in kk])
                  * np.cos(kk)) / (2 * np.pi)
    tdown = -np.sum(dkk * np.array([e_k0_sigma(vdown, L, k) for k in kk])
                    * np.cos(kk)) / (2 * np.pi)

    t_hub = hubbard_t_matrix_integral(tup, tdown, p_on_shell, n_sample=n_sample)
    eff_mass = np.imag(t_hub) / np.imag(1.0 / t_exact(im_ref_a1d_inv))

    def hubbard_u(a1d_inv):
        t = t_exact(a1d_inv)
        with np.errstate(divide="ignore", invalid="ignore"):
            # T = 0 at 1/a1D = 0 gives 1/T = inf and hence U = 0
            return np.where(t == 0, 0.0,
                            1.0 / (eff_mass * np.real(1.0 / np.where(t == 0, 1, t))))

    return Quasi1DResult(hubbard_u, t_matrix=t_exact, t_up=float(tup),
                         t_down=float(tdown), t_hubbard_integral=t_hub,
                         eff_mass_hubbard=float(eff_mass), e0=t_exact.e0,
                         pi_matrix=t_exact.pi_matrix,
                         params=dict(vup=vup, vdown=vdown, ncm_max=ncm_max,
                                     nrel_max=nrel_max, nk_reg=nk_reg,
                                     nk_pole=nk_pole, p_on_shell=p_on_shell,
                                     omega_perp=omega_perp, smax=smax))


def setup_hubbard_u_quasi1d(vup, vdown, p_on_shell, omega_perp, conv_param,
                            **kw):
    """``setupHubbardUQuasi1D`` - all cutoffs driven by one convergence knob."""
    c = int(conv_param)
    return setup_hubbard_u_quasi1d_all_params(
        vup, vdown,
        ncm_max=c * 5 + 5,
        nrel_max=c * 5 + 12,
        nk_reg=c * 3 + 8,
        nk_pole=5 * c + 20,
        p_on_shell=p_on_shell,
        omega_perp=omega_perp,
        smax=100 * c + 200,
        **kw,
    )


def a1d_inv_from_a3d(a3d, omega_perp):
    """``1/a1D`` from the 3D scattering length (confinement-induced resonance).

    As given in the package README::

        lperp  = 1/Sqrt[omegaperp]
        a1dinv = (-2 (lperp/2 (lperp/a3d + Zeta[1/2]/Sqrt[2])))^-1
    """
    lperp = 1.0 / np.sqrt(omega_perp)
    a3d = np.asarray(a3d, dtype=float)
    return 1.0 / (-2.0 * (lperp / 2.0
                          * (lperp / a3d + ZETA_HALF / np.sqrt(2.0))))
