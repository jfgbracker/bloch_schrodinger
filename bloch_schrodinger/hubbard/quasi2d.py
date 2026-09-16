"""Hubbard U for a quasi-2D square optical lattice.

Port of ``2DHubbardParameters.wl``.

The structure mirrors the quasi-1D case, but everything is now a product of an
x- and a y-lattice, so the pair propagator ``Pi`` carries a composite index
``(nu_x, nu_y)`` and the momentum integrals are two-dimensional.  In 2D the
Hubbard pair propagator has a non-vanishing real part, so U is obtained from

    1/U = effMassHubbard * Re[T_exact^-1] + Re[Pi_Hubbard] .

Two-dimensional momentum integrals are split at the inscribed circle of the
Brillouin zone: the part inside the circle is done in polar coordinates (where
the on-shell pole ring is a fixed radius per angle and can be handled with a
pole-symmetric principal-value grid), the remaining corners in Cartesian
coordinates.

Note on the high-energy correction
----------------------------------
The correction integrates the free relative motion over the region *outside*
the ``nrel_max`` box but inside a large cutoff.  It is tempting to rewrite that
region as ``quarter disc - rectangle``; do not.  The transverse mode sum
``quasi2DSum0toInf`` has poles at ``k^2 = c - (2j+1) omega_z``, which lie at
small ``|k|`` - inside both the disc and the rectangle, where they would cancel
analytically but destroy any quadrature.  The literal region never comes near
them, so it is kept literal here.
"""

from __future__ import annotations

import numpy as np

from .bands import V_REC, e_k0_sigma, eig_hc
from .hint import build_hint_tensor, build_hint_tensor_nu00, low_energy_grid_size
from .quadrature import (find_all_crossings, gaussian_quadrature_weights, nd,
                         round_to_multiple, setup_k_int_pv, setup_k_int_reg)
from .special import (quasi2d_sum_0_inf, quasi2d_sum_0_inf_approx,
                      quasi2d_sum_1_inf)
from .tmatrix import TMatrixScan

__all__ = [
    "V_REC",
    "hubbard_inv_free_green_polar",
    "square_polar_radius",
    "hubbard_t_matrix_integral_2d",
    "pi_pole_component_2d",
    "setup_t_matrix_on_shell_quasi2d",
    "setup_hubbard_u_quasi2d_all_params",
    "setup_hubbard_u_quasi2d",
    "log_a2d_inv_from_a3d",
    "Quasi2DResult",
]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _block_flatten(m, n):
    """``ArrayFlatten`` of a rank-4 array ``a[nux, nux', nuy, nuy']``.

    Row index ``(nux, nuy)``, column index ``(nux', nuy')`` - matching
    ``Flatten @ Outer[Times, Hintx, Hinty]`` where ``nux`` runs slowest.
    """
    return m.reshape(n, n, n, n).transpose(0, 2, 1, 3).reshape(n * n, n * n)


def _composite_gauss(edges, n_pt):
    """Composite Gauss-Legendre nodes/weights over consecutive ``edges``."""
    xs, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        x, w = gaussian_quadrature_weights(n_pt, a, b)
        xs.append(x)
        ws.append(w)
    return np.concatenate(xs), np.concatenate(ws)


def _geometric_edges(a, b, n_seg):
    """Geometrically spaced segment edges on ``[a, b]`` (``a > 0``)."""
    return np.geomspace(a, b, n_seg + 1)


# --------------------------------------------------------------------------
# Hubbard-model side
# --------------------------------------------------------------------------
def hubbard_inv_free_green_polar(tupx, tdownx, tupy, tdowny, px, py, k, theta):
    """``HubbardInvFreeGreenFuncPolarCoord``."""
    return (2 * tupx * (np.cos(-k * np.cos(theta)) - np.cos(px))
            + 2 * tupy * (np.cos(-k * np.sin(theta)) - np.cos(py))
            + 2 * tdownx * (np.cos(k * np.cos(theta)) - np.cos(px))
            + 2 * tdowny * (np.cos(k * np.sin(theta)) - np.cos(py)))


def square_polar_radius(theta):
    """``squarePolarRadiusForRad``: distance to the Brillouin-zone boundary."""
    tm = np.mod(theta, np.pi / 2)
    return np.pi / np.cos(tm) if tm <= np.pi / 4 else np.pi / np.cos(np.pi / 2 - tm)


def hubbard_t_matrix_integral_2d(tupx, tdownx, tupy, tdowny, px, py, nk,
                                 ntheta, n_sample=1025):
    """``HubbardTMatrixIntegral`` (2D): the Hubbard pair propagator.

    Polar integration over the Brillouin zone; per angle a principal-value
    radial integral plus ``-i pi`` times the on-shell pole residues.
    """
    theta, dtheta = gaussian_quadrature_weights(ntheta, 0.0, 2 * np.pi)
    total = 0.0 + 0.0j
    for th, dth in zip(theta, dtheta):
        f = lambda k: hubbard_inv_free_green_polar(tupx, tdownx, tupy, tdowny,
                                                   px, py, k, th)
        poles = find_all_crossings(f, 0.0, np.sqrt(2) * np.pi,
                                   n_sample=n_sample)
        # keep only poles inside -pi < kx, ky < pi
        poles = poles[(np.abs(poles * np.cos(th)) < np.pi)
                      & (np.abs(poles * np.sin(th)) < np.pi)]
        k_pv, dk_pv = setup_k_int_pv(nk, poles, 0.0, square_polar_radius(th))
        val = np.sum(dk_pv * k_pv / f(k_pv))
        if poles.size:
            pw = np.array([1.0 / abs(nd(f, kp)) for kp in poles])
            val = val - 1j * np.pi * np.sum(poles * pw)
        total += dth * val
    return total / (2 * np.pi) ** 2


# --------------------------------------------------------------------------
# Pi matrix - pole (nu1 = nu2 = 0) component
# --------------------------------------------------------------------------
def pi_pole_component_2d(vupx, vdownx, vupy, vdowny, p_x, p_y, nk, ntheta,
                         nk_square, ncm_max, nrel_max, n_sample=1025):
    """``PiPoleComponent`` (2D).

    Circle part (radius ``pi``, polar, contains the on-shell ring) plus the
    Brillouin-zone corners (Cartesian, pole free).
    """
    L = low_energy_grid_size(ncm_max, nrel_max)
    n = 2 * ncm_max + 1

    e0 = (e_k0_sigma(vupx, L, -p_x) + e_k0_sigma(vupy, L, -p_y)
          + e_k0_sigma(vdownx, L, p_x) + e_k0_sigma(vdowny, L, p_y))

    def ginv(kx, ky):
        return e0 - (e_k0_sigma(vupx, L, -kx) + e_k0_sigma(vupy, L, -ky)
                     + e_k0_sigma(vdownx, L, kx) + e_k0_sigma(vdowny, L, ky))

    def ginv_polar(k, th):
        return ginv(k * np.cos(th), k * np.sin(th))

    _, eta_x = eig_hc(vupx, vdownx, ncm_max, 0.0)
    _, eta_y = eig_hc(vupy, vdowny, ncm_max, 0.0)

    def pair(hx, hy, w):
        """Sum_i w_i (hx_i (x) hx_i) (x) (hy_i (x) hy_i) as an (n^2, n^2) block."""
        xx = (hx[:, :, None] * hx[:, None, :]).reshape(hx.shape[0], n * n)
        yy = (hy[:, :, None] * hy[:, None, :]).reshape(hy.shape[0], n * n)
        return (xx * np.asarray(w)[:, None]).T @ yy

    # ---------------- circular part -------------------------------------
    theta, dtheta = gaussian_quadrature_weights(ntheta, 0.0, 2 * np.pi)
    circle = np.zeros((n * n, n * n), dtype=complex)
    for th, dth in zip(theta, dtheta):
        f = lambda k: ginv_polar(k, th)
        poles = find_all_crossings(f, 0.0, np.pi, n_sample=n_sample)
        if poles.size == 0:
            raise RuntimeError("Cannot find poles of the free Green's function "
                               f"at theta = {th}")
        k_pv, dk_pv = setup_k_int_pv(nk, poles, 0.0, np.pi)
        hx = build_hint_tensor_nu00(vupx, vdownx, k_pv * np.cos(th),
                                    ncm_max, nrel_max, eta_x)
        hy = build_hint_tensor_nu00(vupy, vdowny, k_pv * np.sin(th),
                                    ncm_max, nrel_max, eta_y)
        w = k_pv * dk_pv / np.array([f(k) for k in k_pv])
        term = pair(hx, hy, w).astype(complex)

        hxp = build_hint_tensor_nu00(vupx, vdownx, poles * np.cos(th),
                                     ncm_max, nrel_max, eta_x)
        hyp = build_hint_tensor_nu00(vupy, vdowny, poles * np.sin(th),
                                     ncm_max, nrel_max, eta_y)
        pw = np.array([1.0 / abs(nd(f, kp)) for kp in poles])
        term = term - 1j * np.pi * pair(hxp, hyp, poles * pw)
        circle += dth * term
    circle /= (2 * np.pi) ** 2

    # ---------------- Brillouin-zone corners ----------------------------
    nk_even = int(round_to_multiple(nk_square, 2))
    k_sq, dk_sq = setup_k_int_reg(nk_even)
    k_pos, dk_pos = k_sq[nk_even // 2:], dk_sq[nk_even // 2:]

    hy_all = build_hint_tensor_nu00(vupy, vdowny, k_sq, ncm_max, nrel_max, eta_y)
    yy = (hy_all[:, :, None] * hy_all[:, None, :]).reshape(nk_even, n * n)
    yy_dk = yy * (dk_sq / (2 * np.pi))[:, None]

    integ = np.empty((nk_even, n * n))
    for iy, ky in enumerate(k_sq):
        r = np.sqrt(np.pi ** 2 - ky ** 2)
        kx_t = (np.pi - r) / np.pi * k_pos + r
        dkx_t = (np.pi - r) / np.pi * dk_pos
        kx = np.concatenate([-kx_t[::-1], kx_t])
        dkx = np.concatenate([dkx_t[::-1], dkx_t])
        hx = build_hint_tensor_nu00(vupx, vdownx, kx, ncm_max, nrel_max, eta_x)
        xx = (hx[:, :, None] * hx[:, None, :]).reshape(kx.size, n * n)
        wx = np.array([1.0 / ginv(k, ky) for k in kx]) * dkx / (2 * np.pi)
        integ[iy] = (xx * wx[:, None]).sum(axis=0)
    square = integ.T @ yy_dk

    return _block_flatten(circle, n) + _block_flatten(square, n)


# --------------------------------------------------------------------------
# high-energy corrections
# --------------------------------------------------------------------------
def _high_energy_2d(e0, ncm_max, nrel_max, omega_z, cutoff, n_pt=48,
                    n_seg=14, n_pt_tail=32):
    """``highEnergyIntegrals_NxNy``.

    ``4/(2 pi)^2`` times the integral over the quarter-plane region outside the
    ``(Lambda_Nx, Lambda_Ny)`` box but inside radius ``Lambda1``, plus the polar
    annulus from ``Lambda1`` out to ``cutoff``.

    Region A (``kx >= Lambda_Nx``) is done in polar coordinates, region B
    (``kx <= Lambda_Nx``, ``ky >= Lambda_Ny``) in Cartesian coordinates; both
    stay far from the poles of the transverse mode sum.
    """
    ns = np.arange(-ncm_max, ncm_max + 1)
    lam1 = np.sqrt(2) * np.pi * (2 * nrel_max + 1)
    lam_even = np.pi * (2 * nrel_max + 1)
    lam_odd = 2 * np.pi * nrel_max

    c_all = e0 - ((2 * np.pi * ns[:, None]) ** 2
                  + (2 * np.pi * ns[None, :]) ** 2) / 4.0
    out = np.zeros_like(c_all)

    t_nodes, t_w = gaussian_quadrature_weights(n_pt, 0.0, 1.0)

    for par_x, lam_x in ((0, lam_even), (1, lam_odd)):
        for par_y, lam_y in ((0, lam_even), (1, lam_odd)):
            mask = np.ix_(ns % 2 == par_x, ns % 2 == par_y)
            c = c_all[mask].ravel()[:, None]

            # --- region A: polar, theta in [0, arccos(lam_x/lam1)] --------
            th_a = np.arccos(lam_x / lam1)
            th, dth = gaussian_quadrature_weights(n_pt, 0.0, th_a)
            r_lo = lam_x / np.cos(th)                       # (n_pt,)
            kk = r_lo[:, None] + (lam1 - r_lo)[:, None] * t_nodes[None, :]
            wk = (lam1 - r_lo)[:, None] * t_w[None, :]
            wa = (dth[:, None] * wk * kk).ravel()           # includes Jacobian k
            ia = (quasi2d_sum_0_inf(c - (kk.ravel()[None, :]) ** 2, -omega_z)
                  * wa[None, :]).sum(axis=1)

            # --- region B: Cartesian, kx in [0, lam_x] --------------------
            kx, wx = gaussian_quadrature_weights(n_pt, 0.0, lam_x)
            y_hi = np.sqrt(np.maximum(lam1 ** 2 - kx ** 2, 0.0))
            ky = lam_y + (y_hi - lam_y)[:, None] * t_nodes[None, :]
            wy = (y_hi - lam_y)[:, None] * t_w[None, :]
            wb = (wx[:, None] * wy).ravel()
            arg = c - (kx[:, None] ** 2 + ky ** 2).ravel()[None, :]
            ib = (quasi2d_sum_0_inf(arg, -omega_z) * wb[None, :]).sum(axis=1)

            # --- polar annulus lam1 -> cutoff -----------------------------
            ke, we = _composite_gauss(_geometric_edges(lam1, cutoff, n_seg),
                                      n_pt_tail)
            tail = (quasi2d_sum_0_inf(c - ke[None, :] ** 2, -omega_z)
                    * (we * ke)[None, :]).sum(axis=1) / (2 * np.pi)

            vals = 4.0 / (2 * np.pi) ** 2 * (ia + ib) + tail
            out[mask] = vals.reshape(out[mask].shape)

    return out


def _radial_mode_integral(a, b, omega_z, which, n_seg=14, n_pt=32):
    """``(1/2 pi) Integral[k quasi2DSum{0,1}toInf[k^2, omega_z], {k, a, b}]``."""
    if a <= 0.0:
        edges = np.linspace(a, b, n_seg + 1)
    else:
        edges = _geometric_edges(a, b, n_seg)
    k, w = _composite_gauss(edges, n_pt)
    f = quasi2d_sum_0_inf if which == 0 else quasi2d_sum_1_inf
    return float(np.sum(w * k * f(k ** 2, omega_z)) / (2 * np.pi))


# --------------------------------------------------------------------------
# T matrix
# --------------------------------------------------------------------------
def setup_t_matrix_on_shell_quasi2d(vupx, vdownx, vupy, vdowny, ncm_max,
                                    nrel_max, nk_reg, nk_pole_circ,
                                    ntheta_pole_circ, nk_pole_square, p_x, p_y,
                                    cutoff, omega_z, n_sample=1025,
                                    quasi2d_sum_mode="faithful", progress=None):
    """``setupTMatrixOnShellQuasi2D``: returns ``T(log(1/a2D))`` as a callable.

    ``quasi2d_sum_mode``
        ``"faithful"`` reproduces the Mathematica switch to the asymptotic
        ``quasi2DSum0toInfApprox`` above ``quasi2DSumApproxPos``;
        ``"exact"`` uses the exact ``gammaln``-based form everywhere (slower,
        and differs from Mathematica at the ~1e-6 relative level of the
        asymptotic form).
    """
    L = low_energy_grid_size(ncm_max, nrel_max)
    n = 2 * ncm_max + 1

    e0 = (e_k0_sigma(vupx, L, -p_x) + e_k0_sigma(vupy, L, -p_y)
          + e_k0_sigma(vdownx, L, p_x) + e_k0_sigma(vdowny, L, p_y))

    _, eta_x = eig_hc(vupx, vdownx, ncm_max, 0.0)
    _, eta_y = eig_hc(vupy, vdowny, ncm_max, 0.0)

    # ---- high-energy correction ----------------------------------------
    hei = _high_energy_2d(e0, ncm_max, nrel_max, omega_z, cutoff)
    xx = eta_x[:, None, :] * eta_x[None, :, :]        # [nux, nux', N]
    yy = eta_y[:, None, :] * eta_y[None, :, :]
    pi_he = np.einsum("abN,cdM,NM->abcd", xx, yy, hei)
    pi_he = _block_flatten(pi_he.reshape(n, n, n, n), n)
    pi_he += _radial_mode_integral(2 * np.pi * nrel_max, cutoff, omega_z, 0) \
        * np.eye(n * n)

    counter_term = _radial_mode_integral(0.0, 2 * np.pi * nrel_max, omega_z, 1) \
        * np.eye(n * n)

    # ---- non-pole contribution -----------------------------------------
    k_reg, dk_reg = setup_k_int_reg(nk_reg)
    hx, eux, edx = build_hint_tensor(vupx, vdownx, k_reg, ncm_max, nrel_max, eta_x)
    hy, euy, edy = build_hint_tensor(vupy, vdowny, k_reg, ncm_max, nrel_max, eta_y)
    n_st = hx.shape[-1]
    ns2 = n_st * n_st

    ax = np.empty((nk_reg, n * n, ns2))
    ay = np.empty((nk_reg, n * n, ns2))
    for ik in range(nk_reg):
        ax[ik] = ((hx[ik][:, None] * hx[ik][None, :]).reshape(n * n, ns2)
                  * (dk_reg[ik] / (2 * np.pi)))
        ay[ik] = ((hy[ik][:, None] * hy[ik][None, :]).reshape(n * n, ns2)
                  * (dk_reg[ik] / (2 * np.pi)))

    ex = e0 - eux[:, :, None] - edx[:, None, :]        # [nk, nst, nst]
    ey = -(euy[:, :, None] + edy[:, None, :])

    # position at which the Mathematica code switches to the asymptotic sum
    pots = [vupx, vdownx, vupy, vdowny]
    smallest = int(np.argmin(pots))
    lm = int(np.ceil((nk_reg + 1) / 2)) - 1
    combined = [eux, edx, euy, edy]
    ref = (combined[smallest][lm, :]
           + sum(combined[o][lm, 0] for o in range(4) if o != smallest)) \
        / (2 * omega_z)
    hits = np.nonzero(ref > 4)[0]
    approx_pos = int(hits[0]) + 1 if hits.size else -1   # 1-based, -1 = none
    if approx_pos != -1 and approx_pos % 2 == 0:
        approx_pos += 1

    pi_non_pole = np.zeros((n * n, n * n))
    total = nk_reg * nk_reg
    done = 0
    for ikx in range(nk_reg):
        for iky in range(nk_reg):
            var = ex[ikx][:, :, None, None] + ey[iky][None, None, :, :]
            if approx_pos == -1 or quasi2d_sum_mode == "exact":
                kern = quasi2d_sum_0_inf(var, -omega_z)
            else:
                kern = quasi2d_sum_0_inf_approx(var, -omega_z)
                s = slice(None, approx_pos)
                kern[s, s, s, s] = quasi2d_sum_0_inf(var[s, s, s, s], -omega_z)
            kern[0, 0, 0, 0] = quasi2d_sum_1_inf(
                np.array(ex[ikx][0, 0] + ey[iky][0, 0]), -omega_z)
            pi_non_pole += ax[ikx] @ kern.reshape(ns2, ns2) @ ay[iky].T
            done += 1
            if progress is not None:
                progress(done, total, "regular contribution")
    pi_non_pole = _block_flatten(pi_non_pole, n)

    # ---- pole contribution ---------------------------------------------
    pi_pole = pi_pole_component_2d(vupx, vdownx, vupy, vdowny, p_x, p_y,
                                   nk_pole_circ, ntheta_pole_circ,
                                   nk_pole_square, ncm_max, nrel_max,
                                   n_sample=n_sample)

    pi_matrix = pi_non_pole + pi_pole + pi_he + counter_term

    hxp = build_hint_tensor_nu00(vupx, vdownx, p_x, ncm_max, nrel_max, eta_x)
    hyp = build_hint_tensor_nu00(vupy, vdowny, p_y, ncm_max, nrel_max, eta_y)
    hp = np.outer(hxp, hyp).ravel()
    log_lam = np.log(2 * np.pi * nrel_max)
    scan = TMatrixScan(pi_matrix, hp)

    def t_matrix(log_a2d_inv):
        # 1/g_2D = -(1/2 pi) Log[Lambda_rel a2D]
        c = -1.0 / (2 * np.pi) * (-np.asarray(log_a2d_inv, dtype=float) + log_lam)
        return scan(c)

    t_matrix.pi_matrix = pi_matrix
    t_matrix.e0 = e0
    t_matrix.hint_p = hp
    t_matrix.approx_pos = approx_pos
    t_matrix.scan = scan
    return t_matrix


class Quasi2DResult:
    """Callable ``U(log(1/a2D))`` that also carries intermediate quantities."""

    def __init__(self, u_func, **kw):
        self._u = u_func
        self.__dict__.update(kw)

    def __call__(self, log_a2d_inv):
        return self._u(log_a2d_inv)


def setup_hubbard_u_quasi2d_all_params(vupx, vdownx, vupy, vdowny, ncm_max,
                                       nrel_max, nk_reg, nk_pole_circ,
                                       ntheta_pole_circ, nk_pole_square, p_x,
                                       p_y, omega_z, nk_hubbard, ntheta_hubbard,
                                       cutoff, nk_hopping=200, n_sample=1025,
                                       quasi2d_sum_mode="faithful",
                                       progress=None):
    """``setupHubbardUQuasi2DAllParams``."""
    L = low_energy_grid_size(ncm_max, nrel_max)
    t_exact = setup_t_matrix_on_shell_quasi2d(
        vupx, vdownx, vupy, vdowny, ncm_max, nrel_max, nk_reg, nk_pole_circ,
        ntheta_pole_circ, nk_pole_square, p_x, p_y, cutoff, omega_z,
        n_sample=n_sample, quasi2d_sum_mode=quasi2d_sum_mode, progress=progress)

    kk, dkk = gaussian_quadrature_weights(nk_hopping, -np.pi, np.pi)

    def hop(v):
        e = np.array([e_k0_sigma(v, L, k) for k in kk])
        return float(-np.sum(dkk * e * np.cos(kk)) / (2 * np.pi))

    tupx, tdownx = hop(vupx), hop(vdownx)
    tupy, tdowny = hop(vupy), hop(vdowny)

    t_hub = hubbard_t_matrix_integral_2d(tupx, tdownx, tupy, tdowny, p_x, p_y,
                                         nk_hubbard, ntheta_hubbard,
                                         n_sample=n_sample)
    eff_mass = np.imag(-t_hub) / np.imag(1.0 / t_exact(0.0))

    def hubbard_u(log_a2d_inv):
        t = t_exact(log_a2d_inv)
        with np.errstate(divide="ignore", invalid="ignore"):
            # T -> 0 (a_2D -> 0, i.e. vanishing coupling) gives 1/T = inf and U = 0.
            # Reachable in practice: a_2D falls exponentially with 1/a_3D, so a small
            # a_3D drives log(1/a_2D) large enough that T underflows to exactly zero.
            inv_t = np.where(t == 0, np.inf, 1.0 / np.where(t == 0, 1.0, t))
            return np.where(t == 0, 0.0,
                            1.0 / (eff_mass * np.real(inv_t) + np.real(t_hub)))

    return Quasi2DResult(hubbard_u, t_matrix=t_exact, t_up_x=tupx,
                         t_down_x=tdownx, t_up_y=tupy, t_down_y=tdowny,
                         t_hubbard_integral=t_hub, eff_mass_hubbard=float(eff_mass),
                         e0=t_exact.e0, pi_matrix=t_exact.pi_matrix,
                         approx_pos=t_exact.approx_pos,
                         params=dict(vupx=vupx, vdownx=vdownx, vupy=vupy,
                                     vdowny=vdowny, ncm_max=ncm_max,
                                     nrel_max=nrel_max, nk_reg=nk_reg,
                                     p_x=p_x, p_y=p_y, omega_z=omega_z,
                                     cutoff=cutoff))


def setup_hubbard_u_quasi2d(vupx, vdownx, vupy, vdowny, p_x, p_y, omega_z,
                            conv_param, **kw):
    """``setupHubbardUQuasi2D`` - all cutoffs driven by one convergence knob."""
    c = int(conv_param)
    return setup_hubbard_u_quasi2d_all_params(
        vupx, vdownx, vupy, vdowny,
        ncm_max=c * 2 + 2,
        nrel_max=c * 3 + 2,
        nk_reg=c * 3 + 4,
        nk_pole_circ=4 + c * 5,
        ntheta_pole_circ=4 + c * 5,
        nk_pole_square=4 + c * 5,
        p_x=p_x, p_y=p_y, omega_z=omega_z,
        nk_hubbard=40 + c * 10,
        ntheta_hubbard=40 + c * 10,
        cutoff=1000 + c * 1000,
        **kw,
    )


def log_a2d_inv_from_a3d(a3d, omega_z):
    """``Log[1/a2D]`` from the 3D scattering length, as in the package README::

        lz = 1/Sqrt[omegaz]
        Log[(lz Sqrt[Pi/0.905] Exp[-(Sqrt[Pi] lz)/(Sqrt[2] a3d)])^-1]
    """
    lz = 1.0 / np.sqrt(omega_z)
    a3d = np.asarray(a3d, dtype=float)
    a2d = lz * np.sqrt(np.pi / 0.905) * np.exp(-(np.sqrt(np.pi) * lz)
                                               / (np.sqrt(2.0) * a3d))
    return np.log(1.0 / a2d)
