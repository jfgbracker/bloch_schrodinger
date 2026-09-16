"""Exact Hubbard U for atoms in a quasi-low-dimensional optical lattice.

Public, unit-aware front end to the Adlong solver.  Everything here speaks the
caller's units (``H = alpha k^2 + V``); the conversion is handled by
:class:`~bloch_schrodinger.hubbard.units.UnitConverter`.

What "exact" means
------------------
``U`` is *not* the Wannier overlap ``g int |w|^4``.  It is defined by requiring
the Hubbard model to reproduce the exact two-body scattering amplitude of the
real lattice in the limit of vanishing relative quasi-momentum: writing the
on-shell T matrix in terms of the pair propagator ``Pi``,

    T_exact^-1 = 1/g - Pi ,        T_Hubbard^-1 = 1/U - Pi_Hubbard ,

the imaginary parts are matched by one factor and ``U`` is read off from the
real parts.  ``Pi`` is built in the basis of two-atom centre-of-mass lattice
states, so *all* bands and *all* transverse harmonic modes are summed over.
The Wannier overlap is the leading Born term of the same quantity, and the two
agree only at weak coupling (see :func:`hubbard_u_wannier_1d`).

Geometry restriction
--------------------
The method as published factorises the pair problem into independent lattice
directions, so it applies to

* a **quasi-1D** lattice (1D lattice + 2D transverse harmonic trap), and
* a **quasi-2D separable square** lattice (``V(x) + V(y)`` + 1D trap along z).

It does **not** apply verbatim to a non-separable in-plane geometry such as a
honeycomb/triangular lattice, whose potential cannot be written as
``V_x(x) + V_y(y)``.  For those, use a matched separable reference lattice to
estimate the beyond-Born correction and say so explicitly -- do not feed a
honeycomb depth into :func:`hubbard_u_quasi2d` and call the result exact.

Reference
---------
H. S. Adlong et al., *Microscopic calculation of Hubbard parameters for quantum
gas microscopes* -- and the Mathematica package ``cold-atom-hubbard-parameters``
that this is a validated port of.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .bands import V_REC, hopping_t
from .quasi1d import setup_hubbard_u_quasi1d
from .quasi2d import setup_hubbard_u_quasi2d
from .units import UnitConverter
from .wannier1d import wannier_function, wannier_u_integral

__all__ = [
    "V_REC",
    "ExactHubbardResult",
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


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------
@dataclass
class ExactHubbardResult:
    """``U`` as a callable, plus everything that went into it.

    Attributes
    ----------
    u : Callable
        ``U(x)`` in the caller's energy units.  ``x`` is ``1/a_1D`` (caller
        inverse-length units) for quasi-1D, or ``log(1/a_2D)`` with ``a_2D`` in
        caller length units for quasi-2D.
    t : float
        Nearest-neighbour hopping of the lowest band, caller energy units.
    converter : UnitConverter
    raw : object
        The underlying solver result, in Adlong units, for cross-checking.
    """

    u: Callable
    t: float
    converter: UnitConverter
    raw: object = field(repr=False, default=None)
    info: dict = field(default_factory=dict)

    def __call__(self, x):
        return self.u(x)

    @property
    def recoil_energy(self) -> float:
        return self.converter.recoil_energy


# ---------------------------------------------------------------------------
# quasi-1D
# ---------------------------------------------------------------------------
def hubbard_u_quasi1d(depth, omega_perp, lattice_constant, alpha=1.0,
                      depth_down=None, p_on_shell=0.05, conv_param=4,
                      **kwargs) -> ExactHubbardResult:
    """Exact Hubbard ``U`` for a quasi-1D lattice.

    Parameters
    ----------
    depth : float
        Lattice depth for the spin-up atoms, **in units of the lattice recoil
        energy** (dimensionless, so no unit conversion is applied to it).
    omega_perp : float
        Transverse harmonic confinement, as the level spacing ``hbar*omega``, in
        the caller's *energy* units.
    lattice_constant : float
        Lattice spacing ``d`` in the caller's *length* units.
    alpha : float
        Caller's kinetic coefficient (``H = alpha k^2 + V``).
    depth_down : float, optional
        Depth seen by the spin-down atoms, in recoils.  Defaults to ``depth``.
    p_on_shell : float
        Relative quasi-momentum at which the amplitudes are matched, in units of
        ``1/d``.  Formally a limit to zero; ~0.05 is the package default.
    conv_param : int
        Convergence knob bundling every cutoff (see the Adlong package).

    Returns
    -------
    ExactHubbardResult
        Whose ``u(a1d_inv)`` takes ``1/a_1D`` in caller inverse-length units and
        returns ``U`` in caller energy units.
    """
    conv = UnitConverter(lattice_constant, alpha)
    v_up = float(depth) * V_REC
    v_down = v_up if depth_down is None else float(depth_down) * V_REC
    w_perp = float(conv.energy_to_adlong(omega_perp))

    res = setup_hubbard_u_quasi1d(v_up, v_down, p_on_shell, w_perp, conv_param,
                                  **kwargs)

    def u(a1d_inv):
        # 1/a_1D: caller inverse length -> units of 1/d
        x = conv.inverse_length_to_adlong(a1d_inv)
        return conv.energy_from_adlong(res(x))

    return ExactHubbardResult(
        u=u,
        t=float(conv.energy_from_adlong(res.t_up)),
        converter=conv,
        raw=res,
        info=dict(depth=depth, depth_down=depth_down, omega_perp=omega_perp,
                  p_on_shell=p_on_shell, conv_param=conv_param,
                  omega_perp_adlong=w_perp),
    )


def hubbard_u_wannier_1d(depth, omega_perp, lattice_constant, alpha=1.0,
                         nmax=40, nk=512):
    """Leading-Born (Wannier) ``U`` for the same quasi-1D problem.

    ``U = g_1D * int |w(x)|^4 dx`` with ``g_1D = -2 hbar^2/(m a_1D)``, i.e. the
    textbook single-band result that :func:`hubbard_u_quasi1d` improves on.
    Returns a callable of ``1/a_1D`` in caller units, plus ``int|w|^4``.
    """
    conv = UnitConverter(lattice_constant, alpha)
    quartic_adlong, norm = wannier_u_integral(float(depth) * V_REC, nmax=nmax,
                                              nk=nk)

    def u(a1d_inv):
        x = conv.inverse_length_to_adlong(a1d_inv)          # d/a_1D
        # g_1D = -2/a_1D in Adlong units; U = g * int|w|^4
        return conv.energy_from_adlong(-2.0 * np.asarray(x, float)
                                       * quartic_adlong)

    return u, quartic_adlong, norm


# ---------------------------------------------------------------------------
# quasi-2D (separable square lattice)
# ---------------------------------------------------------------------------
def hubbard_u_quasi2d(depth, omega_z, lattice_constant, alpha=1.0,
                      depth_y=None, depth_down=None, depth_down_y=None,
                      p_on_shell=0.05, conv_param=3,
                      **kwargs) -> ExactHubbardResult:
    """Exact Hubbard ``U`` for a quasi-2D **separable square** lattice.

    Same conventions as :func:`hubbard_u_quasi1d`; ``omega_z`` is the level
    spacing of the tight confinement along z, in caller energy units.  The
    returned ``u`` takes ``log(1/a_2D)`` with ``a_2D`` in caller length units.

    See the module docstring on why a honeycomb lattice must not be passed here.
    """
    conv = UnitConverter(lattice_constant, alpha)
    vx = float(depth) * V_REC
    vy = vx if depth_y is None else float(depth_y) * V_REC
    vxd = vx if depth_down is None else float(depth_down) * V_REC
    vyd = vy if depth_down_y is None else float(depth_down_y) * V_REC
    wz = float(conv.energy_to_adlong(omega_z))

    res = setup_hubbard_u_quasi2d(vx, vxd, vy, vyd, p_on_shell, p_on_shell, wz,
                                  conv_param, **kwargs)

    log_d = np.log(conv.lattice_constant)

    def u(log_a2d_inv):
        # log(1/a_2D) with a_2D in caller units -> with a_2D in units of d:
        # log(d/a_2D) = log(1/a_2D) + log(d)
        x = np.asarray(log_a2d_inv, dtype=float) + log_d
        return conv.energy_from_adlong(res(x))

    return ExactHubbardResult(
        u=u,
        t=float(conv.energy_from_adlong(res.t_up_x)),
        converter=conv,
        raw=res,
        info=dict(depth=depth, depth_y=depth_y, omega_z=omega_z,
                  p_on_shell=p_on_shell, conv_param=conv_param,
                  omega_z_adlong=wz),
    )


def hubbard_u_wannier_2d_square(depth, omega_z, lattice_constant, alpha=1.0,
                                depth_y=None, nmax=40, nk=512):
    """Leading-Born (Wannier) ``U`` for the separable square quasi-2D lattice.

    ``U = g_3D * (int|w_x|^4)(int|w_y|^4) * int|phi_z|^4`` with
    ``g_3D = 4 pi hbar^2 a_3D / m`` and ``int|phi_z|^4 = 1/(sqrt(2 pi) a_ho)``
    -- the standard formula the notebooks use.  Returns a callable of ``a_3D``
    in caller length units.
    """
    conv = UnitConverter(lattice_constant, alpha)
    qx, _ = wannier_u_integral(float(depth) * V_REC, nmax=nmax, nk=nk)
    dy = depth if depth_y is None else depth_y
    qy, _ = wannier_u_integral(float(dy) * V_REC, nmax=nmax, nk=nk)

    wz = float(conv.energy_to_adlong(omega_z))
    a_ho = 1.0 / np.sqrt(wz)                     # sqrt(hbar/(m omega)) in units of d
    quartic_z = 1.0 / (np.sqrt(2 * np.pi) * a_ho)

    def u(a3d):
        a = conv.length_to_adlong(a3d)           # a_3D in units of d
        return conv.energy_from_adlong(4 * np.pi * a * qx * qy * quartic_z)

    return u, (qx, qy, quartic_z)


# ---------------------------------------------------------------------------
# scattering-length relations
# ---------------------------------------------------------------------------
ZETA_HALF = -1.4603545088095868  # Zeta[1/2]


def a1d_inv_from_a3d(a3d, omega_perp, lattice_constant=1.0, alpha=1.0):
    """``1/a_1D`` from ``a_3D`` (confinement-induced resonance), caller units.

    ``a_1D = -(l_perp^2/a_3D)(1 - C a_3D/l_perp)/2``, ``C = -zeta(1/2)/sqrt(2)``.
    """
    conv = UnitConverter(lattice_constant, alpha)
    w = conv.energy_to_adlong(omega_perp)
    lperp = 1.0 / np.sqrt(w)                              # in units of d
    a = conv.length_to_adlong(a3d)                        # in units of d
    inv_adlong = 1.0 / (-2.0 * (lperp / 2.0
                                * (lperp / a + ZETA_HALF / np.sqrt(2.0))))
    return inv_adlong / conv.lattice_constant             # -> caller 1/length


def log_a2d_inv_from_a3d(a3d, omega_z, lattice_constant=1.0, alpha=1.0):
    """``log(1/a_2D)`` from ``a_3D``, in caller units.

    ``a_2D = l_z sqrt(pi/0.905) exp(-sqrt(pi) l_z / (sqrt(2) a_3D))``.
    """
    conv = UnitConverter(lattice_constant, alpha)
    w = conv.energy_to_adlong(omega_z)
    lz = 1.0 / np.sqrt(w)                                 # units of d
    a = conv.length_to_adlong(a3d)
    a2d = lz * np.sqrt(np.pi / 0.905) * np.exp(-(np.sqrt(np.pi) * lz)
                                               / (np.sqrt(2.0) * a))
    return np.log(1.0 / a2d) - np.log(conv.lattice_constant)


# ---------------------------------------------------------------------------
# helpers shared with the notebooks
# ---------------------------------------------------------------------------
def wannier_profile_1d(depth, x, lattice_constant=1.0, nmax=40, nk=512):
    """Lowest-band Wannier function of a 1D sinusoidal lattice, caller units."""
    conv = UnitConverter(lattice_constant, 1.0)
    xa = conv.length_to_adlong(x)
    w = wannier_function(float(depth) * V_REC, xa, nmax=nmax, nk=nk)
    # |w|^2 dx = 1 in Adlong units -> rescale so it normalises in caller units
    return w / np.sqrt(conv.lattice_constant)


def hopping(depth, lattice_constant=1.0, alpha=1.0, nmax=40, nk=200):
    """Nearest-neighbour hopping ``t`` of a 1D sinusoidal lattice, caller units."""
    conv = UnitConverter(lattice_constant, alpha)
    return float(conv.energy_from_adlong(hopping_t(float(depth) * V_REC, nmax,
                                                   nk=nk)))


def site_trap_frequency(depth, lattice_constant=1.0, alpha=1.0):
    """Harmonic level spacing ``hbar*omega`` at the bottom of a sinusoidal well.

    For ``V = v sin^2(pi x / d)``, expanding about a minimum gives
    ``V ~ v (pi/d)^2 x^2``, i.e. ``hbar*omega = 2 sqrt(v E_rec)`` -- the standard
    deep-lattice result.  Returned in caller energy units, with ``depth`` in
    recoils.
    """
    conv = UnitConverter(lattice_constant, alpha)
    return 2.0 * np.sqrt(float(depth)) * conv.recoil_energy


def wannier_quartic_1d(depth, lattice_constant=1.0, nmax=40, nk=512):
    """``int |w(x)|^4 dx`` of the lowest band, in caller units (1/length)."""
    q, _ = wannier_u_integral(float(depth) * V_REC, nmax=nmax, nk=nk)
    return q / float(lattice_constant)


def wannier_quartic_2d_square(depth, lattice_constant=1.0, nmax=40, nk=512):
    """``int |w(x,y)|^4 d^2r`` for a separable square lattice, caller units.

    Separability gives ``w(x, y) = w(x) w(y)``, so this is the 1D value squared.
    """
    return wannier_quartic_1d(depth, lattice_constant, nmax, nk) ** 2


def matched_square_depth_from_quartic(target_quartic_2d, lattice_constant,
                                      alpha=1.0, bracket=(0.5, 400.0),
                                      nmax=40, nk=512):
    """Depth (in recoils) of a square lattice with a given ``int|w|^4 d^2r``.

    This is the bridge used to attach the exact calculation to a *non-separable*
    lattice (honeycomb, triangular, ...): pick the separable reference lattice
    that reproduces the real lattice's on-site Wannier overlap, so that the
    leading-Born ``U`` of the two agree exactly and any difference between the
    exact and Born results is a pure beyond-Born correction.

    ``lattice_constant`` should be the real lattice's site spacing, so the
    reference also has a comparable band width.

    Raises
    ------
    ValueError
        If the target is not reachable inside ``bracket``.
    """
    from scipy.optimize import brentq

    target = float(target_quartic_2d)

    def f(depth):
        return wannier_quartic_2d_square(depth, lattice_constant, nmax, nk) - target

    lo, hi = bracket
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        raise ValueError(
            f"int|w|^4 = {target:.6g} is not reachable by a square lattice of "
            f"spacing {lattice_constant:.6g} for depths in {bracket} "
            f"(reachable range {target + flo:.6g} .. {target + fhi:.6g})."
        )
    return float(brentq(f, lo, hi, xtol=1e-10))


def matched_square_depth(omega_site, lattice_constant, alpha=1.0):
    """Depth (in recoils) of a sinusoidal lattice with a given on-site frequency.

    Inverts :func:`site_trap_frequency`: ``depth = (hbar omega / 2 E_rec)^2``.
    Used to build a separable reference lattice matched to a non-separable one.
    """
    conv = UnitConverter(lattice_constant, alpha)
    return (np.asarray(omega_site, dtype=float)
            / (2.0 * conv.recoil_energy)) ** 2
