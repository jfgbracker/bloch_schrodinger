"""Unit conversion between this package's conventions and the Adlong solver's.

The two live in different unit systems and mixing them silently gives
plausible-looking but wrong numbers, so every conversion goes through here.

The two systems
---------------
``bloch_schrodinger`` writes the single-particle Hamiltonian as

    H = alpha * k^2 + V ,

with ``alpha`` the kinetic coefficient the user supplies to ``FDSolver`` /
``PWSolver`` / ``Wannier``.  Lengths are in some unit ``L0`` and energies in
some unit ``E0``, both fixed implicitly by the caller (``nninteractions``, for
instance, uses ``alpha = 1``, ``L0 = 1/k_las``, ``E0 = E_rec``).

The Adlong solver (:mod:`bloch_schrodinger.hubbard.quasi1d`, ``.quasi2d``)
works in ``m = hbar = d = 1``, i.e.

    H = k^2 / 2 + V ,

with lengths in units of the lattice spacing ``d`` and energies in units of
``hbar^2 / (m d^2)``.

Deriving the factors
--------------------
Physically ``H = (hbar^2/2m) k^2 + V``.  Matching the caller's form,

    alpha * E0 = hbar^2 / (2 m L0^2)    =>    hbar^2/(2m) = alpha * E0 * L0^2 .

The Adlong energy unit is ``hbar^2/(m d^2)``.  Writing the lattice spacing in
caller length units as ``d = d_c * L0``,

    hbar^2/(m d^2) = 2 * alpha * E0 * L0^2 / (d_c^2 L0^2) = 2 * alpha * E0 / d_c^2 ,

so an energy expressed in caller units converts as

    E_adlong = E_caller * d_c^2 / (2 alpha) ,

and a length as ``L_adlong = L_caller / d_c``.

Sanity check (used as a test): the recoil energy of a lattice of spacing ``d``
is ``E_rec = hbar^2 (pi/d)^2 / 2m``, i.e. ``alpha (pi/d_c)^2`` in caller units.
Converting gives ``alpha (pi/d_c)^2 * d_c^2/(2 alpha) = pi^2/2``, which is
exactly the ``V_REC`` the Adlong solver uses.  Any change to these factors that
breaks that identity is wrong.

Note that *depths quoted in recoil units are dimensionless* and therefore need
no conversion at all -- which is why the public API takes lattice depths that
way.
"""

from __future__ import annotations

import numpy as np

__all__ = ["UnitConverter"]


class UnitConverter:
    """Convert energies and lengths between caller units and the Adlong solver.

    Parameters
    ----------
    lattice_constant : float
        The lattice spacing ``d``, in the caller's length units.
    alpha : float
        The caller's kinetic coefficient, ``H = alpha k^2 + V`` (so
        ``alpha = hbar^2/2m`` in the caller's units).  ``bloch_schrodinger``
        code typically uses ``alpha = 1``.
    """

    def __init__(self, lattice_constant: float, alpha: float = 1.0):
        d_c = float(lattice_constant)
        alpha = float(alpha)
        if not np.isfinite(d_c) or d_c <= 0:
            raise ValueError(f"lattice_constant must be positive, got {d_c}")
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        self.lattice_constant = d_c
        self.alpha = alpha
        # E_adlong = E_caller * d_c^2 / (2 alpha)
        self._energy_to = d_c * d_c / (2.0 * alpha)

    # -- energies ---------------------------------------------------------
    def energy_to_adlong(self, e):
        """Caller energy units -> ``hbar^2/(m d^2)``."""
        return np.asarray(e, dtype=float) * self._energy_to

    def energy_from_adlong(self, e):
        """``hbar^2/(m d^2)`` -> caller energy units."""
        return np.asarray(e, dtype=float) / self._energy_to

    # -- lengths ----------------------------------------------------------
    def length_to_adlong(self, x):
        """Caller length units -> units of the lattice spacing ``d``."""
        return np.asarray(x, dtype=float) / self.lattice_constant

    def length_from_adlong(self, x):
        """Units of the lattice spacing ``d`` -> caller length units."""
        return np.asarray(x, dtype=float) * self.lattice_constant

    def inverse_length_to_adlong(self, x):
        """Caller inverse-length units -> units of ``1/d`` (e.g. ``1/a_1D``)."""
        return np.asarray(x, dtype=float) * self.lattice_constant

    # -- derived quantities ----------------------------------------------
    @property
    def recoil_energy(self) -> float:
        """The lattice recoil ``E_rec = alpha (pi/d_c)^2``, in caller units."""
        return self.alpha * (np.pi / self.lattice_constant) ** 2

    def __repr__(self) -> str:
        return (f"UnitConverter(lattice_constant={self.lattice_constant:.6g}, "
                f"alpha={self.alpha:.6g}, E_rec={self.recoil_energy:.6g})")
