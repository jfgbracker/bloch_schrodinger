"""Shared fixtures. Keeps the suite headless and quiet."""

import matplotlib
import numpy as np
import pytest
import xarray as xr

from bloch_schrodinger.frames import set_cart_factors
from bloch_schrodinger.potential import Potential

matplotlib.use("Agg")

# The recipe of the 'factored' fixture: an invariant lattice times one scale per frame
FACTORS = {"x": ("rho_x", "lambda_x"), "y": ("rho_y", "lambda_y")}


@pytest.fixture(autouse=True)
def _quiet_display(monkeypatch):
    """The plotting functions display ipywidgets on call, which is noise outside a kernel."""
    import bloch_schrodinger.plotting as plotting

    monkeypatch.setattr(plotting, "display", lambda *a, **k: None)


@pytest.fixture
def factored():
    """A moving grid stored as an invariant lattice times a per-frame scale.

    The cartesian coordinates are dropped and the recipe for rebuilding them is left in an
    attribute instead, as a solver stores a frame it does not want to materialise: written out, it
    would cost one full-size array per axis per frame. Deliberately not square, so that a
    transposition cannot pass unnoticed.
    """
    pot = Potential(unitvecs=[[8, 0], [0, 10]], resolution=(12, 14), v0=0)
    rho_x, rho_y = xr.broadcast(pot.x, pot.y)
    tv = np.linspace(0, 1, 4)
    t = xr.DataArray(tv, dims="t", coords={"t": tv})
    lam = 1 + 0.7 * t
    field = (np.exp(-(rho_x**2 + rho_y**2) / 4) * (1 + 0 * t)).transpose("t", "a1", "a2")
    out = field.rename("f").drop_vars(["x", "y"]).assign_coords(
        rho_x=rho_x.drop_vars(["x", "y"], errors="ignore"),
        rho_y=rho_y.drop_vars(["x", "y"], errors="ignore"),
        lambda_x=lam,
        lambda_y=lam * 1.1,
    )
    return set_cart_factors(out, FACTORS)


@pytest.fixture
def materialised(factored):
    """The same moving grid with its cartesian coordinates written out in full, and no recipe.

    Built by hand rather than with 'frames.materialize', which is tested against it.
    """
    out = factored.assign_coords(
        x=(factored.rho_x * factored.lambda_x).transpose("t", "a1", "a2"),
        y=(factored.rho_y * factored.lambda_y).transpose("t", "a1", "a2"),
    )
    out.attrs = {}
    return out


def harmonic(potential, omegas):
    """Set an anisotropic harmonic trap, whose spectrum is known analytically."""
    potential.set(
        sum(
            omegas[i] ** 2 / 2 * coo**2 for i, coo in enumerate(potential.coords)
        )
    )
    return potential


def analytic_levels(omegas, n_levels):
    """The lowest n_levels of sum_i omega_i (n_i + 1/2), for alpha = hbar^2/2m = 1/2."""
    grid = np.arange(12)
    mesh = np.meshgrid(*[grid] * len(omegas), indexing="ij")
    energies = sum(
        omegas[i] * (mesh[i] + 0.5) for i in range(len(omegas))
    ).ravel()
    return np.sort(energies)[:n_levels]
