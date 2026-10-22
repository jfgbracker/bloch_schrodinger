"""The cartesian frame of a field, in the three forms it may be stored in.

Every reader of a field's coordinates goes through the 'frames' module, so the property that
matters most is that the dense and the factored forms of the same moving grid always give the
same answer -- whichever of them a caller happens to hold.
"""

import json

import numpy as np
import pytest
import xarray as xr

from bloch_schrodinger.frames import (
    cart_coord,
    cart_dims,
    cart_extent,
    cart_factors,
    lattice_dims,
    materialize,
    moving_dims,
    set_cart_factors,
)
from bloch_schrodinger.plotting import plot_isosurface
from bloch_schrodinger.potential import Potential

# --- reading and writing the recipe ------------------------------------------------------------

def test_lattice_dims_are_sorted_by_axis_index():
    data = xr.DataArray(np.zeros((1, 1, 1, 1, 1)), dims=("t", "a10", "a2", "a1", "alpha"))
    assert lattice_dims(data) == ["a1", "a2", "a10"]


def test_the_recipe_round_trips_through_the_attribute(factored, materialised):
    """Stored as JSON, since a netCDF attribute cannot hold a mapping, and read back as tuples."""
    assert cart_factors(factored) == {"x": ("rho_x", "lambda_x"), "y": ("rho_y", "lambda_y")}
    assert isinstance(factored.attrs["cart_factors"], str)
    assert json.loads(factored.attrs["cart_factors"])["x"] == ["rho_x", "lambda_x"]
    assert cart_factors(materialised) == {}


def test_declaring_factors_drops_the_dense_coordinate(materialised):
    """The recipe and the coordinate must never both be written, or a reader has two answers."""
    out = set_cart_factors(materialised, {"x": ("rho_x", "lambda_x")})
    assert "x" not in out.coords and "y" in out.coords
    assert cart_factors(out) == {"x": ("rho_x", "lambda_x")}
    # and the caller's array is left as it was
    assert "x" in materialised.coords and materialised.attrs == {}


def test_declaring_factors_adds_to_an_existing_recipe(materialised):
    out = set_cart_factors(materialised, {"x": ("rho_x", "lambda_x")})
    out = set_cart_factors(out, {"y": ("rho_y", "lambda_y")})
    assert cart_factors(out) == {"x": ("rho_x", "lambda_x"), "y": ("rho_y", "lambda_y")}


def test_declaring_an_unknown_factor_raises(factored):
    with pytest.raises(ValueError, match="not coordinates"):
        set_cart_factors(factored, {"x": ("rho_x", "nope")})


# --- the two forms agree -----------------------------------------------------------------------

def test_factored_coordinate_matches_the_materialised_one(factored, materialised):
    for tv in factored.t.values:
        sel = {"t": float(tv)}
        for name in ("x", "y"):
            lean = cart_coord(factored, name, sel)
            full = cart_coord(materialised, name, sel)
            assert lean.dims == ("a1", "a2"), "one frame, not the whole run"
            assert np.array_equal(lean.values, full.transpose(*lean.dims).values)

    # Without a selection, the whole coordinate over every frame
    whole = cart_coord(factored, "x")
    assert np.array_equal(
        whole.transpose("t", "a1", "a2").values, materialised.x.transpose("t", "a1", "a2").values
    )


def test_a_stored_coordinate_wins_over_a_recipe(materialised):
    both = materialised.copy()
    # A deliberately wrong recipe: it must never be consulted while the coordinate is there
    both.attrs["cart_factors"] = json.dumps({"x": ["rho_x", "rho_y"]})
    assert np.array_equal(cart_coord(both, "x").values, materialised.x.values)
    assert cart_dims(both, "x") == materialised.x.dims


def test_dims_and_motion_are_read_through_either_form(factored, materialised):
    assert cart_dims(factored, "x") == ("a1", "a2", "t")
    assert moving_dims(factored, "x") == ["t"]
    assert moving_dims(materialised, "x") == ["t"]

    static = Potential(unitvecs=[[8, 0], [0, 10]], resolution=(6, 7), v0=0)
    assert moving_dims(static.V, "x") == []


def test_an_unknown_coordinate_says_so(factored):
    with pytest.raises(KeyError, match="cart_factors"):
        cart_dims(factored, "z")
    with pytest.raises(KeyError, match="cart_factors"):
        cart_coord(factored, "z")


def test_cart_extent_matches_the_materialised_product(factored, materialised):
    """The span of a factored coordinate is taken from its factors, never by multiplying them out.

    Building the product to measure it would cost exactly the array the factored form exists to
    avoid, so the corners of the factors are used instead. That shortcut is only worth anything
    if it gives the same answer, which is what this checks.
    """
    for name in ("x", "y"):
        assert cart_extent(factored, name) == pytest.approx(cart_extent(materialised, name))


# --- materialize -------------------------------------------------------------------------------

def test_materialize_rebuilds_the_coordinates_and_drops_the_recipe(factored, materialised):
    out = materialize(factored)
    assert "cart_factors" not in out.attrs
    for name in ("x", "y"):
        assert set(out[name].dims) == {"a1", "a2", "t"}
        assert np.allclose(
            out[name].transpose("t", "a1", "a2").values, materialised[name].values
        )


def test_materialize_spans_the_full_mesh_even_for_a_one_axis_factor():
    """A factor running along its own axis only would give a coordinate the plotters cannot use."""
    a1, a2 = np.linspace(-1, 1, 5), np.linspace(-2, 2, 6)
    tv = np.linspace(0, 1, 3)
    field = xr.DataArray(
        np.ones((3, 5, 6)), dims=("t", "a1", "a2"), coords={"t": tv, "a1": a1, "a2": a2}
    ).assign_coords(
        x0=("a1", a1), s=("t", 1 + tv)
    )
    out = materialize(set_cart_factors(field, {"x": ("x0", "s")}))
    assert set(out.x.dims) == {"a1", "a2", "t"}
    assert np.allclose(out.x.sel(t=1.0).transpose("a1", "a2").values[:, 0], 2 * a1)


def test_materialize_can_keep_part_of_the_recipe(factored):
    out = materialize(factored, ["x"])
    assert "x" in out.coords and "y" not in out.coords
    assert cart_factors(out) == {"y": ("rho_y", "lambda_y")}


def test_materialize_refuses_a_name_without_a_recipe(materialised):
    with pytest.raises(ValueError, match="no 'cart_factors' recipe"):
        materialize(materialised, ["x"])


def test_materialize_stays_lazy(factored):
    """An array opened lazily must not be computed just to be given its coordinates."""
    pytest.importorskip("dask")
    lazy = factored.assign_coords(
        lambda_x=factored.lambda_x.chunk(), lambda_y=factored.lambda_y.chunk()
    )
    out = materialize(lazy)
    assert type(out.x.data).__module__.startswith("dask")


# --- a consumer that used to need the dense form -------------------------------------------------

def test_isosurface_reads_a_factored_frame():
    """plot_isosurface regridded each moving frame from dense coordinates it never checked for.

    A factored field went looking for an 'x' that was deliberately not stored, at the first frame
    and again when pinning the scene. Both forms of the same grid must now draw the same surface in
    the same box.
    """
    pot = Potential(unitvecs=[[6, 0, 0], [0, 8, 0], [0, 0, 4]], resolution=(10, 12, 8), v0=0)
    rho = xr.broadcast(pot.x, pot.y, pot.z)
    tv = np.linspace(0, 1, 3)
    lam = xr.DataArray(1 + 0.5 * tv, dims="t", coords={"t": tv})
    field = np.exp(-(rho[0] ** 2 + rho[1] ** 2 + rho[2] ** 2) / 3) * (1 + 0 * lam)
    field = field.drop_vars(["x", "y", "z"]).assign_coords(
        {f"rho_{n}": c.drop_vars(["x", "y", "z"], errors="ignore") for n, c in zip("xyz", rho)}
    ).assign_coords({f"lambda_{n}": lam * (1 + 0.1 * i) for i, n in enumerate("xyz")})
    factored = set_cart_factors(field, {n: (f"rho_{n}", f"lambda_{n}") for n in "xyz"})

    lean = plot_isosurface(factored, cst_bds=True)
    full = plot_isosurface(materialize(factored), cst_bds=True)

    assert np.allclose(lean.data[0].x, full.data[0].x)
    for axis in ("xaxis", "yaxis", "zaxis"):
        assert lean.layout.scene[axis].range == pytest.approx(full.layout.scene[axis].range)
    # pinned to the largest frame, which is the last one here
    widest = float(factored.lambda_x.max())
    assert lean.layout.scene.xaxis.range[1] == pytest.approx(float(pot.x.max()) * widest)
