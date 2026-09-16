"""Tests for Potential.find_minima / Potential.smooth and the minima overlays on the plots.

These cover functionality that lived only in a local working copy until it was ported onto the n-D
Potential, so the checks are written against landscapes whose minima are known by construction
rather than against whatever the implementation currently returns.
"""

import numpy as np
import pytest
from scipy.ndimage import minimum_filter

from bloch_schrodinger.potential import (
    Potential,
    create_parameter,
    optical_honeycomb,
)

L = 4.0


def single_well_1d(resolution=33):
    """-cos(2 pi x / L) over one cell: one minimum, at x = 0.

    An odd resolution is deliberate. The spatial grid is a half-pixel-offset linspace, so x = 0 is a
    grid point only for an odd number of points; with an even one the minimum falls between two
    pixels and is reported twice, which says nothing about the code under test.
    """
    p = Potential(unitvecs=[[L]], resolution=(resolution,), v0=0)
    p.set(-np.cos(2 * np.pi * p.x / L))
    return p


def single_well_2d(resolution=(33, 33)):
    """-cos(2 pi x / L) - cos(2 pi y / L): one minimum, at the origin."""
    p = Potential(unitvecs=[[L, 0], [0, L]], resolution=resolution, v0=0)
    p.set(-np.cos(2 * np.pi * p.x / L) - np.cos(2 * np.pi * p.y / L))
    return p


# --------------------------------------------------------------------------------------
# find_minima
# --------------------------------------------------------------------------------------


def test_finds_the_one_minimum_of_a_1d_well():
    x, v, coords = single_well_1d().find_minima()

    assert x.shape == (1,)
    assert np.allclose(x.values, 0.0)
    assert np.allclose(v.values, -1.0)
    assert np.allclose(coords.a1.values, 0.0)


def test_finds_the_one_minimum_of_a_2d_well():
    x, y, v, coords = single_well_2d().find_minima()

    assert x.shape == (1,)
    assert np.allclose([x.values, y.values], 0.0)
    assert np.allclose(v.values, -2.0)


def test_returns_one_coordinate_per_dimension():
    """The 2D call stays the familiar `x, y, v, coords`, with 1D and 3D one entry shorter/longer."""
    p3d = Potential(unitvecs=np.eye(3).tolist(), resolution=(9, 9, 9), v0=0)
    p3d.set(-np.cos(2 * np.pi * p3d.x) - np.cos(2 * np.pi * p3d.y) - np.cos(2 * np.pi * p3d.z))

    assert len(single_well_1d().find_minima()) == 3
    assert len(single_well_2d().find_minima()) == 4
    assert len(p3d.find_minima()) == 5
    assert list(p3d.find_minima()[-1].data_vars) == ["a1", "a2", "a3"]


def test_lattice_coordinates_select_the_reported_values():
    """The trailing Dataset is the documented way to sample another field at the lattice sites."""
    p = single_well_2d()
    x, y, v, coords = p.find_minima()

    resampled = p.V.sel(a1=coords.a1, a2=coords.a2, method="nearest")

    assert np.allclose(resampled.values, v.values)


def test_parameter_dimensions_survive_and_pad_with_nan():
    s = create_parameter("s", [0.0, 4.0, 12.0])
    _, p = optical_honeycomb(wavelength=1.064, s1=s, s2=s, resolution=(80, 80))
    p.smooth(1, 1)

    x, y, v, coords = p.find_minima()

    assert x.dims == ("s", "minima")
    # s = 0 is a flat landscape: it must contribute no minima rather than every pixel, and the
    # slices that do have minima are padded with NaN up to the largest count.
    counts = [int(np.isfinite(x.isel(s=i)).sum()) for i in range(3)]
    assert counts[0] == 0
    assert counts[1] == counts[2] == 2
    assert np.all(np.isnan(x.isel(s=0).values))


def test_selection_reduces_the_parameter_space():
    s = create_parameter("s", [4.0, 12.0])
    _, p = optical_honeycomb(wavelength=1.064, s1=s, s2=s, resolution=(64, 64))

    x, _, _, _ = p.find_minima({"s": 12.0})

    assert x.dims == ("minima",)


@pytest.mark.parametrize("mode", ["wrap", "nearest"])
@pytest.mark.parametrize("size", [3, 5])
def test_matches_a_direct_minimum_filter_for_every_size_and_mode(size, mode):
    """'size' and 'mode' must reach scipy untouched, and the n-D rewrite must pick out exactly the
    pixels a plain 2D minimum_filter would: anything else means the reshuffling is losing or
    inventing sites."""
    p = Potential(unitvecs=[[L, 0], [0, L]], resolution=(24, 24), v0=0)
    # Asymmetric on purpose, so the two boundary conventions genuinely disagree.
    p.set(
        -np.cos(2 * np.pi * p.x / L + 0.4)
        - 0.6 * np.cos(4 * np.pi * p.y / L)
        + 0.3 * p.x * p.y / L**2
    )

    data = np.real(p.V.transpose("a1", "a2").values)
    expected = data == minimum_filter(data, size=size, mode=mode)

    x, y, v, _ = p.find_minima(size=size, mode=mode)

    assert int(np.isfinite(x).sum()) == int(expected.sum())
    assert np.allclose(np.sort(v.values), np.sort(data[expected]))


def test_the_two_boundary_modes_disagree_on_an_edge_minimum():
    """Guards the check above from going vacuous: 'wrap' and 'nearest' must not be interchangeable."""
    p = Potential(unitvecs=[[L]], resolution=(32,), v0=0)
    p.set(np.cos(2 * np.pi * p.x / L + 0.4))  # minimum sits against the cell edge

    n_wrap = int(np.isfinite(p.find_minima(mode="wrap")[0]).sum())
    n_nearest = int(np.isfinite(p.find_minima(mode="nearest")[0]).sum())

    assert n_wrap == 1  # the edge is a single site once the cell is closed up
    assert n_nearest == 2  # cut open, both ends of the cell look like minima


# --------------------------------------------------------------------------------------
# smooth
# --------------------------------------------------------------------------------------


def test_smoothing_does_not_mix_parameter_values():
    """Each parameter slice must be filtered on its own, not blurred into its neighbours."""
    s = create_parameter("s", [0.0, 4.0, 12.0])
    _, p = optical_honeycomb(wavelength=1.064, s1=s, s2=s, resolution=(48, 48))

    stacked = p.copy()
    stacked.smooth(2, 2)

    for i, value in enumerate([0.0, 4.0, 12.0]):
        alone = p.sel({"s": value})
        alone.smooth(2, 2)
        assert np.allclose(
            stacked.V.isel(s=i).transpose("a1", "a2").values,
            alone.V.transpose("a1", "a2").values,
        )

    # The s = 0 slice is identically zero, so smoothing must leave it that way.
    assert np.allclose(stacked.V.isel(s=0).values, 0.0)


def test_space_units_are_pixel_units_scaled_by_the_grid_spacing():
    p = single_well_2d(resolution=(32, 32))

    in_pixels = p.copy()
    in_pixels.smooth(2, 3)

    in_space = p.copy()
    in_space.smooth(2 * p.da[0], 3 * p.da[1], unit="space")

    assert np.allclose(in_pixels.V.values, in_space.V.values)


def test_a_single_strength_applies_to_every_axis():
    p = single_well_2d(resolution=(32, 32))

    broadcast, explicit, as_tuple = p.copy(), p.copy(), p.copy()
    broadcast.smooth(2)
    explicit.smooth(2, 2)
    as_tuple.smooth((2, 2))

    assert np.allclose(broadcast.V.values, explicit.V.values)
    assert np.allclose(as_tuple.V.values, explicit.V.values)


@pytest.mark.parametrize(
    "call, message",
    [
        (lambda p: p.smooth(1, 2, 3), "expects 1 or n_dims=2"),
        (lambda p: p.smooth(1, 1, unit="furlongs"), "'space' or 'pixel'"),
    ],
)
def test_smooth_rejects_bad_arguments(call, message):
    with pytest.raises(ValueError, match=message):
        call(single_well_2d(resolution=(16, 16)))


# --------------------------------------------------------------------------------------
# plotting overlays
# --------------------------------------------------------------------------------------


def test_plot_overlays_the_minima():
    p = single_well_2d()

    _, ax_plain = p.plot()
    _, ax_minima = p.plot(show_minima=True)

    assert len(ax_minima.collections) == len(ax_plain.collections) + 1


def test_plot_3d_runs_and_can_overlay_the_minima():
    p = single_well_2d()

    fig, ax, cbar = p.plot_3d(get_cbar=True, show_minima=True)

    assert cbar is not None
    assert ax.get_zlabel() == "Potential"


def test_minima_overlay_is_refused_on_a_cut():
    """Minima of a 3D landscape do not lie on a 2D slice of it, so drawing them would be a lie."""
    p3d = Potential(unitvecs=np.eye(3).tolist(), resolution=(8, 8, 8), v0=0)

    with pytest.raises(ValueError, match="only shows a cut"):
        p3d.plot(cart_axes=[0, 1], show_minima=True)


def test_plot_3d_is_refused_outside_2d():
    p3d = Potential(unitvecs=np.eye(3).tolist(), resolution=(8, 8, 8), v0=0)

    with pytest.raises(ValueError, match="n_dims=3"):
        p3d.plot_3d()
