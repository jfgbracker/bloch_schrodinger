"""plot_eigenvector(show_minima=...): the minima of each subplot's potential, marked on it.

Anchored to Potential.find_minima: the scatter must show exactly what it finds at the current
slider position, and follow the sliders.
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402

from bloch_schrodinger.plotting import plot_eigenvector  # noqa: E402
from bloch_schrodinger.potential import Potential, create_parameter  # noqa: E402

L = 4.0


def wells_2d():
    """Two wells per cell along x, whose depth ratio is a parameter."""
    p = Potential(unitvecs=[[L, 0], [0, L]], resolution=(40, 40), v0=0)
    ratio = create_parameter("ratio", [0.5, 1.0, 2.0])
    p.set(-np.cos(2 * np.pi * p.y / L) * (1 + ratio * np.cos(2 * np.pi * p.x / L)
                                           + np.cos(4 * np.pi * p.x / L)))
    return p


def field_on(pot, extra=None):
    """A stand-in eigenvector on the potential's grid, optionally with a dim of its own."""
    f = np.exp(-(pot.x**2 + pot.y**2))
    return f.expand_dims(band=[0, 1]) if extra else f


def scatter_of(ax):
    return next(c for c in ax.collections if c.get_label() == "minima")


def expected(pot, **sel):
    x, y, *_ = pot.find_minima(sel)
    pts = np.column_stack([np.asarray(x).ravel(), np.asarray(y).ravel()])
    return pts[~np.isnan(pts).any(axis=1)]


def same_points(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    key = lambda p: np.lexsort(p.T[::-1])  # noqa: E731
    return a.shape == b.shape and np.allclose(a[key(a)], b[key(b)])


def sliders_of(fig_axes_call, monkeypatch):
    """Run a plot_eigenvector call and capture the slider widgets it displays."""
    shown = {}
    monkeypatch.setattr("bloch_schrodinger.plotting.display",
                        lambda box: shown.setdefault("box", box))
    fig, axes = fig_axes_call()
    widgets = {w.description: w for w in shown["box"].children if hasattr(w, "description")}
    return fig, axes, widgets


def test_minima_on_a_map_are_the_potentials_and_follow_the_sliders(monkeypatch):
    pot = wells_2d()
    fig, axes, widgets = sliders_of(lambda: plot_eigenvector(
        [[field_on(pot)]], [[pot]], show_minima=True, minima_kwargs={"label": "minima"},
    ), monkeypatch)
    sc = scatter_of(axes[0][0])
    assert same_points(sc.get_offsets(), expected(pot, ratio=widgets["ratio"].value))
    widgets["ratio"].value = 2.0
    assert same_points(sc.get_offsets(), expected(pot, ratio=2.0))


def test_sliders_the_potential_does_not_have_are_ignored(monkeypatch):
    """The eigenvector's own 'band' slider must not be passed to the potential's selection."""
    pot = wells_2d()
    fig, axes, widgets = sliders_of(lambda: plot_eigenvector(
        [[field_on(pot, extra=True)]], [[pot]], show_minima=True,
        minima_kwargs={"label": "minima"},
    ), monkeypatch)
    assert "band" in widgets
    widgets["band"].value = 1
    assert same_points(scatter_of(axes[0][0]).get_offsets(),
                       expected(pot, ratio=widgets["ratio"].value))


def test_per_subplot_matrix_marks_only_the_chosen_subplots():
    pot = wells_2d()
    _, axes = plot_eigenvector([[field_on(pot), field_on(pot)]], [[pot, pot]],
                               show_minima=[[True, False]], minima_kwargs={"label": "minima"})
    assert any(c.get_label() == "minima" for c in axes[0][0].collections)
    assert not any(c.get_label() == "minima" for c in axes[0][1].collections)


def test_line_plot_marks_the_minima_on_the_potential_curve():
    pot = Potential(unitvecs=[[L]], resolution=(64,), v0=0)
    pot.set(-np.cos(2 * np.pi * pot.coords[0] / L) - 0.5 * np.cos(4 * np.pi * pot.coords[0] / L))
    field = np.exp(-pot.V.x**2)
    fig, axes = plot_eigenvector([[field]], [[pot]], cart_axes=[0], show_minima=True,
                                 minima_kwargs={"label": "minima"})
    twin = next(a for a in fig.axes if a is not axes[0][0] and a.collections)
    pts = scatter_of(twin).get_offsets()
    x, v, _ = pot.find_minima()
    assert same_points(pts, np.column_stack([np.asarray(x), np.asarray(v)]))


def test_subplots_that_cannot_show_minima_are_skipped():
    """A cut through the 2D potential, and a subplot without a potential, in the same grid as a
    full map: show_minima=True marks the map and leaves the other two alone."""
    pot = wells_2d()
    _, axes = plot_eigenvector(
        [[field_on(pot), field_on(pot), field_on(pot)]], [[pot, pot, None]],
        cart_axes=[[[0, 1], [0], [0, 1]]], show_minima=True, minima_kwargs={"label": "minima"},
    )
    fig = axes[0][0].figure
    marked = [a for a in fig.axes if any(c.get_label() == "minima" for c in a.collections)]
    assert marked == [axes[0][0]]


def test_show_minima_matrix_must_match_plots():
    pot = wells_2d()
    with pytest.raises(ValueError, match="same shape"):
        plot_eigenvector([[field_on(pot)]], [[pot]], show_minima=[[True, True]])

def test_minima_of_a_tiled_potential_are_where_its_wells_are():
    """Regression: find_minima mapped lattice coordinates through the unit vectors, which
    Potential.tile scales by the number of cells while leaving V's a1, a2 in the original cell's
    units -- so on a tiled potential every minimum came out that many times too far out."""
    pot = wells_2d().sel({"ratio": 1.0})
    tiled = pot.tile([(-1, 2), (-1, 2)])
    x, y, v, _ = tiled.find_minima()
    pts = np.column_stack([np.asarray(x), np.asarray(y)])
    pts = pts[~np.isnan(pts).any(axis=1)]
    # every minimum lies on the tiled block (one on the periodic seam is half a step past the last
    # grid point: it is the mean of the pixels either side of the seam)...
    grid = np.column_stack([np.asarray(tiled.V.x).ravel(), np.asarray(tiled.V.y).ravel()])
    half_step = 0.5 * L / 40
    assert np.abs(pts).max() <= np.abs(grid).max() + half_step + 1e-9
    # ...and the single cell's minima, repeated over the 3x3 block, are all among them
    x1, y1, *_ = pot.find_minima()
    for px, py in zip(np.asarray(x1), np.asarray(y1)):
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                d = np.hypot(pts[:, 0] - (px + i * L), pts[:, 1] - (py + j * L))
                assert d.min() < 1e-6
