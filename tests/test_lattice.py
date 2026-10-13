"""bloch_schrodinger.lattice: geometries, snapping found minima onto their sites, the schematic
overlay, and the 'lattice' option of the plotting functions."""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from bloch_schrodinger.lattice import LatticeGeometry, draw_lattice  # noqa: E402
from bloch_schrodinger.plotting import dashboard, plot_eigenvector  # noqa: E402
from bloch_schrodinger.potential import (  # noqa: E402
    create_parameter,
    honeycomb,
    optical_honeycomb,
)

S3 = 3**0.5
A60 = (np.array([1.5, -S3 / 2]), np.array([1.5, S3 / 2]))       # |a| = sqrt(3), a_AB = 1


def labels_at(ax):
    return sorted({t.get_text() for t in ax.texts if t.get_text() in ("A", "B", "C")})


# --- geometries -----------------------------------------------------------------------------

def test_hexagonal_sites_and_bases():
    g = LatticeGeometry.hexagonal(*A60, origin=(-0.5, 0))
    assert np.allclose(g.position("A"), [-0.5, 0]) and np.allclose(g.position("B"), [0.5, 0])
    assert g.nearest([[1.5, 0]])[0] == ["C"]                       # the hexagon centre
    assert g.nn_distance() == pytest.approx(1.0)
    assert g.nn_distance(["A"]) == pytest.approx(S3)                # one sublattice: triangular
    # the same lattice from a 120-degree basis has the same three points
    g120 = LatticeGeometry.hexagonal(A60[0], A60[1] - A60[0], origin=(-0.5, 0))
    for label in "ABC":
        assert g120.nearest([g.position(label)])[1][0] < 1e-12
    # ab picks which neighbour is B
    gs = LatticeGeometry.hexagonal(*A60, origin=(-0.5, 0), ab=(-1, 0))
    assert gs.nearest([[-1.5, 0]])[0] == ["B"] and gs.nearest([[0.5, 0]])[0] == ["C"]
    with pytest.raises(ValueError, match="not a vector"):
        LatticeGeometry.hexagonal(*A60, ab=(0.3, 0))


@pytest.mark.parametrize("kind, a1, a2, nn, n_sites", [
    ("square", [1, 0], [0, 1], 1.0, 1),
    ("checkerboard", [1, 1], [1, -1], 1.0, 2),
    ("kagome", [1, 0], [0.5, S3 / 2], 0.5, 3),
    ("lieb", [1, 0], [0, 1], 0.5, 3),
])
def test_presets(kind, a1, a2, nn, n_sites):
    g = LatticeGeometry.preset(kind, a1, a2)
    assert len(g.labels) == n_sites and g.nn_distance() == pytest.approx(nn)


def test_presets_refuse_the_wrong_lattice():
    with pytest.raises(ValueError, match="degrees"):
        LatticeGeometry.square([1, 0], [0.6, 0.8])
    with pytest.raises(ValueError, match=r"\|a1\| = \|a2\|"):
        LatticeGeometry.hexagonal([1, 0], [1, 2])
    with pytest.raises(ValueError, match="unknown geometry"):
        LatticeGeometry.preset("honeycomb3d", [1, 0], [0, 1])


# --- snapping -------------------------------------------------------------------------------

def test_snap_moves_close_points_and_keeps_the_others():
    g = LatticeGeometry.hexagonal(*A60, origin=(-0.5, 0))
    pts = np.array([[0.53, -0.02],             # B, slightly off
                    [-0.5 + 3.0, 0.04],        # A, a lattice vector away
                    [0.0, 0.0],                # a bond centre: no site
                    [np.nan, np.nan]])         # padding
    with pytest.warns(UserWarning, match="1 point"):
        r = g.snap(pts, tol=0.1)
    assert r.labels == ["B", "A", "", ""]
    assert np.allclose(r.positions[0], [0.5, 0]) and np.allclose(r.positions[1], [2.5, 0])
    assert np.allclose(r.positions[2], [0, 0])                         # kept where found
    assert list(r.on_site) == [True, True, False, False]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g.snap(pts[:2])                                                 # nothing to warn about


def test_find_sites_on_the_optical_honeycomb():
    c, pot = optical_honeycomb(1.0, 3.0, 5.0, resolution=(48, 48))
    a = c["a"]
    sites = pot.find_sites()
    assert sorted(sites.label.values) == ["A", "B"] and sites.on_site.all()
    A = sites.where(sites.label == "A", drop=True)
    assert np.allclose([A.x.item(), A.y.item()], [-a / 2, 0], atol=1e-12)
    assert A.v.item() == sites.v.min().item()                          # A is the s2 (deeper) site
    assert (sites.distance > 0).any()                                  # the grid was off, the sites are not
    # a parameter axis is kept, and tiling keeps the geometry
    s1 = create_parameter("s1", [0.0, 5.0])
    c, pot = optical_honeycomb(1.0, s1, 5.0, resolution=(48, 48))
    found = pot.find_sites()
    assert set(found.label.sel(s1=0.0).values) - {""} == {"A"}         # a triangular lattice
    assert set(found.label.sel(s1=5.0).values) == {"A", "B"}
    assert pot.tile([[0, 2], [0, 2]]).geometry is pot.geometry


def test_pillar_honeycomb_names_its_pillars():
    h = honeycomb(1.0, 0.3, 0.3, res=(60, 60))
    assert h.geometry.nearest([[0, 0.5], [0, -0.5]])[0] == ["A", "B"]


# --- drawing --------------------------------------------------------------------------------

def test_overlay_fills_the_view_and_never_changes_it():
    g = LatticeGeometry.hexagonal(*A60, origin=(-0.5, 0))
    fig, ax = plt.subplots()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, 2)
    ov = draw_lattice(ax, g, occupied=["A", "B"], bonds=True, empty_sites=True, unit_cell=True)
    assert ax.get_xlim() == (-3, 3) and ax.get_ylim() == (-2, 2)
    n_before = sum(t.get_text() == "A" for t in ax.texts)
    assert n_before > 0 and labels_at(ax) == ["A", "B", "C"]           # C as an empty site
    ax.set_xlim(-6, 6)                                                  # zoom out: more sites
    assert sum(t.get_text() == "A" for t in ax.texts) > n_before
    bonds = [c for c in ax.collections if c.__class__.__name__ == "LineCollection"]
    assert len(bonds) == 1 and np.allclose(
        [np.linalg.norm(s[1] - s[0]) for s in bonds[0].get_segments()], 1.0)
    ov.draw(["A"], offsite_points=[[0.0, 0.0]])                         # one sublattice, one stray
    filled = sorted({t.get_text() for t in ax.texts if t.get_text() in ("A", "B", "C") and t.get_alpha() is None})
    assert filled == ["A"] and labels_at(ax) == ["A", "B", "C"]        # B, C now empty sites
    strays = [c for c in ax.collections if c.__class__.__name__ == "PathCollection"
              and np.allclose(c.get_offsets(), [[0.0, 0.0]])]
    assert len(strays) == 1
    plt.close(fig)


@pytest.fixture
def no_display(monkeypatch):
    import bloch_schrodinger.plotting as plotting
    shown = {}
    monkeypatch.setattr(plotting, "display", lambda box: shown.setdefault("box", box))
    import bloch_schrodinger.potential as potential
    monkeypatch.setattr(potential, "display", lambda box: shown.setdefault("box", box))
    return shown


def widgets_of(shown):
    return {w.description: w for w in shown["box"].children if hasattr(w, "description")}


def test_plot_eigenvector_lattice_follows_the_sliders(no_display):
    s1 = create_parameter("s1", [0.0, 5.0])
    c, pot = optical_honeycomb(1.0, s1, 5.0, resolution=(36, 36))
    fig, axes = plot_eigenvector([[pot.V]], [[pot]], templates=[[("real", None)]], lattice=True)
    ax = axes[0][0]
    assert not [x for x in ax.collections if x.__class__.__name__ == "QuadContourSet"]
    w = widgets_of(no_display)["s1"]
    w.value = 0.0
    assert labels_at(ax) == ["A"]
    w.value = 5.0
    assert labels_at(ax) == ["A", "B"]
    plt.close(fig)


def test_lattice_options_without_a_geometry_or_a_potential(no_display):
    c, pot = optical_honeycomb(1.0, 3.0, 5.0, resolution=(36, 36))
    geom, pot.geometry = pot.geometry, None
    with pytest.raises(ValueError, match="lattice=True needs"):
        plot_eigenvector([[pot.V]], [[pot]], lattice=True)
    # no potential: every site of the geometry is drawn
    fig, axes = plot_eigenvector([[pot.V]], [[None]], lattice=geom)
    assert labels_at(axes[0][0]) == ["A", "B", "C"]
    plt.close(fig)


def test_potential_plot_and_dashboard_take_the_lattice(no_display):
    c, pot = optical_honeycomb(1.0, 3.0, 5.0, resolution=(36, 36))
    fig, ax = pot.plot(lattice=True, lattice_kwargs={"bonds": True})
    assert labels_at(ax) == ["A", "B"]
    plt.close(fig)
    with pytest.raises(ValueError, match="2D potential drawn as an x, y map"):
        pot.plot(cart_axes=[0], lattice=True)
    eigva = create_parameter("band", [0, 1]) * 0 + create_parameter("ky", [0.0, 1.0])
    eigve = (pot.V * 0 + 1) * create_parameter("band", [0, 1]) * create_parameter("ky", [0.0, 1.0])
    fig, ax = dashboard(eigva, "ky", [[abs(eigve.sel(band=0))]], pot, "amplitude", lattice=True)
    assert any(labels_at(a) == ["A", "B"] for a in fig.axes)
    plt.close(fig)


def test_size_in_lattice_kwargs_is_the_circle_size(no_display):
    c, pot = optical_honeycomb(1.0, 3.0, 5.0, resolution=(36, 36))
    fig, axes = plot_eigenvector([[pot.V]], [[pot]], lattice=True, lattice_kwargs={"size": 400})
    circles = [x for x in axes[0][0].collections if x.__class__.__name__ == "PathCollection"]
    assert circles and all(np.allclose(x.get_sizes(), 400) for x in circles)
    assert labels_at(axes[0][0]) == ["A", "B"]
    plt.close(fig)
