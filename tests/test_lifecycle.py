"""How long an interactive plot holds on to its data, and how its sliders fire.

Every slider's observer reaches the field being drawn, and the sliders stay in ipywidgets'
registry until closed, so a plot nobody closes keeps its data for the life of the kernel. These
tests pin the three ways a plot lets go -- its figure closing, too many plots being live, and
'close_all' -- and that sliders wait for release before redrawing.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from ipywidgets import FloatSlider
from matplotlib.backend_bases import CloseEvent

import bloch_schrodinger.plotting as plotting
from bloch_schrodinger import utils
from bloch_schrodinger.plotting import close_all, plot_eigenvector
from bloch_schrodinger.potential import Potential, create_parameter


@pytest.fixture(autouse=True)
def _no_live_plots():
    """Start and end every test with no plot registered, whatever the other tests left."""
    close_all()
    yield
    close_all()


@pytest.fixture
def param2d():
    """A 2-D field carrying one parameter dimension."""
    pot = Potential(unitvecs=[[8, 0], [0, 10]], resolution=(12, 14), v0=0)
    omega = create_parameter("omega", np.linspace(1, 2, 3))
    pot.set((pot.x**2 + 2 * pot.y**2) * omega)
    return pot


def draw(field):
    """One interactive plot of 'field', and the registry entry it made."""
    fig, _ = plot_eigenvector([[field.V]], [[field]], [["amplitude"]])
    return fig, plotting._live[-1]


def sliders_of(plot):
    return [w for w in plot.widgets if isinstance(w, FloatSlider)]


def is_closed(widget):
    return widget.comm is None


def test_sliders_redraw_on_release_by_default(param2d):
    _, plot = draw(param2d)
    assert sliders_of(plot)
    assert all(not s.continuous_update for s in sliders_of(plot))


def test_live_scrubbing_can_be_switched_back_on(param2d, monkeypatch):
    monkeypatch.setattr(utils, "CONTINUOUS_UPDATE", True)
    _, plot = draw(param2d)
    assert all(s.continuous_update for s in sliders_of(plot))


def test_a_close_event_releases_the_controls(param2d):
    """What a GUI backend reports when its window is closed."""
    fig, plot = draw(param2d)
    widgets = list(plot.widgets)

    CloseEvent("close_event", fig.canvas)._process()

    assert plot not in plotting._live
    assert plot.widgets == [], "the entry must not keep the data alive either"
    assert all(is_closed(w) for w in widgets)
    plt.close(fig)


def test_plt_close_on_ipympl_releases_the_controls(param2d):
    """ipympl's 'plt.close' closes the canvas widget without emitting a close_event."""
    # Read first: importing ipympl switches the backend by itself
    previous = matplotlib.get_backend()
    pytest.importorskip("ipympl")
    plt.switch_backend("module://ipympl.backend_nbagg")
    try:
        fig, plot = draw(param2d)
        widgets = list(plot.widgets)

        plt.close(fig)

        assert plot not in plotting._live
        assert all(is_closed(w) for w in widgets)
    finally:
        plt.switch_backend(previous)


def test_the_oldest_plot_is_retired_past_the_cap(param2d, monkeypatch):
    monkeypatch.setattr(plotting, "MAX_LIVE_PLOTS", 2)
    figs, plots = zip(*(draw(param2d) for _ in range(3)))
    oldest = sliders_of(plots[0])

    assert list(plotting._live) == list(plots[1:])
    assert all(is_closed(s) for s in oldest)
    # Retired, not closed: the figure stays up showing its last frame
    assert plt.fignum_exists(figs[0].number)
    assert not any(is_closed(s) for p in plots[1:] for s in sliders_of(p))
    for fig in figs:
        plt.close(fig)


def test_close_all_closes_figures_too(param2d):
    figs = [draw(param2d)[0] for _ in range(2)]
    widgets = [w for p in plotting._live for w in p.widgets]

    close_all()

    assert plotting._live == []
    assert all(is_closed(w) for w in widgets)
    assert not any(plt.fignum_exists(f.number) for f in figs)
