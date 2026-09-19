"""How 'dashboard' applies its template to the eigenvector maps and to the potential contours."""

import numpy as np
import pytest
import xarray as xr
from cmcrameri import cm
from matplotlib.collections import QuadMesh

from bloch_schrodinger.plotting import contour_tmpl, dashboard
from bloch_schrodinger.potential import Potential, create_parameter


@pytest.fixture
def system():
    """A swept trap and stand-in eigenpairs: dashboard only needs their shapes and coordinates."""
    pot = Potential(unitvecs=[[4, 0], [0, 4]], resolution=(10, 10), v0=0)
    omega = create_parameter("omega", [1.0, 2.0, 3.0])
    pot.set(omega * (pot.x**2 + pot.y**2))
    eigve = np.exp(-pot.V).expand_dims(band=[0, 1])
    eigva = xr.DataArray(
        np.arange(6.0).reshape(2, 3),
        coords={"band": [0, 1], "omega": omega.values},
        dims=["band", "omega"],
    )
    return pot, eigva, eigve


def meshes(fig):
    return [c for ax in fig.axes for c in ax.collections if isinstance(c, QuadMesh)]


def test_preset_name_styles_the_maps_and_contours(system):
    pot, eigva, eigve = system
    fig, _ = dashboard(eigva, "omega", [[eigve]], pot, "amplitude")

    (mesh,) = meshes(fig)
    assert mesh.get_cmap().name == cm.oslo_r.name
    # The default contour template draws gray dashed lines, matplotlib's own would be colored
    contours = [c for ax in fig.axes for c in ax.collections if not isinstance(c, QuadMesh)]
    assert contours and all(np.allclose(c.get_edgecolor()[:, :3], 0.5, atol=0.01) for c in contours)
    # The maps share the band plot's figure and get no colorbar of their own
    assert len(fig.axes) == 2


def test_dict_template_is_accepted_and_left_untouched(system):
    pot, eigva, eigve = system
    template = {"fkwargs": {"cmap": "magma"}, "colorbar": {"kwargs": {}}}
    fig, _ = dashboard(eigva, "omega", [[eigve]], pot, (template, contour_tmpl(5)))

    assert meshes(fig)[0].get_cmap().name == "magma"
    assert template == {"fkwargs": {"cmap": "magma"}, "colorbar": {"kwargs": {}}}

    fig, _ = dashboard(eigva, "omega", [[eigve]], pot, template)
    assert meshes(fig)[0].get_cmap().name == "magma"
