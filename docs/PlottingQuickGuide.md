# Plotting quick guide

Every plotting function in `bloch_schrodinger.plotting` reads the dimensions of the DataArrays you pass in. **Each dimension that isn't plotted becomes a slider automatically.** You never build sliders yourself: add a parameter with `create_parameter` and a slider for it appears.

Run the plots in Jupyter with the widget backend, and create each figure in its own cell:

```python
%matplotlib widget
import numpy as np
import xarray as xr
from bloch_schrodinger.potential import Potential, create_parameter
from bloch_schrodinger.fdsolver import FDSolver
import bloch_schrodinger.plotting as bp

pot = Potential([[5, 0], [0, 5]], (50, 50))
omega = create_parameter("omega", np.linspace(3, 10, 5))    # -> an "omega" slider
pot.set((pot.x**2 + pot.y**2) * omega)

eigva, eigve = FDSolver(pot, 1 / 2).solve(5)   # eigva: (band, omega), eigve: (band, omega, a1, a2)
```

`cart_axes` picks the spatial axes to plot against, with `0 = x`, `1 = y` and `2 = z`:
- **Two axes** give a 2D map.
- **One axis** gives a line cut.
- Any spatial axis you leave out becomes a slider. This is how you choose where the cut sits.

## Which function to use

| What you want | Call |
|---|---|
| Potential landscape (2D map) | `pot.plot()` or `bp.plot_eigenvector([[pot.V]], [[None]], [["real"]])` |
| Cut through the potential | `pot.plot(cart_axes=[0])` or `bp.plot_eigenvector([[pot.V]], [[None]], cart_axes=[0])` |
| Eigenvalues vs. a parameter | `bp.plot_cuts(eigva, "omega")` |
| Eigenvector maps | `bp.plot_eigenvector([[abs(eigve)**2]], [[pot]], [["amplitude"]])` |
| Cuts through eigenvectors | `bp.plot_eigenvector([[eigve.real]], [[pot]], cart_axes=[0])` |
| Eigenvalues and eigenvectors side by side | `bp.dashboard(eigva, "omega", [[abs(eigve)**2]], pot, "amplitude")` |
| Mode profiles at their energies, over a potential cut | `bp.energy_levels(eigva, eigve, pot)` |
| 3D eigenvector | `bp.plot_isosurface(abs(eigve), xr.ufuncs.angle(eigve), cyclic=True)` |

## Potentials

```python
pot.plot()                                 # 2D map, parameters on sliders
pot.plot(cart_axes=[0])                    # cut along x; y becomes a slider
pot.plot(show_minima=True)                 # mark the local minima
```

`pot.plot` is the quickest option. To get the colormesh templates, or to place the potential in a grid next to other plots, pass `pot.V` to `plot_eigenvector` as an ordinary field (see below).

## Eigenvalues: `plot_cuts`

```python
bp.plot_cuts(eigva, "omega")               # one line per band vs omega
bp.plot_cuts(eigva, "kx", ymin=0, ymax=10) # band structure, other dims on sliders
bp.plot_cuts(eigva.sel(band=0), "omega", groupby=[])      # a single band
bp.plot_cuts(eigva, "kx", linekws=[{"color": "k"}, {"color": "r", "ls": "--"}])  # style cycles over bands
```

- The second argument is the x-axis.
- The dimensions in `groupby` (default `["band"]`) are drawn as separate lines.
- Every other dimension gets a slider.
- To get a band structure, compute over k first, e.g. `solver.create_reciprocal_grid([np.linspace(-np.pi, np.pi, 50), 0])`, then cut along `"kx"`.
- The functions only plot one parameter at a time. A 2D eigenvalue map (such as E over kx and ky) isn't available: plot it as cuts, with the second parameter on a slider.

## Eigenvectors and fields: `plot_eigenvector`

All arguments are matrices (lists of lists), and each entry is one subplot of the grid:

```python
bp.plot_eigenvector(
    plots=[[abs(eigve)**2, eigve.real],
           [pot.V,          None     ]],     # None leaves the subplot empty
    potentials=[[pot, pot],
                [None, None]],               # contour overlay of the potential
    templates=[["amplitude", "real"],
               ["real",      None  ]],
)
```

- **`templates`**:
  - Preset names: `"amplitude"`, `"amplitude - log"`, `"real"`, `"real - log"` and `"phase"`. Use `"phase"` with `xr.ufuncs.angle(eigve)`: `np.angle` returns a bare array without the dimensions the plot needs.
  - Tuples `(colormesh, contour, quiver)` change the overlays too, e.g. `("amplitude", bp.contour_tmpl(5))` draws 5 contour levels.
  - A dict passes keyword arguments straight to matplotlib: `{"fkwargs": {"cmap": "viridis"}, "autoscale": True}`. `create_map` documents every key.
- **Cuts**: `cart_axes=[0]` turns every subplot into a line cut. The potential is drawn dashed, against its own y-axis on the right. Give a matrix of per-subplot axes to mix maps and cuts:

  ```python
  bp.plot_eigenvector([[abs(eigve)**2, abs(eigve)**2]], [[pot, pot]],
                      [["amplitude", None]], cart_axes=[[[0, 1], [0]]])
  ```
- **3D data**: `cart_axes=[0, 2]` shows the x–z plane, and y becomes a slider. `resolutions` sets the grid resolution used when a skewed lattice has to be interpolated.
- **Quivers**: `quivers=[[(U, V)]]` overlays arrows, e.g. a current or a spin texture. This only works on 2D maps.
- **Coupled equations**: eigenvectors have a `field` dimension. Select a field before plotting, e.g. `eigve.sel(field=0)`. If you don't, `field` becomes a slider.

## Combined views

```python
bp.dashboard(eigva, "omega", [[abs(eigve)**2, eigve.real]], pot, "amplitude",
             titles=[["density", "real part"]])
```

The bands are drawn on the left, with a red marker at the current `omega`. The eigenvector maps are on the right, and all panels share the same sliders.

```python
bp.energy_levels(eigva, eigve, pot)   # 2D only
```

This draws a cut through the potential, with each mode's profile placed at its energy. The `offset` and `cut rotation` sliders move the cut line through the plane.

## Tips

- **Too many sliders?** Select the dimension down first: `eigve.sel(band=0)`, `eigva.sel(ky=0)`.
- **Use real data.** Plot `abs(eigve)**2`, `eigve.real` or `xr.ufuncs.angle(eigve)`, not raw complex eigenvectors.
- **Parameters go in `create_parameter`**, so that they have coordinate values. Only dimensions with coordinates can become sliders.
- **Every call returns `(fig, ax)`** (`plot_isosurface` returns a plotly figure instead), so you can adjust titles, limits and labels afterwards with plain matplotlib.
