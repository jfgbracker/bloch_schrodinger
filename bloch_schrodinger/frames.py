"""Where each point of a field sits in space: the cartesian frame of a DataArray.

A field in these packages is a DataArray over lattice dimensions 'a1', 'a2', ... and, optionally,
parameter dimensions -- a time, a swept interaction strength, a beam angle. Its cartesian
coordinates 'x', 'y', 'z' say where each lattice point sits in space. This module is the one place
that knows how those coordinates may be stored, so that everything reading them -- the plotting
functions, a solver's I/O helpers, a potential asked to evaluate itself on a field -- reads them the
same way, instead of each carrying its own copy of the rules.

Three storage forms are understood:

- static and dense: 'x' is a coordinate over the lattice dims only. This is the usual case, and
  what 'Potential' builds.
- moving and dense: 'x' also depends on parameter dims, because the grid itself moves or stretches
  with them -- a comoving or an expanding frame, say. It is then a coordinate over those dims too.
- moving and factored: 'x' is not stored at all, but is the product of other coordinates of the
  same array, typically an invariant lattice coordinate times one scale per frame. Stored as x
  itself that is an array the size of the field; stored as its two factors it is a few kilobytes.
  The recipe lives in the array's 'cart_factors' attribute, as JSON because netCDF attributes
  cannot hold a mapping. 'set_cart_factors' writes it, and 'materialize' turns it back into
  ordinary coordinates.

A stored coordinate always takes precedence over a recipe for the same name, so an array that
happens to carry both is read through the coordinate.
"""

import json
from collections.abc import Sequence

import xarray as xr


def lattice_dims(data: xr.DataArray) -> list[str]:
    """Return the data's own lattice dims (a1, a2, ...), sorted by axis index."""
    return sorted(
        (d for d in data.dims if d[0] == "a" and d[1:].isdigit()),
        key=lambda d: int(d[1:]),
    )


def cart_factors(data: xr.DataArray) -> dict[str, tuple[str, ...]]:
    """The recipe for any cartesian coordinate stored as a product of other coordinates.

    Nothing here knows what the factors mean or what they are called -- only that a coordinate may
    be the product of others on the same array.

    Args:
        data (xr.DataArray): The field.

    Returns:
        dict[str, tuple[str, ...]]: Cartesian coordinate name -> the names of its factors. Empty
        when there are none.
    """
    raw = data.attrs.get("cart_factors")
    if isinstance(raw, str):
        raw = json.loads(raw)
    return {name: tuple(parts) for name, parts in (raw or {}).items()}


def set_cart_factors(
    data: xr.DataArray, factors: dict[str, Sequence[str]]
) -> xr.DataArray:
    """Declare cartesian coordinates as products of other coordinates of the array.

    This is how a frame that moves with the parameters is stored without being materialised: for
    an invariant lattice coordinate 'x0' and one scale 's' per time step, {"x": ("x0", "s")} says
    that x = x0 * s, and every reader of this module then rebuilds x one frame at a time.

    A dense coordinate of the same name is dropped. The recipe and the materialised coordinate must
    never both be written, or a reader would have two answers for the same thing; since the factors
    are what the caller is describing here, they win.

    Args:
        data (xr.DataArray): The field.
        factors (dict[str, Sequence[str]]): Cartesian coordinate name -> the names of the
            coordinates it is the product of. Added to whatever recipe the array already carries.

    Raises:
        ValueError: If a factor is not a coordinate of the array.

    Returns:
        xr.DataArray: A shallow copy of the array, carrying the recipe.
    """
    missing = sorted(
        {part for parts in factors.values() for part in parts if part not in data.coords}
    )
    if missing:
        raise ValueError(
            f"The factor(s) {missing} are not coordinates of the array, so the cartesian "
            "coordinates cannot be rebuilt from them. Assign them first."
        )
    recipe = {name: list(parts) for name, parts in cart_factors(data).items()}
    recipe.update({name: list(parts) for name, parts in factors.items()})

    out = data.copy(deep=False).drop_vars([name for name in factors if name in data.coords])
    out.attrs = {**data.attrs, "cart_factors": json.dumps(recipe)}
    return out


def _factors_of(data: xr.DataArray, name: str) -> tuple[str, ...]:
    """The factors of a cartesian coordinate that is not stored.

    Raises:
        KeyError: If the array declares no recipe for it either.
    """
    factors = cart_factors(data)
    if name not in factors:
        raise KeyError(
            f"'{name}' is neither a coordinate of the array nor declared in its 'cart_factors'"
        )
    return factors[name]


def cart_dims(data: xr.DataArray, name: str) -> tuple[str, ...]:
    """The dims a cartesian coordinate depends on, whether it is stored or factored."""
    if name in data.coords:
        return tuple(data.coords[name].dims)
    return tuple(
        dict.fromkeys(d for part in _factors_of(data, name) for d in data.coords[part].dims)
    )


def cart_coord(
    data: xr.DataArray, name: str, sel: dict | None = None
) -> xr.DataArray:
    """A cartesian coordinate, rebuilt from its factors when it is not stored.

    Each factor is selected down to the frame *before* the product is taken. Multiplying first
    would build the whole outer product -- the very array the factored form exists to avoid -- and
    only then throw all but one frame of it away.

    Args:
        data (xr.DataArray): The field.
        name (str): The cartesian coordinate wanted.
        sel (dict, optional): A position in the parameter space, e.g. the current slider values.
            Entries for dims the coordinate does not depend on are ignored, and the nearest point
            is taken. None gives the coordinate over every frame. Defaults to None.

    Returns:
        xr.DataArray: The coordinate, over the lattice and whatever parameter dims 'sel' left.
    """
    def at(coord: xr.DataArray) -> xr.DataArray:
        if not sel:
            return coord
        sub = {d: v for d, v in sel.items() if d in coord.dims}
        return coord.sel(sub, method="nearest") if sub else coord

    if name in data.coords:
        return at(data.coords[name])
    out = None
    for part in _factors_of(data, name):
        piece = at(data.coords[part])
        out = piece if out is None else out * piece
    return out


def cart_extent(data: xr.DataArray, name: str) -> tuple[float, float]:
    """The full span of a cartesian coordinate, over every frame the array reaches.

    This is the box to pin plot axes to when a moving grid is meant to be seen growing rather
    than rescaled away. Stored dense, the span is simply the coordinate's own min and max.
    Stored as a product of factors, taking it that way would mean building the very outer
    product the factored form exists to avoid, so it is taken from the factors' corners
    instead: the extremes of a product of independent ranges are always attained at some
    combination of the ranges' own extremes.

    Args:
        data (xr.DataArray): The field.
        name (str): The cartesian coordinate wanted.

    Returns:
        tuple[float, float]: The lowest and highest value the coordinate takes anywhere.
    """
    if name in data.coords:
        coord = data.coords[name]
        return float(coord.min()), float(coord.max())

    lo = hi = None
    for part in _factors_of(data, name):
        part_coord = data.coords[part]
        ends = (float(part_coord.min()), float(part_coord.max()))
        if lo is None:
            lo, hi = ends
        else:
            products = [a * b for a in (lo, hi) for b in ends]
            lo, hi = min(products), max(products)
    return lo, hi


def moving_dims(data: xr.DataArray, name: str) -> list[str]:
    """Return the non-lattice dims the cartesian coordinate 'name' depends on.

    A field living on a moving grid carries cartesian coordinates that are functions of the
    parameters, x(t, ...), rather than of the lattice alone. An empty list means the grid is fixed
    and the coordinate can be read once and reused.
    """
    lattice = lattice_dims(data)
    return [d for d in cart_dims(data, name) if d not in lattice]


def materialize(data: xr.DataArray, names: Sequence[str] | None = None) -> xr.DataArray:
    """Write factored cartesian coordinates out as ordinary coordinates.

    The inverse of 'set_cart_factors', for the consumers that want x, y, z as plain coordinates --
    an interpolation onto a cartesian grid, a reduction by position. Lazy factors give a lazy
    coordinate, so an array opened lazily stays lazy.

    Each rebuilt coordinate spans the full lattice mesh, even where one of its factors only runs
    along its own axis: the plotting functions transpose a coordinate onto the whole mesh before
    drawing, and a coordinate carrying only its own axis would raise there.

    Args:
        data (xr.DataArray): The field.
        names (Sequence[str], optional): The cartesian coordinates to rebuild. Defaults to None,
            which takes every one the recipe declares.

    Raises:
        ValueError: If a name has no recipe on the array.

    Returns:
        xr.DataArray: The same array with those coordinates materialised and their recipe removed.
    """
    factors = cart_factors(data)
    names = list(factors) if names is None else list(names)
    missing = [name for name in names if name not in factors]
    if missing:
        raise ValueError(
            f"The array carries no 'cart_factors' recipe for {missing}, so there is nothing to "
            "materialise."
        )

    # A zero-weighted sum over every lattice axis, which is what forces each rebuilt coordinate to
    # span the full spatial mesh rather than only the axes its factors happen to carry.
    spatial = sum(0 * data[d] for d in lattice_dims(data))
    rebuilt = {name: cart_coord(data, name) + spatial for name in names}

    # The recipe and the materialised coordinate must never both be present, or a reader would have
    # two answers for the same thing.
    out = data.copy(deep=False)
    remaining = {name: list(parts) for name, parts in factors.items() if name not in names}
    out.attrs = {k: v for k, v in data.attrs.items() if k != "cart_factors"}
    if remaining:
        out.attrs["cart_factors"] = json.dumps(remaining)
    return out.assign_coords(rebuilt)
