"""Known lattice geometries: the named sites of a 2D lattice, snapping numerically found minima onto
them, and drawing the lattice schematically over a plot.

A potential computed on a grid finds its minima only to within the grid spacing (see
'Potential.find_minima'). When the geometry is known -- a honeycomb's sites sit on the C3-symmetric
points of its triangular Bravais lattice whatever the depths -- those positions are exact, and a
figure is cleaner with the sites drawn where they really are. A 'LatticeGeometry' holds them:

    geom = LatticeGeometry.hexagonal(a1, a2, origin=(-a / 2, 0))     # A at the origin, B, C
    sites = potential.find_sites(geom)                                # minima, snapped and labelled

and 'draw_lattice' (or the 'lattice=' option of the plotting functions) draws the occupied sites as
labelled circles, optionally with the bonds between them, the empty sites and the unit cell.

Sites are labelled by position, not by depth: on a hexagonal lattice the points A, B and C are fixed
points of the geometry, and a well found on the C point is labelled C. Which of them are wells is
the physics (the depths, the polarization); the labels say where they are.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

__all__ = ["LatticeGeometry", "SnapResult", "LatticeOverlay", "draw_lattice", "attach_lattice",
           "resolve_geometry", "PRESETS", "SITE_COLORS"]

PRESETS = ("hexagonal", "square", "checkerboard", "kagome", "lieb")

# Fill and edge colour per label; further labels cycle through matplotlib's default colours
SITE_COLORS = {
    "A": ("#3f6fc4", "#24467f"),
    "B": ("#f2c19b", "#d9822b"),
    "C": ("#6aa84f", "#3d6e2b"),
    "D": ("#a678c9", "#6c3f91"),
}


def _angle(a1: np.ndarray, a2: np.ndarray) -> float:
    return float(np.degrees(np.arccos(a1 @ a2 / np.linalg.norm(a1) / np.linalg.norm(a2))))


def _check(a1, a2, angles: tuple[float, ...], kind: str) -> float:
    """Equal lengths and one of the allowed angles; returns the angle found."""
    l1, l2 = np.linalg.norm(a1), np.linalg.norm(a2)
    if not np.isclose(l1, l2, rtol=1e-3):
        raise ValueError(f"a {kind} lattice needs |a1| = |a2|, got {l1:.6g} and {l2:.6g}")
    ang = _angle(a1, a2)
    for target in angles:
        if abs(ang - target) < 0.1:
            return target
    raise ValueError(f"a {kind} lattice needs a1, a2 at {' or '.join(f'{x:g}' for x in angles)} "
                     f"degrees, got {ang:.3f}")


class SnapResult(NamedTuple):
    """Points matched to the sites of a geometry; one entry per point, in the order given.

    positions: (n, 2) the site position for a point within the tolerance of a site (the image of
        the site nearest to the point), the point itself otherwise.
    labels: the site label, or "" for a point not within the tolerance of any site.
    distances: (n,) the distance from each point to the nearest site.
    on_site: (n,) whether the point was within the tolerance, i.e. snapped.
    """

    positions: np.ndarray
    labels: list[str]
    distances: np.ndarray
    on_site: np.ndarray


class LatticeGeometry:
    """A 2D Bravais lattice with named sites.

    Args:
        a1, a2: The lattice vectors, cartesian. Use the potential's own (`potential.a`), so that
            the geometry has the potential's orientation.
        sites: Label -> fractional coordinates (f1, f2) of the site relative to 'origin', i.e. the
            site sits at origin + f1 a1 + f2 a2 (and at all its lattice translates).
        origin: Cartesian position the fractional coordinates are counted from. Defaults to (0, 0).
        name: A name for messages and the repr.

    The presets ('hexagonal', 'square', 'checkerboard', 'kagome', 'lieb') place the sites from the
    vectors given; 'from_potential' takes the vectors from a Potential.
    """

    def __init__(self, a1, a2, sites: dict, origin=(0.0, 0.0), name: str = "custom"):
        self.a = np.array([a1, a2], dtype=float)
        if self.a.shape != (2, 2):
            raise ValueError("LatticeGeometry is two-dimensional: a1 and a2 must be 2-vectors")
        if np.isclose(abs(np.linalg.det(self.a)), 0):
            raise ValueError("a1 and a2 must not be colinear")
        if not sites:
            raise ValueError("a geometry needs at least one site")
        self.sites = {str(k): np.array(v, dtype=float) for k, v in sites.items()}
        self.origin = np.array(origin, dtype=float)
        self.name = name

    # --- presets ------------------------------------------------------------------------------
    @classmethod
    def hexagonal(cls, a1, a2, origin=(0.0, 0.0), ab=None) -> "LatticeGeometry":
        """The triangular Bravais lattice with its three C3-symmetric points A, B, C.

        A is at 'origin', B at origin + ab and C at origin + 2 ab (= origin - ab, modulo the
        lattice). A honeycomb has two of them occupied, a triangular lattice one; the third is the
        hexagon centre. 'ab' is the A -> B nearest-neighbour vector: by default (a1 + a2) / 3 when
        a1 and a2 are 60 degrees apart, (2 a1 + a2) / 3 when they are 120 degrees apart. Giving it
        explicitly (it must be one of the six A -> B or A -> C vectors) chooses which of the other
        two points is B, e.g. to match the naming of a potential's construction.
        """
        a1, a2 = np.asarray(a1, float), np.asarray(a2, float)
        ang = _check(a1, a2, (60.0, 120.0), "hexagonal")
        d = np.array([1 / 3, 1 / 3]) if ang == 60.0 else np.array([2 / 3, 1 / 3])
        if ab is not None:
            A = np.array([a1, a2])
            f = np.linalg.solve(A.T, np.asarray(ab, float))
            ok = False
            for s in (d, 2 * d):
                diff = f - s
                if np.allclose(diff, np.round(diff), atol=1e-6):
                    d, ok = s, True
                    break
            if not ok:
                raise ValueError(f"ab={ab} is not a vector from one C3 point to another of this lattice")
        return cls(a1, a2, {"A": (0, 0), "B": d % 1, "C": (2 * d) % 1}, origin, "hexagonal")

    @classmethod
    def square(cls, a1, a2, origin=(0.0, 0.0)) -> "LatticeGeometry":
        """A square lattice, one site A per cell."""
        _check(np.asarray(a1, float), np.asarray(a2, float), (90.0,), "square")
        return cls(a1, a2, {"A": (0, 0)}, origin, "square")

    @classmethod
    def checkerboard(cls, a1, a2, origin=(0.0, 0.0)) -> "LatticeGeometry":
        """Two square sublattices A, B; a1, a2 are the primitive vectors (the diagonals of the
        squares the sites form), and B sits at (a1 + a2) / 2."""
        _check(np.asarray(a1, float), np.asarray(a2, float), (90.0,), "checkerboard")
        return cls(a1, a2, {"A": (0, 0), "B": (0.5, 0.5)}, origin, "checkerboard")

    @classmethod
    def kagome(cls, a1, a2, origin=(0.0, 0.0)) -> "LatticeGeometry":
        """Sites A, B, C on the bond midpoints of a triangular lattice: A at the origin, B and C
        half a lattice vector away along the two 60-degree directions."""
        ang = _check(np.asarray(a1, float), np.asarray(a2, float), (60.0, 120.0), "kagome")
        c = (0, 0.5) if ang == 60.0 else (0.5, 0.5)            # a2 / 2, or (a1 + a2) / 2
        return cls(a1, a2, {"A": (0, 0), "B": (0.5, 0), "C": c}, origin, "kagome")

    @classmethod
    def lieb(cls, a1, a2, origin=(0.0, 0.0)) -> "LatticeGeometry":
        """The Lieb lattice: corner site A, edge sites B (at a1 / 2) and C (at a2 / 2)."""
        _check(np.asarray(a1, float), np.asarray(a2, float), (90.0,), "Lieb")
        return cls(a1, a2, {"A": (0, 0), "B": (0.5, 0), "C": (0, 0.5)}, origin, "lieb")

    @classmethod
    def preset(cls, kind: str, a1, a2, origin=(0.0, 0.0), **kwargs) -> "LatticeGeometry":
        """One of PRESETS by name."""
        if kind not in PRESETS:
            raise ValueError(f"unknown geometry {kind!r}: one of {PRESETS}, or build a custom one")
        return getattr(cls, kind)(a1, a2, origin, **kwargs)

    @classmethod
    def from_potential(cls, potential, kind: str, origin=(0.0, 0.0), **kwargs) -> "LatticeGeometry":
        """A preset with the lattice vectors of a 2D Potential."""
        if potential.n_dims != 2:
            raise ValueError("lattice geometries are two-dimensional")
        return cls.preset(kind, potential.a[0], potential.a[1], origin, **kwargs)

    def with_origin(self, origin) -> "LatticeGeometry":
        """The same geometry, moved to another origin."""
        return LatticeGeometry(self.a[0], self.a[1], self.sites, origin, self.name)

    def __repr__(self) -> str:
        def v(x):
            return "(" + ", ".join(f"{float(c):.4g}" for c in x) + ")"
        sites = ", ".join(f"{k}={v(f)}" for k, f in self.sites.items())
        return (f"LatticeGeometry({self.name}, a1={v(self.a[0])}, a2={v(self.a[1])}, "
                f"origin={v(self.origin)}, sites (fractional): {sites})")

    # --- geometry -----------------------------------------------------------------------------
    @property
    def labels(self) -> list[str]:
        return list(self.sites)

    def position(self, label: str) -> np.ndarray:
        """Cartesian position of a site in the cell at the origin."""
        return self.origin + self.sites[label] @ self.a

    def nn_distance(self, labels=None) -> float:
        """The shortest distance between two distinct points of the given sites (all by default),
        lattice translates included."""
        labels = self.labels if labels is None else list(labels)
        shifts = np.array([(i, j) for i in range(-2, 3) for j in range(-2, 3)]) @ self.a
        best = np.inf
        for s in labels:
            for t in labels:
                d = np.linalg.norm(self.position(t) - self.position(s) + shifts, axis=1)
                d = d[d > 1e-9 * np.linalg.norm(self.a)]
                best = min(best, float(d.min()))
        return best

    def points_in(self, xlim, ylim, labels=None, margin: float = 0.0) -> dict[str, np.ndarray]:
        """All lattice points of the given sites inside the box (grown by 'margin'), per label."""
        x0, x1 = min(xlim) - margin, max(xlim) + margin
        y0, y1 = min(ylim) - margin, max(ylim) + margin
        corners = np.array([[x0, y0], [x0, y1], [x1, y0], [x1, y1]])
        inv = np.linalg.inv(self.a)
        out = {}
        for label in (self.labels if labels is None else labels):
            f = (corners - self.position(label)) @ inv
            lo, hi = np.floor(f.min(axis=0)) - 1, np.ceil(f.max(axis=0)) + 1
            n1, n2 = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
            pts = self.position(label) + np.column_stack([n1.ravel(), n2.ravel()]) @ self.a
            keep = (pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)
            out[label] = pts[keep]
        return out

    def nearest(self, points) -> tuple[list[str], np.ndarray, np.ndarray]:
        """For each point (an (n, 2) array): the label of the nearest site, the distance to it, and
        the position of that site's image nearest to the point."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        inv = np.linalg.inv(self.a)
        shifts = np.array([(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float)
        best_d = np.full(len(points), np.inf)
        best_l = [""] * len(points)
        best_p = np.full_like(points, np.nan)
        for label in self.labels:
            f = (points - self.position(label)) @ inv
            base = f - np.round(f)                                   # fractional offset, folded
            cand = base[:, None, :] - shifts[None, :, :]             # and its 8 neighbours
            cart = cand @ self.a
            dist = np.linalg.norm(cart, axis=2)
            k = np.argmin(dist, axis=1)
            d = dist[np.arange(len(points)), k]
            better = d < best_d
            best_d[better] = d[better]
            best_p[better] = points[better] - cart[np.arange(len(points)), k][better]
            for i in np.nonzero(better)[0]:
                best_l[i] = label
        return best_l, best_d, best_p

    def snap(self, points, tol: float = 0.1, warn: bool = True) -> SnapResult:
        """Match points (e.g. numerically found minima) to the sites.

        Args:
            points: (n, 2) cartesian positions; rows with a NaN are passed through as off-site.
            tol: The tolerance, as a fraction of the nearest-neighbour distance of the geometry
                (all its sites). A point closer than that to a site is moved onto it.
            warn: Warn about the points that are not within the tolerance of any site; they are
                kept where they are, and labelled "".
        """
        points = np.atleast_2d(np.asarray(points, dtype=float)).reshape(-1, 2)
        valid = ~np.isnan(points).any(axis=1)
        labels = [""] * len(points)
        dist = np.full(len(points), np.nan)
        pos = points.copy()
        on = np.zeros(len(points), dtype=bool)
        if valid.any():
            lab, d, p = self.nearest(points[valid])
            limit = tol * self.nn_distance()
            idx = np.nonzero(valid)[0]
            for k, i in enumerate(idx):
                dist[i] = d[k]
                if d[k] <= limit:
                    on[i], labels[i], pos[i] = True, lab[k], p[k]
            off = idx[~on[idx]]
            if warn and len(off):
                warnings.warn(
                    f"{len(off)} point(s) not within {tol:g} x the nearest-neighbour distance of any "
                    f"site of the {self.name} geometry, kept where found: "
                    + ", ".join(f"({points[i, 0]:.4g}, {points[i, 1]:.4g})" for i in off[:5])
                    + (" ..." if len(off) > 5 else ""),
                    stacklevel=2,
                )
        return SnapResult(pos, labels, dist, on)


# =============================================================================================
# drawing
# =============================================================================================

class LatticeOverlay:
    """The schematic lattice drawn over an Axes, redrawn on demand (new occupied sites when a
    slider moves) and, by default, whenever the axis limits change, so it always fills the view.

    Args:
        ax: The Axes to draw on; draw after the field, so that its limits are set.
        geometry: The LatticeGeometry.
        sites: Draw the occupied sites as filled circles. Defaults to True.
        labels: Write the label inside each circle. Defaults to True.
        bonds: Dashed lines between nearest-neighbour occupied sites. Defaults to False.
        empty_sites: The geometry's unoccupied sites as faint open circles. Defaults to False.
        unit_cell: The unit cell at the origin, with a1 and a2 as arrows. Defaults to False.
        offsite: Mark points that matched no site (see 'draw'). Defaults to True.
        colors: Label -> fill colour, or -> (fill, edge). Defaults to SITE_COLORS.
        size: Circle size: "auto" (the default) makes the diameter 0.42 x the nearest-neighbour
            distance at the current zoom, within 6 to 30 points; a number is matplotlib's scatter
            's' (points squared), fixed whatever the zoom.
        fontsize: Label size; by default half the circle diameter, within 5 to 11 points.
        site_kwargs, text_kwargs, bond_kwargs, empty_kwargs, offsite_kwargs, cell_kwargs: Extra
            keyword arguments for the circles (scatter), labels (text), bonds (LineCollection),
            empty sites (scatter), off-site points (scatter) and the unit cell (Polygon).
        swap_xy: True when the axes show y horizontally and x vertically (cart_axes [1, 0]).
        follow_limits: Redraw when the axis limits change (zoom, set_xlim). Defaults to True.
        zorder: Base z-order; bonds below it, circles above. Defaults to 4.
    """

    def __init__(self, ax: Axes, geometry: LatticeGeometry, *, sites: bool = True,
                 labels: bool = True, bonds: bool = False, empty_sites: bool = False,
                 unit_cell: bool = False, offsite: bool = True, colors: dict | None = None,
                 size: float | str = "auto", fontsize: float | None = None, site_kwargs: dict | None = None, text_kwargs: dict | None = None,
                 bond_kwargs: dict | None = None, empty_kwargs: dict | None = None,
                 offsite_kwargs: dict | None = None, cell_kwargs: dict | None = None,
                 swap_xy: bool = False, follow_limits: bool = True, zorder: float = 4):
        self.ax, self.geometry = ax, geometry
        self.show_sites, self.show_labels, self.show_bonds = sites, labels, bonds
        self.show_empty, self.show_cell, self.show_offsite = empty_sites, unit_cell, offsite
        self.colors = dict(colors or {})
        self.size, self.fontsize, self.swap, self.zorder = size, fontsize, swap_xy, zorder
        self._s, self._fs = 260.0, 9.0
        self.site_kwargs = site_kwargs or {}
        self.text_kwargs = text_kwargs or {}
        self.bond_kwargs = bond_kwargs or {}
        self.empty_kwargs = empty_kwargs or {}
        self.offsite_kwargs = offsite_kwargs or {}
        self.cell_kwargs = cell_kwargs or {}
        self.artists: list = []
        self.occupied: list[str] = geometry.labels
        self.offsite_points = np.empty((0, 2))
        self._busy = False
        if follow_limits:
            ax.callbacks.connect("xlim_changed", self._limits_changed)
            ax.callbacks.connect("ylim_changed", self._limits_changed)

    # --- helpers ------------------------------------------------------------------------------
    def _xy(self, pts: np.ndarray) -> np.ndarray:
        return pts[:, ::-1] if self.swap else pts

    def _color(self, label: str) -> tuple:
        c = self.colors.get(label)
        if c is None:
            c = SITE_COLORS.get(label)
        if c is None:
            k = self.geometry.labels.index(label) if label in self.geometry.labels else 0
            c = (f"C{k % 10}", "black")
        return (c, "black") if isinstance(c, str) else tuple(c)

    def _scale(self, xlim) -> None:
        """Circle area and font size for this zoom."""
        if self.size == "auto":
            ax = self.ax
            try:
                ax.apply_aspect()
            except Exception:
                pass
            width_pt = ax.bbox.width * 72 / ax.figure.dpi
            span = abs(xlim[1] - xlim[0]) or 1.0
            diam = 0.42 * self.geometry.nn_distance() * width_pt / span
            diam = float(np.clip(diam, 6, 30))
            self._s = diam**2
        else:
            self._s = float(self.size)
            diam = self._s**0.5
        self._fs = self.fontsize if self.fontsize is not None else float(np.clip(diam / 2, 5, 11))

    def _limits_changed(self, ax):
        if not self._busy:
            self.draw(self.occupied, self.offsite_points)

    def remove(self) -> None:
        for a in self.artists:
            try:
                a.remove()
            except (ValueError, NotImplementedError):
                pass
        self.artists = []

    # --- drawing ------------------------------------------------------------------------------
    def draw(self, occupied=None, offsite_points=None) -> None:
        """(Re)draw with these occupied site labels (all of the geometry's when None) and these
        off-site points (an (n, 2) array of positions that matched no site)."""
        g, ax = self.geometry, self.ax
        self.occupied = g.labels if occupied is None else [lab for lab in g.labels if lab in set(occupied)]
        self.offsite_points = (np.empty((0, 2)) if offsite_points is None
                               else np.asarray(offsite_points, dtype=float).reshape(-1, 2))
        self._busy = True
        try:
            self.remove()
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            box_x, box_y = (ylim, xlim) if self.swap else (xlim, ylim)
            self._scale(xlim)
            nn = g.nn_distance()
            pts = g.points_in(box_x, box_y, margin=nn)
            occ = {lab: pts[lab] for lab in self.occupied}

            if self.show_bonds and len(self.occupied):
                self._bonds(occ)
            if self.show_empty:
                empty = [lab for lab in g.labels if lab not in self.occupied]
                for lab in empty:
                    if len(pts[lab]):
                        self._circles(lab, pts[lab], empty=True)
            if self.show_sites:
                for lab, p in occ.items():
                    if len(p):
                        self._circles(lab, p)
            if self.show_offsite and len(self.offsite_points):
                kw = dict(marker="x", color="0.35", s=50, linewidths=1.5, zorder=self.zorder + 2)
                kw.update(self.offsite_kwargs)
                xy = self._xy(self.offsite_points)
                self.artists.append(ax.scatter(xy[:, 0], xy[:, 1], **kw))
            if self.show_cell:
                self._cell()
            ax.set_xlim(xlim)                       # the overlay never changes the view
            ax.set_ylim(ylim)
        finally:
            self._busy = False
        if ax.figure is not None:
            ax.figure.canvas.draw_idle()

    def _circles(self, label: str, p: np.ndarray, empty: bool = False) -> None:
        fill, edge = self._color(label)
        xy = self._xy(p)
        if empty:
            kw = dict(s=self._s, facecolors="none", edgecolors=edge, linewidths=1.0, alpha=0.45,
                      zorder=self.zorder)
            kw.update(self.empty_kwargs)
        else:
            kw = dict(s=self._s, c=fill, edgecolors=edge, linewidths=1.2, zorder=self.zorder + 1)
            kw.update(self.site_kwargs)
        self.artists.append(self.ax.scatter(xy[:, 0], xy[:, 1], **kw))
        if self.show_labels:
            r, g, b = mcolors.to_rgb(fill)
            light = 0.299 * r + 0.587 * g + 0.114 * b > 0.6           # dark text on light fills
            tkw = dict(ha="center", va="center", fontsize=self._fs,
                       color="0.15" if light else "white", fontweight="bold",
                       zorder=self.zorder + 1.5, clip_on=True)
            if empty:
                tkw.update(color=edge, alpha=0.6, fontweight="normal")
            tkw.update(self.text_kwargs)
            for x, y in xy:
                self.artists.append(self.ax.text(x, y, label, **tkw))

    def _bonds(self, occ: dict[str, np.ndarray]) -> None:
        allp = np.concatenate([p for p in occ.values() if len(p)] or [np.empty((0, 2))])
        if len(allp) < 2:
            return
        d0 = self.geometry.nn_distance(self.occupied)
        diff = allp[:, None, :] - allp[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        i, j = np.nonzero(np.triu(np.isclose(dist, d0, rtol=1e-3), k=1))
        segs = np.stack([self._xy(allp[i]), self._xy(allp[j])], axis=1)
        kw = dict(colors="black", linestyles=(0, (4, 3)), linewidths=0.9, alpha=0.8,
                  zorder=self.zorder - 0.5)
        kw.update(self.bond_kwargs)
        lc = LineCollection(segs, **kw)
        self.ax.add_collection(lc, autolim=False)
        self.artists.append(lc)

    def _cell(self) -> None:
        g = self.geometry
        o = g.origin
        corners = np.array([o, o + g.a[0], o + g.a[0] + g.a[1], o + g.a[1]])
        kw = dict(closed=True, fill=False, edgecolor="0.2", linewidth=1.0, linestyle="-",
                  zorder=self.zorder - 0.5)
        kw.update(self.cell_kwargs)
        poly = Polygon(self._xy(corners), **kw)
        self.ax.add_patch(poly)
        self.artists.append(poly)
        for k, name in enumerate(("$a_1$", "$a_2$")):
            tip = self._xy(np.array([o + g.a[k]]))[0]
            base = self._xy(np.array([o]))[0]
            ann = self.ax.annotate("", xy=tip, xytext=base, zorder=self.zorder + 2,
                                   arrowprops=dict(arrowstyle="-|>", color="0.1", lw=1.2))
            lab = self.ax.text(*(base + 0.55 * (tip - base)), name, fontsize=9, color="0.1",
                               ha="left", va="bottom", zorder=self.zorder + 2, clip_on=True)
            self.artists += [ann, lab]


def draw_lattice(ax: Axes, geometry: LatticeGeometry, occupied=None, offsite_points=None,
                 **style) -> LatticeOverlay:
    """Draw the schematic lattice over 'ax' and return the overlay (call its 'draw' to update).

    Args:
        ax: The Axes, with the field already drawn (the overlay fills its current limits).
        geometry: The LatticeGeometry.
        occupied: The site labels to draw as occupied; all of the geometry's by default. From
            found minima: `potential.find_sites(geometry)`, then its labels.
        offsite_points: Positions to mark as matching no site.
        **style: See LatticeOverlay: sites, labels, bonds, empty_sites, unit_cell, offsite, colors,
            size, *_kwargs, swap_xy, follow_limits, zorder.
    """
    ov = LatticeOverlay(ax, geometry, **style)
    ov.draw(occupied, offsite_points)
    return ov


# =============================================================================================
# the plotting functions' 'lattice' option
# =============================================================================================

def resolve_geometry(lattice, potential=None) -> LatticeGeometry | None:
    """The geometry a plot's 'lattice' option asks for: None for False/None, the potential's own
    for True, or the LatticeGeometry given."""
    if lattice is None or lattice is False:
        return None
    if lattice is True:
        geom = getattr(potential, "geometry", None) if potential is not None else None
        if geom is None:
            raise ValueError("lattice=True needs a potential with a 'geometry' (a LatticeGeometry): "
                             "set potential.geometry, or pass the geometry as lattice=...")
        return geom
    if isinstance(lattice, LatticeGeometry):
        return lattice
    raise TypeError(f"lattice must be a bool or a LatticeGeometry, got {type(lattice).__name__}")


def attach_lattice(ax: Axes, geometry: LatticeGeometry, potential=None, slider_values=None,
                   swap_xy: bool = False, occupied="minima", tol: float = 0.1,
                   minima_size: int = 3, minima_mode: str = "wrap", **style):
    """Draw the lattice over 'ax' and return the function that redraws it for new slider values.

    Which sites are drawn as occupied: with occupied="minima" (the default) and a potential, the
    sites its minima snap to at the current slider values (see Potential.find_sites); minima that
    match no site are marked as off-site. Without a potential, or with occupied="all", every site
    of the geometry; or a list of labels. minima_size and minima_mode are find_minima's size and
    mode; the other keyword arguments are LatticeOverlay's styles.
    """
    ov = LatticeOverlay(ax, geometry, swap_xy=swap_xy, **style)
    dynamic = isinstance(occupied, str) and occupied == "minima" and potential is not None
    if isinstance(occupied, str) and occupied not in ("minima", "all"):
        raise ValueError("occupied must be 'minima', 'all' or a list of site labels")
    spatial = [f"a{k + 1}" for k in range(potential.n_dims)] if potential is not None else []
    param_dims = [d for d in potential.V.dims if d not in spatial] if potential is not None else []
    first = [True]

    def update(**values):
        if not dynamic:
            ov.draw(None if isinstance(occupied, str) else occupied)
            return
        sel = {d: values[d] for d in param_dims if d in values}
        found = potential.find_sites(geometry, sel, tol=tol, size=minima_size, mode=minima_mode,
                                     warn=first[0])
        first[0] = False
        on = np.asarray(found["on_site"].values).ravel()
        labels = np.asarray(found["label"].values).ravel()
        xf = np.asarray(found["x_found"].values, float).ravel()
        yf = np.asarray(found["y_found"].values, float).ravel()
        valid = ~(np.isnan(xf) | np.isnan(yf))
        occ = sorted({str(lab) for lab, o in zip(labels, on) if o})
        off = np.column_stack([xf, yf])[valid & ~on]
        ov.draw(occ, off)

    update(**(slider_values or {}))
    ov.update = update
    return ov
