"""Fields over a four-dimensional coordinate domain and the three-dimensional hypersurfaces cut from it.

A field is sampled on a regular grid of the coordinates (t, x, y, z), independently of how it will be shown. A
hypersurface fixes one coordinate, x^mu = c, and keeps the other three; the order in which those three are laid out
(which one varies fastest in the exported dataset) is chosen separately and does not change what the hypersurface is.

Conventions, fixed here for the whole repository:
  * coordinates are ordered (t, x, y, z), the order of `algebra.point` and `algebra.coordinates`;
  * the metric signature is (+, -, -, -), the spacetime part of the algebra's metric;
  * lengths and times are in units of the hole's mass M (G = c = 1);
  * arrays on the 4D grid are indexed [t, x, y, z, components...]; arrays on a section follow VTK's point order,
    [third, second, first, components...] for display axes (first, second, third), the first varying fastest.

Field values are one of three kinds:
  * "scalar": one number per point;
  * "vector": the four components of a vector along (t, x, y, z), in the coordinate basis or the frame basis
    (`basis`); on a section it is shown by its components along the display axes, in display order, next to all four;
  * "components": named components (`labels`) of an algebraic object, such as the generator coefficients of a
    curvature 2-form on the coordinate planes. Their labels name physical axes, so reordering the display axes leaves
    them as they are.
Reordering the display axes permutes how a vector is drawn, never what it is: the four-component array is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AXES = ("t", "x", "y", "z")
SIGNATURE = (1, -1, -1, -1)
UNITS = "M"                                                     # geometric units, G = c = 1
KINDS = ("scalar", "vector", "components")


def axis_index(name: str) -> int:
    if name not in AXES:
        raise ValueError(f"unknown axis {name!r}; the axes are {AXES}")
    return AXES.index(name)


# --- the four-dimensional grid and fields on it ----------------------------------------------------
@dataclass(frozen=True)
class Grid4:
    """A regular grid of the coordinates (t, x, y, z): one strictly increasing, evenly spaced array per axis."""
    axes: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

    def __post_init__(self):
        if len(self.axes) != 4:
            raise ValueError("a Grid4 has four axes, (t, x, y, z)")
        for name, values in zip(AXES, self.axes):
            values = np.asarray(values, dtype=np.float64)
            if values.ndim != 1 or len(values) < 1:
                raise ValueError(f"axis {name}: a non-empty one-dimensional array is needed")
            steps = np.diff(values)
            if len(values) > 1 and (np.any(steps <= 0) or np.abs(steps - steps[0]).max() > 1e-9 * abs(steps[0])):
                raise ValueError(f"axis {name}: values must increase evenly")

    @classmethod
    def regular(cls, bounds, shape, offsets=(0.0, 0.0, 0.0, 0.0)) -> "Grid4":
        """`bounds` (lower, upper) and `shape` (number of points) per axis in (t, x, y, z) order, each axis shifted by
        its offset (a small spatial offset keeps the points off a singularity at the origin)."""
        return cls(tuple(np.linspace(low, high, n) + offset for (low, high), n, offset in zip(bounds, shape, offsets)))

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(len(values) for values in self.axes)

    @property
    def origin(self) -> tuple[float, float, float, float]:
        return tuple(float(values[0]) for values in self.axes)

    @property
    def spacing(self) -> tuple[float, float, float, float]:
        return tuple(float(values[1] - values[0]) if len(values) > 1 else 1.0 for values in self.axes)

    def axis(self, name: str) -> np.ndarray:
        return np.asarray(self.axes[axis_index(name)])

    def points(self) -> np.ndarray:
        """Every grid point as (t, x, y, z), shape (N, 4), in the grid's index order (z fastest)."""
        mesh = np.meshgrid(*self.axes, indexing="ij")
        return np.stack([values.ravel() for values in mesh], axis=-1)

    def metadata(self) -> dict:
        return coordinate_metadata() | dict(shape=list(self.shape), origin=list(self.origin),
                                            spacing=list(self.spacing))


def coordinate_metadata() -> dict:
    """The coordinate conventions every exported dataset records."""
    return dict(coordinates=list(AXES), ordering="t, x, y, z", signature=list(SIGNATURE), units=UNITS,
                unit_note="geometric units G = c = 1, lengths and times in units of the mass M")


@dataclass(frozen=True)
class FieldInfo:
    """What a field's values are: its kind, the basis of its components and their labels."""
    kind: str = "scalar"
    basis: str = ""
    labels: tuple[str, ...] = ()

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown field kind {self.kind!r}; the kinds are {KINDS}")

    def metadata(self) -> dict:
        out = dict(kind=self.kind)
        if self.basis:
            out["basis"] = self.basis
        if self.labels:
            out["labels"] = list(self.labels)
        return out


VECTOR = FieldInfo("vector", "coordinate", AXES)


@dataclass(frozen=True)
class Field4:
    """A field sampled on a `Grid4`: values [t, x, y, z, components...]."""
    grid: Grid4
    values: np.ndarray
    info: FieldInfo = field(default_factory=FieldInfo)

    def __post_init__(self):
        if tuple(self.values.shape[:4]) != self.grid.shape:
            raise ValueError(f"values {self.values.shape} do not start with the grid's shape {self.grid.shape}")
        if self.info.kind == "vector" and self.values.shape[4:] != (4,):
            raise ValueError("a vector field has four components along (t, x, y, z)")


# --- hypersurfaces ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class Hypersurface:
    """The hypersurface x^mu = c, `fixed_axis` = mu, `coordinate` = c, laid out along `order`, a permutation of the
    three remaining axes (default: their (t, x, y, z) order), the first varying fastest in the dataset."""
    fixed_axis: str
    coordinate: float
    order: tuple[str, str, str] | None = None

    def __post_init__(self):
        axis_index(self.fixed_axis)
        if self.order is not None and sorted(self.order) != sorted(self.remaining):
            raise ValueError(f"order {self.order} is not a permutation of the remaining axes {self.remaining}")

    @property
    def remaining(self) -> tuple[str, str, str]:
        """The three axes the hypersurface keeps, in (t, x, y, z) order: the physical hypersurface."""
        return tuple(name for name in AXES if name != self.fixed_axis)

    @property
    def display(self) -> tuple[str, str, str]:
        """The display axes, first fastest: the orientation of the exported dataset."""
        return tuple(self.order) if self.order is not None else self.remaining

    @property
    def temporal(self) -> bool:
        """True for a spatial snapshot t = c, the only kind a sequence of which is a time series."""
        return self.fixed_axis == "t"

    def metadata(self) -> dict:
        return dict(fixed_axis=self.fixed_axis, coordinate=float(self.coordinate), display_axes=list(self.display),
                    remaining_axes=list(self.remaining), spatial_snapshot=self.temporal)


@dataclass
class Section:
    """Fields on a hypersurface, ready for a VTK ImageData: `axes` the coordinate values along the display axes,
    `data[name]` shaped [third, second, first, components...] (first fastest), `info[name]` what each array is."""
    hypersurface: Hypersurface
    axes: tuple[np.ndarray, np.ndarray, np.ndarray]
    data: dict[str, np.ndarray]
    info: dict[str, FieldInfo]

    @property
    def dims(self) -> tuple[int, int, int]:
        return tuple(len(values) for values in self.axes)

    @property
    def origin(self) -> tuple[float, float, float]:
        return tuple(float(values[0]) for values in self.axes)

    @property
    def spacing(self) -> tuple[float, float, float]:
        return tuple(float(values[1] - values[0]) if len(values) > 1 else 1.0 for values in self.axes)

    def metadata(self) -> dict:
        return coordinate_metadata() | self.hypersurface.metadata() | dict(
            dims=list(self.dims), origin=list(self.origin), spacing=list(self.spacing),
            fields={name: self.info[name].metadata() for name in self.data})


def section_points(hypersurface: Hypersurface, axes: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[int, ...]]:
    """The events (t, x, y, z) of a hypersurface over the coordinate values `axes[name]` of its remaining axes, shape
    (N, 4) in the hypersurface's canonical index order (remaining axes in (t, x, y, z) order, the last fastest), and
    that canonical shape."""
    remaining = [np.asarray(axes[name], dtype=np.float64) for name in hypersurface.remaining]
    mesh = np.meshgrid(*remaining, indexing="ij")
    points = np.empty(mesh[0].shape + (4,))
    points[..., axis_index(hypersurface.fixed_axis)] = hypersurface.coordinate
    for name, values in zip(hypersurface.remaining, mesh):
        points[..., axis_index(name)] = values
    return points.reshape(-1, 4), mesh[0].shape


def shown(hypersurface: Hypersurface, info: dict[str, FieldInfo]) -> dict[str, FieldInfo]:
    """What a section holds for fields described by `info`: each vector field by its components along the display
    axes, in display order, under its own name, and all four as `name_txyz`; every other field as it is."""
    out = {}
    for name, kind in info.items():
        if kind.kind == "vector":
            out[name] = FieldInfo("vector", kind.basis, hypersurface.display)
            out[name + "_txyz"] = FieldInfo("components", kind.basis, AXES)
        else:
            out[name] = kind
    return out


def orient(hypersurface: Hypersurface, axes: dict[str, np.ndarray], canonical: dict[str, np.ndarray],
           info: dict[str, FieldInfo]) -> Section:
    """A section from arrays in the hypersurface's canonical order ([a, b, c, components...] for its remaining axes
    a, b, c in (t, x, y, z) order), transposed to VTK order for the display axes, vectors shown as `shown` says."""
    display = hypersurface.display
    # canonical axis k holds remaining[k]; VTK order lists the display axes slowest first.
    permutation = [hypersurface.remaining.index(name) for name in reversed(display)]
    data = {}
    for name, values in canonical.items():
        values = np.asarray(values)
        moved = values.transpose(permutation + list(range(3, values.ndim)))
        if info.get(name, FieldInfo()).kind == "vector":
            data[name] = moved[..., [axis_index(axis) for axis in display]]
            data[name + "_txyz"] = moved
        else:
            data[name] = moved
    kinds = shown(hypersurface, {name: info.get(name, FieldInfo()) for name in canonical})
    return Section(hypersurface, tuple(np.asarray(axes[name]) for name in display), data, kinds)


def extract(fields: dict[str, Field4], hypersurface: Hypersurface, tolerance: float = 1e-9) -> Section:
    """The section of fields sampled on one grid: the grid's own slice when the coordinate lies on a grid line (within
    `tolerance` of the spacing), else linear interpolation between the two neighboring slices, the interpolation that
    reproduces a sampled field exactly when it is linear along the fixed axis and makes no assumption beyond the
    samples. (Fields with a closed form, such as those of `export`, are better evaluated on the hypersurface itself.)"""
    grids = {id(value.grid) for value in fields.values()}
    if len(grids) != 1:
        raise ValueError("all fields must share one grid")
    grid = next(iter(fields.values())).grid
    k = axis_index(hypersurface.fixed_axis)
    values = grid.axes[k]
    c = float(hypersurface.coordinate)
    step = grid.spacing[k]
    nearest = int(np.argmin(np.abs(values - c)))
    if abs(values[nearest] - c) <= tolerance * step:
        weights = [(nearest, 1.0)]
    elif values[0] < c < values[-1]:
        upper = int(np.searchsorted(values, c))
        fraction = (c - values[upper - 1]) / (values[upper] - values[upper - 1])
        weights = [(upper - 1, 1.0 - fraction), (upper, fraction)]
    else:
        raise ValueError(f"{hypersurface.fixed_axis} = {c} lies outside the grid [{values[0]}, {values[-1]}]")
    canonical = {}
    for name, value in fields.items():
        parts = [weight * np.take(value.values, index, axis=k) for index, weight in weights]
        result = parts[0] if len(parts) == 1 else parts[0] + parts[1]
        if value.values.dtype.kind in "iub":                       # masks: valid only where both neighbors are
            result = np.min([np.take(value.values, index, axis=k) for index, _ in weights], axis=0)
        canonical[name] = result
    axes = {name: grid.axis(name) for name in hypersurface.remaining}
    return orient(hypersurface, axes, canonical, {name: value.info for name, value in fields.items()})
