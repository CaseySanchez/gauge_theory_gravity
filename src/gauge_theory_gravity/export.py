"""Export spacetime fields to ParaView as VTKHDF: three-dimensional hypersurfaces of the four-dimensional spacetime.

    PYTHONPATH=src python -m gauge_theory_gravity.export flyby --n 48 --frames 16 --out export/flyby
    PYTHONPATH=src python -m gauge_theory_gravity.export kerr_flyby --n 48 --frames 16 --fixed x --order tyz
    PYTHONPATH=src python -m gauge_theory_gravity.export growing_mass --n 48 --frames 16 --components

Every field is a function on the coordinate domain (t, x, y, z) (`hypersurface`). A dataset is the hypersurface
x^mu = c for one fixed coordinate mu, laid out along a chosen order of the other three:
  * fixed t: a spatial snapshot over (x, y, z); a sequence of them is a time series, written with a
    `.vtkhdf.series` that ParaView's time slider steps through;
  * fixed x, y or z: a section over time and the two remaining spatial axes, e.g. (t, y, z); a sequence of them is
    not a time series, so it gets an index, `stem.sections.json`, and no time metadata.
The fields are evaluated on the hypersurface itself, pointwise, by the same extensor code as everything else; a 4D
grid of samples can also be evaluated once (`evaluate_grid`) and cut with `hypersurface.extract`, which interpolates
between grid slices. `stem.json` records the coordinate conventions and, per field, its kind, basis and labels.

Open `export/flyby/flyby.vtkhdf.series` in ParaView (>= 5.12 reads `.vtkhdf` and `.series`).

Cost (CPU, measured on a 4000-point chunk): ~0.65 ms per grid point, so ~70 s per section at n = 48.
Fields are chunked and evaluated with `jit(vmap(...))`; the chunk size bounds memory.

Fields (point data; magnitudes are L2 norms of the blade coefficients, so they depend on the frame):
  weyl2_half      <F F~> of the normal conformal connection (f = Schouten) = C^2 / 2   (gauge invariant)
  weyl2_dual_half C.*C / 2, the imaginary part of the twistor trace of the curvature squared over the omega
                  half (`twistor.weyl_invariant`), whose real part is weyl2_half; zero for static spherical holes,
                  Im 24 M^2 / (r - i a cos theta)^6 for Kerr                             (gauge invariant)
  riem2_half      <F F~> of the Lorentz-only connection = Riemann^2 / 2                 (gauge invariant)
  ricci_scalar    R                                                                      (gauge invariant)
  F_M F_P F_K F_D |sector| of the normal connection's curvature: Lorentz (Weyl), translation (torsion = 0),
                  special conformal (Cotton), dilation (= 0)
  shift           the translation-sector vector e(d_t), frame components along the display axes (on t = c the
                  spatial part e^i_0, the PG shift for the Schwarzschild examples); shift_txyz all four
  log10_*         sign-preserving log10(|x| + 1e-12) of the invariants, for color mapping
  valid           1 where every field is finite (0 at the singularity)
With `components`, also the curvature itself, per sector S: S_components, the generator coefficients of
F(d_mu, d_nu) on the six coordinate planes, labeled `G(mu nu)` for generator G (M_tx ... D) and plane mu nu.
"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import h5py
import numpy as np
from numga.extensor import Extensor

from . import algebra as al
from . import cartan, conformal as cf, hypersurface as hs, invariants as iv, twistor as tw
from .cartan import Coframe
from .curvature import curvature

INVARIANTS = ("weyl2_half", "weyl2_dual_half", "riem2_half", "ricci_scalar")
SECTOR_NAMES = ("M", "P", "K", "D")
PLANES = [(a, b) for a in range(4) for b in range(a + 1, 4)]           # the coordinate planes mu < nu


def component_labels(sector_name: str) -> tuple[str, ...]:
    """Labels of a sector's components: generator, then the coordinate plane it is read on, plane fastest."""
    generators = al.LABELS[al.SECTORS[sector_name]]
    return tuple(f"{generator}({hs.AXES[a]}{hs.AXES[b]})" for generator in generators for a, b in PLANES)


def field_info(components: bool = False) -> dict[str, hs.FieldInfo]:
    """What each exported array is."""
    info = {name: hs.FieldInfo() for name in INVARIANTS + tuple("F_" + s for s in SECTOR_NAMES)}
    info |= {"log10_" + name: hs.FieldInfo() for name in INVARIANTS}
    info["shift"] = hs.FieldInfo("vector", "frame", hs.AXES)
    info["valid"] = hs.FieldInfo()
    if components:
        info |= {f"F_{s}_components": hs.FieldInfo("components", "generator x coordinate plane", component_labels(s))
                 for s in SECTOR_NAMES}
    return info


# ------------------------------------------------------------------------------------------
# fields
# ------------------------------------------------------------------------------------------

def field_function(coframe: Coframe, components: bool = False) -> Callable[[Extensor], dict[str, Extensor]]:
    """position (a batch of them is fine) -> the extensors the exported fields are read from. With `components`, also
    the curvature on the coordinate planes, `planes`: F(d_mu, d_nu) for every pair, a bivector per pair."""
    lorentz = cartan.lorentz_connection(coframe)
    normal = cf.connection(coframe, cf.schouten(coframe))
    strength = curvature(normal)
    normal_density = iv.quadratic_invariant(coframe, normal)
    lorentz_density = iv.quadratic_invariant(coframe, lorentz)
    twistor_density = tw.weyl_invariant(coframe, normal)
    directions = al.DIRECTIONS

    def fields(x: Extensor) -> dict[str, Extensor]:
        curved = strength(x)
        out = {
            "weyl2_half": normal_density(x),
            "weyl2_dual_half": tw.imaginary_part(twistor_density(x)),
            "riem2_half": lorentz_density(x),
            "ricci_scalar": cf.ricci_scalar(coframe, x),
            "F_M": al.sector(curved, "M"), "F_P": al.sector(curved, "P"),
            "F_K": al.sector(curved, "K"), "F_D": al.sector(curved, "D"),
            "shift": coframe(x)(al.time),
        }
        if components:
            out["planes"] = curved[..., None, None](directions[:, None], directions[None, :]).cast(al.Bivector)
        return out

    return fields


def readout(fields: dict[str, Extensor]) -> dict[str, np.ndarray]:
    """The numpy arrays written to file, per grid point (leading axis): scalars, sector magnitudes (L2 norm of the
    coefficients), the four frame components of e(d_t) and, when the curvature on the coordinate planes was
    evaluated, each sector's generator coefficients on the six planes (generator-major)."""
    out = {name: np.asarray(fields[name].kernel).reshape(len(fields[name].kernel), -1)[:, 0]
           for name in INVARIANTS}
    for name in ("F_M", "F_P", "F_K", "F_D"):
        kernel = np.asarray(fields[name].kernel)
        out[name] = np.sqrt((kernel.reshape(len(kernel), -1) ** 2).sum(axis=1))
    out["shift"] = np.asarray(al.coordinates(fields["shift"]))
    if "planes" in fields:
        generators = al.to_generators(np.asarray(fields["planes"].kernel))              # (N, 4, 4, 15)
        on_planes = np.stack([generators[:, a, b] for a, b in PLANES], axis=1)          # (N, 6, 15)
        for name in SECTOR_NAMES:
            block = on_planes[:, :, al.SECTORS[name]]                                    # (N, 6, generators)
            out[f"F_{name}_components"] = block.transpose(0, 2, 1).reshape(len(block), -1)
    return out


def evaluate_points(coframe: Coframe, points: np.ndarray, chunk: int = 2048,
                    components: bool = False) -> dict[str, np.ndarray]:
    """Every exported field at spacetime points (N, 4), as numpy arrays with a leading axis of N. Non-finite points
    are set to zero in every field and flagged by `valid`. The points go through the extensor code in chunks, each
    one batch of positions under `jit`."""
    import jax

    @jax.jit
    def evaluate(positions: Extensor) -> dict[str, Extensor]:
        return field_function(coframe, components)(positions)

    parts = []
    for start in range(0, len(points), chunk):
        block = points[start:start + chunk]
        padding = chunk - len(block)                                   # keep one compiled shape
        if padding:
            block = np.concatenate([block, np.repeat(block[-1:], padding, axis=0)])
        values = readout(evaluate(al.point(block)))
        parts.append({name: array[:chunk - padding] for name, array in values.items()})
    flat = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
    valid = np.all([np.isfinite(array).reshape(len(array), -1).all(axis=1) for array in flat.values()], axis=0)
    data = {name: np.where(valid.reshape((-1,) + (1,) * (array.ndim - 1)), array, 0.0) for name, array in flat.items()}
    data["valid"] = valid.astype(np.uint8)
    return data


def with_logarithms(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The sign-preserving log10 of each invariant added."""
    return data | {"log10_" + name: np.sign(data[name]) * np.log10(np.abs(data[name]) + 1e-12) for name in INVARIANTS}


def evaluate_section(coframe: Coframe, hypersurface: hs.Hypersurface, axes: dict[str, np.ndarray],
                     chunk: int = 2048, components: bool = False) -> hs.Section:
    """Every field on a hypersurface, evaluated at its own events: `axes[name]` the coordinate values along each of
    its remaining axes. Laid out along the hypersurface's display axes."""
    points, shape = hs.section_points(hypersurface, axes)
    flat = with_logarithms(evaluate_points(coframe, points, chunk, components))
    canonical = {name: array.reshape(*shape, *array.shape[1:]) for name, array in flat.items()}
    return hs.orient(hypersurface, axes, canonical, field_info(components))


def evaluate_grid(coframe: Coframe, grid: hs.Grid4, chunk: int = 2048,
                  components: bool = False) -> dict[str, hs.Field4]:
    """Every field sampled on a 4D grid, for cutting with `hypersurface.extract`."""
    flat = with_logarithms(evaluate_points(coframe, grid.points(), chunk, components))
    info = field_info(components)
    return {name: hs.Field4(grid, array.reshape(*grid.shape, *array.shape[1:]), info[name])
            for name, array in flat.items()}


def evaluate_slice(coframe: Coframe, t: float, axes, chunk: int = 2048) -> dict:
    """Every field on the grid `axes = (xs, ys, zs)` at lab time t, shaped (nz, ny, nx) (and (nz, ny, nx, 3) for the
    shift), x varying fastest, plus the log10 versions and `valid`: the spatial snapshot t = c."""
    section = evaluate_section(coframe, hs.Hypersurface("t", t), dict(zip("xyz", axes)), chunk)
    return section.data


# ------------------------------------------------------------------------------------------
# VTKHDF writer
# ------------------------------------------------------------------------------------------

def write_vtkhdf_imagedata(path, dims, origin, spacing, point_data: dict, metadata: dict | None = None) -> None:
    """Write a regular grid as a VTKHDF ImageData file (VTK >= 9.1 / ParaView >= 5.10).
    dims = (nx, ny, nz) points; point_data[name] shaped (nz, ny, nx) or (nz, ny, nx, ncomp), x fastest. `metadata`
    (JSON) goes in an attribute of the file's root, outside the VTKHDF group, where readers do not look."""
    nx, ny, nz = dims
    with h5py.File(path, "w") as f:
        g = f.create_group("VTKHDF")
        g.attrs["Version"] = np.array([2, 0], dtype=np.int64)
        g.attrs.create("Type", np.bytes_("ImageData"), dtype=h5py.string_dtype("ascii", 9))
        g.attrs["WholeExtent"] = np.array([0, nx - 1, 0, ny - 1, 0, nz - 1], dtype=np.int64)
        g.attrs["Origin"] = np.asarray(origin, dtype=np.float64)
        g.attrs["Spacing"] = np.asarray(spacing, dtype=np.float64)
        g.attrs["Direction"] = np.eye(3, dtype=np.float64).ravel()
        pd = g.create_group("PointData")
        for name, a in point_data.items():
            a = np.asarray(a)
            if a.shape[:3] != (nz, ny, nx):
                raise ValueError(f"{name}: shape {a.shape} does not match (nz, ny, nx) = {(nz, ny, nx)}")
            if a.ndim > 4:
                a = a.reshape(nz, ny, nx, -1)
            if a.dtype == np.float64:
                a = a.astype(np.float32)
            pd.create_dataset(name, data=a, compression="gzip", compression_opts=4)
        if metadata is not None:
            f.attrs["metadata"] = json.dumps(metadata)


def write_vtkhdf_polydata(path, points: np.ndarray, lines: list[np.ndarray], point_data: dict,
                          cell_data: dict | None = None, metadata: dict | None = None) -> None:
    """Write polylines as a VTKHDF PolyData file: `points` (N, 3), `lines` the point indices of each polyline,
    `point_data[name]` (N,) or (N, ncomp), `cell_data[name]` one row per line."""
    points = np.asarray(points, dtype=np.float64)
    connectivity = np.concatenate(lines).astype(np.int64) if lines else np.zeros(0, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum([len(line) for line in lines])]).astype(np.int64)
    with h5py.File(path, "w") as f:
        g = f.create_group("VTKHDF")
        g.attrs["Version"] = np.array([2, 0], dtype=np.int64)
        g.attrs.create("Type", np.bytes_("PolyData"), dtype=h5py.string_dtype("ascii", 8))
        g.create_dataset("NumberOfPoints", data=np.array([len(points)], dtype=np.int64))
        g.create_dataset("Points", data=points)
        for kind in ("Vertices", "Lines", "Polygons", "Strips"):
            cells = g.create_group(kind)
            used = kind == "Lines"
            cells.create_dataset("NumberOfCells", data=np.array([len(lines) if used else 0], dtype=np.int64))
            cells.create_dataset("NumberOfConnectivityIds",
                                 data=np.array([len(connectivity) if used else 0], dtype=np.int64))
            cells.create_dataset("Offsets", data=offsets if used else np.zeros(1, dtype=np.int64))
            cells.create_dataset("Connectivity", data=connectivity if used else np.zeros(0, dtype=np.int64))
        for group_name, data, count in (("PointData", point_data, len(points)),
                                        ("CellData", cell_data or {}, len(lines))):
            group = g.create_group(group_name)
            for name, a in data.items():
                a = np.asarray(a)
                if len(a) != count:
                    raise ValueError(f"{name}: {len(a)} rows, {count} expected")
                if a.dtype == np.float64:
                    a = a.astype(np.float32)
                group.create_dataset(name, data=a)
        if metadata is not None:
            f.attrs["metadata"] = json.dumps(metadata)


def write_section(path, section: hs.Section) -> None:
    """One hypersurface as a VTKHDF ImageData file, its display axes as VTK's x, y, z."""
    write_vtkhdf_imagedata(path, section.dims, section.origin, section.spacing, section.data, section.metadata())


def write_series(directory, stem, times) -> Path:
    """`stem.vtkhdf.series`: ParaView's JSON file series for per-step files `stem_0000.vtkhdf`, ...; only for
    spatial snapshots at successive times."""
    directory = Path(directory)
    series = {"file-series-version": "1.0",
              "files": [{"name": f"{stem}_{i:04d}.vtkhdf", "time": float(t)} for i, t in enumerate(times)]}
    path = directory / f"{stem}.vtkhdf.series"
    path.write_text(json.dumps(series, indent=2))
    return path


def write_index(directory, stem, hypersurfaces: list[hs.Hypersurface]) -> Path:
    """`stem.sections.json`: the files of a sequence of hypersurfaces that is not a time series, each with its fixed
    axis, coordinate and display axes."""
    directory = Path(directory)
    index = dict(time_series=False, sections=[dict(name=f"{stem}_{i:04d}.vtkhdf", **surface.metadata())
                                              for i, surface in enumerate(hypersurfaces)])
    path = directory / f"{stem}.sections.json"
    path.write_text(json.dumps(index, indent=2))
    return path


def export_sections(coframe, hypersurfaces: list[hs.Hypersurface], grid: hs.Grid4, out, stem, chunk=2048,
                    components: bool = False, meta: dict | None = None, via_grid: bool = False) -> Path:
    """Evaluate the fields on each hypersurface over the grid's values of its remaining axes and write one file per
    hypersurface, plus a `.vtkhdf.series` when they are spatial snapshots (a time series) and a `.sections.json`
    index otherwise. With `via_grid`, the fields are sampled once on the whole 4D grid and each hypersurface cut from
    it (interpolating between grid slices); otherwise each is evaluated at its own events."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sampled = evaluate_grid(coframe, grid, chunk, components) if via_grid else None
    for i, surface in enumerate(hypersurfaces):
        if sampled is not None:
            section = hs.extract(sampled, surface)
        else:
            section = evaluate_section(coframe, surface, {name: grid.axis(name) for name in surface.remaining},
                                       chunk, components)
        write_section(out / f"{stem}_{i:04d}.vtkhdf", section)
        print(f"{stem}: section {i + 1}/{len(hypersurfaces)}  {surface.fixed_axis} = {surface.coordinate:g}",
              flush=True)
    shown = hs.shown(hypersurfaces[0], field_info(components))
    record = dict(grid=grid.metadata(), fields={name: value.metadata() for name, value in shown.items()},
                  hypersurfaces=[surface.metadata() for surface in hypersurfaces])
    (out / f"{stem}.json").write_text(json.dumps(record | dict(case=meta or {}), indent=2))
    if all(surface.temporal for surface in hypersurfaces):
        return write_series(out, stem, [surface.coordinate for surface in hypersurfaces])
    return write_index(out, stem, hypersurfaces)


def box_grid(times, n=48, half=10.0, center=(0.0, 0.0, 0.0)) -> hs.Grid4:
    """The 4D grid of a box: n points on each spatial axis over center +- half (shifted by 1e-3 of a cell, which
    keeps r = 0 off the grid), n points in time over the range of `times`."""
    offset = 1e-3 * half / n
    bounds = [(min(times), max(times))] + [(c - half, c + half) for c in center]
    shape = (n if len(times) > 1 and min(times) < max(times) else 1, n, n, n)
    return hs.Grid4.regular(bounds, shape, (0.0, offset, offset, offset))


def export_sequence(coframe, times, out, stem, n=48, half=10.0, center=(0.0, 0.0, 0.0), chunk=2048,
                    meta: dict | None = None, components: bool = False) -> Path:
    """Spatial snapshots t = time of an n^3 grid of half-width `half`, written as a time series."""
    surfaces = [hs.Hypersurface("t", float(t)) for t in times]
    return export_sections(coframe, surfaces, box_grid(times, n, half, center), out, stem, chunk, components, meta)


# ------------------------------------------------------------------------------------------
# the spacetimes
# ------------------------------------------------------------------------------------------

def spinning_flyby(speed: float = 0.6, mass: float = 1.0, spin: float = 0.9) -> Coframe:
    """A Kerr hole (spin along +z) moving along +x at the given speed: vacuum, with a nonzero C.*C."""
    return cartan.boosted(cartan.kerr(mass, spin), speed)


def flyby(speed: float = 0.6, mass: float = 1.0) -> Coframe:
    """Schwarzschild (mass M) moving along +x at the given speed, passing the origin at t = 0 (a boosted PG tetrad).
    Vacuum: Weyl^2 is the boosted r^-6 profile, Ricci = 0, Cotton = 0."""
    return cartan.boosted(cartan.schwarzschild(mass), speed)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", choices=["flyby", "kerr_flyby", "growing_mass"])
    ap.add_argument("--n", type=int, default=48, help="grid points per axis")
    ap.add_argument("--frames", type=int, default=16, help="number of hypersurfaces along the fixed axis")
    ap.add_argument("--half", type=float, default=None, help="half-width of the box (default 12 flyby, 8 growing_mass)")
    ap.add_argument("--fixed", choices=list(hs.AXES), default="t", help="the coordinate each hypersurface fixes")
    ap.add_argument("--order", default=None, help="display order of the remaining axes, e.g. tyz (default t,x,y,z order)")
    ap.add_argument("--components", action="store_true", help="also export each sector's curvature components")
    ap.add_argument("--via-grid", action="store_true", help="sample the 4D grid once and cut it (interpolating)")
    ap.add_argument("--out", default=None, help="output directory (default export/<case>)")
    a = ap.parse_args(argv)
    out = a.out or f"export/{a.case}"
    if a.case in ("flyby", "kerr_flyby"):
        half = a.half or 12.0
        times = (-half, half)
        coframe = flyby() if a.case == "flyby" else spinning_flyby()
        meta = dict(case=a.case, speed=0.6, mass=1.0, spin=0.0 if a.case == "flyby" else 0.9)
    else:
        half = a.half or 8.0
        times = (-6.0, 6.0)
        coframe = cartan.growing_mass()
        meta = dict(case="growing_mass", mass=1.0, growth=1.0, at=0.0, width=2.0)
    grid = box_grid(times, a.n, half)
    k = hs.axis_index(a.fixed)
    low, high = (times if a.fixed == "t" else (grid.axes[k][0], grid.axes[k][-1]))
    order = tuple(a.order) if a.order else None
    surfaces = [hs.Hypersurface(a.fixed, float(c), order) for c in np.linspace(low, high, a.frames)]
    path = export_sections(coframe, surfaces, grid, out, a.case, components=a.components, meta=meta,
                           via_grid=a.via_grid)
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1:])
