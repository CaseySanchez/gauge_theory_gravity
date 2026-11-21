"""Export spacetime slices to ParaView as VTKHDF ImageData, one file per time step plus a `.vtkhdf.series`.

    PYTHONPATH=src python -m gauge_theory_gravity.export flyby --n 48 --frames 16 --out export/flyby
    PYTHONPATH=src python -m gauge_theory_gravity.export growing_mass --n 48 --frames 16 --out export/growing_mass

Open `export/flyby/flyby.vtkhdf.series` in ParaView (>= 5.12 reads `.vtkhdf` and `.series`); the time slider then
steps through the slices.  Each step is the field content of the CSTA connection on the spatial grid at
fixed lab time t, evaluated pointwise by the same extensor code as everything else.

Cost (CPU, measured on a 4000-point chunk): ~0.65 ms per grid point, so ~70 s per frame at n = 48.
Fields are chunked and evaluated with `jit(vmap(...))`; the chunk size bounds memory.

Fields (point data; magnitudes are L2 norms of the blade coefficients, so they depend on the frame):
  weyl2_half      <F F~> of the normal conformal connection (f = Schouten) = C^2 / 2   (gauge invariant)
  riem2_half      <F F~> of the Lorentz-only connection = Riemann^2 / 2                 (gauge invariant)
  ricci_scalar    R                                                                      (gauge invariant)
  F_M F_P F_K F_D |sector| of the normal connection's curvature: Lorentz (Weyl), translation (torsion = 0),
                  special conformal (Cotton), dilation (= 0)
  shift           translation-sector vector e^i_0 (3 components; the PG shift for the Schwarzschild examples)
  log10_*         sign-preserving log10(|x| + 1e-12) of the invariants, for colour mapping
  valid           1 where every field is finite (0 at the singularity)
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
from . import cartan, conformal as cf, invariants as iv
from .cartan import Coframe
from .curvature import curvature

INVARIANTS = ("weyl2_half", "riem2_half", "ricci_scalar")


# ------------------------------------------------------------------------------------------
# fields
# ------------------------------------------------------------------------------------------

def field_function(coframe: Coframe) -> Callable[[Extensor], dict[str, Extensor]]:
    """position (a batch of them is fine) -> the extensors the exported fields are read from."""
    lorentz = cartan.lorentz_connection(coframe)
    normal = cf.connection(coframe, cf.schouten(coframe))
    strength = curvature(normal)
    normal_density = iv.quadratic_invariant(coframe, normal)
    lorentz_density = iv.quadratic_invariant(coframe, lorentz)

    def fields(x: Extensor) -> dict[str, Extensor]:
        curved = strength(x)
        return {
            "weyl2_half": normal_density(x),
            "riem2_half": lorentz_density(x),
            "ricci_scalar": cf.ricci_scalar(coframe, x),
            "F_M": al.sector(curved, "M"), "F_P": al.sector(curved, "P"),
            "F_K": al.sector(curved, "K"), "F_D": al.sector(curved, "D"),
            "shift": coframe(x)(al.time),
        }

    return fields


def readout(fields: dict[str, Extensor]) -> dict[str, np.ndarray]:
    """The numpy arrays written to file, per grid point (leading axis): scalars, sector magnitudes (L2 norm of the
    coefficients) and the shift e^i_0."""
    out = {name: np.asarray(fields[name].kernel).reshape(len(fields[name].kernel), -1)[:, 0]
           for name in INVARIANTS}
    for name in ("F_M", "F_P", "F_K", "F_D"):
        kernel = np.asarray(fields[name].kernel)
        out[name] = np.sqrt((kernel.reshape(len(kernel), -1) ** 2).sum(axis=1))
    out["shift"] = al.coordinates(fields["shift"])[:, 1:]
    return out


def evaluate_points(coframe: Coframe, points: np.ndarray, chunk: int = 2048) -> dict[str, np.ndarray]:
    """Every exported field at spacetime points (N, 4), as numpy arrays with a leading axis of N. Non-finite points
    are set to zero in every field and flagged by `valid`. The points go through the extensor code in chunks, each
    one batch of positions under `jit`."""
    import jax

    @jax.jit
    def evaluate(positions: Extensor) -> dict[str, Extensor]:
        return field_function(coframe)(positions)

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


def evaluate_slice(coframe: Coframe, t: float, axes, chunk: int = 2048) -> dict:
    """Evaluate every field on the grid `axes = (xs, ys, zs)` at lab time t.  Returns arrays shaped
    (nz, ny, nx) (and (nz, ny, nx, 3) for the shift), x varying fastest, plus the log10 versions and `valid`."""
    xs, ys, zs = axes
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    points = np.stack([np.full(X.size, t), X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    flat = evaluate_points(coframe, points, chunk)
    data = {name: array.reshape(*X.shape, *array.shape[1:]) for name, array in flat.items()}
    for name in INVARIANTS:
        data["log10_" + name] = np.sign(data[name]) * np.log10(np.abs(data[name]) + 1e-12)
    return data


# ------------------------------------------------------------------------------------------
# VTKHDF writer
# ------------------------------------------------------------------------------------------

def write_vtkhdf_imagedata(path, dims, origin, spacing, point_data: dict) -> None:
    """Write a regular grid as a VTKHDF ImageData file (VTK >= 9.1 / ParaView >= 5.10).
    dims = (nx, ny, nz) points; point_data[name] shaped (nz, ny, nx) or (nz, ny, nx, ncomp), x fastest."""
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
            if a.dtype == np.float64:
                a = a.astype(np.float32)
            pd.create_dataset(name, data=a, compression="gzip", compression_opts=4)


def write_series(directory, stem, times) -> Path:
    """`stem.vtkhdf.series`: ParaView's JSON file series for per-step files `stem_0000.vtkhdf`, ..."""
    directory = Path(directory)
    series = {"file-series-version": "1.0",
              "files": [{"name": f"{stem}_{i:04d}.vtkhdf", "time": float(t)} for i, t in enumerate(times)]}
    path = directory / f"{stem}.vtkhdf.series"
    path.write_text(json.dumps(series, indent=2))
    return path


def export_sequence(coframe, times, out, stem, n=48, half=10.0, center=(0.0, 0.0, 0.0), chunk=2048,
                    meta: dict | None = None) -> Path:
    """Evaluate the fields on an n^3 grid of half-width `half` at each time and write the series."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    axes = tuple(np.linspace(c - half, c + half, n) + 1e-3 * half / n for c in center)   # keep r = 0 off the grid
    origin = tuple(a[0] for a in axes)
    spacing = tuple(a[1] - a[0] for a in axes)
    for i, t in enumerate(times):
        data = evaluate_slice(coframe, float(t), axes, chunk)
        write_vtkhdf_imagedata(out / f"{stem}_{i:04d}.vtkhdf", (n, n, n), origin, spacing, data)
        print(f"{stem}: step {i + 1}/{len(times)}  t = {t:g}", flush=True)
    if meta:
        (out / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return write_series(out, stem, times)


# ------------------------------------------------------------------------------------------
# the two spacetimes
# ------------------------------------------------------------------------------------------

def flyby(speed: float = 0.6, mass: float = 1.0) -> Coframe:
    """Schwarzschild (mass M) moving along +x at the given speed, passing the origin at t = 0 (a boosted PG tetrad).
    Vacuum: Weyl^2 is the boosted r^-6 profile, Ricci = 0, Cotton = 0."""
    return cartan.boosted(cartan.schwarzschild(mass), speed)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", choices=["flyby", "growing_mass"])
    ap.add_argument("--n", type=int, default=48, help="grid points per axis")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--half", type=float, default=None, help="half-width of the box (default 12 flyby, 8 growing_mass)")
    ap.add_argument("--out", default=None, help="output directory (default export/<case>)")
    a = ap.parse_args(argv)
    out = a.out or f"export/{a.case}"
    if a.case == "flyby":
        half = a.half or 12.0
        times = np.linspace(-half, half, a.frames)
        path = export_sequence(flyby(), times, out, "flyby", a.n, half, meta=dict(case="flyby", speed=0.6, mass=1.0))
    else:
        half = a.half or 8.0
        times = np.linspace(-6.0, 6.0, a.frames)
        path = export_sequence(cartan.growing_mass(), times, out, "growing_mass", a.n, half,
                               meta=dict(case="growing_mass", mass=1.0, growth=1.0, at=0.0, width=2.0))
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1:])
