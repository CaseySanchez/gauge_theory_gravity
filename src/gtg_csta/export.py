"""Export spacetime slices to ParaView as VTKHDF ImageData, one file per time step plus a `.vtkhdf.series`.

    PYTHONPATH=src python -m gtg_csta.export flyby --n 48 --frames 16 --out export/flyby
    PYTHONPATH=src python -m gtg_csta.export growing_mass --n 48 --frames 16 --out export/growing_mass

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
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np

from . import algebra as al
from . import cartan, conformal as cf, invariants as iv
from .curvature import curvature

INVARIANTS = ("weyl2_half", "riem2_half", "ricci_scalar")


# ------------------------------------------------------------------------------------------
# fields
# ------------------------------------------------------------------------------------------

def field_function(coframe):
    """x4 (jnp array) -> dict of the exported fields at that spacetime point (scalars and the shift vector)."""
    lorentz = cartan.lorentz_connection(coframe)
    schouten = cf.schouten(coframe)
    normal = cf.connection(coframe, schouten)
    F_normal = curvature(normal)
    I_lorentz = iv.quadratic_invariant(coframe, lorentz)
    I_normal = iv.quadratic_invariant(coframe, normal)

    def fields(k):
        x = al.point(k)
        F = F_normal(x)
        out = {
            "weyl2_half": I_normal(x).kernel.reshape(()),
            "riem2_half": I_lorentz(x).kernel.reshape(()),
            "ricci_scalar": cf.ricci_scalar(coframe, x).kernel.reshape(()),
        }
        for s in "MPKD":
            out["F_" + s] = jnp.sqrt(jnp.sum(al.sector(F, s).kernel ** 2))
        out["shift"] = coframe(x).kernel[1:, 0]
        return out
    return fields


def evaluate_slice(coframe, t: float, axes, chunk: int = 2048) -> dict:
    """Evaluate every field on the grid `axes = (xs, ys, zs)` at lab time t.  Returns arrays shaped
    (nz, ny, nx) (and (nz, ny, nx, 3) for the shift), x varying fastest, plus `valid`."""
    xs, ys, zs = axes
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    pts = np.stack([np.full(X.size, t), X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
    f = jax.jit(jax.vmap(field_function(coframe)))
    parts = []
    for i in range(0, len(pts), chunk):
        block = pts[i:i + chunk]
        pad = chunk - len(block)                                   # keep one compiled shape
        if pad:
            block = np.concatenate([block, np.repeat(block[-1:], pad, axis=0)])
        res = jax.tree_util.tree_map(np.asarray, f(jnp.asarray(block)))
        parts.append(jax.tree_util.tree_map(lambda a: a[:chunk - pad], res))
    flat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    shape = X.shape
    valid = np.all([np.isfinite(flat[k]).reshape(len(flat[k]), -1).all(axis=1) for k in flat], axis=0)
    data = {}
    for k, v in flat.items():
        v = np.where(valid.reshape((-1,) + (1,) * (v.ndim - 1)), v, 0.0)
        data[k] = v.reshape(*shape, *v.shape[1:])
    for k in INVARIANTS:
        data["log10_" + k] = np.sign(data[k]) * np.log10(np.abs(data[k]) + 1e-12)
    data["valid"] = valid.reshape(shape).astype(np.uint8)
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

def flyby(v: float = 0.6, M: float = 1.0):
    """Schwarzschild (mass M) moving along +x at speed v, passing the origin at t = 0 (a boosted PG tetrad).
    Vacuum: Weyl^2 is the boosted r^-6 profile, Ricci = 0, Cotton = 0."""
    return cartan.coframe_field(cartan.boosted(cartan.pg_matrix(lambda t: M), v))


def growing_mass(M0: float = 1.0, dM: float = 1.0, t0: float = 0.0, width: float = 2.0):
    """PG-form geometry with M(t) = M0 + dM/2 (1 + tanh((t - t0)/width)).  A toy with matter (Ricci != 0), so
    the special-conformal sector (Cotton) is switched on while the mass changes."""
    return cartan.coframe_field(cartan.pg_matrix(lambda t: M0 + 0.5 * dM * (1.0 + jnp.tanh((t - t0) / width))))


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
        path = export_sequence(flyby(), times, out, "flyby", a.n, half, meta=dict(case="flyby", v=0.6, M=1.0))
    else:
        half = a.half or 8.0
        times = np.linspace(-6.0, 6.0, a.frames)
        path = export_sequence(growing_mass(), times, out, "growing_mass", a.n, half,
                               meta=dict(case="growing_mass", M0=1.0, dM=1.0, t0=0.0, width=2.0))
    print("wrote", path)


if __name__ == "__main__":
    main(sys.argv[1:])
