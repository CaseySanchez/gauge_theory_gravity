"""VTKHDF export.  The writer test reads files back with VTK's own reader (vtk must be installed)."""
import json

import numpy as np
import pytest

vtk = pytest.importorskip("vtk")

import gtg_csta  # noqa: F401
from gtg_csta import export as ex


def _read(path):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    r = vtk.vtkHDFReader()
    r.SetFileName(str(path))
    r.Update()
    return r.GetOutput(), vtk_to_numpy


def test_writer_layout_is_read_back_by_vtk_with_correct_point_order(tmp_path):
    """Write the coordinates themselves as point data; VTK must place point i at the coordinates stored for it."""
    nx, ny, nz = 5, 4, 3
    origin, spacing = (-1.0, 2.0, 0.5), (0.5, 0.25, 2.0)
    xs, ys, zs = (np.arange(n) * s + o for n, s, o in zip((nx, ny, nz), spacing, origin))
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    data = {"x": X, "y": Y, "z": Z, "pos": np.stack([X, Y, Z], axis=-1), "flag": (X > 0).astype(np.uint8)}
    path = tmp_path / "g.vtkhdf"
    ex.write_vtkhdf_imagedata(path, (nx, ny, nz), origin, spacing, data)
    img, to_np = _read(path)
    assert img.GetDimensions() == (nx, ny, nz)
    pts = np.array([img.GetPoint(i) for i in range(img.GetNumberOfPoints())])
    pd = img.GetPointData()
    for name, col in (("x", 0), ("y", 1), ("z", 2)):
        assert np.allclose(to_np(pd.GetArray(name)), pts[:, col], atol=1e-6), name
    assert np.allclose(to_np(pd.GetArray("pos")), pts, atol=1e-6)
    assert np.array_equal(to_np(pd.GetArray("flag")), (pts[:, 0] > 0).astype(np.uint8))


def test_series_file(tmp_path):
    p = ex.write_series(tmp_path, "run", [0.0, 0.5, 1.0])
    s = json.loads(p.read_text())
    assert s["file-series-version"] == "1.0"
    assert [f["name"] for f in s["files"]] == ["run_0000.vtkhdf", "run_0001.vtkhdf", "run_0002.vtkhdf"]
    assert [f["time"] for f in s["files"]] == [0.0, 0.5, 1.0]


@pytest.mark.slow
def test_tiny_slice_of_schwarzschild_flyby_has_expected_fields(tmp_path):
    """Pointwise fields on a 3x3x3 grid: vacuum has Ricci = 0, Cotton = 0, torsion = 0, and the invariant is the
    boosted Kretschmann.  Not part of the default run (it triggers JAX compilation of the curvature chain)."""
    out = ex.export_sequence(ex.flyby(v=0.6), [0.0], tmp_path, "tiny", n=3, half=6.0, chunk=8)
    img, to_np = _read(tmp_path / "tiny_0000.vtkhdf")
    pd = img.GetPointData()
    assert np.abs(to_np(pd.GetArray("ricci_scalar"))).max() < 1e-8
    assert to_np(pd.GetArray("F_K")).max() < 1e-8 and to_np(pd.GetArray("F_P")).max() < 1e-8
    assert out.name == "tiny.vtkhdf.series"
