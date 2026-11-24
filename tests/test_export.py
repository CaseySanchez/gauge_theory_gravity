"""VTKHDF export.  The writer test reads files back with VTK's own reader (vtk must be installed)."""
import json

import numpy as np
import pytest

vtk = pytest.importorskip("vtk")

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import export as ex


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


def test_dual_weyl_field_is_kerr_chern_pontryagin_and_zero_for_schwarzschild():
    """weyl2_dual_half at two events, evaluated eagerly through the export's own field function and readout."""
    from gauge_theory_gravity import algebra as al, cartan
    mass, spin = 1.0, 0.9
    events = np.array([[0.0, 3.0, 1.0, 2.0], [0.0, -2.5, 1.5, -1.0]])
    kerr = ex.readout(ex.field_function(cartan.kerr(mass, spin))(al.point(events)))
    radius = np.asarray(cartan.kerr_schild_fields(al.point(events), mass, spin)[0].kernel)[..., 0]
    expected = 24 * mass ** 2 / (radius - 1j * spin * events[:, 3] / radius) ** 6
    assert np.abs(kerr["weyl2_dual_half"] - expected.imag).max() < 1e-10 * np.abs(expected).max()
    assert np.abs(kerr["weyl2_half"] - expected.real).max() < 1e-10 * np.abs(expected).max()
    schwarzschild = ex.readout(ex.field_function(cartan.schwarzschild(mass))(al.point(events)))
    assert np.abs(schwarzschild["weyl2_dual_half"]).max() < 1e-12


@pytest.mark.slow
def test_small_slice_of_schwarzschild_flyby_has_expected_fields(tmp_path):
    """Fields on a 4x4x4 grid (even, so no point lies near the singularity): in vacuum Ricci = 0, Cotton = 0,
    torsion = 0, and the normal-connection density equals the Lorentz-only one. Triggers JAX compilation of the whole
    curvature chain on a batch of positions."""
    out = ex.export_sequence(ex.flyby(speed=0.6), [0.0], tmp_path, "small", n=4, half=6.0, chunk=16)
    img, to_np = _read(tmp_path / "small_0000.vtkhdf")
    pd = img.GetPointData()
    field = lambda name: to_np(pd.GetArray(name)).astype(float)
    assert np.abs(field("ricci_scalar")).max() < 1e-8
    assert field("F_K").max() < 1e-8 and field("F_P").max() < 1e-8 and field("F_D").max() < 1e-8
    assert np.abs(field("weyl2_half") - field("riem2_half")).max() < 1e-8 * (1 + np.abs(field("riem2_half")).max())
    assert field("riem2_half").max() > 1e-5 and field("valid").min() == 1
    assert out.name == "small.vtkhdf.series"


def test_sector_components_rebuild_the_curvature_on_each_coordinate_plane():
    """Each sector's generator coefficients on the six coordinate planes, summed against the generators, give back
    that sector of F(d_mu, d_nu); the four sectors together give F(d_mu, d_nu)."""
    from gauge_theory_gravity import algebra as al, cartan
    from gauge_theory_gravity.curvature import curvature
    from gauge_theory_gravity import conformal as cf
    coframe = cartan.kerr(1.0, 0.9)
    events = np.array([[0.0, 3.0, 1.0, 2.0], [0.5, -2.5, 1.5, -1.0]])
    data = ex.readout(ex.field_function(coframe, components=True)(al.point(events)))
    strength = curvature(cf.connection(coframe, cf.schouten(coframe)))(al.point(events))
    total = np.zeros((2, 6, 15))
    for name in ex.SECTOR_NAMES:
        generators = al.LABELS[al.SECTORS[name]]
        values = data[f"F_{name}_components"].reshape(2, len(generators), 6)
        assert len(ex.component_labels(name)) == values.shape[1] * 6
        coefficients = np.zeros((2, 6, 15))
        coefficients[:, :, al.SECTORS[name]] = values.transpose(0, 2, 1)
        blades = al.to_blades(coefficients)
        sector = al.sector(strength, name)
        for plane, (a, b) in enumerate(ex.PLANES):
            expected = np.asarray(sector(al.DIRECTIONS[a], al.DIRECTIONS[b]).kernel)
            assert np.abs(blades[:, plane] - expected).max() < 1e-10 * (1 + np.abs(expected).max())
        total += blades
    for plane, (a, b) in enumerate(ex.PLANES):
        expected = np.asarray(strength(al.DIRECTIONS[a], al.DIRECTIONS[b]).kernel)
        assert np.abs(total[:, plane] - expected).max() < 1e-10 * (1 + np.abs(expected).max())


def test_polylines_are_read_back_by_vtk(tmp_path):
    from vtkmodules.util.numpy_support import vtk_to_numpy
    points = np.array([[0.0, 0, 0], [1, 0, 0], [2, 1, 0], [0, 5, 5], [0, 6, 6]])
    lines = [np.array([0, 1, 2]), np.array([3, 4])]
    path = tmp_path / "rays.vtkhdf"
    ex.write_vtkhdf_polydata(path, points, lines, {"s": np.arange(5.0), "v": points * 2},
                             {"ray": np.array([7, 8])}, {"note": "test"})
    reader = vtk.vtkHDFReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    assert poly.GetNumberOfPoints() == 5 and poly.GetNumberOfLines() == 2
    read = np.array([poly.GetPoint(i) for i in range(5)])
    assert np.allclose(read, points)
    assert np.allclose(vtk_to_numpy(poly.GetPointData().GetArray("v")), points * 2)
    assert np.array_equal(vtk_to_numpy(poly.GetCellData().GetArray("ray")), [7, 8])
    ids = vtk.vtkIdList()
    poly.GetCellPoints(1, ids)
    assert [ids.GetId(i) for i in range(ids.GetNumberOfIds())] == [3, 4]


@pytest.mark.slow
def test_a_section_at_fixed_x_is_the_fields_at_its_events_and_matches_the_cut_grid(tmp_path):
    """The hypersurface x = c over (t, y, z), laid out as (z, t, y): its fields are those evaluated at its events,
    and cutting a 4D grid sampled on the same lines gives the same arrays."""
    from gauge_theory_gravity import hypersurface as hs
    coframe = ex.flyby(speed=0.6)
    grid = ex.box_grid((-3.0, 3.0), 4, 6.0)
    surface = hs.Hypersurface("x", float(grid.axes[1][1]), ("z", "t", "y"))
    direct = ex.export_sections(coframe, [surface], grid, tmp_path / "direct", "d", chunk=64, components=True)
    cut = ex.export_sections(coframe, [surface], grid, tmp_path / "grid", "g", chunk=64, components=True,
                             via_grid=True)
    assert direct.name == "d.sections.json" and cut.name == "g.sections.json"
    img, to_np = _read(tmp_path / "direct" / "d_0000.vtkhdf")
    other, _ = _read(tmp_path / "grid" / "g_0000.vtkhdf")
    pd, od = img.GetPointData(), other.GetPointData()
    for k in range(pd.GetNumberOfArrays()):
        name = pd.GetArrayName(k)
        assert np.abs(to_np(pd.GetArray(name)).astype(float) - to_np(od.GetArray(name)).astype(float)).max() < 1e-6
    full = to_np(pd.GetArray("shift_txyz"))
    shown = to_np(pd.GetArray("shift"))
    assert np.allclose(shown, full[:, [3, 0, 2]])                       # e(d_t) along z, t, y
    points = np.array([img.GetPoint(i) for i in range(img.GetNumberOfPoints())])   # (z, t, y)
    events = np.stack([points[:, 1], np.full(len(points), surface.coordinate), points[:, 2], points[:, 0]], axis=-1)
    expected = ex.evaluate_points(coframe, events, chunk=64)
    assert np.abs(to_np(pd.GetArray("riem2_half")) - expected["riem2_half"]).max() < 1e-5 * (
        1 + np.abs(expected["riem2_half"]).max())
