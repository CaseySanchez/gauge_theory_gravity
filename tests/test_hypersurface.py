"""The 4D grid and its hypersurfaces: ordering, extraction (on grid lines and between them), display permutations,
and the VTK files written from a section."""
import json

import numpy as np
import pytest

from gauge_theory_gravity import hypersurface as hs


def grid():
    return hs.Grid4.regular([(-1.0, 1.0), (0.0, 3.0), (-2.0, 2.0), (1.0, 2.0)], (3, 4, 5, 6))


def coordinate_fields(g):
    """Fields that are their own coordinates: a linear scalar, the position vector, a mask."""
    points = g.points().reshape(*g.shape, 4)
    linear = points @ np.array([1.0, 2.0, -3.0, 0.5]) + 7.0
    mask = (points[..., 1] > 1.0).astype(np.uint8)
    return {"linear": hs.Field4(g, linear), "position": hs.Field4(g, points, hs.VECTOR),
            "mask": hs.Field4(g, mask)}


def test_grid_order_shape_spacing_and_metadata():
    g = grid()
    assert g.shape == (3, 4, 5, 6)
    assert np.allclose(g.spacing, (1.0, 1.0, 1.0, 0.2)) and g.origin == (-1.0, 0.0, -2.0, 1.0)
    points = g.points()
    assert points.shape == (3 * 4 * 5 * 6, 4)
    assert np.allclose(points[1] - points[0], (0, 0, 0, 0.2))           # z fastest
    assert np.allclose(points[6] - points[0], (0, 0, 1.0, 0))
    meta = g.metadata()
    assert meta["coordinates"] == ["t", "x", "y", "z"] and meta["signature"] == [1, -1, -1, -1]
    assert meta["units"] == "M"
    with pytest.raises(ValueError):
        hs.Grid4((np.array([0.0, 1.0, 3.0]),) + g.axes[1:])          # uneven axis


@pytest.mark.parametrize("axis", hs.AXES)
def test_extraction_on_a_grid_line_is_the_grid_slice(axis):
    g = grid()
    rng = np.random.default_rng(0)
    values = rng.normal(size=g.shape)
    k = hs.axis_index(axis)
    index = 1
    surface = hs.Hypersurface(axis, float(g.axes[k][index]))
    section = hs.extract({"f": hs.Field4(g, values)}, surface)
    expected = np.take(values, index, axis=k).transpose(2, 1, 0)        # VTK order: last remaining axis slowest
    assert np.array_equal(section.data["f"], expected)
    assert section.dims == tuple(g.shape[hs.axis_index(name)] for name in surface.remaining)
    assert section.hypersurface.temporal == (axis == "t")


@pytest.mark.parametrize("axis", hs.AXES)
def test_extraction_between_grid_lines_interpolates_linear_fields_exactly(axis):
    g = grid()
    k = hs.axis_index(axis)
    c = 0.37 * g.axes[k][0] + 0.63 * g.axes[k][1]
    surface = hs.Hypersurface(axis, c)
    section = hs.extract(coordinate_fields(g), surface)
    positions = section.data["position_txyz"]
    assert np.allclose(positions[..., k], c)
    assert np.allclose(section.data["linear"], positions @ np.array([1.0, 2.0, -3.0, 0.5]) + 7.0)
    assert section.data["mask"].dtype == np.uint8
    with pytest.raises(ValueError):
        hs.extract(coordinate_fields(g), hs.Hypersurface(axis, g.axes[k][-1] + 1.0))


def test_display_order_permutes_the_layout_and_the_shown_vector_components_only():
    """Point (i, j, l) of a section with display axes (a1, a2, a3) lies at the a1, a2, a3 coordinates of index l, j, i;
    the shown vector lists the components along a1, a2, a3; the four-component array is the same physical vector."""
    g = grid()
    fields = coordinate_fields(g)
    canonical = hs.extract(fields, hs.Hypersurface("y", 0.0))
    for order in [("x", "t", "z"), ("z", "x", "t"), ("t", "z", "x")]:
        surface = hs.Hypersurface("y", 0.0, order)
        section = hs.extract(fields, surface)
        assert section.hypersurface.display == order
        axes = section.axes
        mesh = np.meshgrid(*axes, indexing="ij")                        # [first, second, third]
        full = section.data["position_txyz"]
        for slot, name in enumerate(order):
            along = mesh[slot].transpose(2, 1, 0)
            assert np.allclose(full[..., hs.axis_index(name)], along)
            assert np.allclose(section.data["position"][..., slot], along)
        assert section.info["position"].labels == order
        # the same events carry the same values, whatever the layout
        assert np.isclose(np.sort(section.data["linear"].ravel()), np.sort(canonical.data["linear"].ravel())).all()
    with pytest.raises(ValueError):
        hs.Hypersurface("y", 0.0, ("x", "y", "z"))


def test_section_points_cover_the_hypersurface():
    surface = hs.Hypersurface("z", 0.5)
    axes = {"t": np.array([0.0, 1.0]), "x": np.array([-1.0, 0.0, 1.0]), "y": np.array([2.0])}
    points, shape = hs.section_points(surface, axes)
    assert shape == (2, 3, 1) and points.shape == (6, 4)
    assert np.allclose(points[:, 3], 0.5) and np.allclose(points[:3, 1], [-1.0, 0.0, 1.0])


def test_section_files_are_read_back_by_vtk_in_display_order(tmp_path):
    vtk = pytest.importorskip("vtk")
    from vtkmodules.util.numpy_support import vtk_to_numpy
    from gauge_theory_gravity import export as ex
    g = grid()
    surface = hs.Hypersurface("x", 1.5, ("z", "t", "y"))
    section = hs.extract(coordinate_fields(g), surface)
    path = tmp_path / "s.vtkhdf"
    ex.write_section(path, section)
    reader = vtk.vtkHDFReader()
    reader.SetFileName(str(path))
    reader.Update()
    image = reader.GetOutput()
    assert image.GetDimensions() == section.dims
    points = np.array([image.GetPoint(i) for i in range(image.GetNumberOfPoints())])
    shown = vtk_to_numpy(image.GetPointData().GetArray("position"))
    assert np.allclose(shown, points, atol=1e-6)                       # (z, t, y) drawn along VTK's x, y, z
    full = vtk_to_numpy(image.GetPointData().GetArray("position_txyz"))
    assert np.allclose(full[:, 1], 1.5)
    import h5py
    with h5py.File(path) as f:
        meta = json.loads(f.attrs["metadata"])
    assert meta["fixed_axis"] == "x" and meta["display_axes"] == ["z", "t", "y"]
    assert meta["spatial_snapshot"] is False and meta["fields"]["position"]["labels"] == ["z", "t", "y"]


def test_only_spatial_snapshots_make_a_time_series(tmp_path):
    from gauge_theory_gravity import export as ex
    surfaces = [hs.Hypersurface("x", c) for c in (0.0, 1.0)]
    index = json.loads(ex.write_index(tmp_path, "run", surfaces).read_text())
    assert index["time_series"] is False
    assert [s["coordinate"] for s in index["sections"]] == [0.0, 1.0]
    assert not list(tmp_path.glob("*.series"))
