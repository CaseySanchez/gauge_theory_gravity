"""Ray paths with their twistors, under the extrinsic transport (the accumulated versor) and the intrinsic one (the
twistor itself): the invariants each keeps, their agreement, convergence to a fine reference, and the PolyData file."""
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, twistor as tw
from gauge_theory_gravity.examples import rays, render as R

HOLE = R.schwarzschild_hole()


def start():
    return rays.camera_rays(HOLE, 5, 1, 30.0, 0.5, np.radians(15.0))


@pytest.fixture(scope="module")
def paths():
    return rays.trace_paths(HOLE, start(), 6, 10, base_step=0.2, adaptive=False)


@pytest.fixture(scope="module")
def observed(paths):
    return rays.observables(HOLE, paths)


@pytest.fixture(scope="module")
def intrinsic():
    return rays.trace_paths(HOLE, start(), 6, 10, base_step=0.2, adaptive=False, transport="intrinsic")


def test_records_and_affine_parameter(observed):
    o = observed
    assert o["coordinates"].shape == (7, 5, 4) and o["lambda"].shape == (7, 5)
    assert np.allclose(o["lambda"][:, 0], -0.2 * 10 * np.arange(7))          # fixed steps, backward
    assert np.allclose(o["redshift"][0], 0.0) and np.allclose(o["energy_drift"][0], 0.0)


def test_the_twistor_and_its_momentum_stay_null_and_on_the_ray(observed):
    o = observed
    assert np.abs(o["nullity"]).max() < 1e-12
    assert np.abs(o["null_residual"]).max() < 1e-10 * np.abs(o["momentum"]).max() ** 2
    assert o["incidence"].max() < 1e-9                                        # omega drifts at O(h^4) only


def test_versor_stays_in_the_group_and_starts_at_the_identity(observed):
    o = observed
    assert o["versor_residual"].max() < 1e-10
    assert np.allclose(o["transport_trace"][0], [1.0, 0.0])
    assert np.allclose(o["relative_trace"][:, 0], [1.0, 0.0])                 # the first ray against itself
    frame = np.stack([o["frame_" + name] for name, _ in rays.FRAME], axis=-2)  # [records, rays, 6, 6]
    gram = np.einsum("...ia,ab,...jb->...ij", frame, al.ETA6, frame)
    assert np.abs(gram - al.ETA6).max() < 1e-10                               # U e_a ~U stays orthonormal


def test_versor_transport_carries_the_camera_twistor(paths):
    """Z = U Z0 is the camera twistor at the first record and, later on, agrees with a run at half the step to the
    integrator's truncation error."""
    twistors = rays.twistors_of(paths)
    first = np.asarray(tw.coordinates(twistors[0])) - np.asarray(tw.coordinates(start()[1]))
    assert np.abs(first).max() < 1e-14
    fine = rays.twistors_of(rays.trace_paths(HOLE, start(), 6, 20, base_step=0.1, adaptive=False))
    later = np.abs(tw.coordinates(twistors[-1]) - tw.coordinates(fine[-1])).max()
    assert later < 1e-8 * np.abs(tw.coordinates(fine[-1])).max()


def test_intrinsic_transport_keeps_the_twistor_on_the_ray_and_forms_no_versor(intrinsic):
    """The additive step builds each stage's generator from the twistor it moves, so omega stays zero to round-off,
    and with it h(Z, Z) = 2 Re(conj(omega) . pi), though the step is not unitary."""
    o = rays.observables(HOLE, intrinsic)
    assert intrinsic["versors"] is None and "versor" not in o and "versor_residual" not in o
    assert np.abs(tw.coordinates(intrinsic["twistors"][0]) - tw.coordinates(start()[1])).max() < 1e-15
    assert o["incidence"].max() < 1e-12
    assert np.abs(o["nullity"]).max() < 1e-12
    assert np.abs(o["energy_drift"]).max() < 1e-6 and np.abs(o["angular_momentum_drift"]).max() < 1e-6


def test_intrinsic_and_extrinsic_transports_agree(paths, intrinsic):
    """The two integrate one equation, dZ = A Z: the events and twistors agree to truncation error."""
    assert paths["transport"] == "extrinsic" and intrinsic["transport"] == "intrinsic"
    events = np.abs(al.coordinates(paths["events"][-1]) - al.coordinates(intrinsic["events"][-1])).max()
    extrinsic_twistor = tw.coordinates(rays.twistors_of(paths)[-1])
    relative = np.abs(tw.coordinates(rays.twistors_of(intrinsic)[-1]) - extrinsic_twistor).max()
    assert events < 1e-8 and relative < 1e-8 * np.abs(extrinsic_twistor).max()


def test_paths_file(tmp_path, paths):
    vtk = pytest.importorskip("vtk")
    from vtkmodules.util.numpy_support import vtk_to_numpy
    path = tmp_path / "rays.vtkhdf"
    rays.write_paths(path, HOLE, paths, ("t", "x", "z"))
    reader = vtk.vtkHDFReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    assert poly.GetNumberOfLines() == 5 and poly.GetNumberOfPoints() == 35
    points = np.array([poly.GetPoint(i) for i in range(poly.GetNumberOfPoints())])
    coordinates = vtk_to_numpy(poly.GetPointData().GetArray("coordinates"))
    assert np.allclose(points, coordinates[:, [0, 1, 3]], atol=1e-4 * np.abs(coordinates).max())
    for name in ("momentum", "screen_x", "omega_re", "pi_im", "versor", "frame_p", "relative_trace"):
        assert poly.GetPointData().GetArray(name) is not None, name


@pytest.mark.slow
def test_versor_transport_converges_at_fourth_order():
    """Positions at four records over lambda = -16, against steps of 0.025: halving the step cuts the error
    sixteenfold, and the omega drift falls with it."""
    def run(size, every):
        return rays.observables(HOLE, rays.trace_paths(HOLE, start(), 4, every, size, adaptive=False))

    reference = run(0.025, 160)["coordinates"]
    coarse, fine = run(0.4, 10), run(0.2, 20)
    errors = [np.abs(o["coordinates"] - reference).max() for o in (coarse, fine)]
    assert 12 < errors[0] / errors[1] < 20, errors
    assert 12 < coarse["incidence"].max() / fine["incidence"].max() < 20
