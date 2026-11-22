"""The ray tracer's physics on tiny scenes: conserved quantities along rays, the transported screen, the disk's
redshift and the polarization (unit intensity), for Schwarzschild (PG tetrad) and Kerr (Kerr-Schild gauge field)."""
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al
from gauge_theory_gravity.examples import light, render as R

HOLES = {"schwarzschild": R.schwarzschild_hole(), "kerr": R.kerr_hole(1.0, 0.9)}


@pytest.fixture(scope="module", params=list(HOLES))
def traced(request):
    hole = HOLES[request.param]
    scene, disk, _, _ = R.disk_scene(hole, 12, 9, 30.0, 0.5, 1800, 14.0, np.radians(15.0))
    return hole, scene, disk


def test_energy_and_angular_momentum_are_conserved_from_camera_to_disk(traced):
    hole, scene, _ = traced
    assert scene["hit"].sum() > 20
    assert R.conservation_error(hole, scene) < 5e-3


def test_screen_stays_orthonormal_and_across_the_ray(traced):
    _, scene, _ = traced
    hit = scene["hit"]
    _, k, screen = R.readout(scene["crossing"])
    s1, s2 = screen(al.mv.x.dual()), screen(al.mv.y.dual())
    read = lambda v: np.asarray(v.kernel)[..., 0][hit]
    assert np.abs(read(s1 | s1) + 1).max() < 5e-3 and np.abs(read(s2 | s2) + 1).max() < 5e-3
    for value in (s1 | s2, s1 | k, s2 | k):
        assert np.abs(read(value)).max() < 5e-3


def test_frequency_ratio_matches_conserved_quantities(traced):
    """1/g = u^t (k_t + Omega k_phi): the gas's frame velocity paired with the transported momentum, against the
    photon's energy and angular momentum taken at the camera."""
    hole, scene, disk = traced
    hit = scene["hit"]
    energy, angular = R.killing_momenta(hole, *R.readout(scene["start"])[:2])
    position = scene["crossing"][0]
    speed = light.keplerian_speed(disk.radius, hole.spin, hole.mass)
    flow = al.time + al.mv.scalar(speed[:, None]) * R.azimuth(position[hit])
    local = hole.geometry(position[hit])(flow)
    time_component = 1 / np.sqrt(np.asarray((local | local).kernel)[..., 0])
    expected = 1 / (time_component * (energy[hit] + speed * angular[hit]))
    assert np.abs(disk.ratio / expected - 1).max() < 5e-3
    assert disk.ratio.min() > 0.1 and disk.ratio.max() < 2


def test_polarization_has_unit_intensity_and_transmission_in_range(traced):
    _, scene, _ = traced
    hit = scene["hit"]
    coherency = R.disk_coherency(*R.readout(scene["crossing"]), 0.8)
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    passed = np.asarray(R.transmitted(coherency, R.analyzer_slices(angles)[:, None]).kernel)[..., 0][:, hit]
    assert np.abs(passed[:4] + passed[4:] - 1).max() < 1e-12
    assert passed.min() >= 0.1 - 1e-12 and passed.max() <= 0.9 + 1e-12


def test_local_twistor_stays_incident_with_the_ray(traced):
    """Local twistor transport by the normal conformal connection keeps omega = 0 along a null geodesic."""
    _, scene, _ = traced
    assert R.twistor_incidence(scene) < 5e-3


def test_camera_screen_read_from_the_twistor_is_the_cameras_right_and_up():
    """The camera turns each twistor's phase so that its flag reads screen x as the camera's right (and screen y as
    its up), each up to a multiple of the ray's momentum."""
    hole = HOLES["kerr"]
    (_, twistor), _ = R.static_camera(hole, 30.0, 0.5, 6, 4, np.radians(15.0))
    _, momentum, right, up, _ = R.camera_view(hole, 30.0, 0.5, 6, 4, np.radians(15.0))
    screen = R.screen_frame(twistor)
    k = np.asarray(al.coordinates(momentum))
    for axis, expected in ((al.mv.x.dual(), right), (al.mv.y.dual(), up)):
        difference = np.asarray(al.coordinates(screen(axis))) - np.asarray(al.coordinates(expected))
        along = difference[:, :1] / k[:, :1]
        assert np.abs(difference - along * k).max() < 1e-12


def test_page_thorne_temperature():
    assert abs(light.innermost_stable_orbit(0.0) - 6.0) < 1e-12
    radius = np.linspace(6.0, 20.0, 400)
    temperature = light.disk_temperature(radius, 0.0, 1.0)
    assert temperature[0] == 0 and temperature.argmax() > 0
    assert abs(radius[temperature.argmax()] - 9.55) < 0.1          # the Schwarzschild thin disk peaks near 9.5 M
