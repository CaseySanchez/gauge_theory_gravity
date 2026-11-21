"""Kerr in the CSTA: gauge theory gravity's Kerr-Schild position gauge field as the translation sector, its Levi-Civita
rotation gauge field as the Lorentz sector.  Cross-checked against numga's own Kerr example
(`numga/examples/relativity/kerr/core.py`, numga's STA `x+y+z+t-`, the opposite overall sign)."""
import sys
from pathlib import Path

import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, cartan, conformal as cf, curvature as cv, invariants as iv

M = 1.0
CASES = {"a=0.6": (0.6, [0.3, 1.7, -0.9, 1.1]), "a=0.9": (0.9, [-0.4, 2.5, 0.8, -1.3])}


def test_kerr_schild_form():
    """r, f and l against the coordinate formulas; l flat-null; g = eta - 2 f l l."""
    a, x4 = 0.6, np.array([0.3, 1.7, -0.9, 1.1])
    radius, profile, null = cartan.kerr_schild_fields(al.point(x4), M, a)
    _, x, y, z = x4
    rho2 = x * x + y * y + z * z
    r2 = 0.5 * (rho2 - a * a + np.sqrt((rho2 - a * a) ** 2 + 4 * a * a * z * z))
    r = np.sqrt(r2)
    outward = np.array([(r * x + a * y) / (r2 + a * a), (r * y - a * x) / (r2 + a * a), z / r])
    assert abs(al.scalar(radius) - r) < 1e-14 and abs(al.scalar(profile) - M * r ** 3 / (r2 ** 2 + a * a * z * z)) < 1e-14
    assert np.abs(al.coordinates(null) - np.r_[1.0, -outward]).max() < 1e-14
    assert abs(al.scalar(null.squared())) < 1e-15
    lowered = al.ETA4 @ np.r_[1.0, -outward]
    g = al.components(cartan.metric(cartan.kerr(M, a))(al.point(x4)))[0]
    assert np.abs(g - (al.ETA4 - 2 * al.scalar(profile) * np.outer(lowered, lowered))).max() < 1e-14


@pytest.mark.parametrize("name", CASES)
def test_kerr_is_torsion_free_vacuum_with_zero_normal_f_and_weyl_squared_kretschmann(name):
    a, x4 = CASES[name]
    geometry, x = cartan.kerr(M, a), al.point(x4)
    strength = cv.curvature(cf.connection(geometry))(x)
    for sector in "PKD":
        assert al.sector_size(strength, sector) < 1e-12                      # torsion-free, Cotton and dilation zero
    assert abs(al.scalar(cf.ricci_scalar(geometry, x))) < 1e-12
    f, residual, rank = cf.solve_normal_f(geometry, x)
    assert rank == 16 and residual < 1e-12 and np.abs(np.asarray(f.kernel)).max() < 1e-12
    density = al.scalar(iv.quadratic_invariant(geometry, cf.connection(geometry, cf.schouten(geometry)))(x))
    kretschmann = al.scalar(cartan.kerr_kretschmann(x, M, a))
    assert abs(2 * density / kretschmann - 1) < 1e-12                        # C^2 = Kretschmann in vacuum


def test_zero_spin_kerr_has_the_schwarzschild_kretschmann():
    x4 = np.array([0.3, 1.7, -0.9, 1.1])
    geometry = cartan.kerr(M, 0.0)
    density = al.scalar(iv.quadratic_invariant(geometry, cartan.lorentz_connection(geometry))(al.point(x4)))
    assert abs(2 * density - 48 * M ** 2 / np.linalg.norm(x4[1:]) ** 6) < 1e-12


@pytest.fixture(scope="module")
def numga_kerr():
    root = Path(__file__).resolve().parents[1] / "numga"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return pytest.importorskip("examples.relativity.kerr.core")


def _sta_reader(K):
    """Coordinates of an STA direction in numga's Kerr example: each read by its basis vector, normalised on its axis."""
    mv = K.mv
    axes = [K.time, mv.x.dual(), mv.y.dual(), mv.z.dual()]
    vectors = [mv.t, mv.x, mv.y, mv.z]
    zero = mv.antivector(np.zeros(4))

    def read(direction):
        direction = zero + direction
        return np.array([float(np.asarray((v & direction).kernel).ravel()[0]) / float(np.asarray((v & axis).kernel).ravel()[0])
                         for v, axis in zip(vectors, axes)])

    return axes, read


@pytest.mark.parametrize("name", CASES)
def test_matches_numga_kerr_gauge_field_and_rotation_gauge_field(numga_kerr, name):
    """numga's `gauge_inverse` is our coframe, and its closed-form rotation gauge bivector along e(d_mu) turns
    directions exactly as the Levi-Civita connection we obtain by differentiating the coframe.  Bivectors are
    compared by their action on directions, which is the same in either overall sign of the metric."""
    K = numga_kerr
    a, x4 = CASES[name]
    axes, read = _sta_reader(K)
    event = axes[0] * x4[0] + axes[1] * x4[1] + axes[2] * x4[2] + axes[3] * x4[3]
    _, profile, null, profile_gradient, null_gradient, null_curl = K.field_components(event, K.mv.xy.dual() * a, M)
    gauge_inverse = K.gauge_inverse(profile, null)
    theirs = np.stack([read(gauge_inverse(axis)) for axis in axes], axis=1)
    geometry, x = cartan.kerr(M, a), al.point(x4)
    assert np.abs(theirs - al.components(geometry(x))).max() < 1e-14
    connection = cartan.lorentz_connection(geometry)(x)
    for mu, axis in enumerate(axes):
        rotation = K.rotation(profile, null, profile_gradient, null_gradient, null_curl, gauge_inverse(axis))
        turns_theirs = np.stack([read(rotation.commutator(other)) for other in axes], axis=1)
        ours = connection(al.DIRECTIONS[mu])
        turns_ours = np.stack([al.coordinates(ours.commutator(al.DIRECTIONS[k])) for k in range(4)], axis=1)
        assert np.abs(turns_theirs - turns_ours).max() < 1e-13
        assert np.abs(turns_ours).max() > 1e-2
