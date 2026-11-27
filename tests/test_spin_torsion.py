"""Phase 5 of docs/PLAN_DIRAC.md: spin-torsion. The first check that separates Gauge Theory Gravity from general
relativity.

The CSTA action with Dirac matter is S1 + beta S2 + gamma S3 + 4 kappa |det e| L_D: Lasenby, Doran and Gull's
det(h)^-1 (1/2 R - kappa L_D) times -4, since 1/2 R = -S1/4 (`test_dirac_conventions`, `test_gtg_dirac_reference`).

  * The Levi-Civita connection plus LDG's contorsion, Omega(d_mu) += 1/2 kappa g_mu . S (their 4.41, 4.53), solves the
    Lorentz rows of the combined equations, under either transport; the opposite sign does not.
  * Its torsion is nonzero, T(a, b) = -n_o _| F_P(a, b) = kappa (a ^ b) . S on the frame: the torsion is the spin density
    (their 4.37, D ^ h_bar(a) = kappa S . h_bar(a)). Its trace vanishes, so the minimally coupled action gives the
    minimally coupled Dirac equation: LDG's consistency loop (4.32-4.35).
  * Beyond GTG, under the full transport: the special-conformal equation, 12 delta - 4 beta tr F delta + 4 beta F^T
    + 4 kappa C = 0 (S2 = -2 sqrt(g) ((tr F)^2 - F^a_b F^b_a) for F = f e^-1), is solved by
    F^b_a = (1/beta)(1 + kappa tr C / 3) delta^b_a - (kappa/beta) C^b_a, with tr C = sqrt 2 sigma: the Dirac field shifts
    f away from e/beta by its scalar density and its tensor bilinear.
"""
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, cartan, conformal as cf, curvature as cv, dirac as dr, first_order as fo
from test_gtg_dirac_reference import GEOMETRIES, csta_spinor

KAPPA, MASS = 0.3, 0.7
M = al.SECTORS["M"]


def with_spin(geometry, sign=1.0, special=None):
    """Levi-Civita, the coframe, f (default e/4), and the contorsion sign * 1/2 kappa g_mu . S."""
    base = cf.connection(geometry, special or (lambda q: 0.25 * geometry(q)))

    def omega(position):
        connection = base(position)
        spin = dr.spin_trivector(csta_spinor(position))
        legs = dr.coframe_of(connection)(al.DIRECTIONS).dual_inverse()                    # [4] Vector, g_mu
        contorsion = sum(((0.5 * KAPPA * sign * (legs[mu] * spin).restrict_grade(2)).cast(al.Bivector)
                          * (al.SLICES[mu] & al.Direction) for mu in range(4)), 0 * connection)
        return connection + contorsion
    return omega


def lorentz_rows(omega, x4, which):
    gravity = fo.euler_lagrange_connection(omega, x4)[0]                                  # S1; S2, S3 have no M rows
    matter = dr.euler_lagrange_connection(omega, csta_spinor, x4, mass=MASS, which=which)
    return gravity[M], 4 * KAPPA * matter[M]


@pytest.mark.parametrize("name", ["painleve_gullstrand", "kerr"])
@pytest.mark.parametrize("which", dr.CONNECTIONS)
def test_ldg_contorsion_solves_the_lorentz_equations(name, which):
    geometry, x4 = GEOMETRIES[name]
    gravity, matter = lorentz_rows(with_spin(geometry), x4, which)
    assert np.abs(matter).max() > 0.5
    assert np.abs(gravity + matter).max() < 1e-12


def test_opposite_contorsion_does_not():
    geometry, x4 = GEOMETRIES["kerr"]
    gravity, matter = lorentz_rows(with_spin(geometry, sign=-1.0), x4, "full")
    assert np.abs(gravity + matter).max() > 1.0


@pytest.mark.parametrize("name", ["painleve_gullstrand", "kerr"])
def test_torsion_is_the_spin_density_and_traceless(name):
    geometry, x4 = GEOMETRIES[name]
    x = al.point(x4)
    omega = with_spin(geometry)
    torsion = -(al.n_o | al.sector(cv.curvature(omega)(x), "P"))                          # Vector <- (a, b)
    frame = dr.frame_at(omega(x))
    spin = dr.spin_trivector(csta_spinor(x))
    largest = 0.0
    for a in range(4):
        for b in range(4):
            value = torsion(frame[a], frame[b]).cast(al.Vector)
            expected = KAPPA * ((al.SPACETIME[a] ^ al.SPACETIME[b]) * spin).restrict_grade(1)
            largest = max(largest, float(np.abs(np.asarray(value.kernel)).max()))
            assert np.abs(np.asarray((value - expected).kernel)).max() < 1e-12
    assert largest > 0.1
    assert np.abs(np.asarray(dr.torsion_trace(omega, x).kernel)).max() < 1e-13
    minimal = dr.act(dr.I, dr.dirac_operator(omega, csta_spinor, x, "lorentz")) - MASS * csta_spinor(x)
    lorentz = dr.spinor_equation(omega, csta_spinor, x, MASS, "lorentz")
    assert np.abs(np.asarray((lorentz - minimal).kernel)).max() < 1e-13


def test_s2_is_quadratic_in_the_mixed_special_conformal_field():
    geometry, x4 = GEOMETRIES["kerr"]
    x = al.point(x4)
    coframe = np.asarray(al.components(geometry(x)))
    rng = np.random.default_rng(2)
    for _ in range(3):
        special = al.CTX.extensor(cartan.CoframeType, rng.normal(size=(4, 4)) * 0.3)
        F = np.asarray(al.components(special)) @ np.linalg.inv(coframe)
        value = float(fo.densities(cf.connection(geometry, lambda q: special), x)[1])
        assert abs(value + 2 * abs(np.linalg.det(coframe)) * (np.trace(F) ** 2 - np.trace(F @ F))) < 1e-12


def test_dirac_field_shifts_the_special_conformal_field():
    beta, gamma = 4.0, 0.25
    geometry, x4 = GEOMETRIES["kerr"]
    x = al.point(x4)
    Z = csta_spinor(x)
    current = np.asarray(dr.special_conformal_current(Z).kernel).reshape(4, 4)                 # [a, b] = C^b_a
    sigma = float(np.asarray(dr.scalar_density(Z).kernel).reshape(-1)[0])
    assert abs(np.trace(current) - np.sqrt(2) * sigma) < 1e-12
    F = (1 + KAPPA * np.trace(current) / 3) / beta * np.eye(4) - KAPPA / beta * current.T          # [b, a] = F^b_a
    coframe = geometry(x)
    special = sum((float(F[a, b]) * al.DIRECTIONS[a] * (al.SLICES[b] & coframe) for a in range(4) for b in range(4)),
                  0 * coframe)
    omega = cf.connection(geometry, lambda q: special)
    gravity = fo.euler_lagrange_connection(omega, x4)
    matter = dr.euler_lagrange_connection(omega, csta_spinor, x4, mass=MASS, which="full")
    K = al.SECTORS["K"]
    assert np.abs(4 * KAPPA * matter[K]).max() > 1.0
    assert np.abs((gravity[0] + beta * gravity[1] + gamma * gravity[2] + 4 * KAPPA * matter)[K]).max() < 1e-12
