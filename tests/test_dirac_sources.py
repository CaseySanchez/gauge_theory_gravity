"""Phase 4 of docs/PLAN_DIRAC.md: the matter sources of the gravitational equations, the Dirac part of
delta S / delta Omega^k_mu sector by sector, against Lasenby, Doran and Gull.

  * Lorentz rows: |det e| S . g^mu, S = 1/2 psi I gamma_3 ~psi the spin trivector, g^mu = h_bar(e^mu): their spin tensor
    S(a) = S . h_bar(a) (4.27, 4.33, 4.37). The same under either transport.
  * Translation rows: the oracle's d(det(h)^-1 L)/d e (4.16, 4.20), and in closed form -|det e| E^mu_b T^b_a with
    T(a) = <a.D psi I gamma_3 ~psi>_1 - a L. Under the full transport D omits the translations, whose action on the
    spinor does not depend on the coframe, and L carries the compensator terms.
  * Special-conformal rows: zero with the Lorentz coupling; |det e| E^mu_b C^b_a with the full transport, C the
    special-conformal current, its symmetric part (sqrt 2 / 4) sigma eta and the rest the tensor bilinear.
  * Dilation rows: zero under either transport.
"""
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, dirac as dr
import gtg_dirac_reference as ref
from test_dirac_action import dilation
from test_dirac_equation import SPECIAL, connection, volume
from test_gtg_dirac_reference import GEOMETRIES, csta_spinor, lorentz_kernels, oracle_spinor, traceable

MASS = 0.7
M, P, K, D = (al.SECTORS[s] for s in "MPKD")


def setup(name="kerr", torsion=1):
    geometry, x4 = GEOMETRIES[name]
    omega = connection(geometry, torsion, lambda q: 0.25 * geometry(q) + SPECIAL, dilation)
    return geometry, x4, al.point(x4), omega


def size(e):
    return float(np.abs(np.asarray(e.kernel)).max())


@pytest.fixture(scope="module", params=dr.CONNECTIONS)
def sources(request):
    geometry, x4, x, omega = setup()
    return request.param, dr.euler_lagrange_connection(omega, csta_spinor, x4, mass=MASS, which=request.param)


def test_native_spin_trivector_is_ldg():
    x4 = GEOMETRIES["kerr"][1]
    S = dr.spin_trivector(csta_spinor(al.point(x4)))
    assert size(S) > 0.5 and size(S - ref.spin_trivector(oracle_spinor, x4)) < 1e-13


def test_lorentz_rows_are_the_spin_tensor(sources):
    which, equations = sources
    geometry, x4, x, omega = setup()
    S = dr.spin_trivector(csta_spinor(x))
    up = dr.reciprocal_coordinate_frame(omega(x))
    for mu in range(4):
        rows = sum((equations[M][k, mu] * al.RECIPROCALS[k] for k in range(6)), 0 * al.GENS[0])
        assert size(rows - volume(geometry, x4) * (S * up[mu]).restrict_grade(2)) < 1e-12


def test_lorentz_translation_rows_are_the_oracle_h_equation_source():
    for name in ("painleve_gullstrand", "kerr"):
        geometry, x4 = GEOMETRIES[name]
        equations = dr.euler_lagrange_connection(connection(geometry, 1), csta_spinor, x4, mass=MASS, which="lorentz")
        oracle = ref.coframe_derivative(traceable(geometry), lorentz_kernels(geometry, 1), oracle_spinor, x4, MASS)
        assert np.abs(oracle).max() > 1.0 and np.abs(equations[P] - oracle).max() < 1e-12


def test_translation_rows_are_the_stress_energy(sources):
    which, equations = sources
    geometry, x4, x, omega = setup()
    stress = np.asarray(dr.stress_energy(omega, csta_spinor, x, MASS, which).kernel).reshape(4, 4)     # [a, b]
    inverse = np.asarray(al.components(dr.coframe_of(omega(x)).inverse()))                            # [mu, a]
    expected = -volume(geometry, x4) * np.einsum("mb,ab->am", inverse, stress)
    assert np.abs(equations[P] - expected).max() < 1e-12


def test_special_conformal_and_dilation_rows(sources):
    which, equations = sources
    geometry, x4, x, omega = setup()
    assert np.abs(equations[D]).max() < 1e-12
    if which == "lorentz":
        assert np.abs(equations[K]).max() < 1e-12
        return
    current = np.asarray(dr.special_conformal_current(csta_spinor(x)).kernel).reshape(4, 4)          # [a, b]
    inverse = np.asarray(al.components(dr.coframe_of(omega(x)).inverse()))
    expected = volume(geometry, x4) * np.einsum("mb,ab->am", inverse, current)
    assert np.abs(equations[K]).max() > 0.1 and np.abs(equations[K] - expected).max() < 1e-12


def test_special_conformal_current_splits_into_mass_and_tensor_parts():
    Z = csta_spinor(al.point(GEOMETRIES["kerr"][1]))
    lowered = dr.SQUARES[:, None] * np.asarray(dr.special_conformal_current(Z).kernel).reshape(4, 4).T  # C_ab
    sigma = float(np.asarray(dr.scalar_density(Z).kernel).reshape(-1)[0])
    assert np.abs(0.5 * (lowered + lowered.T) - np.sqrt(2) / 4 * sigma * al.ETA4).max() < 1e-12
    assert np.abs(0.5 * (lowered - lowered.T)).max() > 1e-2


def test_lorentz_rows_do_not_depend_on_the_transport():
    geometry, x4, x, omega = setup()
    full = dr.euler_lagrange_connection(omega, csta_spinor, x4, mass=MASS, which="full")
    lorentz = dr.euler_lagrange_connection(omega, csta_spinor, x4, mass=MASS, which="lorentz")
    assert np.abs(full[M] - lorentz[M]).max() < 1e-12
