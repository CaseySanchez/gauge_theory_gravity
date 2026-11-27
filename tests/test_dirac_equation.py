"""Phase 3 of docs/PLAN_DIRAC.md: the spinor field equation of the CSTA Dirac action, against Lasenby, Doran and Gull.

  * With the Lorentz coupling the CSTA spinor equation is the oracle's, blade by blade through `dirac.from_sta`, on
    Painleve-Gullstrand and Kerr with and without torsion: LDG's coupled Dirac equation (3.62).
  * In closed form it is I Gamma^a D_a Z + 1/2 I Gamma(tau) Z = m Z, with tau the torsion trace read from the translation
    sector of the curvature; tau is LDG's (3.65), and it vanishes without torsion, whatever the dilation field, so the
    minimally coupled equation follows.
  * The full transport adds the h-Hermitian part of what the other sectors contribute, an algebraic term: for
    f = k e it is the compensator mass, sqrt 2 (1 + k) Z.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, cartan, conformal as cf, dirac as dr, twistor as tw
import gtg_dirac_reference as ref
from test_dirac_action import dilation
from test_gtg_dirac_reference import BLADES, GEOMETRIES, TWIST, csta_spinor, lorentz_kernels, oracle_spinor, traceable

MASS = 0.7
SPECIAL = al.CTX.extensor(cartan.CoframeType, np.random.default_rng(8).normal(size=(4, 4)) * 0.2)   # with antisymmetric part
INDICES = [int(np.flatnonzero(np.abs(np.asarray(b.kernel)) > 0.5)[0]) for b in BLADES]
SIGNS = [float(np.asarray(b.kernel)[i]) for b, i in zip(BLADES, INDICES)]
# c = MAP psi: the real coordinates of from_sta(psi) along dirac.UNITS, for psi along BLADES.
MAP = np.stack([np.concatenate([tw.coordinates(dr.from_sta(b)).real, tw.coordinates(dr.from_sta(b)).imag])
                for b in BLADES], axis=1)


def connection(geometry, torsion, special=None, dilation_field=None):
    if special is None:
        special = lambda q: 0.25 * geometry(q)                                          # noqa: E731

    def omega(position):
        return cf.connection(geometry, special, dilation_field)(position) + torsion * TWIST
    return omega


def volume(geometry, x4):
    return abs(np.linalg.det(np.asarray(traceable(geometry)(jnp.asarray(x4)))))


def predicted(geometry, x4, equation):
    return np.array([2 * volume(geometry, x4) * tw.as_complex(dr.bar(u, equation)).real for u in dr.UNITS])


@pytest.mark.parametrize("name", ["painleve_gullstrand", "kerr"])
@pytest.mark.parametrize("torsion", [0, 1])
def test_lorentz_coupled_spinor_equation_is_ldg(name, torsion):
    geometry, x4 = GEOMETRIES[name]
    csta = dr.euler_lagrange_spinor(connection(geometry, torsion), csta_spinor, x4, mass=MASS, which="lorentz")
    oracle = ref.euler_lagrange(traceable(geometry), lorentz_kernels(geometry, torsion), oracle_spinor, x4, MASS)
    by_blade = np.array([oracle[i] * s for i, s in zip(INDICES, SIGNS)])
    assert np.abs(by_blade).max() > 1.0
    assert np.abs(by_blade - MAP.T @ csta).max() < 1e-13


@pytest.mark.parametrize("name", ["painleve_gullstrand", "kerr"])
def test_torsion_trace_is_ldg_and_blind_to_dilation(name):
    geometry, x4 = GEOMETRIES[name]
    x = al.point(x4)
    oracle = ref.coefficients(ref.torsion_trace(traceable(geometry), lorentz_kernels(geometry, 1), x4))
    csta = ref.coefficients(dr.torsion_trace(connection(geometry, 1, lambda q: SPECIAL, dilation), x))
    assert np.abs(oracle).max() > 1e-2 and np.abs(csta - oracle).max() < 1e-13
    free = ref.coefficients(dr.torsion_trace(connection(geometry, 0, lambda q: SPECIAL, dilation), x))
    assert np.abs(free).max() < 1e-13


@pytest.mark.parametrize("which", dr.CONNECTIONS)
def test_closed_form_equation(which):
    """General f (with an antisymmetric part), a dilation field with curl, and torsion."""
    geometry, x4 = GEOMETRIES["kerr"]
    omega = connection(geometry, 1, lambda q: 0.25 * geometry(q) + SPECIAL, dilation)
    equations = dr.euler_lagrange_spinor(omega, csta_spinor, x4, mass=MASS, which=which)
    expected = predicted(geometry, x4, dr.spinor_equation(omega, csta_spinor, al.point(x4), MASS, which))
    assert np.abs(equations).max() > 1.0
    assert np.abs(equations - expected).max() < 1e-12


def test_torsion_free_equation_is_minimally_coupled():
    geometry, x4 = GEOMETRIES["kerr"]
    x = al.point(x4)
    omega = connection(geometry, 0, dilation_field=dilation)
    lorentz = dr.spinor_equation(omega, csta_spinor, x, MASS, "lorentz")
    minimal = dr.act(dr.I, dr.dirac_operator(omega, csta_spinor, x, "lorentz")) - MASS * csta_spinor(x)
    assert np.abs(np.asarray((lorentz - minimal).kernel)).max() < 1e-13


@pytest.mark.parametrize("k", [0.0, 0.25])
def test_full_transport_adds_the_compensator_mass(k):
    geometry, x4 = GEOMETRIES["painleve_gullstrand"]
    x = al.point(x4)
    omega = connection(geometry, 1, lambda q: k * geometry(q), dilation)
    difference = dr.spinor_equation(omega, csta_spinor, x, MASS, "full") - dr.spinor_equation(omega, csta_spinor, x, MASS, "lorentz")
    assert np.abs(np.asarray((difference - np.sqrt(2) * (1 + k) * csta_spinor(x)).kernel)).max() < 1e-13
