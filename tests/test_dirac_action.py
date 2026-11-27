"""Phase 1 of docs/PLAN_DIRAC.md: the Dirac density of a CSTA connection and its spinor field equation.

`dirac.density` is |det e| (Re h(Z, I Gamma^a D_a Z) - m h(Z, Z)), with the coframe read from DX = Omega x n_o and D the
full twistor transport ('full', the target) or its Lorentz sector ('lorentz', Lasenby, Doran and Gull's coupling, the
reference). Checked here:
  * in flat space the spinor equation is Dirac's, 2 Re h(U_k, (I Gamma^a d_a - m) Z), and plane waves solve it;
  * the full transport is the Lorentz one with the mass shifted by the compensators, m -> m - sqrt 2 (1 + k) for
    f = k e, whatever the dilation field (here with curl);
  * both densities are invariant under local Lorentz transformations of (Omega, Z).
"""
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, cartan, conformal as cf, dirac as dr, fields, gauge, twistor as tw

X4 = [0.3, 0.2, -0.5, 0.7]
X_HOLE = [0.3, 1.7, -0.9, 1.1]
MASS = 0.7
RAW = fields.polynomial(al.Even, seed=3, scale=0.4)


def spinor(x):
    return tw.twistor(RAW(x))


def flat(x):
    return (cartan.Direction & al.INFINITY).dual().cast(al.Bivector)


def dilation(x):
    return (0.2 * cartan.coordinate(x, 1) * (al.SLICES[0] & al.Direction)
            + 0.1 * cartan.coordinate(x, 0) * (al.SLICES[2] & al.Direction))                # d b != 0


def number(e):
    return float(np.asarray(e.kernel).reshape(-1)[0])


def test_coframe_is_read_from_the_translation_sector():
    geometry, x = cartan.schwarzschild(1.0), al.point(X_HOLE)
    omega = cf.connection(geometry, lambda y: 0.25 * geometry(y), dilation)
    assert np.abs(al.components(dr.coframe_of(omega(x))) - al.components(geometry(x))).max() < 1e-14


def test_flat_spinor_equation_is_dirac():
    x = al.point(X4)
    equations = dr.euler_lagrange_spinor(flat, spinor, X4, mass=MASS, which="lorentz")
    operator = dr.act(dr.I, dr.dirac_operator(flat, spinor, x, "lorentz")) - MASS * spinor(x)
    expected = [2 * tw.as_complex(dr.bar(unit, operator.cast(dr.Twistor))).real for unit in dr.UNITS]
    assert np.abs(equations - np.asarray(expected)).max() < 1e-12


def test_flat_plane_wave_solves_dirac():
    """Z = exp(-I p.x) (Gamma^a p_a + m) W with p^2 = m^2."""
    k = np.array([0.3, -0.2, 0.5])
    momentum = np.array([np.sqrt(MASS ** 2 + k @ k), *(-k)])                               # p_a, lowered
    amplitude = ((dr.GAMMA_UP * momentum).sum(axis=0) + MASS) * tw.from_coordinates([1, 0.5j, -0.3, 0.2 + 0.1j])

    def wave(x):
        phase = sum(momentum[m] * cartan.coordinate(x, m) for m in range(4))
        return ((phase.cos() - phase.sin() * dr.I) * amplitude).cast(dr.Twistor)

    assert np.abs(dr.euler_lagrange_spinor(flat, wave, X4, mass=MASS, which="lorentz")).max() < 1e-12


@pytest.mark.parametrize("k", [0.0, 0.25])
def test_full_transport_shifts_the_mass_by_the_compensators(k):
    geometry, x = cartan.schwarzschild(1.0), al.point(X_HOLE)
    omega = cf.connection(geometry, lambda y: k * geometry(y), dilation)
    shift = np.sqrt(2) * (1 + k)
    full = float(dr.density(omega, spinor, x, MASS, "full"))
    lorentz = float(dr.density(omega, spinor, x, MASS - shift, "lorentz"))
    assert abs(full - lorentz) < 1e-11 * max(1.0, abs(full))
    equations_full = dr.euler_lagrange_spinor(omega, spinor, X_HOLE, mass=MASS, which="full")
    equations_lorentz = dr.euler_lagrange_spinor(omega, spinor, X_HOLE, mass=MASS - shift, which="lorentz")
    assert np.abs(equations_full - equations_lorentz).max() < 1e-10


@pytest.mark.parametrize("which", dr.CONNECTIONS)
def test_density_is_invariant_under_local_lorentz_transformations(which):
    geometry, x = cartan.schwarzschild(1.0), al.point(X_HOLE)
    omega = cf.connection(geometry, lambda y: 0.25 * geometry(y), dilation)
    generators = fields.polynomial(al.SpacetimeBivector, seed=9, scale=0.3)

    def versor(y):
        return generators(y).cast(al.Bivector).exp()

    moved_omega = gauge.transform_connection(omega, versor)

    def moved_spinor(y):
        return (versor(y) * spinor(y)).cast(dr.Twistor)

    before = float(dr.density(omega, spinor, x, MASS, which))
    after = float(dr.density(moved_omega, moved_spinor, x, MASS, which))
    assert abs(before - after) < 1e-11 * max(1.0, abs(before))
