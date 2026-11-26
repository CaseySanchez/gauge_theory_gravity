"""Plan for 11.4, phase 4: does adding the quadratic (Weyl) action change the theory?

* Second order: the compensator action plus alpha sqrt(g) C^2. Einstein metrics still solve it (Bach-flat), and so do
  their Weyl rescalings with their scale, since sqrt(g) C^2 is Weyl-invariant. Linearized about flat space, a
  transverse-traceless plane wave e = 1 + eps H cos(k.x) gets L = -2 k^2 H from sqrt(g) R and 4 k^4 H from sqrt(g) C^2
  (k^2 = omega^2 - |q|^2). With Einstein-Hilbert's coefficient -1/12 in the compensator action,
  L = (k^2/6)(1 + 24 alpha k^2), and the propagator 6 [1/k^2 - 1/(k^2 - m^2)], m^2 = -1/(24 alpha), has a massive
  spin-2 pole with the opposite residue: Stelle's ghost, non-tachyonic only for alpha < 0. The scale adds no mode
  (its equation is the trace), so the compensator does not remove the ghost; alpha = 0 is general relativity.
* First order: S1 + beta S2 + gamma S3 + mu <F^F^X^I>. Einstein metrics still solve it, with f = k e,
  k = (1 - 4 mu gamma) / beta and Lambda = 6 (gamma - k): the quadratic term only shifts the special-conformal field.
"""
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import cartan, conformal as cf, first_order as fo, variation as vr
from test_weyl_rescaling import rescaled

H = np.zeros((4, 4))
H[1, 1], H[2, 2] = 1.0, -1.0                                              # TT, "plus", along z
POLARIZATION = (vr.UNITS * H).sum(axis=(0, 1))
Q = 0.7
EPS = 1e-4


def _wave(omega, amplitude):
    def coframe(x):
        phase = omega * cartan.coordinate(x, 0) - Q * cartan.coordinate(x, 3)
        return cartan.Direction + (amplitude * phase.cos()) * POLARIZATION
    return coframe


def linear_response(density, omega):
    """d E / d eps at x = 0 (where cos(k.x) = 1), by a central difference."""
    plus = vr.euler_lagrange(density, _wave(omega, EPS), [0.0, 0.0, 0.0, 0.0])
    minus = vr.euler_lagrange(density, _wave(omega, -EPS), [0.0, 0.0, 0.0, 0.0])
    return (plus - minus) / (2 * EPS)


def _first_order_mixture(geometry, x4, k, beta, gamma, mu):
    omega = cf.connection(geometry, lambda x: k * geometry(x))
    terms = fo.euler_lagrange_connection(omega, x4)
    quadratic = fo.euler_lagrange_connection(omega, x4, fo.macdowell_mansouri)
    return terms[0] + beta * terms[1] + gamma * terms[2] + mu * quadratic


# --- first order -----------------------------------------------------------------------------------------------------
def test_quadratic_term_keeps_schwarzschild_with_a_shifted_special_conformal_field():
    mu, gamma, beta = 0.5, 0.25, 2.0                                       # Lambda = 0: gamma = k
    k = (1 - 4 * mu * gamma) / beta
    assert k == gamma
    equations = _first_order_mixture(cartan.schwarzschild(1.0), [0.3, 1.7, -0.9, 1.1], k, beta, gamma, mu)
    assert np.abs(equations).max() < 1e-13


@pytest.mark.slow
def test_quadratic_term_keeps_de_sitter():
    mu, gamma, beta, hubble = 1.0, 0.625, -3.0, 0.5
    k = (1 - 4 * mu * gamma) / beta
    assert abs(6 * (gamma - k) - 3 * hubble ** 2) < 1e-15                 # Lambda = 3 H^2
    equations = _first_order_mixture(cartan.de_sitter(hubble), [0.7, 0.4, -0.7, 0.9], k, beta, gamma, mu)
    assert np.abs(equations).max() < 1e-13


# --- second order ----------------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_weyl_squared_vanishes_on_weyl_rescaled_schwarzschild():
    """Bach-flatness is Weyl-invariant, so the rescaled solution of the compensator action solves the mixture too."""
    assert np.abs(vr.euler_lagrange(vr.weyl_squared, rescaled(cartan.schwarzschild(1.0)), [0.3, 1.7, -0.9, 1.1])).max() < 1e-13


@pytest.mark.slow
def test_linearized_einstein_hilbert_is_k_squared():
    omega = 1.0
    k2 = omega ** 2 - Q ** 2
    response = linear_response(vr.einstein_hilbert, omega)
    assert np.abs(response - (-2 * k2) * H).max() < 1e-8


@pytest.mark.slow
def test_linearized_weyl_squared_is_k_to_the_fourth():
    omega = 1.3
    k2 = omega ** 2 - Q ** 2
    response = linear_response(vr.weyl_squared, omega)
    assert np.abs(response - 4 * k2 ** 2 * H).max() < 1e-6


@pytest.mark.slow
def test_null_waves_are_massless_for_both():
    for density in (vr.einstein_hilbert, vr.weyl_squared):
        assert np.abs(linear_response(density, Q)).max() < 1e-9


def test_massive_mode_is_a_ghost():
    """From the two linear responses (closed forms checked above): the compensator action with alpha C^2 has
    L(k^2) = -2 (-1/12) k^2 + 4 alpha k^4; its massive pole has the residue of opposite sign to the graviton's."""
    alpha = -0.1
    mass_squared = -1 / (24 * alpha)

    def inverse(k2):
        return 1 / ((-1 / 12) * (-2 * k2) + alpha * 4 * k2 ** 2)

    small = 1e-6
    graviton = small * inverse(small)
    massive = small * inverse(mass_squared + small)
    assert mass_squared > 0 and graviton > 0 and massive < 0 and abs(graviton + massive) < 1e-4
