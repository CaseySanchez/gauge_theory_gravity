"""Plan for 11.4, phase 3A: the compensator action and Einstein's equations.

S[e, sigma] = integral of sqrt|g| sigma^-4 (I . I + c), with I the scale tractor of sigma.  It is Weyl-invariant, so the
scale equation is the trace of the coframe equation (E_sigma sigma + E[a, mu] e^a_mu = 0, off shell).  In the gauge
sigma = 1 the coframe equation is -1/12 times Einstein-Hilbert's plus the cosmological term,

    E[a, mu] = sqrt|g| e_{a nu} (G^{mu nu}/6 + c g^{mu nu}),

so the field equations are G + 6 c g = 0: Einstein's equations with Lambda = -6 c (in this signature and sign of R,
de Sitter with Hubble rate H has G = 3 H^2 g), and on shell I . I = -R/12 = -2 c.  A Weyl-rescaled solution with its
scale, (e^lambda e, e^lambda), solves them too.
"""
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, cartan, compensator as cp, variation as vr
from gauge_theory_gravity._array import conformal as acf
from test_variation import einstein_reference
from test_weyl_rescaling import rescaled, weight

H = 0.5
X_LAPSE = [1.3, 0.4, -0.7, 0.9]
X_HOLE = [0.3, 1.7, -0.9, 1.1]
X_DS = [0.7, 0.4, -0.7, 0.9]


def _sigma(x):
    return weight(x).exp()


def _equations(geometry, scale, x4, cosmological=0.0):
    coframe, by_scale = vr.euler_lagrange_with_scale(vr.compensator_density(cosmological), geometry, scale, x4)
    sigma = float(np.asarray(scale(al.point(x4)).kernel).ravel()[0])
    noether = by_scale * sigma + np.sum(coframe * al.components(geometry(al.point(x4))))
    return coframe, by_scale, noether


def test_unit_scale_gives_einstein_equations():
    """At sigma = 1 the coframe equation is -1/12 of Einstein-Hilbert's: (1/6) sqrt(g) e G, against the array index
    computation; the scale equation is its trace."""
    coframe, by_scale, noether = _equations(cartan.static_lapse(), cp.unit_scale, X_LAPSE)
    reference = -einstein_reference(acf.static_lapse(), X_LAPSE) / 12
    assert np.abs(coframe - reference).max() < 1e-14 and np.abs(reference).max() > 1e-2
    assert abs(noether) < 1e-14 and abs(by_scale) > 1e-3


def test_de_sitter_solves_with_the_cosmological_term():
    """c = -H^2/2, i.e. Lambda = -6c = 3 H^2; the tractor norm on shell is H^2 = -2c."""
    coframe, by_scale, _ = _equations(cartan.de_sitter(H), cp.unit_scale, X_DS, -H ** 2 / 2)
    assert np.abs(coframe).max() < 1e-14 and abs(by_scale) < 1e-14


def test_weyl_rescaled_schwarzschild_with_its_scale_is_a_solution():
    """e^{2 lambda} g is not Einstein, but with sigma = e^lambda it is the same solution in another Weyl gauge."""
    coframe, by_scale, _ = _equations(rescaled(cartan.schwarzschild(1.0)), _sigma, X_HOLE)
    assert np.abs(coframe).max() < 1e-14 and abs(by_scale) < 1e-14


@pytest.mark.slow
def test_weyl_rescaled_schwarzschild_with_the_wrong_scale_is_not():
    coframe, _, _ = _equations(rescaled(cartan.schwarzschild(1.0)), cp.unit_scale, X_HOLE)
    assert np.abs(coframe).max() > 1e-3


@pytest.mark.slow
def test_de_sitter_without_the_cosmological_term_is_not_a_solution():
    coframe, by_scale, noether = _equations(cartan.de_sitter(H), cp.unit_scale, X_DS)
    assert np.abs(coframe).max() > 0.1 and abs(noether) < 1e-14


@pytest.mark.slow
def test_scale_equation_is_the_trace_off_shell():
    """The Noether identity of Weyl invariance with a nonconstant scale and c != 0, on a geometry that solves nothing."""
    coframe, by_scale, noether = _equations(cartan.static_lapse(), _sigma, X_LAPSE, 0.3)
    assert abs(noether) < 1e-14 and abs(by_scale) > 0.1


@pytest.mark.slow
def test_kerr_is_a_solution():
    coframe, by_scale, _ = _equations(cartan.kerr(1.0, 0.9), cp.unit_scale, [-0.4, 2.5, 0.8, -1.3])
    assert np.abs(coframe).max() < 1e-13 and abs(by_scale) < 1e-13
