"""Spec 11.3: the dilation gauge field, through a Weyl rescaling of the metric.

A Weyl rescaling g -> e^{2 lambda} g (coframe e -> e^lambda e) changes the normal conformal connection only by a CSTA
gauge transformation, so the dilation and special-conformal sectors are gauge, not new degrees of freedom here:

    V = exp(-lambda/2 n_o ^ n_inf) exp(1/2 Upsilon ^ n_o),    Upsilon = grad lambda in the frame of e.

Taken one factor at a time: the dilation alone scales the translation sector by e^lambda and the special-conformal
sector by e^-lambda, leaves the Lorentz sector alone, and switches on the dilation sector, b = d lambda; the special
conformal factor then removes b and shifts the Lorentz sector and the Schouten tensor to those of the rescaled metric.
Weyl squared, a density of weight -4, scales by e^{-4 lambda}.
"""
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, cartan, conformal as cf, curvature as cv, fields as fl, gauge
from gauge_theory_gravity import invariants as iv

CASES = {
    "schw": (cartan.schwarzschild(1.0), [0.3, 1.7, -0.9, 1.1]),      # vacuum: the rescaled metric is not
    "lapse": (cartan.static_lapse(), [1.3, 0.4, -0.7, 0.9]),
}


def weight(x):
    """lambda(x), a smooth scalar depending on all four coordinates."""
    c = cartan.coordinate
    return 0.15 * (c(x, 0) + c(x, 1)).sin() + 0.1 * c(x, 2) * c(x, 3)


def rescaled(base):
    return cartan.Geometry(lambda x: weight(x).exp() * base.coframe(x), lambda x: base.frame(x) / weight(x).exp())


def dilation(x):
    return (cf.Dilation * (-0.5 * weight(x))).exp()


def special_conformal(base):
    """exp(1/2 Upsilon ^ n_o), Upsilon_a = E_a(lambda) raised with the flat metric to a frame vector."""
    gradient = fl.derivative(weight)

    def versor(x):
        frame, step = cartan.frame_at(base, x), gradient(x)
        upsilon = sum((al.ETA4[a, a] * step(frame(al.DIRECTIONS[a])) * al.SPACETIME[a] for a in range(4)),
                      0 * al.SPACETIME[0])                                        # [] Vector
        return ((0.5 * upsilon) ^ al.n_o).exp()
    return versor


def normal(geometry):
    return cf.connection(geometry, cf.schouten(geometry))


def _close(a, b):
    return np.abs(al.components(a) - al.components(b)).max()


@pytest.mark.parametrize("name", CASES)
def test_weyl_rescaling_is_a_gauge_transformation_of_the_normal_connection(name):
    base, x4 = CASES[name]
    x, scaled = al.point(x4), rescaled(base)

    def versor(x):
        return dilation(x) * special_conformal(base)(x)

    transformed = gauge.transform_connection(normal(base), versor)(x)
    assert _close(transformed, normal(scaled)(x)) < 1e-13
    assert _close(transformed, normal(base)(x)) > 1e-2
    strength = cv.curvature(normal(scaled))(x)
    assert al.sector_size(strength, "P") < 1e-13 and al.sector_size(strength, "D") < 1e-13     # still normal


@pytest.mark.parametrize("name", CASES)
def test_dilation_alone_switches_on_b_equal_to_d_lambda(name):
    base, x4 = CASES[name]
    x = al.point(x4)
    before = normal(base)(x)
    after = gauge.transform_connection(normal(base), dilation)(x)
    scale = np.exp(al.scalar(weight(x)))
    assert _close(al.sector(after, "M"), al.sector(before, "M")) < 1e-14
    assert _close(al.sector(after, "P"), al.sector(cf.connection(rescaled(base))(x), "P")) < 1e-14   # e^lambda e
    assert _close(al.sector(after, "K"), al.sector(before, "K") / scale) < 1e-14
    b = (fl.derivative(weight)(x) * cf.Dilation).cast(al.Bivector)
    assert _close(al.sector(after, "D"), b) < 1e-14 and al.sector_size(after, "D") > 1e-2
    assert al.sector_size(cv.curvature(gauge.transform_connection(normal(base), dilation))(x), "D") < 1e-13


@pytest.mark.parametrize("name", CASES)
def test_weyl_squared_has_weight_minus_four(name):
    base, x4 = CASES[name]
    x = al.point(x4)
    ratio = al.scalar(iv.weyl_squared(rescaled(base), x)) / al.scalar(iv.weyl_squared(base, x))
    assert abs(ratio * np.exp(4 * al.scalar(weight(x))) - 1) < 1e-12
