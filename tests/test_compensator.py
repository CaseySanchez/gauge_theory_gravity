"""Spec 11.4, first step: the scale tractor (infinity twistor) as the compensator of the conformal connection.

`compensator.scale_tractor` builds I = sigma n_inf + grad sigma + rho n_o from a scale sigma.  Checked: it is the
gauge image of n_inf under the Weyl-rescaling versor of `test_weyl_rescaling.py` (so it is Weyl-covariant), its norm
is -R/12 at sigma = 1 (the Einstein-Hilbert density, up to the factor), and it is parallel for the normal connection
exactly on Einstein metrics: Ricci-flat with I = n_inf, de Sitter with I . I = H^2 = Lambda/3.  For metrics that are
not conformally Einstein the integrability condition F x I = 0 leaves no candidate with sigma = I . n_o != 0.
"""
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, cartan, compensator as cp, conformal as cf
from test_weyl_rescaling import dilation, normal, rescaled, special_conformal, weight

H = 0.5
EINSTEIN = {
    "schw": (cartan.schwarzschild(1.0), [0.3, 1.7, -0.9, 1.1], 0.0),
    "kerr": (cartan.kerr(1.0, 0.9), [-0.4, 2.5, 0.8, -1.3], 0.0),
    "de_sitter": (cartan.de_sitter(H), [0.7, 0.4, -0.7, 0.9], H ** 2),
}
NOT_EINSTEIN = {
    "lapse": (cartan.static_lapse(), [1.3, 0.4, -0.7, 0.9]),
    "growing": (cartan.growing_mass(), [0.3, 1.7, -0.9, 1.1]),
}


def _size(e):
    return float(np.abs(np.asarray(e.kernel)).max())


def _sigma(x):
    return weight(x).exp()


def test_wave_operator_on_de_sitter():
    """box sigma = sigma_tt + 3 H sigma_t - e^{-2Ht} lap sigma, for sigma = e^{0.3 t} + 0.2 x^2 (signature + - - -)."""
    def sigma(x):
        return (0.3 * cartan.coordinate(x, 0)).exp() + 0.2 * cartan.coordinate(x, 1) ** 2

    t = 0.7
    expected = (0.09 + 0.9 * H) * np.exp(0.3 * t) - 0.4 * np.exp(-2 * H * t)
    box = al.scalar(cp.wave_operator(cartan.de_sitter(H), sigma)(al.point([t, 0.4, -0.7, 0.9])))
    assert abs(box - expected) < 1e-13


def test_scale_tractor_of_rescaled_schwarzschild_is_the_gauge_image_of_n_inf():
    """On e^lambda times Schwarzschild the Einstein scale is sigma = e^lambda; its tractor is V >> n_inf, parallel."""
    base, x = cartan.schwarzschild(1.0), al.point([0.3, 1.7, -0.9, 1.1])
    scaled = rescaled(base)
    tractor = cp.scale_tractor(scaled, _sigma)
    image = (dilation(x) * special_conformal(base)(x)) >> al.n_inf
    assert _size(tractor(x) - image) < 1e-14
    assert _size(cp.tractor_derivative(normal(scaled), tractor)(x)) < 1e-14
    assert _size(tractor(x) - al.n_inf) > 1e-3


def test_scale_tractor_is_weyl_covariant_off_shell():
    """On a metric that is not Einstein: I(e^lambda e, e^lambda sigma) = V >> I(e, sigma), so I . I is Weyl-invariant."""
    base, x = cartan.static_lapse(), al.point([1.3, 0.4, -0.7, 0.9])
    before = cp.scale_tractor(base, cp.unit_scale)(x)
    after = cp.scale_tractor(rescaled(base), _sigma)(x)
    assert _size(after - ((dilation(x) * special_conformal(base)(x)) >> before)) < 1e-13
    assert abs(al.scalar(after | after) - al.scalar(before | before)) < 1e-14


@pytest.mark.parametrize("name", [*NOT_EINSTEIN, "de_sitter"])
def test_unit_scale_norm_is_minus_R_over_12(name):
    geometry, x4 = {**NOT_EINSTEIN, **{k: v[:2] for k, v in EINSTEIN.items()}}[name]
    x = al.point(x4)
    tractor = cp.scale_tractor(geometry, cp.unit_scale)(x)
    ricci = al.scalar(cf.ricci_scalar(geometry, x))
    assert abs(al.scalar(tractor | tractor) + ricci / 12) < 1e-14 and abs(ricci) > 1e-3


@pytest.mark.parametrize("name", EINSTEIN)
def test_einstein_metrics_have_a_parallel_scale_tractor(name):
    geometry, x4, norm = EINSTEIN[name]
    x = al.point(x4)
    tractor = cp.scale_tractor(geometry, cp.unit_scale)
    assert _size(cp.tractor_derivative(normal(geometry), tractor)(x)) < 1e-13
    assert abs(al.scalar(tractor(x) | tractor(x)) - norm) < 1e-14                  # Lambda / 3
    if norm == 0:
        assert _size(tractor(x) - al.n_inf) < 1e-14


@pytest.mark.parametrize("name", NOT_EINSTEIN)
def test_metrics_not_conformally_einstein_have_no_scale(name):
    """The candidates F x I = 0 at x span only n_o, whose scale I . n_o vanishes; the unit scale is not parallel."""
    geometry, x4 = NOT_EINSTEIN[name]
    x = al.point(x4)
    candidates = cp.parallel_candidates(normal(geometry), x)
    assert candidates and max(abs(al.scalar(v | al.n_o)) for v in candidates) < 1e-9
    assert _size(cp.tractor_derivative(normal(geometry), cp.scale_tractor(geometry, cp.unit_scale))(x)) > 1e-2


def test_schwarzschild_candidates_include_a_scale():
    candidates = cp.parallel_candidates(normal(cartan.schwarzschild(1.0)), al.point([0.3, 1.7, -0.9, 1.1]))
    assert max(abs(al.scalar(v | al.n_o)) for v in candidates) > 0.5
