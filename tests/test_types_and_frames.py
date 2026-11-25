"""Orientation typing of the spacetime directions, and the closed-form frame fields of the geometries."""
import numpy as np
import pytest
from numga.gatype.errors import GATypeError

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, cartan, fields as fl

X = al.point([1.3, 0.4, -0.7, 0.9])
GEOMETRIES = {
    "schwarzschild": (cartan.schwarzschild(1.0), al.point([0.3, 1.7, -0.9, 1.1])),
    "growing_mass": (cartan.growing_mass(), al.point([0.3, 1.7, -0.9, 1.1])),
    "frw": (cartan.friedmann(), X),
    "de_sitter": (cartan.de_sitter(), X),
    "lapse": (cartan.static_lapse(), X),
    "flyby": (cartan.boosted(cartan.schwarzschild(1.0), 0.6), al.point([0.3, 1.7, -0.9, 1.1])),
    "kerr": (cartan.kerr(1.0, 0.7), al.point([0.3, 1.7, -0.9, 1.1])),
}


def test_directions_are_intrinsic_and_slices_read_them():
    assert al.Direction.orientation.name == "intrinsic" and al.Slice.orientation.name == "extrinsic"
    assert np.abs(np.asarray((al.SLICES & al.DIRECTIONS[:, None]).kernel)[..., 0] - np.eye(4)).max() == 0
    # The flat metric on directions, d_mu | d_nu = eta.
    assert np.abs(al.components(al.ALG.gatype(al.Direction) | al.ALG.gatype(al.Direction))[0] - al.ETA4).max() == 0
    assert np.abs(al.coordinates(al.point([0.3, -1.0, 2.0, 0.5])) - np.array([0.3, -1.0, 2.0, 0.5])).max() < 1e-15


def test_orientation_is_checked():
    """A slice is not a direction: adding them, or feeding a slice to the connection, fails before any arithmetic."""
    with pytest.raises(GATypeError):
        _ = X + al.t
    with pytest.raises(GATypeError):
        fl.random_connection_field(0)(X)(al.t)


def test_conformal_generators_from_joins_with_origin_and_infinity():
    for k, v in enumerate(al.SPACETIME):
        d = al.DIRECTIONS[k]
        assert np.abs(al._kernel((d & al.INFINITY).dual()) - al._kernel(v ^ al.n_inf)).max() < 1e-15
        assert np.abs(al._kernel((d & al.ORIGIN).dual()) - al._kernel(v ^ al.n_o)).max() < 1e-15


def test_slices_are_metric_free_duals_and_the_flat_metric_is_explicit():
    """The slices are fixed by the pairing alone (they come out as t, -x, -y, -z), and eta enters only through the
    duality map FLAT_METRIC, whose form eta(d_mu) & d_nu is the metric and which equals its own adjugate."""
    for k, (sign, vector) in enumerate(zip([1, -1, -1, -1], al.SPACETIME)):
        assert np.abs(np.asarray((al.SLICES[k] - sign * vector).kernel)).max() == 0
    assert al.FLAT_METRIC.gatype.orientation.name == "extrinsic"
    assert np.abs(al.components(al.FLAT_METRIC & al.Direction)[0] - al.ETA4).max() == 0
    assert np.abs(np.asarray((al.FLAT_METRIC - al.FLAT_METRIC.adjugate()).kernel)).max() == 0


@pytest.mark.parametrize("name", GEOMETRIES)
def test_metric_is_the_flat_duality_map_pulled_back_by_the_adjugate(name):
    """c & e(v) == e^adj(c) & v for any slice c, with no metric or inverse; g = e^adj eta e is symmetric as a duality
    map, g == g^adj, and its form g(v) & w is the metric e | e."""
    geometry, x = GEOMETRIES[name]
    e = geometry(x)
    reader, v = 0.4 * al.SLICES[0] - 1.3 * al.SLICES[2] + 0.7 * al.SLICES[3], al.point([0.2, -0.5, 1.1, 0.3])
    assert abs(al.scalar(reader & e(v)) - al.scalar(e.adjugate()(reader) & v)) < 1e-13
    g = cartan.metric_map(geometry)(x)
    assert np.abs(np.asarray((g - g.adjugate()).kernel)).max() < 1e-13
    assert np.abs(al.components(cartan.metric(geometry)(x))[0] - al.components(e | e)[0]).max() < 1e-13


@pytest.mark.parametrize("name", GEOMETRIES)
def test_closed_form_frame_inverts_the_coframe(name):
    geometry, x = GEOMETRIES[name]
    coframe, frame = al.components(geometry(x)), al.components(cartan.frame_at(geometry, x))
    assert np.abs(coframe @ frame - np.eye(4)).max() < 1e-13
    assert np.abs(frame - al.components(geometry(x).inverse())).max() < 1e-12
