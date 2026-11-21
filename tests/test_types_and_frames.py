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


@pytest.mark.parametrize("name", GEOMETRIES)
def test_closed_form_frame_inverts_the_coframe(name):
    geometry, x = GEOMETRIES[name]
    coframe, frame = al.components(geometry(x)), al.components(cartan.frame_at(geometry, x))
    assert np.abs(coframe @ frame - np.eye(4)).max() < 1e-13
    assert np.abs(frame - al.components(geometry(x).inverse())).max() < 1e-12
