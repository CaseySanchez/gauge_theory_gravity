"""Spec 11.2: GTG's gauge freedom in the displacement field h, as CSTA gauge transformations.

Gauge theory gravity has two gauge freedoms: displacements (a change of the position coordinates, under which h and
the rotation gauge field are pulled back) and rotations (a local Lorentz rotor R).  Schwarzschild in Doran and
Lasenby's Newtonian (Painleve-Gullstrand) gauge and in the Kerr-Schild (ingoing Eddington-Finkelstein) gauge are two
such gauges of one solution.  The displacement is t_KS = t_PG + F(r) with

    F(r) = 4M ln((sqrt r + sqrt 2M) / sqrt 2M) - 2 sqrt(2 M r),    F'(r) = -s / (1 + s),   s = sqrt(2M/r),

(the difference of the two time coordinates, t_KS = t_S + 2M ln|r/2M - 1| and t_PG = t_S + 2 sqrt(2Mr)
+ 2M ln|(sqrt r - sqrt 2M)/(sqrt r + sqrt 2M)|, smooth across r = 2M), and the rotation is a radial boost of rapidity
ln(1 + s), worked out by hand from the two time covectors: e_KS^0 = (1 + s + s^2/2)/(1 + s) e^0 - (s + s^2/2)/(1 + s) e^r.

Checked: the pulled-back Kerr-Schild coframe is the PG coframe boosted by R, and the full CSTA connection (Lorentz
plus translation sector) pulled back is the PG connection transformed by R with the spec's law
Omega' = R Omega ~R - 2 (a . grad R) ~R.  Also Doran and Lasenby's closed form of the PG rotation gauge field.
"""
import numpy as np
import pytest

import gauge_theory_gravity  # noqa: F401
from gauge_theory_gravity import algebra as al, cartan, conformal as cf, fields as fl, gauge

M = 1.0
POINTS = {"outside": [0.3, 1.7, -0.9, 1.1], "inside": [0.2, 0.6, -0.5, 0.8]}     # r = 2.23 M and r = 1.12 M


def _s(x):
    return (2 * M / cartan.spatial_radius(x)).square_root()                        # [] Scalar+, sqrt(2M/r)


def displacement(x):
    """The position gauge transformation from PG to Kerr-Schild coordinates: (t, x) -> (t + F(r), x)."""
    r, s = cartan.spatial_radius(x), _s(x)
    return x + (4 * M * ((1 + s) / s).log() - 2 * (2 * M * r).square_root()) * cartan.time


def rotation(x):
    """The radial boost of rapidity ln(1 + sqrt(2M/r)) taking the PG frame to the pulled-back Kerr-Schild frame."""
    radial = (cartan.SpacePart(x) / cartan.spatial_radius(x)).dual_inverse()    # [] Vector, unit outward
    return ((al.t ^ radial) * (0.5 * (1 + _s(x)).log())).exp()                    # [] Rotor+


def _pulled_back(field):
    """A field of maps on directions, pulled back across the displacement: x -> field(phi(x))(D phi(x))."""
    jacobian = fl.derivative(displacement)

    def pulled(x):
        return field(displacement(x))(jacobian(x))
    return pulled


def test_displacement_derivative_is_the_closed_form():
    """D phi = identity + F'(r) d_t (r^ . ), F' = -s/(1+s), against autodiff."""
    x4 = np.array(POINTS["outside"])
    r = np.linalg.norm(x4[1:])
    s = np.sqrt(2 * M / r)
    expected = np.eye(4)
    expected[0, 1:] = -s / (1 + s) * x4[1:] / r
    assert np.abs(al.components(fl.derivative(displacement)(al.point(x4))) - expected).max() < 1e-14


@pytest.mark.parametrize("where", POINTS)
def test_kerr_schild_pulled_back_is_pg_boosted(where):
    """The displacement carries the metric of one gauge to the other, and the coframes differ by the boost R."""
    x = al.point(POINTS[where])
    pg, pulled = cartan.schwarzschild(M), _pulled_back(cartan.kerr(M, 0.0).coframe)
    assert np.abs(al.components(cartan.metric(pulled)(x)) - al.components(cartan.metric(pg)(x))).max() < 1e-14
    boosted = al.components(rotation(x) >> pg.coframe(x))
    assert np.abs(al.components(pulled(x)) - boosted).max() < 1e-14
    assert np.abs(boosted - al.components(pg.coframe(x))).max() > 0.1                # the boost is not trivial


@pytest.mark.parametrize("where", POINTS)
def test_full_connection_changes_gauge_by_the_spec_law(where):
    """Pulled-back Kerr-Schild CSTA connection == R-transform of the PG CSTA connection, translation sector included."""
    x = al.point(POINTS[where])
    pulled = _pulled_back(cf.connection(cartan.kerr(M, 0.0)))(x)
    transformed = gauge.transform_connection(cf.connection(cartan.schwarzschild(M)), rotation)(x)
    assert np.abs(al.components(pulled) - al.components(transformed)).max() < 1e-13
    assert np.abs(al.components(pulled) - al.components(cf.connection(cartan.schwarzschild(M))(x))).max() > 0.1


def test_pg_rotation_gauge_field_is_doran_lasenby_closed_form():
    """In the PG frame the rotation gauge field is a pure boost: Omega(E_0) = 0 and, along E_k, it turns d_0 into
    -B_ik d_i and d_i into -B_ik d_0, with B = grad beta = sqrt(2M/r^3) (1 - 3/2 r^ r^) (Cartan's first structure
    equation for e^i = dx^i + beta^i dt).  `commutator` with a direction is the covariant derivative's action."""
    x4 = np.array(POINTS["outside"])
    n = x4[1:] / np.linalg.norm(x4[1:])
    B = np.sqrt(2 * M / np.linalg.norm(x4[1:]) ** 3) * (np.eye(3) - 1.5 * np.outer(n, n))
    pg, x = cartan.schwarzschild(M), al.point(x4)
    omega, frame = cartan.lorentz_connection(pg)(x), pg.frame(x)
    for k in range(4):
        along = omega(frame(al.DIRECTIONS[k]))
        turns = np.stack([al.coordinates(along.commutator(al.DIRECTIONS[j])) for j in range(4)], axis=1)
        expected = np.zeros((4, 4))
        if k:
            expected[1:, 0] = expected[0, 1:] = -B[:, k - 1]
        assert np.abs(turns - expected).max() < 1e-14
